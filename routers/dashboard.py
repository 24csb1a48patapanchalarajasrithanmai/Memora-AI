"""
routers/dashboard.py - Fixed: all analytics endpoints the frontend expects
"""
from fastapi import APIRouter
from database import SessionLocal, QuizAttempt, TopicPerformance, QASession
from rl_agent import get_performance_summary
from datetime import datetime, timedelta
import numpy as np

router = APIRouter()


@router.get("/summary")
async def summary():
    return get_performance_summary()


@router.get("/timeline")
async def timeline(days: int = 30):
    db = SessionLocal()
    since = datetime.utcnow() - timedelta(days=days)
    attempts = (db.query(QuizAttempt).filter(QuizAttempt.created_at >= since)
                  .order_by(QuizAttempt.created_at.asc()).all())
    db.close()
    daily = {}
    for a in attempts:
        k = a.created_at.strftime("%Y-%m-%d")
        daily.setdefault(k, {"correct": 0, "total": 0})
        daily[k]["total"] += 1
        if a.is_correct: daily[k]["correct"] += 1
    return {"timeline": [{"date": k, "accuracy": round(v["correct"]/v["total"]*100,1) if v["total"] else 0,
                           "total": v["total"]} for k, v in sorted(daily.items())]}


@router.get("/topic_breakdown")
async def topic_breakdown():
    db = SessionLocal()
    all_tp = db.query(TopicPerformance).all()
    db.close()
    return {"topics": [{"topic": t.topic[:30], "accuracy": round(t.accuracy*100,1),
                         "attempts": t.attempts, "mastery_level": t.mastery_level,
                         "avg_time": round(t.avg_time,1)} for t in all_tp]}


@router.get("/topic_improvement")
async def topic_improvement(topic: str):
    db = SessionLocal()
    attempts = (db.query(QuizAttempt).filter_by(topic=topic)
                  .order_by(QuizAttempt.created_at.asc()).all())
    db.close()
    progress, correct = [], 0
    for i, a in enumerate(attempts):
        if a.is_correct: correct += 1
        progress.append(round(correct/(i+1)*100, 1))
    return {"topic": topic, "progress": progress}


@router.get("/train_test_split")
async def train_test_split():
    db = SessionLocal()
    attempts = db.query(QuizAttempt).order_by(QuizAttempt.created_at.asc()).all()
    db.close()
    if not attempts:
        return {"train_accuracy": 0, "test_accuracy": 0, "train_count": 0,
                "test_count": 0, "total": 0, "learning_efficiency": 0}
    split = int(len(attempts) * 0.7)
    train, test = attempts[:split], attempts[split:]
    def acc(lst): return round(sum(1 for a in lst if a.is_correct)/len(lst)*100,1) if lst else 0
    ta, tea = acc(train), acc(test)
    return {"train_accuracy": ta, "test_accuracy": tea,
            "train_count": len(train), "test_count": len(test),
            "total": len(attempts), "learning_efficiency": round(tea-ta,1)}


@router.get("/difficulty_progression")
async def difficulty_progression():
    db = SessionLocal()
    attempts = db.query(QuizAttempt).order_by(QuizAttempt.created_at.asc()).all()
    db.close()
    dm = {"easy": 1, "medium": 2, "hard": 3}
    return {"progression": [{"attempt": i+1, "difficulty_num": dm.get(a.difficulty,2),
                              "difficulty": a.difficulty, "is_correct": a.is_correct,
                              "topic": a.topic[:20]} for i, a in enumerate(attempts)]}


@router.get("/qa_stats")
async def qa_stats():
    db = SessionLocal()
    sessions = db.query(QASession).all()
    db.close()
    total = len(sessions)
    with_fb = [s for s in sessions if s.feedback_relevant is not None]
    relevant = sum(1 for s in with_fb if s.feedback_relevant)
    qd = {"good": 0, "average": 0, "poor": 0}
    for s in sessions:
        if s.feedback_quality in qd: qd[s.feedback_quality] += 1
    return {"total_questions": total, "with_feedback": len(with_fb),
            "relevance_rate": round(relevant/len(with_fb)*100,1) if with_fb else 0,
            "quality_distribution": qd}
