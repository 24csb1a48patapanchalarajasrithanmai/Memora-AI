"""
rl_agent.py - Fixed:
  - get_common_mistakes now imports QuizAttempt properly inside the function
  - get_performance_summary() added for dashboard
  - get_topics_due_for_retest() added
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from database import TopicPerformance, SessionLocal

ALPHA = 0.3; GAMMA = 0.9; EPSILON_START = 0.3; EPSILON_MIN = 0.05
ACTIONS = ["easy", "medium", "hard"]
WEAK_THRESHOLD = 0.5; STRONG_THRESHOLD = 0.8
RETEST_WEAK = 1; RETEST_MODERATE = 3; RETEST_STRONG = 7


def _epsilon(attempts: int) -> float:
    return max(EPSILON_MIN, EPSILON_START * max(0, 1 - attempts / 30))


def _compute_reward(is_correct: bool, difficulty: str, time_taken: float) -> float:
    base = {"easy": (0.3, -0.2), "medium": (0.6, -0.4), "hard": (1.0, -0.6)}
    cr, wr = base.get(difficulty, (0.6, -0.4))
    r = cr if is_correct else wr
    if time_taken < 30: r += 0.1
    elif time_taken > 60: r -= 0.1
    return float(np.clip(r, -1.0, 1.0))


def get_or_create_topic(db, topic: str) -> TopicPerformance:
    tp = db.query(TopicPerformance).filter_by(topic=topic).first()
    if not tp:
        tp = TopicPerformance(topic=topic)
        db.add(tp); db.commit(); db.refresh(tp)
    return tp


def select_difficulty(topic: str) -> str:
    db = SessionLocal()
    tp = get_or_create_topic(db, topic)
    q = np.array([tp.q_value_easy, tp.q_value_medium, tp.q_value_hard])
    eps = _epsilon(tp.attempts)
    db.close()
    return np.random.choice(ACTIONS) if np.random.rand() < eps else ACTIONS[int(np.argmax(q))]


def update_q_values(topic: str, difficulty: str, is_correct: bool, time_taken: float) -> dict:
    db = SessionLocal()
    tp = get_or_create_topic(db, topic)
    q = np.array([tp.q_value_easy, tp.q_value_medium, tp.q_value_hard])
    idx = ACTIONS.index(difficulty)
    reward = _compute_reward(is_correct, difficulty, time_taken)
    q[idx] += ALPHA * (reward + GAMMA * np.max(q) - q[idx])
    tp.q_value_easy, tp.q_value_medium, tp.q_value_hard = float(q[0]), float(q[1]), float(q[2])

    prev_acc = tp.accuracy
    tp.attempts += 1
    if is_correct: tp.correct += 1
    tp.accuracy = tp.correct / tp.attempts
    tp.avg_time = (tp.avg_time * (tp.attempts - 1) + time_taken) / tp.attempts

    if tp.accuracy >= STRONG_THRESHOLD and tp.attempts >= 3:
        tp.mastery_level = "strong"
        tp.next_quiz_due = datetime.utcnow() + timedelta(days=RETEST_STRONG)
    elif tp.accuracy >= WEAK_THRESHOLD:
        tp.mastery_level = "moderate"
        tp.next_quiz_due = datetime.utcnow() + timedelta(days=RETEST_MODERATE)
    else:
        tp.mastery_level = "weak"
        tp.next_quiz_due = datetime.utcnow() + timedelta(days=RETEST_WEAK)

    tp.updated_at = datetime.utcnow()
    db.commit()

    result = {
        "accuracy": tp.accuracy, "mastery_level": tp.mastery_level,
        "reward": reward, "improvement": tp.accuracy - prev_acc,
        "next_difficulty": ACTIONS[int(np.argmax(q))],
    }
    db.close()
    return result


def get_weak_strong_topics() -> Tuple[List[str], List[str]]:
    db = SessionLocal()
    all_tp = db.query(TopicPerformance).all()
    db.close()
    weak   = [t.topic for t in all_tp if t.mastery_level == "weak"]
    strong = [t.topic for t in all_tp if t.mastery_level == "strong"]
    return weak, strong


def get_common_mistakes(topic: str) -> List[str]:
    """Returns questions the user got wrong for this topic."""
    from database import QuizAttempt  # local import avoids circular dep
    db = SessionLocal()
    attempts = db.query(QuizAttempt).filter(
        QuizAttempt.topic == topic,
        QuizAttempt.is_correct == False
    ).order_by(QuizAttempt.id.desc()).limit(10).all()
    db.close()
    return [a.question for a in attempts]


def get_topics_due_for_retest() -> List[str]:
    db = SessionLocal()
    now = datetime.utcnow()
    due = db.query(TopicPerformance).filter(TopicPerformance.next_quiz_due <= now).all()
    db.close()
    return [t.topic for t in due]


def get_performance_summary() -> dict:
    db = SessionLocal()
    all_tp = db.query(TopicPerformance).all()
    db.close()
    if not all_tp:
        return {"topics": [], "overall_accuracy": 0.0, "mastery_breakdown": {"weak":0,"moderate":0,"strong":0}}
    topics_data = [{
        "topic": tp.topic, "accuracy": round(tp.accuracy * 100, 1),
        "attempts": tp.attempts, "correct": tp.correct,
        "avg_time": round(tp.avg_time, 1), "mastery_level": tp.mastery_level,
        "next_quiz_due": tp.next_quiz_due.isoformat() if tp.next_quiz_due else None,
    } for tp in all_tp]
    overall = np.mean([t["accuracy"] for t in topics_data]) if topics_data else 0.0
    mb = {lvl: sum(1 for t in topics_data if t["mastery_level"] == lvl)
          for lvl in ("weak", "moderate", "strong")}
    return {"topics": topics_data, "overall_accuracy": round(float(overall), 1), "mastery_breakdown": mb}
