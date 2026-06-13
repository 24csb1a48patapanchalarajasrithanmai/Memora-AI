"""
routers/quiz.py - Fixed:
  - /topics endpoint added
  - /submit endpoint added (per-question, not batch)
  - /lesson endpoint added
  - /recommendations endpoint added
  - /due endpoint added
  - /history/{topic} endpoint added
  - rl_agent.get_common_mistakes properly imported
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from ingest import retrieve_chunks, get_all_topics, get_chunks_by_topic
from llm import generate_quiz, generate_lesson, generate_learning_recommendations
from rl_agent import (select_difficulty, update_q_values, get_weak_strong_topics,
                      get_common_mistakes, get_topics_due_for_retest)
from database import SessionLocal, QuizAttempt

router = APIRouter()


class StartQuizRequest(BaseModel):
    topic: str
    n_questions: int = 5
    force_difficulty: Optional[str] = None


class SubmitAnswerRequest(BaseModel):
    topic: str
    question: str
    correct_answer: str
    user_answer: str
    difficulty: str
    time_taken: float


class LessonRequest(BaseModel):
    topic: str


@router.get("/topics")
async def list_topics():
    return {"topics": get_all_topics()}


@router.get("/due")
async def due_topics():
    return {"due_topics": get_topics_due_for_retest()}


@router.post("/start")
async def start_quiz(req: StartQuizRequest):
    weak, strong = get_weak_strong_topics()
    if req.force_difficulty:
        difficulty = req.force_difficulty
    elif req.topic in weak:
        difficulty = "easy"
    elif req.topic in strong:
        difficulty = "hard"
    else:
        difficulty = select_difficulty(req.topic)

    # Merge topic-specific chunks + query-based retrieval
    topic_chunks = get_chunks_by_topic(req.topic)
    query_chunks = retrieve_chunks(req.topic, top_k=8)
    seen, combined = set(), []
    for c in topic_chunks + query_chunks:
        key = c.get("chunk_id", c.get("text", "")[:40])
        if key not in seen:
            seen.add(key); combined.append(c)

    if not combined:
        raise HTTPException(404, f"No content found for topic '{req.topic}'. Upload PDFs first.")

    mistakes = get_common_mistakes(req.topic)
    questions = generate_quiz(combined, req.topic, difficulty, req.n_questions, mistakes)

    return {"topic": req.topic, "difficulty": difficulty, "questions": questions}


@router.post("/submit")
async def submit_answer(req: SubmitAnswerRequest):
    is_correct = req.user_answer.strip().lower() == req.correct_answer.strip().lower()

    rl = update_q_values(req.topic, req.difficulty, is_correct, req.time_taken)

    db = SessionLocal()
    db.add(QuizAttempt(
        topic=req.topic, question=req.question,
        correct_answer=req.correct_answer, user_answer=req.user_answer,
        is_correct=is_correct, difficulty=req.difficulty, time_taken=req.time_taken,
    ))
    db.commit(); db.close()

    return {
        "is_correct": is_correct, "correct_answer": req.correct_answer,
        "mastery_level": rl["mastery_level"], "accuracy": rl["accuracy"],
        "next_difficulty": rl["next_difficulty"], "reward": rl["reward"],
        "improvement": rl["improvement"],
    }


@router.post("/lesson")
async def get_lesson(req: LessonRequest):
    chunks = retrieve_chunks(req.topic, top_k=6) + get_chunks_by_topic(req.topic)
    if not chunks:
        raise HTTPException(404, "No content for this topic.")
    # Deduplicate
    seen, combined = set(), []
    for c in chunks:
        k = c.get("text", "")[:40]
        if k not in seen: seen.add(k); combined.append(c)
    return generate_lesson(combined, req.topic)


@router.get("/recommendations")
async def get_recommendations():
    weak, strong = get_weak_strong_topics()
    recs = generate_learning_recommendations(weak, strong)
    return {"weak_topics": weak, "strong_topics": strong, "recommendations": recs}


@router.get("/history/{topic}")
async def topic_history(topic: str, limit: int = 50):
    db = SessionLocal()
    attempts = (db.query(QuizAttempt).filter_by(topic=topic)
                  .order_by(QuizAttempt.created_at.asc()).limit(limit).all())
    db.close()
    return [{"is_correct": a.is_correct, "difficulty": a.difficulty,
             "time_taken": a.time_taken, "created_at": a.created_at.isoformat()} for a in attempts]
