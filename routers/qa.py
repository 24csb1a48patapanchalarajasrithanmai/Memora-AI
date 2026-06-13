"""
routers/qa.py - Fixed:
  - returns session_id in /ask response
  - feedback endpoint accepts {session_id, relevant, quality}
  - history endpoint exists
  - feedback-adjusted retrieval uses .get('text') safely
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ingest import retrieve_chunks
from llm import answer_question, summarize_content
from database import SessionLocal, QASession

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str
    include_summary: bool = False
    top_k: int = 8


class FeedbackRequest(BaseModel):
    session_id: int
    relevant: bool
    quality: str = "good"  # good / average / poor


def _adjust_by_feedback(chunks):
    """Re-rank chunks based on historical feedback."""
    db = SessionLocal()
    past = db.query(QASession).filter(QASession.feedback_relevant.isnot(None)).all()
    db.close()
    if not past:
        return chunks
    scores = {}
    for s in past:
        adj = 1 if s.feedback_relevant else -1
        for snippet in (s.relevant_chunks or []):
            # stored as plain strings (first 200 chars of chunk text)
            key = snippet if isinstance(snippet, str) else snippet.get("text", "")
            scores[key] = scores.get(key, 0) + adj
    def chunk_score(c):
        text = c.get("text", "")
        for k, v in scores.items():
            if k and k in text:
                return v
        return 0
    filtered = [c for c in chunks if chunk_score(c) > -5]
    return sorted(filtered, key=chunk_score, reverse=True) or chunks


@router.post("/ask")
async def ask_question(req: QuestionRequest):
    chunks = retrieve_chunks(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(404, "No content found. Upload study materials first.")
    chunks = _adjust_by_feedback(chunks)

    result = answer_question(req.question, chunks)
    summary = ""
    if req.include_summary:
        summary = await summarize_content(chunks)

    db = SessionLocal()
    session = QASession(
        question=req.question,
        answer=result.get("answer", ""),
        summary=summary,
        relevant_chunks=[c.get("text", "")[:200] for c in chunks],
    )
    db.add(session); db.commit(); db.refresh(session)
    sid = session.id; db.close()

    # Deduplicate chunk sources
    seen, chunk_sources = set(), []
    for c in chunks:
        key = c.get("source", "?")
        if key not in seen:
            seen.add(key)
            chunk_sources.append({"source": key, "topic": c.get("topic", "?")})

    return {
        "session_id": sid,
        "answer": result.get("answer", ""),
        "explanation": result.get("explanation", ""),
        "key_concepts": result.get("key_concepts", []),
        "summary": summary,
        "sources": chunk_sources,
        "answer_sources": result.get("sources", []),
    }


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    db = SessionLocal()
    s = db.query(QASession).filter_by(id=req.session_id).first()
    if not s:
        db.close(); raise HTTPException(404, "Session not found.")
    s.feedback_relevant = req.relevant
    s.feedback_quality  = req.quality
    db.commit(); db.close()
    return {"status": "feedback recorded"}


@router.get("/history")
async def get_history(limit: int = 20):
    db = SessionLocal()
    sessions = db.query(QASession).order_by(QASession.id.desc()).limit(limit).all()
    db.close()
    return [{
        "id": s.id, "question": s.question,
        "answer": (s.answer or "")[:300],
        "feedback_relevant": s.feedback_relevant,
        "feedback_quality": s.feedback_quality,
        "created_at": s.created_at.isoformat(),
    } for s in sessions]