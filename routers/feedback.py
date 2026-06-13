from fastapi import APIRouter
from pydantic import BaseModel
from database import SessionLocal, QASession

router = APIRouter()

class RetrievalFeedbackRequest(BaseModel):
    query: str
    was_helpful: bool
    comment: str = ""

@router.post("/retrieval")
async def submit_retrieval_feedback(fb: RetrievalFeedbackRequest):
    db = SessionLocal()
    s = (db.query(QASession).filter(QASession.question == fb.query)
           .order_by(QASession.id.desc()).first())
    if s: s.feedback_relevant = fb.was_helpful
    db.commit(); db.close()
    return {"status": "feedback recorded", "query": fb.query}
