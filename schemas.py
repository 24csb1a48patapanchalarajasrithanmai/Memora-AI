from pydantic import BaseModel, Field
from typing import List, Optional

# =========================
# COMMON MODELS
# =========================

class Chunk(BaseModel):
    text: str
    source: str

# =========================
# UPLOAD MODELS
# =========================

class UploadResponse(BaseModel):
    status: str
    chunks: List[Chunk] = Field(default_factory=list)
    total_chunks: int = 0

# =========================
# QA MODELS
# =========================

class QuestionRequest(BaseModel):
    question: str
    include_summary: bool = False

class QuestionResponse(BaseModel):
    answer: str
    explanation: str = ""
    summary: str = ""

# =========================
# QUIZ MODELS
# =========================

class QuizQuestion(BaseModel):
    type: str
    question: str
    options: List[str] = Field(default_factory=list)
    answer: str
    explanation: str = ""
    concept: str = ""

class StartQuizRequest(BaseModel):
    topic: str
    n_questions: int = 5
    force_difficulty: Optional[str] = None

class StartQuizResponse(BaseModel):
    topic: str
    difficulty: str
    questions: List[QuizQuestion] = Field(default_factory=list)

# =========================
# DASHBOARD MODELS
# =========================

class TopicImprovementResponse(BaseModel):
    topic: str
    progress: List[float] = Field(default_factory=list)

# =========================
# FEEDBACK MODELS
# =========================

class RetrievalFeedbackRequest(BaseModel):
    query: str
    was_helpful: bool
    comment: str = ""

class FeedbackResponse(BaseModel):
    status: str
    query: str
