"""
database.py - SQLite database setup with SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./edumind.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    total_chunks = Column(Integer, default=0)
    topics = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, index=True)
    question = Column(Text)
    correct_answer = Column(String)
    user_answer = Column(String)
    is_correct = Column(Boolean, default=False)
    difficulty = Column(String, default="medium")  # easy/medium/hard
    time_taken = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class TopicPerformance(Base):
    __tablename__ = "topic_performance"
    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, unique=True, index=True)
    accuracy = Column(Float, default=0.0)
    attempts = Column(Integer, default=0)
    correct = Column(Integer, default=0)
    avg_time = Column(Float, default=0.0)
    mastery_level = Column(String, default="weak")  # weak/moderate/strong
    next_quiz_due = Column(DateTime, default=datetime.utcnow)
    q_value_easy = Column(Float, default=0.5)
    q_value_medium = Column(Float, default=0.5)
    q_value_hard = Column(Float, default=0.5)
    updated_at = Column(DateTime, default=datetime.utcnow)


class QASession(Base):
    __tablename__ = "qa_sessions"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    answer = Column(Text)
    summary = Column(Text)
    relevant_chunks = Column(JSON, default=[])
    feedback_relevant = Column(Boolean, nullable=True)
    feedback_quality = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    id = Column(Integer, primary_key=True, index=True)
    text_hash = Column(String, unique=True, index=True)
    embedding = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()