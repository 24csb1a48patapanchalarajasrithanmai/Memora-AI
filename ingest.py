"""
ingest.py - MEMORA AI
  - PDF and PPTX text extraction
  - Sentence-aware chunking
  - FAISS vector storage
  - Embedding cache
  - Lazy loading of SentenceTransformer model
"""

import os
import pickle
import re
import hashlib
from typing import List

import fitz
import faiss
import numpy as np

try:
    from pptx import Presentation
    _PPTX_AVAILABLE = True
except ImportError:
    _PPTX_AVAILABLE = False

from sentence_transformers import SentenceTransformer

from database import SessionLocal, EmbeddingCache, Document
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

FAISS_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/chunks.pkl"

os.makedirs("vector_store", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_model = None


def get_model():
    """
    Load SentenceTransformer only when it is actually needed.

    This prevents the large PyTorch / embedding model from loading
    during FastAPI startup.
    """
    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")

    return _model


# ─────────────────────────────────────────────────────────────────────────────
# FAISS STORE
# ─────────────────────────────────────────────────────────────────────────────

index = None
chunks_store: List[dict] = []


def load_store():
    global index, chunks_store

    if os.path.exists(FAISS_PATH) and os.path.exists(CHUNKS_PATH):

        index = faiss.read_index(FAISS_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            chunks_store = pickle.load(f)

    else:

        index = faiss.IndexFlatIP(384)
        chunks_store = []


def save_store():

    faiss.write_index(index, FAISS_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks_store, f)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:

    doc = fitz.open(path)

    text = "\n".join(
        page.get_text()
        for page in doc
    )

    doc.close()

    return text


def _extract_pptx(path: str) -> str:

    if not _PPTX_AVAILABLE:
        raise RuntimeError(
            "python-pptx is not installed. "
            "Run: pip install python-pptx"
        )

    prs = Presentation(path)

    slides = []

    for i, slide in enumerate(prs.slides, 1):

        parts = [f"[Slide {i}]"]

        if slide.shapes.title and slide.shapes.title.text.strip():
            parts.append(
                slide.shapes.title.text.strip()
            )

        for shape in slide.shapes:

            if shape.has_text_frame:

                for para in shape.text_frame.paragraphs:

                    line = para.text.strip()

                    if line:
                        parts.append(line)

        slides.append(
            "\n".join(parts)
        )

    return "\n\n".join(slides)


def extract_text(path: str) -> str:

    ext = os.path.splitext(path)[1].lower()

    if ext in (".pptx", ".ppt"):
        return _extract_pptx(path)

    return _extract_pdf(path)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def _infer_topic(text: str) -> str:

    first = text.strip().split("\n")[0][:80]

    clean = re.sub(
        r"[^a-zA-Z0-9 ]",
        "",
        first
    ).strip()

    return clean if len(clean) > 3 else "General"


def split_chunks(
    text: str,
    source: str
) -> List[dict]:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current = []
    length = 0
    cid = 0

    for sentence in sentences:

        sentence_length = len(
            sentence.split()
        )

        if (
            length + sentence_length > CHUNK_SIZE
            and current
        ):

            chunk_text = " ".join(current)

            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": cid,
                "topic": _infer_topic(chunk_text)
            })

            cid += 1

            overlap = " ".join(
                chunk_text.split()[-CHUNK_OVERLAP:]
            )

            current = overlap.split()

            length = len(current)

        current.append(sentence)

        length += sentence_length

    if current:

        chunk_text = " ".join(current)

        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_id": cid,
            "topic": _infer_topic(chunk_text)
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(
    texts: List[str]
) -> np.ndarray:

    db = SessionLocal()

    result = [None] * len(texts)

    to_compute_idx = []
    to_compute_texts = []

    # Check database cache first
    for i, text in enumerate(texts):

        text_hash = hashlib.md5(
            text.encode()
        ).hexdigest()

        cached = (
            db.query(EmbeddingCache)
            .filter_by(text_hash=text_hash)
            .first()
        )

        if cached:

            result[i] = np.array(
                cached.embedding,
                dtype=np.float32
            )

        else:

            to_compute_idx.append(i)
            to_compute_texts.append(text)

    # Only load the model if new embeddings are required
    if to_compute_texts:

        model = get_model()

        embeddings = model.encode(
            to_compute_texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        for idx, text, embedding in zip(
            to_compute_idx,
            to_compute_texts,
            embeddings
        ):

            text_hash = hashlib.md5(
                text.encode()
            ).hexdigest()

            db.add(
                EmbeddingCache(
                    text_hash=text_hash,
                    embedding=embedding.tolist()
                )
            )

            result[idx] = embedding.astype(
                np.float32
            )

        db.commit()

    db.close()

    return np.array(
        result,
        dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(
    path: str,
    filename: str
) -> dict:

    load_store()

    text = extract_text(path)

    if not text.strip():
        raise ValueError(
            "No text extracted from file."
        )

    chunks = split_chunks(
        text,
        filename
    )

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    index.add(embeddings)

    chunks_store.extend(chunks)

    save_store()

    topics = list(
        set(
            chunk["topic"]
            for chunk in chunks
        )
    )

    # Save document information
    db = SessionLocal()

    existing = (
        db.query(Document)
        .filter_by(filename=filename)
        .first()
    )

    if existing:

        existing.total_chunks = len(chunks)
        existing.topics = topics

    else:

        db.add(
            Document(
                filename=filename,
                total_chunks=len(chunks),
                topics=topics
            )
        )

    db.commit()
    db.close()

    return {
        "status": "success",
        "chunks": chunks,
        "total_chunks": len(chunks),
        "topics": topics
    }


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    top_k: int = 5
) -> List[dict]:

    load_store()

    if index is None or index.ntotal == 0:
        return []

    # Load embedding model only when a query actually needs it
    model = get_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, i in zip(
        scores[0],
        indices[0]
    ):

        if 0 <= i < len(chunks_store):

            chunk = dict(
                chunks_store[i]
            )

            chunk["score"] = float(score)

            results.append(chunk)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TOPIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_topics() -> List[str]:

    load_store()

    return list(
        set(
            chunk.get(
                "topic",
                "General"
            )
            for chunk in chunks_store
        )
    )


def get_chunks_by_topic(
    topic: str
) -> List[dict]:

    load_store()

    return [
        chunk
        for chunk in chunks_store
        if chunk.get("topic") == topic
    ]
