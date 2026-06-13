"""
routers/upload.py - Fixed: added /documents endpoint, proper error handling
"""
import os, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from ingest import ingest_pdf

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


ALLOWED = {".pdf", ".pptx", ".ppt"}

@router.post("/")
async def upload_pdf(file: UploadFile = File(...)):
    name, raw_ext = os.path.splitext(file.filename)
    ext = raw_ext.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Only PDF, PPTX, and PPT files are supported.")
    # Save with lowercased extension so ingest.py extension checks always match
    safe_filename = name + ext
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = ingest_pdf(filepath, safe_filename)
        return {"status": "success", "filename": file.filename,
                "chunks": result["total_chunks"], "topics": result.get("topics", [])}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/documents")
async def list_documents():
    from database import SessionLocal, Document
    db = SessionLocal()
    docs = db.query(Document).order_by(Document.id.desc()).all()
    db.close()
    return [{"id": d.id, "filename": d.filename, "total_chunks": d.total_chunks,
             "topics": d.topics or [], "created_at": d.created_at.isoformat()} for d in docs]
