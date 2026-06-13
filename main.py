"""
main.py - Fixed:
  - routers imported from routers/ sub-package
  - static files served correctly
  - startup initialises DB
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn, os

from routers import upload, qa, quiz, dashboard, feedback
from database import init_db

app = FastAPI(title="EduMind AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(upload.router,    prefix="/api/upload",    tags=["upload"])
app.include_router(qa.router,        prefix="/api/qa",        tags=["qa"])
app.include_router(quiz.router,      prefix="/api/quiz",      tags=["quiz"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(feedback.router,  prefix="/api/feedback",  tags=["feedback"])

# Serve frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup():
    init_db()

@app.get("/")
async def root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "EduMind AI running — place index.html in static/"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
