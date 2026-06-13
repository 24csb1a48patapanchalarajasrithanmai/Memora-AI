# MEMORA AI – Intelligent Learning Platform

MEMORA AI is an AI-powered learning platform that enables users to upload study materials, ask questions, generate quizzes, and track learning progress.

The system combines **Retrieval-Augmented Generation (RAG)**, a **local Large Language Model (Ollama)**, and **adaptive learning strategies** to create a personalized learning experience.

---

## Features

* Upload PDF, PPT, and PPTX files
* Ask questions based on uploaded content
* AI-powered document summarization
* Adaptive quiz generation
* Performance tracking dashboard
* Feedback-based learning improvement
* Teaching mode for concept explanation
* Personalized revision support

---

## Setup Instructions

### 1. Install Python

Use **Python 3.10 or Python 3.11**.

Check your installation:

```bash
python --version
```

---

### 2. Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install and Run Ollama

```bash
ollama run llama3.2
```

Keep Ollama running in the background.

---

### 5. Configure Environment

Create a `.env` file:

```text
OLLAMA_MODEL=llama3.2
VECTOR_DB=faiss
```

---

### 6. Run the Application

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

---

### 7. Open in Browser

```
http://localhost:8000
```

---

## Project Structure

```text
MEMORA-AI/
│
├── README.md
├── main.py
├── ingest.py
├── llm.py
├── rl_agent.py
├── database.py
├── schemas.py
├── requirements.txt
├── .env.example
│
├── routers/
│   ├── upload.py
│   ├── qa.py
│   ├── quiz.py
│   ├── dashboard.py
│   └── feedback.py
│
├── static/
│   └── index.html
│
├── uploads/
├── vector_store/
└── data/
```

---

## API Endpoints

* POST `/api/upload/`
* GET `/api/upload/documents`
* POST `/api/qa/ask`
* POST `/api/quiz/start`
* POST `/api/quiz/submit`
* GET `/api/quiz/topics`
* GET `/api/dashboard/summary`

---

## Technologies Used

### Programming Language

* Python

### Backend

* FastAPI

### AI / ML

* Retrieval-Augmented Generation (RAG)
* Ollama (Local LLM)
* Reinforcement Learning
* Adaptive Learning

### Vector Search

* FAISS

### Document Processing

* PDF and PPT ingestion

---

## Notes

* Ollama must be running before using QA and quiz features.
* Works completely with a local language model.
* Supports personalized learning workflows.
* Can be extended to support multiple users and cloud deployment.

---

## Future Improvements

* Multi-document chat
* Voice interaction
* Cloud deployment
* Personalized learning analytics
* Multi-user support
* Mobile application integration

---

## Project Status

 **Currently Under Development**

---

## Summary

MEMORA AI is an intelligent learning assistant that integrates semantic retrieval, local large language models, and adaptive learning techniques to help students study more effectively through document understanding, question answering, and personalized revision.
