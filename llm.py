import os, re, json, asyncio
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_client       = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
_async_client = AsyncOpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")

# System message: helpful but grounded
_JSON_SYSTEM = (
    "You are a helpful educational AI. "
    "Answer using the provided context from uploaded study materials. "
    "If the context contains relevant information, use it to answer fully. "
    "Only say 'Not found in uploaded materials' if the context has absolutely zero relevant content. "
    "Return ONLY valid JSON. No markdown, no extra text."
)


# ================= CORE =================
def _call_llm_sync(prompt: str, max_tokens: int = 1024) -> str:
    try:
        resp = _client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error: {e}]"


async def _call_llm_async(prompt: str, max_tokens: int = 1024) -> str:
    try:
        resp = await _async_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error: {e}]"


def _safe_json(raw: str):
    """Robustly extract JSON from LLM output that may be wrapped in strings or markdown."""
    if not raw or raw.startswith("[LLM Error"):
        return None
    try:
        # 1. Strip markdown code fences
        clean = re.sub(r"```json|```", "", raw).strip()
        # 2. If the whole thing is a JSON string (llama3.2 wraps in quotes), unwrap it
        if clean.startswith('"') and clean.endswith('"'):
            try:
                clean = json.loads(clean)  # unwrap the outer string
            except Exception:
                pass
        # 3. Try direct parse
        return json.loads(clean)
    except Exception:
        pass
    # 4. Find first {...} or [...] block
    for pattern in (r'(\{.*\})', r'(\[.*\])'):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return None


# ================= QA =================
def answer_question(question: str, context_chunks: List[dict]) -> dict:
    context = "\n\n".join(
        f"[Source: {c.get('source', 'Unknown')} | Topic: {c.get('topic', '?')}]\n{c.get('text', '')[:500]}"
        for c in context_chunks[:5]
    )

    prompt = (
        f"Context from uploaded study materials:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Read the context carefully and answer the question as fully as possible.\n"
        "- The answer may be spread across multiple context chunks — combine them.\n"
        "- Use your understanding of the context, not just exact word matches.\n"
        "- Only set answer to 'Not found in uploaded materials.' if there is truly NO relevant content at all.\n"
        "- List filenames from [Source:] labels you used in 'sources'.\n\n"
        'Return ONLY this JSON: {"answer":"...","explanation":"...","key_concepts":["..."],"sources":["filename"]}'
    )

    raw    = _call_llm_sync(prompt, 600)
    parsed = _safe_json(raw)
    # Last resort: try to extract answer text from raw if it contains JSON-like content
    if not parsed:
        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        answer_text = answer_match.group(1) if answer_match else "Could not parse response. Please try again."
        parsed = {"answer": answer_text, "explanation": "", "key_concepts": [], "sources": []}
    return parsed


# ================= SUMMARY =================
async def summarize_content(context_chunks: List[dict], topic: Optional[str] = None) -> str:
    try:
        # Filter chunks to only those relevant to the topic (by score, highest first)
        ranked = sorted(context_chunks, key=lambda c: c.get("score", 0), reverse=True)
        top = ranked[:3]

        sources = list(dict.fromkeys(c.get("source", "Unknown") for c in top))
        context_parts = [
            f"[Source: {c.get('source', 'Unknown')}]\n{c.get('text', '')[:500]}"
            for c in top
        ]
        combined_context = "\n\n".join(context_parts)
        source_list = ", ".join(sources)

        suffix = f" about {topic}" if topic else ""
        prompt = (
            f"Context from uploaded materials (these are the most relevant chunks):\n{combined_context}\n\n"
            f"Write a clear 3-sentence summary{suffix} using ONLY the text above. "
            "Do not mix in unrelated topics. Do not add outside knowledge. "
            f"End your response with exactly this line: Sources: {source_list}"
        )
        resp = await _async_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You summarize study material strictly from provided context. No outside knowledge. Plain text only. Never mix topics."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=280,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Summary Error: {e}]"


# ================= QUIZ =================
def generate_quiz(
    context_chunks: List[dict],
    topic: str,
    difficulty: str = "medium",
    n_questions: int = 5,
    mistakes: Optional[List[str]] = None,
) -> List[dict]:

    context = "\n".join(
        f"[Source: {c.get('source','?')}]\n{c.get('text', '')[:300]}"
        for c in context_chunks[:5]
    )

    mistake_hint = (
        f"Prioritise these weak areas: {'; '.join(mistakes[:3])}\n"
        if mistakes else ""
    )

    prompt = (
        f"Topic: {topic} | Difficulty: {difficulty} | Questions: {n_questions}\n"
        f"{mistake_hint}"
        f"Context from uploaded materials (use ONLY this):\n{context}\n\n"
        "Rules:\n"
        "- Questions must be based ONLY on the context above.\n"
        "- EXACTLY 4 meaningful options (no A/B/C/D labels)\n"
        "- 'answer' must exactly match one option\n"
        "- 'explanation' must quote or closely paraphrase the source material\n"
        "- 'source' must be the filename the question came from\n\n"
        "Return JSON array:\n"
        f'[{{"type":"mcq","question":"...","options":["opt1","opt2","opt3","opt4"],'
        f'"answer":"opt1","explanation":"...","concept":"{topic}","source":"filename"}}]'
    )

    raw = _call_llm_sync(prompt, n_questions * 250)

    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        data  = json.loads(match.group() if match else clean)

        valid = []
        for q in data:
            if not (isinstance(q, dict) and "question" in q):
                continue
            opts = q.get("options", [])
            ans  = q.get("answer", "")
            if opts and ans and ans not in opts:
                lower_map = {o.lower(): o for o in opts}
                q["answer"] = lower_map.get(ans.lower(), opts[0])
            q.setdefault("type", "mcq")
            q.setdefault("options", [])
            q.setdefault("explanation", "")
            q.setdefault("concept", topic)
            q.setdefault("source", "")
            valid.append(q)

        return valid

    except Exception:
        return [{
            "type": "mcq",
            "question": f"Which statement about {topic} is correct?",
            "options": [
                f"{topic} is a core concept.",
                f"{topic} is unrelated to this subject.",
                f"{topic} only applies in advanced cases.",
                f"{topic} has no practical use.",
            ],
            "answer": f"{topic} is a core concept.",
            "explanation": "Fallback question — LLM output could not be parsed.",
            "concept": topic,
            "source": "",
        }]


# ================= LESSON =================
def generate_lesson(context_chunks: List[dict], topic: str) -> dict:
    context = "\n".join(
        f"[Source: {c.get('source','?')}]\n{c.get('text', '')[:300]}"
        for c in context_chunks[:4]
    )

    prompt = (
        f"Topic: {topic}\n"
        f"Context from uploaded materials (use ONLY this):\n{context}\n\n"
        "Do not add outside knowledge.\n"
        f'Return: {{"title":"...","overview":"2 sentences from context","key_points":["..."],'
        f'"analogy":"1 sentence","remember":"1 sentence","sources":["filename"]}}'
    )

    raw    = _call_llm_sync(prompt, 500)
    parsed = _safe_json(raw)
    return parsed if parsed else {
        "title": topic, "overview": raw[:300],
        "key_points": [], "analogy": "", "remember": "", "sources": [],
    }


# ================= RECOMMENDATIONS =================
def generate_learning_recommendations(weak: List[str], strong: List[str]) -> str:
    prompt = (
        f"Weak topics: {', '.join(weak[:5]) or 'None'}\n"
        f"Strong topics: {', '.join(strong[:5]) or 'None'}\n"
        "Give 3 specific study tips based on these topics. Be concise."
    )
    return _call_llm_sync(prompt, 250)