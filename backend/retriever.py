import os
from typing import List, Optional
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv
from db import get_supabase

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env")
    return genai.Client(api_key=api_key)


def _get_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in .env")
    return Groq(api_key=api_key)


def _rewrite_query(question: str, history: List[dict]) -> str:
    """
    Use the LLM to rewrite the user's latest message into a fully self-contained
    search query, resolving any pronouns or references from conversation history.
    Returns the rewritten query string.
    """
    if not history:
        return question  # No history — raw question is already standalone

    # Only use last 6 turns to keep the prompt lean
    recent = history[-6:]
    history_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in recent
    )

    prompt = f"""Given this conversation history:
{history_text}

And this latest user message: "{question}"

Rewrite the latest message as a single, fully self-contained search query that captures what the user is actually looking for — resolving any references like "that", "it", "the second point", "tell me more", etc.

Reply with ONLY the rewritten query. No explanation, no punctuation changes, no quotes."""

    client = _get_groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten if rewritten else question


def _expand_query(question: str) -> List[str]:
    """
    Generate 2 alternative phrasings of the question to improve retrieval recall.
    Returns [original, alt1, alt2].
    """
    prompt = f"""Generate 2 alternative search queries for this question, capturing the same intent with different wording.

Question: "{question}"

Reply with ONLY the 2 alternatives, one per line. No numbering, no explanation."""

    client = _get_groq()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=120,
        )
        lines = [
            l.strip() for l in response.choices[0].message.content.strip().splitlines()
            if l.strip()
        ]
        alternatives = lines[:2]
    except Exception:
        alternatives = []

    return [question] + alternatives


def _embed_queries(queries: List[str]) -> List[List[float]]:
    """Embed a list of query strings in one batched API call."""
    client = _get_gemini()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=queries,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return [e.values for e in result.embeddings]


def _vector_search(embedding: List[float], top_k: int, threshold: float) -> List[dict]:
    """Run pgvector cosine similarity search, return chunk dicts."""
    supabase = get_supabase()
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": embedding,
            "match_count": top_k,
            "match_threshold": threshold,
        },
    ).execute()

    chunks = []
    for row in result.data:
        chunks.append({
            "content":    row["content"],
            "url":        row.get("metadata", {}).get("url", ""),
            "similarity": row.get("similarity", 0),
        })
    return chunks


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    history: Optional[List[dict]] = None,
    expand: bool = True,
) -> List[dict]:
    """
    Retrieve the most relevant chunks for a question.

    Improvements over naive retrieval:
    1. Conversation-aware — rewrites the query to resolve history references
       before searching ("tell me more about that" → specific standalone query).
    2. Query expansion — generates 2 alternative phrasings, retrieves for each,
       merges and deduplicates to improve recall on vague or ambiguous questions.

    Returns deduplicated chunks sorted by similarity, capped at top_k.
    """
    history = history or []

    # Step 1 — resolve history references into a standalone query
    standalone_query = _rewrite_query(question, history)

    # Step 2 — expand into multiple phrasings
    queries = _expand_query(standalone_query) if expand else [standalone_query]

    # Step 3 — embed all queries in one batched call
    embeddings = _embed_queries(queries)

    # Step 4 — retrieve for each embedding, merge and deduplicate
    seen_content: set = set()
    all_chunks: List[dict] = []

    for embedding in embeddings:
        for chunk in _vector_search(embedding, top_k=top_k, threshold=similarity_threshold):
            key = chunk["content"][:120]
            if key not in seen_content:
                seen_content.add(key)
                all_chunks.append(chunk)

    # Step 5 — sort by similarity, cap at top_k
    all_chunks.sort(key=lambda c: c["similarity"], reverse=True)
    return all_chunks[:top_k]
