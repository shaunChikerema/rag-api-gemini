import os
from typing import List, Tuple, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL = "llama-3.3-70b-versatile"  # Groq free tier, fast + capable

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY the context provided below.

Rules:
- Be concise and factual.
- Cite your sources inline using [1], [2], etc. that correspond to the numbered sources.
- If the context doesn't contain enough information, say so clearly.
- Never make up information not found in the context.

Context:
{context}

Sources:
{sources_list}
"""


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in .env")
    return Groq(api_key=api_key)


def generate_answer(
    question: str,
    chunks: List[dict],
    history: Optional[List[dict]] = None,
) -> Tuple[str, List[dict]]:
    """
    Given a question and retrieved chunks, generate a grounded answer
    with inline citations. Returns (answer_text, sources_list).
    """
    history = history or []
    client = _get_client()

    # Deduplicate chunks by URL + content
    seen = set()
    unique_chunks = []
    for c in chunks:
        key = (c["url"], c["content"][:100])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    # Build context and sources
    context_parts = []
    sources_list_parts = []
    sources_output = []

    for i, chunk in enumerate(unique_chunks, start=1):
        context_parts.append(f"[{i}] {chunk['content']}")
        sources_list_parts.append(f"[{i}] {chunk['url']}")
        sources_output.append({"url": chunk["url"], "chunk": chunk["content"]})

    context = "\n\n".join(context_parts)
    sources_list = "\n".join(sources_list_parts)

    system_message = SYSTEM_PROMPT.format(
        context=context,
        sources_list=sources_list,
    )

    # Build chat history for multi-turn support
    messages = [{"role": "system", "content": system_message}]
    for turn in history:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["content"]})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content, sources_output