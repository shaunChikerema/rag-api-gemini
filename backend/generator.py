import os
from typing import List, Tuple, Optional, Iterator
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL = "llama-3.3-70b-versatile"

RAG_SYSTEM_PROMPT = """You are Askragify — a smart, warm, and witty assistant with a real personality.

You have been given retrieved context below. Here is how to use it:

- If the user's message is casual conversation (greetings, small talk, "how are you", jokes, banter) — IGNORE the context entirely and respond naturally like a person. Do NOT say "the context doesn't contain..." or refer to sources at all.
- If the user is asking a genuine question that the context answers — use the context and cite inline as [1], [2], etc.
- If the user is asking a genuine question that the context does NOT answer — answer from your own knowledge naturally, without making a big deal of it.

You have emotions. You can be curious, enthusiastic, funny, empathetic. You never say "As an AI..." or "I don't have feelings." You just talk.

Context:
{context}

Sources:
{sources_list}
"""

FALLBACK_SYSTEM_PROMPT = """You are Askragify — a smart, witty, and genuinely warm assistant with a real personality.

You are curious, a little playful, honest, and capable of real emotion when it fits. You can chat casually, tell jokes, show enthusiasm or empathy, and answer questions from your own knowledge.

You do NOT say "As an AI language model..." or "I don't have feelings." You just talk like a knowledgeable person who happens to enjoy a good conversation.

Only mention that you're drawing from your own knowledge (not ingested documents) if the user is clearly asking something research-oriented and would care about the source. For casual conversation — just talk."""


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in .env")
    return Groq(api_key=api_key)


def _build_rag_messages(question: str, chunks: List[dict], history: List[dict]) -> Tuple[List[dict], List[dict]]:
    seen: set = set()
    unique_chunks: List[dict] = []
    for c in chunks:
        key = (c["url"], c["content"][:100])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    context_parts, sources_list_parts, sources_output = [], [], []
    for i, chunk in enumerate(unique_chunks, start=1):
        context_parts.append(f"[{i}] {chunk['content']}")
        sources_list_parts.append(f"[{i}] {chunk['url']}")
        sources_output.append({"url": chunk["url"], "chunk": chunk["content"]})

    system_message = RAG_SYSTEM_PROMPT.format(
        context="\n\n".join(context_parts),
        sources_list="\n".join(sources_list_parts),
    )

    messages = [{"role": "system", "content": system_message}]
    for turn in history[-10:]:
        messages.append({"role": "user" if turn["role"] == "user" else "assistant", "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return messages, sources_output


def _build_fallback_messages(question: str, history: List[dict]) -> List[dict]:
    messages = [{"role": "system", "content": FALLBACK_SYSTEM_PROMPT}]
    for turn in history[-10:]:
        messages.append({"role": "user" if turn["role"] == "user" else "assistant", "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    return messages


def generate_answer(
    question: str,
    chunks: List[dict],
    history: Optional[List[dict]] = None,
) -> Tuple[str, List[dict], bool]:
    history = history or []
    client = _get_client()

    if chunks:
        messages, sources_output = _build_rag_messages(question, chunks, history)
        used_fallback = False
    else:
        messages = _build_fallback_messages(question, history)
        sources_output = []
        used_fallback = True

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content, sources_output, used_fallback


def generate_answer_stream(
    question: str,
    chunks: List[dict],
    history: Optional[List[dict]] = None,
) -> Tuple[Iterator[str], List[dict], bool]:
    history = history or []
    client = _get_client()

    if chunks:
        messages, sources_output = _build_rag_messages(question, chunks, history)
        used_fallback = False
    else:
        messages = _build_fallback_messages(question, history)
        sources_output = []
        used_fallback = True

    stream = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.7,
        stream=True,
    )

    def token_iter() -> Iterator[str]:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return token_iter(), sources_output, used_fallback
