import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
from db import get_supabase

load_dotenv()

EMBEDDING_MODEL = "models/text-embedding-004"  # 768 dimensions, free tier
BATCH_SIZE = 50  # Gemini embedding batch limit


def _get_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env")
    genai.configure(api_key=api_key)
    return genai


def embed_and_store(chunks: List[dict]) -> int:
    """
    Take a list of chunk dicts, embed each one using Gemini,
    and insert into the Supabase `documents` table.
    Returns the number of chunks stored.
    """
    ai = _get_genai()
    supabase = get_supabase()

    # Deduplicate by content to avoid storing duplicates on re-ingest
    seen_content = set()
    unique_chunks = []
    for c in chunks:
        key = c["content"][:200]
        if key not in seen_content:
            seen_content.add(key)
            unique_chunks.append(c)

    total = 0

    # Process in batches
    for i in range(0, len(unique_chunks), BATCH_SIZE):
        batch = unique_chunks[i: i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        # Embed batch
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="retrieval_document",
        )

        rows = []
        for j, embedding in enumerate(result["embedding"]):
            rows.append({
                "content":   batch[j]["content"],
                "embedding": embedding,  # list of 768 floats
                "metadata": {
                    "url":   batch[j].get("url", ""),
                    "label": batch[j].get("label", ""),
                },
            })

        supabase.table("documents").insert(rows).execute()
        total += len(rows)

    return total
