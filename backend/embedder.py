import os
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from db import get_supabase

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 50  # Gemini embedding batch limit


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env")
    return genai.Client(api_key=api_key)


def embed_and_store(chunks: List[dict]) -> int:
    """
    Take a list of chunk dicts, embed each one using Gemini,
    and insert into the Supabase `documents` table.
    Returns the number of chunks stored.
    """
    client = _get_client()
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
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        rows = []
        for j, embedding_obj in enumerate(result.embeddings):
            rows.append({
                "content":   batch[j]["content"],
                "embedding": embedding_obj.values,  # list of 768 floats
                "metadata": {
                    "url":   batch[j].get("url", ""),
                    "label": batch[j].get("label", ""),
                },
            })

        supabase.table("documents").insert(rows).execute()
        total += len(rows)

    return total