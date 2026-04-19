import os
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from db import get_supabase

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 50


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env")
    return genai.Client(api_key=api_key)


def embed_and_store(chunks: List[dict]) -> int:
    client = _get_client()
    supabase = get_supabase()

    seen_content = set()
    unique_chunks = []
    for c in chunks:
        key = c["content"][:200]
        if key not in seen_content:
            seen_content.add(key)
            unique_chunks.append(c)

    total = 0
    for i in range(0, len(unique_chunks), BATCH_SIZE):
        batch = unique_chunks[i: i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )

        rows = []
        for j, embedding_obj in enumerate(result.embeddings):
            rows.append({
                "content": batch[j]["content"],
                "embedding": embedding_obj.values,
                "metadata": {
                    "url": batch[j].get("url", ""),
                    "label": batch[j].get("label", ""),
                },
            })

        supabase.table("documents").insert(rows).execute()
        total += len(rows)

    return total


def list_sources() -> List[dict]:
    """Return all distinct sources with chunk counts."""
    supabase = get_supabase()
    result = supabase.table("documents").select("metadata").execute()

    counts = {}
    for row in result.data:
        url = row.get("metadata", {}).get("url", "unknown")
        label = row.get("metadata", {}).get("label", "")
        if url not in counts:
            counts[url] = {"url": url, "label": label, "chunks": 0}
        counts[url]["chunks"] += 1

    return sorted(counts.values(), key=lambda x: x["chunks"], reverse=True)


def delete_source(url: str) -> int:
    """Delete all chunks for a given source URL."""
    supabase = get_supabase()
    # Supabase doesn't support filtering by JSONB in delete easily,
    # so we fetch IDs first then delete by ID
    result = supabase.table("documents").select("id, metadata").execute()
    ids_to_delete = [
        row["id"] for row in result.data
        if row.get("metadata", {}).get("url") == url
    ]
    if ids_to_delete:
        supabase.table("documents").delete().in_("id", ids_to_delete).execute()
    return len(ids_to_delete)
