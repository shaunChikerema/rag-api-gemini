import os
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
from db import get_supabase

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env")
    return genai.Client(api_key=api_key)


def retrieve_chunks(question: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[dict]:
    """
    Embed the question with Gemini and find the top_k most similar chunks
    using pgvector cosine similarity via Supabase RPC.
    Returns list of {content, url, similarity}
    """
    client = _get_client()

    # Embed the question (use retrieval_query task type for better results)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = result.embeddings[0].values

    # Call the pgvector match function in Supabase
    supabase = get_supabase()
    rpc_result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "match_threshold": similarity_threshold,
        },
    ).execute()

    chunks = []
    for row in rpc_result.data:
        chunks.append({
            "content":    row["content"],
            "url":        row.get("metadata", {}).get("url", ""),
            "similarity": row.get("similarity", 0),
        })

    return chunks