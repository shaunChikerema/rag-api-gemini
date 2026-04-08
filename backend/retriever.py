import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
from db import get_supabase

load_dotenv()

EMBEDDING_MODEL = "models/text-embedding-004"


def _get_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env")
    genai.configure(api_key=api_key)
    return genai


def retrieve_chunks(question: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[dict]:
    """
    Embed the question with Gemini and find the top_k most similar chunks
    using pgvector cosine similarity via Supabase RPC.
    Returns list of {content, url, similarity}
    """
    _get_genai()

    # Embed the question (use retrieval_query task type for better results)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=question,
        task_type="retrieval_query",
    )
    query_embedding = result["embedding"]

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
