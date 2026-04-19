import json
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from scraper import scrape_url
from embedder import embed_and_store
from retriever import retrieve_chunks
from generator import generate_answer, generate_answer_stream
from conversations import save_turn, load_history, clear_session, list_sessions

load_dotenv()

app = FastAPI(title="RAG Research API", version="4.0.0")

_raw_origins = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
security = HTTPBearer(auto_error=False)


def require_admin(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    if _ADMIN_TOKEN:
        if not credentials or credentials.credentials != _ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing admin token")


# ── Models ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: str
    label: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    chunks_stored: int


class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.5
    stream: Optional[bool] = False
    session_id: Optional[str] = "default"  # for conversation memory


class Source(BaseModel):
    url: str
    chunk: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    used_fallback: bool = False  # true when answered from model knowledge


class HistoryResponse(BaseModel):
    session_id: str
    history: List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "RAG API running",
        "model": "llama-3.3-70b-versatile",
        "embeddings": "gemini-embedding-001",
        "version": "4.0.0",
        "features": [
            "conversation-aware retrieval",
            "query expansion",
            "streaming",
            "persistent conversation memory",
            "fallback to model knowledge",
        ],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Scrape a URL, chunk, embed with Gemini, store in pgvector."""
    try:
        chunks = scrape_url(req.url, label=req.label)
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from URL")
        count = embed_and_store(chunks)
        return IngestResponse(message="Ingestion complete", chunks_stored=count)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Retrieve relevant chunks and generate a grounded answer with citations.

    - Merges client-provided history with any persisted history for the session
    - Falls back to model knowledge when no chunks match
    - Pass stream=true for SSE token streaming
    - Pass session_id to scope conversation memory (default: "default")

    SSE event format (stream=true):
      data: {"type": "token",   "content": "..."}
      data: {"type": "sources", "sources": [...], "used_fallback": bool}
      data: {"type": "done"}
    """
    try:
        # Merge client history with persisted session history
        # Persisted history takes precedence for older turns; client history
        # may include the very latest turn not yet saved.
        persisted = load_history(session_id=req.session_id)

        # Deduplicate: if client sent history that overlaps with persisted, use persisted
        # Simple strategy: if persisted is non-empty, use it; otherwise use client history
        history = persisted if persisted else (req.history or [])

        # Retrieve chunks — passing history so query rewriting works
        chunks = retrieve_chunks(
            req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            history=history,
            expand=True,
        )

        # Persist the user's question
        save_turn("user", req.question, session_id=req.session_id)

        # ── Streaming ──
        if req.stream:
            token_iter, sources_output, used_fallback = generate_answer_stream(
                req.question, chunks, history=history
            )

            async def event_stream():
                full_answer = []
                for token in token_iter:
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                # Persist the assistant's full answer once streaming completes
                save_turn("assistant", "".join(full_answer), session_id=req.session_id)

                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_output, 'used_fallback': used_fallback})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # ── Non-streaming ──
        answer, sources_output, used_fallback = generate_answer(
            req.question, chunks, history=history
        )

        # Persist assistant's answer
        save_turn("assistant", answer, session_id=req.session_id)

        return QueryResponse(answer=answer, sources=sources_output, used_fallback=used_fallback)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str = "default"):
    """
    Load persisted conversation history for a session.
    The frontend calls this on startup to restore the chat.
    """
    try:
        history = load_history(session_id=session_id)
        return HistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def get_sessions():
    """List all conversation session IDs."""
    try:
        return {"sessions": list_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{session_id}")
async def delete_history(session_id: str = "default"):
    """Clear conversation history for a session."""
    try:
        deleted = clear_session(session_id=session_id)
        return {"message": f"Cleared {deleted} turns for session '{session_id}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear", dependencies=[Depends(require_admin)])
async def clear_documents():
    """Clear all stored document chunks. Requires ADMIN_TOKEN if set."""
    try:
        from db import get_supabase
        client = get_supabase()
        client.table("documents").delete().neq("id", 0).execute()
        return {"message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
