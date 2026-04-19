import json
from fastapi import FastAPI, HTTPException, Depends, Security, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
from dotenv import load_dotenv

from scraper import scrape_url, scrape_pdf
from embedder import embed_and_store, list_sources, delete_source
from retriever import retrieve_chunks
from generator import generate_answer, generate_answer_stream
from conversations import save_turn, load_history, clear_session, list_sessions

load_dotenv()

app = FastAPI(title="RAG Research API", version="5.0.0")

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
    similarity_threshold: Optional[float] = 0.75
    stream: Optional[bool] = False
    session_id: Optional[str] = "default"


class Source(BaseModel):
    url: str
    chunk: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    used_fallback: bool = False


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
        "version": "5.0.0",
        "features": [
            "conversation-aware retrieval",
            "query expansion",
            "streaming",
            "persistent conversation memory",
            "fallback to model knowledge",
            "pdf upload",
            "knowledge base listing",
            "source deletion",
        ],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
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


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...), label: Optional[str] = None):
    """Upload and ingest a PDF file."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        chunks = scrape_pdf(tmp_path, label=label or file.filename)
        os.unlink(tmp_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")
        count = embed_and_store(chunks)
        return IngestResponse(message="PDF ingestion complete", chunks_stored=count)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def get_sources():
    """List all ingested sources with chunk counts."""
    try:
        sources = list_sources()
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sources")
async def delete_source_endpoint(url: str):
    """Delete all chunks for a specific source URL."""
    try:
        deleted = delete_source(url)
        return {"message": f"Deleted {deleted} chunks for {url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        persisted = load_history(session_id=req.session_id)
        history = persisted if persisted else (req.history or [])

        chunks = retrieve_chunks(
            req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            history=history,
            expand=True,
        )

        save_turn("user", req.question, session_id=req.session_id)

        if req.stream:
            token_iter, sources_output, used_fallback = generate_answer_stream(
                req.question, chunks, history=history
            )

            async def event_stream():
                full_answer = []
                for token in token_iter:
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                save_turn("assistant", "".join(full_answer), session_id=req.session_id)
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_output, 'used_fallback': used_fallback})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        answer, sources_output, used_fallback = generate_answer(
            req.question, chunks, history=history
        )
        save_turn("assistant", answer, session_id=req.session_id)
        return QueryResponse(answer=answer, sources=sources_output, used_fallback=used_fallback)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str = "default"):
    try:
        history = load_history(session_id=session_id)
        return HistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def get_sessions():
    try:
        return {"sessions": list_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{session_id}")
async def delete_history(session_id: str = "default"):
    try:
        deleted = clear_session(session_id=session_id)
        return {"message": f"Cleared {deleted} turns for session '{session_id}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear", dependencies=[Depends(require_admin)])
async def clear_documents():
    try:
        from db import get_supabase
        client = get_supabase()
        client.table("documents").delete().neq("id", 0).execute()
        return {"message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
