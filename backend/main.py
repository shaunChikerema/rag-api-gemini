from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from scraper import scrape_url
from embedder import embed_and_store
from retriever import retrieve_chunks
from generator import generate_answer

load_dotenv()

app = FastAPI(title="RAG Research API (Gemini)", version="2.0.0")

# CORS — restrict origins via env var in production
# e.g. CORS_ORIGINS=https://myapp.com,https://myapp2.com
_raw_origins = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional bearer token auth for destructive endpoints
_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
security = HTTPBearer(auto_error=False)


def require_admin(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    if _ADMIN_TOKEN:
        if not credentials or credentials.credentials != _ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing admin token")


# ── Request / Response models ─────────────────────────────────────────────────

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


class Source(BaseModel):
    url: str
    chunk: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RAG API running", "model": "gemini-1.5-flash", "embeddings": "text-embedding-004"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Scrape a URL, chunk it, embed with Gemini, and store in pgvector."""
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
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    try:
        chunks = retrieve_chunks(
            req.question,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
        )
        if not chunks:
            return QueryResponse(
                answer="I don't have enough information to answer that question.",
                sources=[]
            )
        answer, sources = generate_answer(req.question, chunks, history=req.history)
        return QueryResponse(answer=answer, sources=sources)
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
