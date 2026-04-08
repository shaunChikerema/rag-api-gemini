-- Run this in your Supabase SQL editor before starting the API
-- NOTE: Using 768 dimensions for Google Gemini text-embedding-004

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Drop old table if switching from OpenAI (1536 dims → 768 dims)
-- WARNING: This deletes all stored documents. Comment out if fresh install.
drop table if exists documents;

-- 3. Create documents table
create table if not exists documents (
  id        bigserial primary key,
  content   text              not null,
  embedding vector(768)       not null,
  metadata  jsonb             default '{}'
);

-- 4. Index for fast cosine similarity search
create index if not exists documents_embedding_idx
  on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 5. Match function used by the retriever
create or replace function match_documents(
  query_embedding  vector(768),
  match_count      int     default 5,
  match_threshold  float   default 0.5
)
returns table (
  id         bigint,
  content    text,
  metadata   jsonb,
  similarity float
)
language sql stable
as $$
  select
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where 1 - (embedding <=> query_embedding) > match_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
