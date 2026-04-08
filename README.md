# RAG Research API (Gemini)

A full Retrieval-Augmented Generation pipeline built with Python FastAPI, Google Gemini, pgvector, and Supabase. **100% free to run.**

## Stack
- **FastAPI** — REST API
- **BeautifulSoup** — web scraping
- **Google Gemini** — embeddings (`text-embedding-004`) + generation (`gemini-1.5-flash`)
- **pgvector** — vector similarity search on Supabase Postgres
- **Supabase** — hosted Postgres + vector store

## Setup

### 1. Supabase
Run `supabase_setup.sql` in your Supabase SQL editor.
> ⚠️ This uses **768 dimensions** (Gemini). If you previously ran the OpenAI version (1536 dims), the SQL will drop and recreate the table.

### 2. Gemini API Key (Free)
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in → **"Get API key"**
3. Copy the key

### 3. Environment
```bash
cp backend/.env.example backend/.env
# Fill in your keys
```

Your `.env` should look like:
```
GEMINI_API_KEY=AIza...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Optional: protect the DELETE /clear endpoint
ADMIN_TOKEN=your_secret_token_here

# Optional: restrict CORS in production
CORS_ORIGINS=https://myapp.com
```

### 4. Install & Run
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

API runs at: http://localhost:8000

## API Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|-------------------------------------|
| GET    | /         | Health check                        |
| POST   | /ingest   | Scrape URL → chunk → embed → store  |
| POST   | /query    | Question → retrieve → generate      |
| DELETE | /clear    | Clear all stored documents          |

## Example Usage

### Ingest a URL
```json
POST /ingest
{
  "url": "https://example.com/article",
  "label": "docs"
}
```

### Ask a question
```json
POST /query
{
  "question": "What is the main topic?",
  "history": [],
  "top_k": 5,
  "similarity_threshold": 0.5
}
```

### Clear all documents (requires ADMIN_TOKEN if set)
```
DELETE /clear
Authorization: Bearer your_secret_token_here
```

## Interactive API Docs
Visit http://localhost:8000/docs for the auto-generated Swagger UI.
