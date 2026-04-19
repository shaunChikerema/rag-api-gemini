"""
conversations.py — persistent conversation memory via Supabase.

Supabase table required (run once in the SQL editor):

    create table conversations (
        id          bigserial primary key,
        session_id  text not null,
        role        text not null check (role in ('user', 'assistant')),
        content     text not null,
        created_at  timestamptz default now()
    );

    create index on conversations (session_id, created_at);

That's it — no vectors, no RLS needed unless you want per-user isolation.
"""

from typing import List, Optional
from db import get_supabase

TABLE = "conversations"
DEFAULT_SESSION = "default"
MAX_HISTORY = 40  # max turns to load (20 pairs)


def save_turn(role: str, content: str, session_id: str = DEFAULT_SESSION) -> None:
    """Persist a single conversation turn."""
    supabase = get_supabase()
    supabase.table(TABLE).insert({
        "session_id": session_id,
        "role": role,
        "content": content,
    }).execute()


def load_history(session_id: str = DEFAULT_SESSION, limit: int = MAX_HISTORY) -> List[dict]:
    """
    Load the most recent `limit` turns for a session, oldest-first
    (so they can be passed directly to the LLM as message history).
    """
    supabase = get_supabase()
    result = (
        supabase.table(TABLE)
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    # Reverse so oldest turn is first
    turns = list(reversed(result.data))
    return [{"role": t["role"], "content": t["content"]} for t in turns]


def list_sessions() -> List[str]:
    """Return all distinct session IDs, most recently active first."""
    supabase = get_supabase()
    result = (
        supabase.table(TABLE)
        .select("session_id, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    seen = set()
    sessions = []
    for row in result.data:
        sid = row["session_id"]
        if sid not in seen:
            seen.add(sid)
            sessions.append(sid)
    return sessions


def clear_session(session_id: str = DEFAULT_SESSION) -> int:
    """Delete all turns for a session. Returns number of rows deleted."""
    supabase = get_supabase()
    result = (
        supabase.table(TABLE)
        .delete()
        .eq("session_id", session_id)
        .execute()
    )
    return len(result.data)
