import httpx
from bs4 import BeautifulSoup
from typing import List, Optional
import re

CHUNK_SIZE = 2000     # characters per chunk (~500 tokens, good for Gemini)
CHUNK_OVERLAP = 200  # overlap to preserve context across chunks


def scrape_url(url: str, label: Optional[str] = None) -> List[dict]:
    """
    Fetch a URL, extract clean text, split into overlapping chunks.
    Returns a list of chunk dicts: {content, url, label}
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    with httpx.Client(timeout=15, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    # Validate content type
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}. Only HTML pages are supported.")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Extract and clean text
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = _split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    return [
        {"content": chunk, "url": url, "label": label or ""}
        for chunk in chunks
    ]


def _split_into_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, always breaking at word boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size

        # Break at word boundary if not at end of text
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks
