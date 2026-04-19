import httpx
from bs4 import BeautifulSoup
from typing import List, Optional
import re

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def scrape_url(url: str, label: Optional[str] = None) -> List[dict]:
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

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}. Only HTML pages are supported.")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = _split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    return [{"content": chunk, "url": url, "label": label or ""} for chunk in chunks]


def scrape_pdf(path: str, label: Optional[str] = None) -> List[dict]:
    """Extract text from a PDF file and split into chunks."""
    try:
        import pypdf
    except ImportError:
        raise RuntimeError("pypdf is required for PDF ingestion. Add it to requirements.txt.")

    text_parts = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
    if not text:
        return []

    chunks = _split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    source_label = label or "pdf"
    return [{"content": chunk, "url": f"pdf://{source_label}", "label": source_label} for chunk in chunks]


def _split_into_chunks(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks
