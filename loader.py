"""
loader.py — extract text from PDF, TXT, and Markdown files.
Returns a list of Document chunks ready for embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    text: str
    source: str       # filename
    chunk_id: int
    metadata: dict


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks by word boundaries.
    chunk_size and overlap are measured in words.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def _clean(text: str) -> str:
    """Remove excessive whitespace and control characters."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_txt(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _clean(raw)
    chunks = _chunk_text(text)
    return [
        Document(text=c, source=path.name, chunk_id=i, metadata={"type": "txt"})
        for i, c in enumerate(chunks)
    ]


def load_pdf(path: Path) -> list[Document]:
    """
    Extract text from PDF using pypdf (pure Python).
    Falls back to a clear error message if pypdf isn't installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(str(path))
    pages_text: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            pages_text.append(_clean(t))

    full_text = "\n\n".join(pages_text)
    chunks = _chunk_text(full_text)
    return [
        Document(
            text=c,
            source=path.name,
            chunk_id=i,
            metadata={"type": "pdf", "total_pages": len(reader.pages)},
        )
        for i, c in enumerate(chunks)
    ]


def load_markdown(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip markdown syntax for cleaner embeddings
    text = re.sub(r'#{1,6}\s', '', raw)          # headings
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text) # code
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # bold/italic
    text = _clean(text)
    chunks = _chunk_text(text)
    return [
        Document(text=c, source=path.name, chunk_id=i, metadata={"type": "md"})
        for i, c in enumerate(chunks)
    ]


LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".md": load_markdown,
}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(LOADERS)}")
    return LOADERS[suffix](path)


def load_directory(directory: Path) -> list[Document]:
    """Load all supported files from a directory (non-recursive)."""
    docs: list[Document] = []
    for suffix in LOADERS:
        for file in sorted(directory.glob(f"*{suffix}")):
            docs.extend(load_file(file))
    return docs
