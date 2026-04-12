"""
Chunker — split documents into overlapping text chunks.

Supports: .txt, .md, .py, .json, .pdf (basic text extraction)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED = {".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml", ".toml", ".rst"}


@dataclass
class Chunk:
    text: str
    source: str       # file path
    chunk_index: int
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(
            text=d["text"],
            source=d["source"],
            chunk_index=d["chunk_index"],
            start_char=d["start_char"],
            end_char=d["end_char"],
        )


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Split text into overlapping chunks by character count.
    Tries to split on paragraph boundaries where possible.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to end on a paragraph break
        if end < len(text):
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + overlap:
                end = para_break + 2

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                chunk_index=index,
                start_char=start,
                end_char=end,
            ))
            index += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


def chunk_file(
    path: Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Read a file and return its chunks."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix in SUPPORTED:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        return []

    return chunk_text(text, source=str(path), chunk_size=chunk_size, overlap=overlap)


def chunk_directory(
    directory: Path,
    chunk_size: int = 500,
    overlap: int = 100,
    recursive: bool = True,
) -> list[Chunk]:
    """Chunk all supported files in a directory."""
    directory = Path(directory)
    glob = "**/*" if recursive else "*"
    all_chunks = []

    for path in sorted(directory.glob(glob)):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            chunks = chunk_file(path, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)

    return all_chunks


def _extract_pdf(path: Path) -> str:
    """Basic PDF text extraction without external deps."""
    try:
        import urllib.request
        # Try pdfminer if available
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        pass

    # Fallback: read raw bytes and extract ASCII text runs
    raw = path.read_bytes()
    import re
    text_runs = re.findall(rb"\(([^\)]{4,})\)", raw)
    return " ".join(
        run.decode("latin-1", errors="ignore")
        for run in text_runs
        if not any(c < 32 for c in run)
    )
