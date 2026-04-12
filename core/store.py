"""
Vector store — save and load chunks + their embeddings.

No external vector DB. Uses numpy for similarity search and JSON for metadata.
Chunks and vectors are stored separately:
  data/chunks.json   — chunk text + metadata
  data/vectors.npy   — embedding matrix (one row per chunk)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.chunker import Chunk


DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
VECTORS_FILE = DATA_DIR / "vectors.npy"


def save(chunks: list[Chunk], vectors: list[list[float]], data_dir: Path | None = None) -> None:
    """Persist chunks and their vectors to disk."""
    d = Path(data_dir) if data_dir else DATA_DIR
    d.mkdir(parents=True, exist_ok=True)

    chunks_path = d / "chunks.json"
    vectors_path = d / "vectors.npy"

    chunks_path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(str(vectors_path), np.array(vectors, dtype=np.float32))


def load(data_dir: Path | None = None) -> tuple[list[Chunk], np.ndarray]:
    """Load chunks and vectors from disk. Returns (chunks, matrix)."""
    d = Path(data_dir) if data_dir else DATA_DIR
    chunks_path = d / "chunks.json"
    vectors_path = d / "vectors.npy"

    if not chunks_path.exists() or not vectors_path.exists():
        return [], np.array([], dtype=np.float32)

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(r) for r in raw]
    vectors = np.load(str(vectors_path))
    return chunks, vectors


def count(data_dir: Path | None = None) -> int:
    """Return number of stored chunks."""
    d = Path(data_dir) if data_dir else DATA_DIR
    chunks_path = d / "chunks.json"
    if not chunks_path.exists():
        return 0
    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    return len(raw)


def clear(data_dir: Path | None = None) -> None:
    """Delete all stored chunks and vectors."""
    d = Path(data_dir) if data_dir else DATA_DIR
    for f in [d / "chunks.json", d / "vectors.npy"]:
        if f.exists():
            f.unlink()
