"""
Tests for core/store.py — no Ollama required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.chunker import Chunk
from core.store import save, load, count, clear


def make_chunk(text: str = "hello", source: str = "test.txt", index: int = 0) -> Chunk:
    return Chunk(text=text, source=source, chunk_index=index, start_char=0, end_char=len(text))


def make_vectors(n: int, dims: int = 4) -> list[list[float]]:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dims)).astype(np.float32).tolist()


# ── save + load roundtrip ─────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load_basic(self, tmp_path):
        chunks = [make_chunk("chunk A", index=0), make_chunk("chunk B", index=1)]
        vectors = make_vectors(2)
        save(chunks, vectors, tmp_path)

        loaded_chunks, loaded_vecs = load(tmp_path)
        assert len(loaded_chunks) == 2
        assert loaded_vecs.shape == (2, 4)

    def test_roundtrip_text_preserved(self, tmp_path):
        chunks = [make_chunk("Hello world", index=0)]
        vectors = make_vectors(1)
        save(chunks, vectors, tmp_path)

        loaded_chunks, _ = load(tmp_path)
        assert loaded_chunks[0].text == "Hello world"

    def test_roundtrip_metadata_preserved(self, tmp_path):
        c = Chunk(text="x", source="my/file.txt", chunk_index=7, start_char=100, end_char=200)
        save([c], make_vectors(1), tmp_path)

        loaded, _ = load(tmp_path)
        assert loaded[0].source == "my/file.txt"
        assert loaded[0].chunk_index == 7
        assert loaded[0].start_char == 100
        assert loaded[0].end_char == 200

    def test_roundtrip_vectors_preserved(self, tmp_path):
        vecs = [[1.0, 2.0, 3.0, 4.0]]
        save([make_chunk()], vecs, tmp_path)

        _, loaded_vecs = load(tmp_path)
        assert loaded_vecs.shape == (1, 4)
        assert abs(float(loaded_vecs[0][0]) - 1.0) < 1e-4

    def test_unicode_text_preserved(self, tmp_path):
        chunks = [make_chunk("héllo wörld 🎉", index=0)]
        save(chunks, make_vectors(1), tmp_path)

        loaded, _ = load(tmp_path)
        assert loaded[0].text == "héllo wörld 🎉"

    def test_many_chunks(self, tmp_path):
        n = 50
        chunks = [make_chunk(f"chunk {i}", index=i) for i in range(n)]
        vectors = make_vectors(n, dims=8)
        save(chunks, vectors, tmp_path)

        loaded_chunks, loaded_vecs = load(tmp_path)
        assert len(loaded_chunks) == n
        assert loaded_vecs.shape == (n, 8)

    def test_load_missing_returns_empty(self, tmp_path):
        chunks, vecs = load(tmp_path)
        assert chunks == []
        assert vecs.size == 0

    def test_load_missing_chunks_file(self, tmp_path):
        # Only vectors file exists — should return empty
        np.save(str(tmp_path / "vectors.npy"), np.zeros((3, 4)))
        chunks, vecs = load(tmp_path)
        assert chunks == []

    def test_load_missing_vectors_file(self, tmp_path):
        # Only chunks file exists — should return empty
        (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
        chunks, vecs = load(tmp_path)
        assert chunks == []

    def test_save_creates_data_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        save([make_chunk()], make_vectors(1), nested)
        assert (nested / "chunks.json").exists()
        assert (nested / "vectors.npy").exists()

    def test_overwrite_replaces_data(self, tmp_path):
        save([make_chunk("first")], make_vectors(1), tmp_path)
        save([make_chunk("second"), make_chunk("third")], make_vectors(2), tmp_path)

        loaded, vecs = load(tmp_path)
        assert len(loaded) == 2
        assert loaded[0].text == "second"

    def test_float32_dtype(self, tmp_path):
        save([make_chunk()], [[1.0, 2.0, 3.0]], tmp_path)
        _, vecs = load(tmp_path)
        assert vecs.dtype == np.float32


# ── count ─────────────────────────────────────────────────────────────────────

class TestCount:
    def test_count_zero_when_empty(self, tmp_path):
        assert count(tmp_path) == 0

    def test_count_after_save(self, tmp_path):
        chunks = [make_chunk(index=i) for i in range(7)]
        save(chunks, make_vectors(7), tmp_path)
        assert count(tmp_path) == 7

    def test_count_missing_file(self, tmp_path):
        assert count(tmp_path / "nonexistent") == 0


# ── clear ─────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_removes_files(self, tmp_path):
        save([make_chunk()], make_vectors(1), tmp_path)
        assert (tmp_path / "chunks.json").exists()
        assert (tmp_path / "vectors.npy").exists()

        clear(tmp_path)
        assert not (tmp_path / "chunks.json").exists()
        assert not (tmp_path / "vectors.npy").exists()

    def test_clear_empty_store_no_error(self, tmp_path):
        # Should not raise if files don't exist
        clear(tmp_path)

    def test_clear_resets_count(self, tmp_path):
        save([make_chunk(index=i) for i in range(5)], make_vectors(5), tmp_path)
        assert count(tmp_path) == 5
        clear(tmp_path)
        assert count(tmp_path) == 0

    def test_load_after_clear_returns_empty(self, tmp_path):
        save([make_chunk()], make_vectors(1), tmp_path)
        clear(tmp_path)
        chunks, vecs = load(tmp_path)
        assert chunks == []
        assert vecs.size == 0
