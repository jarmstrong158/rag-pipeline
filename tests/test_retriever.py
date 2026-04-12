"""
Tests for core/retriever.py — no Ollama required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.chunker import Chunk
from core.retriever import cosine_similarity, retrieve, retrieve_deduplicated


def make_chunk(text: str, source: str = "test.txt", index: int = 0) -> Chunk:
    return Chunk(text=text, source=source, chunk_index=index, start_char=0, end_char=len(text))


def make_vector(dims: int = 4, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dims).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ── cosine_similarity ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        scores = cosine_similarity(v, v.reshape(1, -1))
        assert abs(scores[0] - 1.0) < 1e-5

    def test_opposite_vectors_score_minus_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        neg = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(v, neg)
        assert abs(scores[0] - (-1.0)) < 1e-5

    def test_orthogonal_vectors_score_zero(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([[0.0, 1.0]], dtype=np.float32)
        scores = cosine_similarity(v, m)
        assert abs(scores[0]) < 1e-5

    def test_multiple_rows(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array([
            [1.0, 0.0, 0.0],  # identical → 1.0
            [0.0, 1.0, 0.0],  # orthogonal → 0.0
            [-1.0, 0.0, 0.0], # opposite → -1.0
        ], dtype=np.float32)
        scores = cosine_similarity(q, matrix)
        assert len(scores) == 3
        assert abs(scores[0] - 1.0) < 1e-5
        assert abs(scores[1]) < 1e-5
        assert abs(scores[2] - (-1.0)) < 1e-5

    def test_zero_vector_no_crash(self):
        q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        m = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(q, m)
        assert np.isfinite(scores[0])

    def test_1d_matrix_handled(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([1.0, 0.0], dtype=np.float32)  # 1D, not 2D
        scores = cosine_similarity(q, m)
        assert abs(scores[0] - 1.0) < 1e-5

    def test_returns_ndarray(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        scores = cosine_similarity(q, m)
        assert isinstance(scores, np.ndarray)


# ── retrieve ──────────────────────────────────────────────────────────────────

class TestRetrieve:
    def _make_store(self, n: int = 5, dims: int = 8):
        """Make n chunks with random normalized vectors."""
        chunks = [make_chunk(f"chunk {i}", index=i) for i in range(n)]
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, dims)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return chunks, vecs

    def test_empty_store_returns_empty(self):
        result = retrieve([1.0, 0.0], [], np.array([], dtype=np.float32))
        assert result == []

    def test_returns_top_k(self):
        chunks, vecs = self._make_store(10)
        query = vecs[0].tolist()  # exact match for chunk 0
        results = retrieve(query, chunks, vecs, top_k=3)
        assert len(results) == 3

    def test_top_k_capped_at_store_size(self):
        chunks, vecs = self._make_store(3)
        results = retrieve(vecs[0].tolist(), chunks, vecs, top_k=100)
        assert len(results) == 3

    def test_exact_match_ranks_first(self):
        chunks, vecs = self._make_store(5)
        # Query with exact copy of chunk 2's vector → chunk 2 should be #1
        query = vecs[2].tolist()
        results = retrieve(query, chunks, vecs, top_k=5)
        top_chunk, top_score = results[0]
        assert top_chunk.chunk_index == 2
        assert abs(top_score - 1.0) < 1e-4

    def test_sorted_by_score_descending(self):
        chunks, vecs = self._make_store(5)
        results = retrieve(vecs[0].tolist(), chunks, vecs, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_chunk_and_float_tuples(self):
        chunks, vecs = self._make_store(3)
        results = retrieve(vecs[0].tolist(), chunks, vecs, top_k=2)
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    def test_min_score_filters_results(self):
        chunks, vecs = self._make_store(5)
        # With very high min_score only exact match should pass
        results = retrieve(vecs[0].tolist(), chunks, vecs, top_k=5, min_score=0.999)
        assert len(results) == 1
        assert results[0][1] >= 0.999

    def test_min_score_zero_no_filter(self):
        chunks, vecs = self._make_store(5)
        results = retrieve(vecs[0].tolist(), chunks, vecs, top_k=5, min_score=0.0)
        assert len(results) == 5

    def test_top_k_one(self):
        chunks, vecs = self._make_store(5)
        results = retrieve(vecs[0].tolist(), chunks, vecs, top_k=1)
        assert len(results) == 1


# ── retrieve_deduplicated ─────────────────────────────────────────────────────

class TestRetrieveDeduplicated:
    def test_deduplicates_by_source(self):
        # Two chunks from same source — only one should appear
        source_a = "doc_a.txt"
        source_b = "doc_b.txt"
        chunks = [
            make_chunk("a1", source=source_a, index=0),
            make_chunk("a2", source=source_a, index=1),
            make_chunk("b1", source=source_b, index=0),
        ]
        # Make vectors where a1 and a2 both score high
        vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        query = vecs[0].tolist()
        results = retrieve_deduplicated(query, chunks, vecs, top_k=3)

        sources = [c.source for c, _ in results]
        assert len(sources) == len(set(sources))  # no duplicates

    def test_returns_up_to_top_k(self):
        chunks = [
            make_chunk(f"chunk {i}", source=f"doc_{i}.txt", index=i)
            for i in range(10)
        ]
        rng = np.random.default_rng(7)
        vecs = rng.standard_normal((10, 8)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        results = retrieve_deduplicated(vecs[0].tolist(), chunks, vecs, top_k=4)
        assert len(results) <= 4

    def test_empty_store(self):
        result = retrieve_deduplicated([1.0, 0.0], [], np.array([], dtype=np.float32))
        assert result == []
