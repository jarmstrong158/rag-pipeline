"""
Retriever — find the most relevant chunks for a query using cosine similarity.

No external vector DB. Pure numpy cosine similarity search.
"""

from __future__ import annotations

import numpy as np

from core.chunker import Chunk


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row in matrix.
    Returns a 1-D array of similarity scores, one per row.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid divide-by-zero
    denom = query_norm * matrix_norms
    denom = np.where(denom == 0, 1e-9, denom)

    return (matrix @ query_vec) / denom


def retrieve(
    query_vector: list[float],
    chunks: list[Chunk],
    vectors: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[Chunk, float]]:
    """
    Return the top-K most similar chunks for a query vector.

    Args:
        query_vector: The embedded query as a float list.
        chunks: The stored chunks (parallel to vectors).
        vectors: The embedding matrix (one row per chunk).
        top_k: Number of results to return.
        min_score: Discard results below this similarity threshold.

    Returns:
        List of (Chunk, score) tuples sorted by score descending.
    """
    if not chunks or vectors.size == 0:
        return []

    q = np.array(query_vector, dtype=np.float32)
    scores = cosine_similarity(q, vectors)

    # Get top-K indices sorted by score descending
    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= min_score:
            results.append((chunks[idx], score))

    return results


def retrieve_deduplicated(
    query_vector: list[float],
    chunks: list[Chunk],
    vectors: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[Chunk, float]]:
    """
    Like retrieve(), but deduplicate by source file — at most one chunk per source.
    Useful when a single document dominates the results.
    """
    results = retrieve(query_vector, chunks, vectors, top_k=top_k * 3, min_score=min_score)

    seen_sources: set[str] = set()
    deduped = []
    for chunk, score in results:
        if chunk.source not in seen_sources:
            seen_sources.add(chunk.source)
            deduped.append((chunk, score))
        if len(deduped) >= top_k:
            break

    return deduped
