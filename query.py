"""
query.py — CLI tool to ask a question and get an answer from the RAG pipeline.

Usage:
  python query.py "What is the capital of France?"
  python query.py "Explain the bond system" --top-k 8
  python query.py "Summarize this" --stream
  python query.py "What?" --no-generate      # retrieval only, show chunks
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum similarity score")
    parser.add_argument("--stream", action="store_true", help="Stream the response token by token")
    parser.add_argument("--no-generate", action="store_true", help="Show retrieved chunks only, skip generation")
    parser.add_argument("--model", default=None, help="Override the generation model")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory")

    args = parser.parse_args()

    from core import embedder, store, retriever, generator

    # Load store
    chunks, vectors = store.load(args.data_dir)
    if not chunks:
        print("The vector store is empty. Run: python ingest.py <path>", file=sys.stderr)
        sys.exit(1)

    # Embed query
    try:
        embedder.require_available()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching {len(chunks)} chunks...")
    t0 = time.time()
    query_vector = embedder.embed(args.question)
    results = retriever.retrieve(
        query_vector,
        chunks,
        vectors,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print(f"Retrieved {len(results)} chunks  ({time.time()-t0:.2f}s)\n")

    if not results:
        print("No relevant chunks found.")
        return

    if args.no_generate:
        _print_chunks(results)
        return

    # Generate
    gen_model = args.model or generator.DEFAULT_MODEL
    try:
        generator.require_available(gen_model)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nChunks that would have been used:")
        _print_chunks(results)
        sys.exit(1)

    print(f"Answer (via {gen_model}):")
    print("-" * 60)

    t1 = time.time()
    answer = generator.generate(
        args.question,
        results,
        model=gen_model,
        stream=args.stream,
    )
    elapsed = time.time() - t1

    if not args.stream:
        print(answer)

    print("-" * 60)
    print(f"Generated in {elapsed:.1f}s")
    print(f"\nSources:")
    for chunk, score in results:
        print(f"  [{score:.3f}] {chunk.source}  (chunk {chunk.chunk_index})")


def _print_chunks(results: list) -> None:
    for i, (chunk, score) in enumerate(results, 1):
        print(f"[{i}] score={score:.3f}  source={chunk.source}  chunk={chunk.chunk_index}")
        print(f"    {chunk.text[:200].replace(chr(10), ' ')}...")
        print()


if __name__ == "__main__":
    main()
