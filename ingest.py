"""
ingest.py — CLI tool to chunk and embed documents into the vector store.

Usage:
  python ingest.py <path>                  # ingest a file or directory
  python ingest.py <path> --chunk-size 800
  python ingest.py <path> --overlap 150
  python ingest.py --clear                 # wipe the store
  python ingest.py --count                 # show stored chunk count
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk (default: 500)")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap between chunks (default: 100)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")
    parser.add_argument("--clear", action="store_true", help="Clear the vector store")
    parser.add_argument("--count", action="store_true", help="Show number of stored chunks")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory")

    args = parser.parse_args()

    # Lazy imports after arg parsing
    from core import chunker, embedder, store

    if args.clear:
        store.clear(args.data_dir)
        print("Vector store cleared.")
        return

    if args.count:
        n = store.count(args.data_dir)
        print(f"Stored chunks: {n}")
        return

    if not args.path:
        parser.print_help()
        sys.exit(1)

    path = Path(args.path)
    if not path.exists():
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # Preflight: embedder must be available
    try:
        embedder.require_available()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Chunk
    print(f"Chunking {'directory' if path.is_dir() else 'file'}: {path}")
    t0 = time.time()

    if path.is_dir():
        chunks = chunker.chunk_directory(
            path,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recursive=not args.no_recursive,
        )
    else:
        chunks = chunker.chunk_file(path, chunk_size=args.chunk_size, overlap=args.overlap)

    if not chunks:
        print("No supported files found or all files were empty.")
        return

    print(f"  {len(chunks)} chunks from {len({c.source for c in chunks})} file(s)  ({time.time()-t0:.1f}s)")

    # Load existing store so we can append
    existing_chunks, existing_vectors = store.load(args.data_dir)

    # Embed
    print(f"Embedding {len(chunks)} chunks via Ollama...")
    t1 = time.time()

    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)

    print(f"  Done  ({time.time()-t1:.1f}s)")

    # Merge + save
    all_chunks = existing_chunks + chunks
    import numpy as np
    if existing_vectors.size > 0:
        all_vectors = existing_vectors.tolist() + vectors
    else:
        all_vectors = vectors

    store.save(all_chunks, all_vectors, args.data_dir)

    total = store.count(args.data_dir)
    print(f"Saved. Total chunks in store: {total}")


if __name__ == "__main__":
    main()
