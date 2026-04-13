# How the RAG Pipeline Works

RAG stands for Retrieval-Augmented Generation. Instead of asking an LLM to answer from its training data, RAG first searches your own documents for relevant passages, then asks the LLM to answer using only those passages. This keeps answers grounded in your actual content.

## The Four Steps

### 1. Ingest
You point the pipeline at a file or directory. The chunker reads every supported file (.txt, .md, .py, .json, etc.) and splits the text into overlapping chunks of about 500 characters. Overlap (default 100 chars) means no sentence gets cut off at a boundary. Each chunk records its source file and position.

### 2. Embed
Every chunk is converted into a vector — a list of ~768 numbers — using the `nomic-embed-text` model running locally via llama-cpp-python. Text with similar meaning produces similar vectors. This is how the pipeline understands meaning, not just keywords. Vectors are stored in `data/vectors.npy` alongside `data/chunks.json`.

### 3. Retrieve
When you ask a question, the question itself is embedded into a vector using the same model. The pipeline then computes cosine similarity between your question vector and every stored chunk vector. The top-k most similar chunks are returned. This finds chunks that are *about* the same topic as your question, even if they use different words.

### 4. Generate
The retrieved chunks are assembled into a prompt: "Here are relevant passages: [chunk1] [chunk2] ... Answer this question using only these passages: [your question]". This prompt is sent to the Llama 3.2 3B Instruct model (also running locally via llama-cpp-python). The model reads only those chunks and writes an answer grounded in them.

## The Components

- **chunker.py** — splits files into overlapping text chunks
- **embedder.py** — converts text to vectors using nomic-embed-text GGUF
- **generator.py** — sends prompt + chunks to Llama 3.2 3B GGUF, returns answer
- **retriever.py** — cosine similarity search over the stored vectors
- **store.py** — saves and loads chunks.json, vectors.npy, and meta.json
- **api.py** — FastAPI server on port 8000, exposes all operations as REST endpoints
- **mcp_server.py** — MCP server for Claude Desktop, thin HTTP client over api.py
- **ui.html** — browser UI with progress bar showing the three pipeline stages

## What Makes a Good Question

Questions work best when they match the vocabulary and content of your ingested documents. If you ingested Python code, ask about function names, parameters, and what the code does. If you ingested documentation, ask conceptual questions. The pipeline finds passages that are *semantically similar* to your question — so the closer your question is to how the source material talks about a topic, the better the answer.

## Why Local Models

All inference runs on your machine. No API calls, no data leaving your computer, no ongoing cost. The tradeoff is speed — the 3B Llama model takes 30–90 seconds to generate on CPU.

## The Store

- `data/chunks.json` — all ingested text chunks with source metadata
- `data/vectors.npy` — numpy array of embeddings, one row per chunk
- `data/meta.json` — just the chunk count, for fast status checks without loading everything
