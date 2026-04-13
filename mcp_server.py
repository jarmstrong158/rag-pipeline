#!/usr/bin/env python3
"""
RAG Pipeline MCP Server

Thin HTTP client over api.py. Starts the API server automatically and opens
the browser so the user can see live progress. All logic lives in api.py —
this file only routes MCP tool calls to REST endpoints.

Tools:
  rag_query    — ask a question, get an answer from your ingested documents
  rag_ingest   — ingest a file or directory into the vector store
  rag_status   — how many chunks are stored, are the models ready
  rag_clear    — wipe the vector store
"""

import json
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import webbrowser

API_BASE = "http://localhost:8000"
API_PY   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.py")

_browser_opened = False


TOOLS = [
    {
        "name": "rag_query",
        "description": (
            "Ask a question and get an answer from your ingested documents. "
            "The pipeline embeds the question, finds the most relevant chunks, "
            "and generates a grounded answer using the local LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer"},
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 5)", "default": 5},
            },
            "required": ["question"],
        },
    },
    {
        "name": "rag_ingest",
        "description": (
            "Ingest a file or directory of documents into the vector store. "
            "Supported: .txt .md .py .js .json .yaml .toml .rst .pdf"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to a file or directory"},
                "chunk_size": {"type": "integer", "description": "Characters per chunk (default 500)", "default": 500},
                "overlap": {"type": "integer", "description": "Overlap between chunks (default 100)", "default": 100},
            },
            "required": ["path"],
        },
    },
    {
        "name": "rag_status",
        "description": "Show how many chunks are stored and whether the models are ready.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "rag_clear",
        "description": "Wipe the entire vector store. All ingested documents will be removed.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _is_api_running() -> bool:
    try:
        urllib.request.urlopen(f"{API_BASE}/health", timeout=2)
        return True
    except Exception:
        return False


def _ensure_api():
    """Start api.py if not already running, then open browser on first call."""
    global _browser_opened

    if not _is_api_running():
        env = os.environ.copy()
        env["RAG_NO_BROWSER"] = "1"
        subprocess.Popen(
            [sys.executable, API_PY],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Wait up to 30 s for the server to come up
        for _ in range(30):
            time.sleep(1)
            if _is_api_running():
                break

    if not _browser_opened:
        webbrowser.open(API_BASE)
        _browser_opened = True


def _api(method: str, path: str, body: dict | None = None) -> dict:
    _ensure_api()
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        try:
            detail = json.loads(body_text).get("detail", body_text)
        except Exception:
            detail = body_text
        return {"error": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"error": str(e)}


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_rag_query(args: dict) -> dict:
    question = args.get("question", "").strip()
    if not question:
        return {"error": "question is required"}
    _ensure_api()
    import urllib.parse
    url = f"{API_BASE}/?q={urllib.parse.quote(question)}"
    webbrowser.open(url)
    return {"status": "Query submitted to UI", "question": question, "url": url}


def handle_rag_ingest(args: dict) -> dict:
    path = args.get("path", "").strip()
    if not path:
        return {"error": "path is required"}
    return _api("POST", "/ingest", {
        "path": path,
        "chunk_size": int(args.get("chunk_size", 500)),
        "overlap": int(args.get("overlap", 100)),
    })


def handle_rag_status(_args: dict) -> dict:
    return _api("GET", "/health")


def handle_rag_clear(_args: dict) -> dict:
    return _api("POST", "/clear")


HANDLERS = {
    "rag_query":  handle_rag_query,
    "rag_ingest": handle_rag_ingest,
    "rag_status": handle_rag_status,
    "rag_clear":  handle_rag_clear,
}


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "rag-pipeline", "version": "1.0.0"},
                },
            }

        elif method == "notifications/initialized":
            continue

        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            handler = HANDLERS.get(tool_name)

            if handler is None:
                result = {"error": f"Unknown tool: {tool_name}"}
                is_error = True
            else:
                try:
                    result = handler(tool_args)
                    is_error = "error" in result
                except Exception as e:
                    result = {"error": f"Tool '{tool_name}' failed: {e}"}
                    is_error = True

            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": is_error,
                },
            }

        elif method.startswith("notifications/"):
            continue

        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
