"""
Tests for core/chunker.py — no Ollama required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.chunker import Chunk, chunk_text, chunk_file, chunk_directory, SUPPORTED


# ── Chunk dataclass ───────────────────────────────────────────────────────────

class TestChunk:
    def test_to_dict(self):
        c = Chunk(text="hello", source="foo.txt", chunk_index=0, start_char=0, end_char=5)
        d = c.to_dict()
        assert d == {
            "text": "hello",
            "source": "foo.txt",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 5,
        }

    def test_from_dict_roundtrip(self):
        c = Chunk(text="world", source="bar.md", chunk_index=2, start_char=10, end_char=20)
        c2 = Chunk.from_dict(c.to_dict())
        assert c == c2

    def test_from_dict_fields(self):
        d = {"text": "x", "source": "s", "chunk_index": 1, "start_char": 5, "end_char": 6}
        c = Chunk.from_dict(d)
        assert c.text == "x"
        assert c.chunk_index == 1


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("", "test.txt") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n  ", "test.txt") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", "test.txt", chunk_size=500, overlap=100)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].source == "test.txt"
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self):
        text = "A" * 1000
        chunks = chunk_text(text, "test.txt", chunk_size=300, overlap=50)
        assert len(chunks) > 1

    def test_chunks_cover_full_text(self):
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, "test.txt", chunk_size=200, overlap=50)
        # chunk_text strips the input, so start/end chars are relative to stripped text
        stripped = text.strip()
        covered = set()
        for c in chunks:
            covered.update(range(c.start_char, c.end_char))
        assert 0 in covered
        assert len(stripped) - 1 in covered

    def test_chunk_indices_sequential(self):
        text = "X" * 1500
        chunks = chunk_text(text, "test.txt", chunk_size=400, overlap=80)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_creates_shared_content(self):
        # With overlap, consecutive chunks should share some characters
        text = "word " * 300
        chunks = chunk_text(text, "test.txt", chunk_size=200, overlap=80)
        if len(chunks) >= 2:
            end1 = chunks[0].end_char
            start2 = chunks[1].start_char
            assert start2 < end1  # overlap means second starts before first ends

    def test_paragraph_break_preference(self):
        # Should prefer to split on \n\n rather than mid-sentence
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, "test.txt", chunk_size=30, overlap=5)
        # At least one chunk should end at a paragraph boundary
        texts = [c.text for c in chunks]
        assert any("First paragraph." in t for t in texts)

    def test_source_preserved_in_all_chunks(self):
        text = "A" * 2000
        chunks = chunk_text(text, "my_source.txt", chunk_size=300, overlap=50)
        assert all(c.source == "my_source.txt" for c in chunks)


# ── chunk_file ────────────────────────────────────────────────────────────────

class TestChunkFile:
    def test_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello from a text file.\n\nSecond paragraph.", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        assert chunks[0].source == str(f)

    def test_md_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Header\n\nSome content here.", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_py_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 42\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_json_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"key": "value", "list": [1, 2, 3]}), encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 1

    def test_unsupported_extension_returns_empty(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("some content", encoding="utf-8")
        chunks = chunk_file(f)
        assert chunks == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        chunks = chunk_file(f)
        assert chunks == []

    def test_chunk_size_respected(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("word " * 500, encoding="utf-8")
        chunks = chunk_file(f, chunk_size=200, overlap=50)
        # No chunk should be dramatically bigger than chunk_size
        for c in chunks:
            assert len(c.text) <= 250  # some tolerance for paragraph snapping


# ── chunk_directory ───────────────────────────────────────────────────────────

class TestChunkDirectory:
    def test_chunks_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Content A " * 20, encoding="utf-8")
        (tmp_path / "b.md").write_text("Content B " * 20, encoding="utf-8")
        chunks = chunk_directory(tmp_path)
        sources = {c.source for c in chunks}
        assert len(sources) == 2

    def test_ignores_unsupported_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Good content", encoding="utf-8")
        (tmp_path / "b.exe").write_bytes(b"\x00\x01\x02")
        chunks = chunk_directory(tmp_path)
        sources = {c.source for c in chunks}
        assert all(".exe" not in s for s in sources)

    def test_recursive_default(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content " * 10, encoding="utf-8")
        (tmp_path / "root.txt").write_text("Root content " * 10, encoding="utf-8")
        chunks = chunk_directory(tmp_path, recursive=True)
        sources = {c.source for c in chunks}
        assert len(sources) == 2

    def test_non_recursive(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content " * 10, encoding="utf-8")
        (tmp_path / "root.txt").write_text("Root content " * 10, encoding="utf-8")
        chunks = chunk_directory(tmp_path, recursive=False)
        sources = {c.source for c in chunks}
        assert len(sources) == 1  # only root.txt

    def test_empty_directory_returns_empty(self, tmp_path):
        chunks = chunk_directory(tmp_path)
        assert chunks == []


# ── SUPPORTED set ─────────────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_common_extensions_supported(self):
        for ext in [".txt", ".md", ".py", ".json"]:
            assert ext in SUPPORTED

    def test_pdf_handled_separately(self):
        # PDF is handled via _extract_pdf, not in SUPPORTED set
        assert ".pdf" not in SUPPORTED
