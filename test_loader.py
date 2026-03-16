"""Tests for loader and chunking logic — no API key required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import _chunk_text, _clean, load_txt


def test_chunk_text_basic():
    words = ["word"] * 500
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 100
    print(f"✓ chunking produced {len(chunks)} chunks")


def test_chunk_text_overlap():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=100, overlap=20)
    # First word of chunk 2 should be word at index 80 (100-20)
    assert chunks[1].split()[0] == "w80"
    print("✓ overlap is correct")


def test_clean():
    raw = "hello   world\r\n\r\n\r\nfoo   bar"
    cleaned = _clean(raw)
    assert "   " not in cleaned
    assert "\r" not in cleaned
    print("✓ _clean removes extra whitespace")


def test_load_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world. " * 500)
    docs = load_txt(f)
    assert len(docs) > 0
    assert all(d.source == "test.txt" for d in docs)
    print(f"✓ load_txt produced {len(docs)} chunks")


if __name__ == "__main__":
    test_chunk_text_basic()
    test_chunk_text_overlap()
    test_clean()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_load_txt(Path(d))
    print("\nAll tests passed.")
