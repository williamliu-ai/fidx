from fidx.chunker import Chunk, chunk_text, extract_title


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_doc_is_single_chunk():
    text = "# Title\n\nA short note."
    assert chunk_text(text) == [Chunk(0, 0, len(text))]


def test_long_doc_chunks_cover_text_with_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(30))
    chunks = chunk_text(text, target=1000, overlap=150, window=300)
    assert len(chunks) > 3
    assert chunks[0].pos == 0
    last = chunks[-1]
    assert last.pos + last.length == len(text)
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.pos < prev.pos + prev.length  # overlap
        assert cur.pos > prev.pos  # forward progress


def test_prefers_heading_break():
    filler = "word " * 150
    text = f"# One\n{filler}\n## Two\n{filler}\n## Three\n{filler}"
    chunks = chunk_text(text, target=900, overlap=100, window=400)
    heading_positions = {text.index("## Two"), text.index("## Three")}
    chunk_ends = {c.pos + c.length for c in chunks}
    assert chunk_ends & heading_positions, "expected at least one chunk to end at a heading"


def test_does_not_split_inside_code_fence():
    code = "```python\n" + "x = 1\n" * 120 + "```\n"
    text = "intro " * 100 + "\n\n" + code + "\n\n" + "outro " * 100
    fence_start = text.index("```python")
    fence_end = text.index("```\n", fence_start + 1)
    for c in chunk_text(text, target=800, overlap=100, window=300):
        assert not (fence_start < c.pos <= fence_end), "chunk boundary inside code fence"


def test_extract_title_prefers_heading():
    assert extract_title("intro\n\n## Real Title\nbody", "fallback") == "Real Title"
    assert extract_title("just a first line\nmore", "fallback") == "just a first line"
    assert extract_title("", "fallback") == "fallback"
