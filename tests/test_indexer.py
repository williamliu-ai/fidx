# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from fidx import indexer


def index_all(conn, store, embedder, root, name="notes"):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, name, root)
    indexer.sync_collection(conn, store, name, stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return stats


def counts(conn, store):
    docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
    chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return docs, chunks, store.count()


def test_initial_index(conn, store, embedder, corpus_dir):
    stats = index_all(conn, store, embedder, corpus_dir)
    assert stats.added == 3 and stats.removed == 0
    docs, chunks, vectors = counts(conn, store)
    assert docs == 3
    assert chunks == vectors > 0


def test_incremental_noop(conn, store, embedder, corpus_dir):
    index_all(conn, store, embedder, corpus_dir)
    stats = indexer.IndexStats()
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    assert stats.added == stats.updated == stats.removed == 0
    assert stats.unchanged == 3
    assert stats.embedded_chunks == 0


def test_update_and_delete(conn, store, embedder, corpus_dir):
    index_all(conn, store, embedder, corpus_dir)
    (corpus_dir / "auth.md").write_text("# Authentication\n\nNow with passkeys.")
    (corpus_dir / "recipes.md").unlink()

    stats = indexer.IndexStats()
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    assert stats.updated == 1 and stats.removed == 1

    docs, chunks, vectors = counts(conn, store)
    assert docs == 2
    assert chunks == vectors
    body = conn.execute("SELECT body FROM documents WHERE relpath='auth.md'").fetchone()["body"]
    assert "passkeys" in body


def test_remove_collection_cleans_everything(conn, store, embedder, corpus_dir):
    index_all(conn, store, embedder, corpus_dir)
    indexer.remove_collection(conn, store, "notes")
    assert counts(conn, store) == (0, 0, 0)


def test_remove_collection_before_any_embed(tmp_path, corpus_dir):
    # A db that has never run `index` has no vector store; remove must not crash.
    from fidx import db as dbmod
    from fidx.vector_store import NullStore

    conn = dbmod.connect(tmp_path / "fresh.db")
    dbmod.init_schema(conn)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, NullStore(), "notes", stats)
    indexer.remove_collection(conn, NullStore(), "notes")
    assert conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"] == 0


def _body_for_chunk_count(target):
    # Grow in small steps; chunking is greedy sequential, so the count rises
    # by at most one per small increment and every integer is hit exactly.
    from fidx.chunker import chunk_text

    parts = ["# Big document\n"]
    i = 0
    while len(chunk_text("\n\n".join(parts))) < target:
        i += 1
        parts.append(f"Paragraph {i}: " + "lorem ipsum dolor sit amet " * 8)
        assert i < 100000, "could not reach target chunk count"
    body = "\n\n".join(parts)
    while len(chunk_text(body)) > target:
        body = body[: -100]
    assert len(chunk_text(body)) == target
    return body


def _index_one_doc(conn, store, embedder, tmp_path, name, body):
    root = tmp_path / name.replace(".", "_")
    root.mkdir(exist_ok=True)
    (root / name).write_text(body)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, f"col_{name}", root)
    indexer.sync_collection(conn, store, f"col_{name}", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return stats


def _drop_tail(conn, store, relpath, keep_seqs):
    row = conn.execute("SELECT id FROM documents WHERE relpath = ?", (relpath,)).fetchone()
    doomed = [
        r["id"]
        for r in conn.execute(
            "SELECT id, seq FROM chunks WHERE doc_id = ? AND seq >= ?", (row["id"], keep_seqs)
        )
    ]
    store.delete(doomed)
    conn.execute("DELETE FROM chunks WHERE id IN (%s)" % ",".join("?" * len(doomed)), doomed)
    conn.commit()
    return len(doomed)


def test_resume_fully_covered_doc_is_noop(conn, store, embedder, corpus_dir, tmp_path):
    body = _body_for_chunk_count(300)
    _index_one_doc(conn, store, embedder, tmp_path, "big.md", body)
    before = counts(conn, store)
    stats = indexer.IndexStats()
    indexer.embed_pending(conn, store, embedder, stats)
    assert stats.embedded_chunks == 0
    assert counts(conn, store) == before


def test_resume_partial_document_creates_tail(conn, store, embedder, corpus_dir, tmp_path):
    from fidx.chunker import chunk_text

    body = _body_for_chunk_count(300)
    _index_one_doc(conn, store, embedder, tmp_path, "big.md", body)
    # Simulate a crash after the first 256-chunk flush committed: only the
    # first-flush prefix survives; the uncommitted tail rolled back.
    missing = _drop_tail(conn, store, "big.md", 256)
    assert missing == 300 - 256
    stats = indexer.IndexStats()
    indexer.embed_pending(conn, store, embedder, stats)
    docs, chunks, vectors = counts(conn, store)
    assert chunks == len(chunk_text(body))
    assert chunks == vectors
    assert stats.embedded_chunks == missing


def test_resume_flush_boundary_256_and_257(conn, store, embedder, corpus_dir, tmp_path):
    from fidx.chunker import chunk_text

    for target in (256, 257):
        name = f"edge{target}.md"
        body = _body_for_chunk_count(target)
        _index_one_doc(conn, store, embedder, tmp_path, name, body)
        # Simulate a crash right after the first flush had committed: for 257
        # that leaves a 256-row prefix; for 256 the doc was already whole.
        keep = min(256, target)
        _drop_tail(conn, store, name, keep)
        stats = indexer.IndexStats()
        indexer.embed_pending(conn, store, embedder, stats)
        n = conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE doc_id = "
            "(SELECT id FROM documents WHERE relpath = ?)",
            (name,),
        ).fetchone()["n"]
        assert n == target, f"target={target}"
        assert stats.embedded_chunks == target - keep
    docs, chunks, vectors = counts(conn, store)
    assert chunks == vectors


def test_resume_nul_body_partial_document_creates_tail(conn, store, embedder, corpus_dir, tmp_path):
    from fidx.chunker import chunk_text

    # Leading NUL: SQLite LENGTH() stops at the first NUL while Python
    # chunk offsets count the whole string, so coverage comparison must
    # still detect the missing tail.
    body = "\x00" + _body_for_chunk_count(369)
    assert len(chunk_text(body)) == 369
    _index_one_doc(conn, store, embedder, tmp_path, "nulbig.md", body)
    # Simulate a crash after the first 256-chunk flush committed: only the
    # first-flush prefix survives; the uncommitted tail rolled back.
    missing = _drop_tail(conn, store, "nulbig.md", 256)
    assert missing == 369 - 256
    stats = indexer.IndexStats()
    indexer.embed_pending(conn, store, embedder, stats)
    docs, chunks, vectors = counts(conn, store)
    assert chunks == len(chunk_text(body))
    assert chunks == vectors
    assert stats.embedded_chunks == missing


def test_zero_chunk_document_still_indexed(conn, store, embedder, corpus_dir, tmp_path):
    index_all(conn, store, embedder, corpus_dir)
    (corpus_dir / "fresh.md").write_text("# Fresh note\n\nBrand new content here.")
    stats = indexer.IndexStats()
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    row = conn.execute("SELECT id FROM documents WHERE relpath='fresh.md'").fetchone()
    assert row is not None
    n = conn.execute("SELECT count(*) AS n FROM chunks WHERE doc_id = ?", (row["id"],)).fetchone()["n"]
    assert n > 0
    docs, chunks, vectors = counts(conn, store)
    assert chunks == vectors


def test_empty_body_document_indexes_with_zero_chunks(conn, store, embedder, corpus_dir, tmp_path):
    # Empty bodies chunk to nothing, so the document keeps zero chunk rows
    # (COALESCE(MAX(...), -1) edge: -1 < LENGTH('') = 0 reselects it every
    # run) — indexing must still succeed and stay a no-op on re-run.
    _index_one_doc(conn, store, embedder, tmp_path, "empty.md", "")
    row = conn.execute("SELECT id FROM documents WHERE relpath='empty.md'").fetchone()
    assert row is not None
    n = conn.execute("SELECT count(*) AS n FROM chunks WHERE doc_id = ?", (row["id"],)).fetchone()["n"]
    assert n == 0
    stats = indexer.IndexStats()
    indexer.embed_pending(conn, store, embedder, stats)
    assert stats.embedded_chunks == 0
    assert conn.execute("SELECT count(*) AS n FROM chunks WHERE doc_id = ?", (row["id"],)).fetchone()["n"] == 0
    docs, chunks, vectors = counts(conn, store)
    assert chunks == vectors


def test_repair_orphan_chunks(conn, store, embedder, corpus_dir):
    # Simulate an interrupted embed: chunks committed, vectors missing.
    index_all(conn, store, embedder, corpus_dir)
    victim = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]
    store.delete([victim])
    conn.commit()
    stats = indexer.IndexStats()
    indexer.embed_pending(conn, store, embedder, stats)
    docs, chunks, vectors = counts(conn, store)
    assert chunks == vectors
    assert stats.embedded_chunks == 1
