# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Indexed chunk locations remain additive across new and legacy databases."""

import hashlib

import numpy as np
import pytest

from fidx import daemon, db as dbmod, indexer, search
from fidx.vector_store import SqliteVecStore


class FixedEmbedder:
    def embed_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


@pytest.mark.parametrize("body,last_line", [
    ("only one line", 1),
    ("first\r\nsecond\r\nlast\r\n", 3),
    ("first\n\n", 2),
])
def test_indexed_chunk_line_range_ignores_trailing_empty_line(conn, store, embedder,
                                                               body, last_line):
    conn.execute(
        "INSERT INTO documents(collection, relpath, title, body, hash) "
        "VALUES ('notes', 'note.md', '', ?, 'hash')", (body,))
    indexer.embed_pending(conn, store, embedder, indexer.IndexStats())
    rows = conn.execute("SELECT line_start, line_end FROM chunks ORDER BY seq").fetchall()
    assert rows[0]["line_start"] == 1
    assert rows[-1]["line_end"] == last_line


def test_first_and_last_chunks_get_inclusive_lines(conn, store, embedder):
    body = "first\n" + "middle\n" * 350 + "last\n"
    conn.execute(
        "INSERT INTO documents(collection, relpath, title, body, hash) "
        "VALUES ('notes', 'long.md', '', ?, 'hash')", (body,))
    indexer.embed_pending(conn, store, embedder, indexer.IndexStats())
    rows = conn.execute("SELECT line_start, line_end FROM chunks ORDER BY seq").fetchall()
    assert len(rows) > 1
    assert rows[0]["line_start"] == 1
    assert rows[-1]["line_end"] == 352


def test_vector_and_hybrid_emit_best_chunk_but_lexical_omits(conn, store, embedder, tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("# Header\nneedle token\nlast line\n")
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", root)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)

    vector = search.search_vector(conn, store, embedder, "needle token")
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    assert vector[0].matched_chunk_id == chunk_id
    assert (vector[0].line_start, vector[0].line_end) == (1, 3)

    for mode in ("vector", "hybrid"):
        payload = daemon.run_search(conn, store, embedder, {
            "query": "needle token", "mode": mode, "limit": 10})
        assert payload["results"][0]["locations"] == [{"line_start": 1, "line_end": 3}]
    lexical = daemon.run_search(conn, store, embedder, {
        "query": "needle token", "mode": "lexical", "limit": 10})
    assert "locations" not in lexical["results"][0]


def test_hybrid_keeps_vector_location_when_lexical_result_arrives_first():
    lexical = search.Result(1, "notes", "note.md", "Note", "same", 0.9)
    vector = search.Result(1, "notes", "note.md", "Note", "same", 0.8,
                           matched_chunk_id=7, line_start=4, line_end=5)
    fused = search.rrf_fuse([("lexical", 1.0, [lexical]),
                             ("vector", 1.0, [vector])], limit=1)
    assert (fused[0].matched_chunk_id, fused[0].line_start, fused[0].line_end) == (7, 4, 5)


def _legacy_db(path, roots):
    conn = dbmod.connect(path)
    conn.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema_version', '1');
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, collection TEXT NOT NULL, relpath TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL, hash TEXT NOT NULL,
            mtime REAL NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
            UNIQUE(collection, relpath));
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, seq INTEGER NOT NULL,
            pos INTEGER NOT NULL, length INTEGER NOT NULL, UNIQUE(doc_id, seq));
    """)
    store = SqliteVecStore(conn)
    store.ensure(2)
    ids = []
    for name, root in roots.items():
        path = root / "note.md"
        body = path.read_text()
        doc_id = conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash, mtime) "
            "VALUES (?, 'note.md', '', ?, ?, ?)",
            (name, body, hashlib.sha256(body.encode()).hexdigest(), path.stat().st_mtime),
        ).lastrowid
        ids.append(conn.execute(
            "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, 0, 0, ?)",
            (doc_id, len(body))).lastrowid)
    store.add(ids, np.tile([1.0, 0.0], (len(ids), 1)).astype(np.float32))
    return conn, store


def test_old_schema_search_omits_locations_then_partial_reindex_backfills(tmp_path):
    roots = {}
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        (root / "note.md").write_text("first\nlast\n")
        roots[name] = root
    conn, store = _legacy_db(tmp_path / "old.db", roots)
    try:
        before = search.search_vector(conn, store, FixedEmbedder(), "query")
        assert [r.collection for r in before] == ["a", "b"]
        assert all(r.line_start is None for r in before)
        assert all("locations" not in r for r in daemon.serialize_results(before, "vector"))

        dbmod.init_schema(conn)
        assert dbmod.get_meta(conn, "schema_version") == "2"
        assert {"line_start", "line_end"} <= {
            row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        assert all(row["line_start"] is None for row in conn.execute("SELECT line_start FROM chunks"))
        for name, root in roots.items():
            indexer.add_collection(conn, name, root)

        stats = indexer.IndexStats()
        indexer.sync_collection(conn, store, "a", stats)
        assert stats.embedded_chunks == 0
        mixed = search.search_vector(conn, store, FixedEmbedder(), "query")
        assert [(r.collection, r.line_start, r.line_end) for r in mixed] == [
            ("a", 1, 2), ("b", None, None)]
        assert [(r.collection, r.score) for r in mixed] == [
            (r.collection, r.score) for r in before]

        indexer.sync_collection(conn, store, "b", indexer.IndexStats())
        assert [(r.collection, r.line_start, r.line_end) for r in
                search.search_vector(conn, store, FixedEmbedder(), "query")] == [
            ("a", 1, 2), ("b", 1, 2)]
    finally:
        conn.close()
