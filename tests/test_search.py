# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import pytest
import numpy as np

from fidx import db as dbmod, indexer, search
from fidx.search import Result, fts_match_expr, rrf_fuse
from fidx.vector_store import SqliteVecStore


@pytest.fixture()
def indexed(conn, store, embedder, corpus_dir):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return conn


def test_lexical_finds_keyword(indexed):
    results = search.search_lexical(indexed, "carbonara")
    assert results and results[0].relpath == "recipes.md"
    assert results[0].docid and results[0].snippet


def test_lexical_survives_fts_special_chars(indexed):
    for query in ['weird "quoted phrase', "a-b (c) AND OR NOT *", "  ", "!!!"]:
        search.search_lexical(indexed, query)  # must not raise


def test_fts_match_expr():
    assert fts_match_expr("hello world") == '"hello" OR "world"'
    assert fts_match_expr('"exact phrase" extra') == '"exact phrase" OR "extra"'
    assert fts_match_expr("...") is None
    # FTS5 operators inside user quotes must not leak into the expression.
    assert fts_match_expr('"relpath:* NOT x"') == '"relpath NOT x"'  # NOT is literal inside quotes
    assert fts_match_expr('"*"') is None


def test_vector_finds_shared_vocabulary(indexed, store, embedder):
    results = search.search_vector(indexed, store, embedder, "users log in with OAuth tokens")
    assert results and results[0].relpath == "auth.md"


def test_hybrid_ranks_target_first(indexed, store, embedder):
    results = search.search_hybrid(indexed, store, embedder,
                                   "production deployment rolling restarts")
    assert results and results[0].relpath == "deploy.md"
    assert "lexical" in results[0].sources or "vector" in results[0].sources


def test_collection_scoping(indexed, store, embedder, tmp_path):
    other = tmp_path / "emails"
    other.mkdir()
    (other / "mail.md").write_text("# Mail\n\ncarbonara lunch invite")
    stats = indexer.IndexStats()
    indexer.add_collection(indexed, "emails", other)
    indexer.sync_collection(indexed, store, "emails", stats)
    indexer.embed_pending(indexed, store, embedder, stats)

    only_emails = search.search_lexical(indexed, "carbonara", collections=["emails"])
    assert {r.collection for r in only_emails} == {"emails"}
    both = search.search_lexical(indexed, "carbonara")
    assert {r.collection for r in both} == {"emails", "notes"}


def _result(doc_id, score=0.5):
    return Result(doc_id=doc_id, collection="c", relpath=f"{doc_id}.md", title="t",
                  docid=str(doc_id), score=score)


def test_rrf_fusion_prefers_doc_in_both_lists():
    lex = [_result(1), _result(2), _result(3)]
    vec = [_result(2), _result(4)]
    fused = rrf_fuse([("lexical", 1.0, lex), ("vector", 1.0, vec)], limit=10)
    assert fused[0].doc_id == 2  # present in both lists beats single-source rank 1
    assert set(fused[0].sources) == {"lexical", "vector"}


def test_rrf_respects_limit():
    lex = [_result(i) for i in range(20)]
    assert len(rrf_fuse([("lexical", 1.0, lex)], limit=5)) == 5


def test_rrf_equal_scores_use_collection_and_path_not_content_hash():
    a = Result(1, "a", "z.md", "", "same", 0.5)
    b = Result(2, "b", "a.md", "", "same", 0.5)
    lists = [("lexical", 1.0, [b, a]), ("vector", 1.0, [a, b])]
    assert [(r.collection, r.relpath) for r in rrf_fuse(lists, 2)] == [
        ("a", "z.md"), ("b", "a.md")]


def test_lexical_equal_rank_has_stable_cutoff(tmp_path):
    orders = [("z.md", "a.md"), ("a.md", "z.md")]
    for index, order in enumerate(orders):
        conn = dbmod.connect(tmp_path / f"lex-{index}.db")
        try:
            dbmod.init_schema(conn)
            for path in order:
                conn.execute(
                    "INSERT INTO documents(collection, relpath, title, body, hash) "
                    "VALUES ('notes', ?, '', 'shared search text', 'same')", (path,))
            assert [r.relpath for r in search.search_lexical(conn, "shared", limit=1)] == ["a.md"]
        finally:
            conn.close()


class _FixedEmbedder:
    def embed_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def _equal_vector_index(path, paths):
    conn = dbmod.connect(path)
    dbmod.init_schema(conn)
    store = SqliteVecStore(conn)
    store.ensure(2)
    ids = []
    for relpath in paths:
        cur = conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash) "
            "VALUES ('notes', ?, '', 'same body', 'same')", (relpath,))
        ids.append(conn.execute(
            "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, 0, 0, 9)",
            (cur.lastrowid,)).lastrowid)
    store.add(ids, np.tile([1.0, 0.0], (len(ids), 1)).astype(np.float32))
    return conn, store


@pytest.mark.parametrize("size", [13, 201])
def test_sqlite_vec_tied_cutoff_is_complete_across_reindexes(tmp_path, size):
    paths = [f"{i:03d}.md" for i in range(size)]
    observed = []
    for index, order in enumerate((list(reversed(paths)), paths)):
        conn, store = _equal_vector_index(tmp_path / f"vec-{size}-{index}.db", order)
        try:
            windows = []
            original_knn = store.knn

            def recording_knn(qvec, k):
                windows.append(k)
                return original_knn(qvec, k)

            store.knn = recording_knn
            observed.append([r.relpath for r in search.search_vector(
                conn, store, _FixedEmbedder(), "query", limit=3)])
            assert len(windows) > 1
            assert max(windows) >= size
        finally:
            conn.close()
    assert observed == [["000.md", "001.md", "002.md"]] * 2


def test_vector_equal_chunk_scores_choose_lowest_sequence(tmp_path):
    conn = dbmod.connect(tmp_path / "chunks.db")
    try:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        store.ensure(2)
        body = "first text second text"
        doc_id = conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash) "
            "VALUES ('notes', 'a.md', '', ?, 'same')", (body,)).lastrowid
        ids = [conn.execute(
            "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, ?, ?, 10)",
            (doc_id, seq, pos)).lastrowid for seq, pos in [(1, 11), (0, 0)]]
        store.add(ids, np.tile([1.0, 0.0], (2, 1)).astype(np.float32))
        assert search.search_vector(conn, store, _FixedEmbedder(), "query", limit=1)[0].snippet == "first text"
    finally:
        conn.close()


def test_sqlite_vec_tie_cohort_beyond_k_limit_degrades_to_full_scan(tmp_path):
    paths = [f"{i:05d}.md" for i in range(5000)]
    observed = []
    for index, order in enumerate((list(reversed(paths)), paths)):
        conn, store = _equal_vector_index(tmp_path / f"big-{index}.db", order)
        try:
            observed.append([r.relpath for r in search.search_vector(
                conn, store, _FixedEmbedder(), "q", limit=3)])
        finally:
            conn.close()
    assert observed == [["00000.md", "00001.md", "00002.md"]] * 2


def test_sqlite_vec_limit_1024_stays_within_k_limit(tmp_path):
    rng = np.random.default_rng(0)
    conn = dbmod.connect(tmp_path / "wide.db")
    try:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        store.ensure(2)
        ids = []
        for i in range(4200):
            doc_id = conn.execute(
                "INSERT INTO documents(collection, relpath, title, body, hash) "
                "VALUES ('n', ?, '', 'x', 'h')", (f"{i:05d}.md",)).lastrowid
            ids.append(conn.execute(
                "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, 0, 0, 1)",
                (doc_id,)).lastrowid)
        vectors = rng.normal(size=(4200, 2)).astype(np.float32)
        store.add(ids, vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        assert len(search.search_vector(conn, store, _FixedEmbedder(), "q", limit=1024)) == 1024
    finally:
        conn.close()


def test_sqlite_vec_widening_anchors_on_cap_boundary(tmp_path):
    conn = dbmod.connect(tmp_path / "separate-ties.db")
    try:
        dbmod.init_schema(conn)
        store = SqliteVecStore(conn)
        store.ensure(2)
        ids = []
        vectors = [[1.0, 0.0]] * 13
        vectors += [[float(np.cos(a)), float(np.sin(a))] for a in np.linspace(0.1, 1.0, 10)]
        vectors += [[0.0, 1.0]] * 7  # unrelated tie spanning ranks 24 and 25
        for i in range(len(vectors)):
            doc_id = conn.execute(
                "INSERT INTO documents(collection, relpath, title, body, hash) "
                "VALUES ('n', ?, '', 'x', 'h')", (f"{i:03d}.md",)).lastrowid
            ids.append(conn.execute(
                "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, 0, 0, 1)",
                (doc_id,)).lastrowid)
        store.add(ids, np.array(vectors, dtype=np.float32))
        windows = []
        original_knn = store.knn

        def recording_knn(qvec, k):
            windows.append(k)
            return original_knn(qvec, k)

        store.knn = recording_knn
        search.search_vector(conn, store, _FixedEmbedder(), "q", limit=3)
        assert windows == [13, 25]
    finally:
        conn.close()
