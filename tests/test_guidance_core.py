# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The dark guidance core observes searches without changing their output."""

from __future__ import annotations

import math
import sqlite3

import pytest

from fidx import daemon, db, guidance, indexer, search, vector_store
from fidx.search import Result, SearchTrace


def _result(doc_id: int, score: float) -> Result:
    return Result(doc_id, "docs", f"{doc_id}.md", "", str(doc_id), score)


def _request(mode: str) -> dict:
    return {"cmd": "search", "query": "deployment", "mode": mode,
            "collections": [], "limit": 1, "min_score": None, "truncate": None}


def test_trace_captures_both_channels_and_full_fused_pool_without_output_drift(
    conn, store, embedder, corpus_dir,
):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    request = _request("hybrid")
    baseline = daemon.run_search(conn, store, embedder, request)
    trace = SearchTrace()
    observed = daemon.run_search(conn, store, embedder, request, trace=trace)
    assert observed == baseline
    assert trace.count_relation == "gte"
    assert trace.lexical and trace.vector
    assert len(trace.fused_pool) > len(observed["results"])
    direct = search.search_hybrid(conn, store, embedder, request["query"], limit=1)
    assert [(r.collection, r.relpath, r.score) for r in trace.fused_pool[:1]] == [
        (r.collection, r.relpath, r.score) for r in direct
    ]


@pytest.mark.parametrize("mode", ["lexical", "vector"])
def test_trace_exists_for_single_channel_modes(conn, store, embedder, corpus_dir, mode):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    trace = SearchTrace()
    baseline = daemon.run_search(conn, store, embedder, _request(mode))
    observed = daemon.run_search(conn, store, embedder, _request(mode), trace=trace)
    assert observed == baseline
    assert trace.count_relation == "gte"
    assert bool(trace.lexical) == (mode == "lexical")
    assert bool(trace.vector) == (mode == "vector")
    assert trace.fused_pool
    qpp = guidance.compute_qpp(trace, term_coverage_ratio=1.0)
    assert qpp.channel_agreement == 1.0
    assert math.isfinite(guidance.miss_score(qpp, {"flatness": 1.0}))


def test_reused_trace_does_not_leak_a_previous_query(conn, store, embedder, corpus_dir):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    trace = SearchTrace()
    daemon.run_search(conn, store, embedder, _request("hybrid"), trace=trace)
    assert trace.lexical and trace.vector
    daemon.run_search(conn, store, embedder, _request("lexical"), trace=trace)
    assert trace.lexical and not trace.vector
    conn.execute("UPDATE documents SET active = 0")
    daemon.run_search(conn, store, embedder, _request("hybrid"), trace=trace)
    assert not trace.lexical and not trace.vector and not trace.fused_pool


def test_trace_keeps_capped_fetch_union_tie_order_and_one_vector_query(monkeypatch):
    calls = []
    lexical = [_result(i, 1.0) for i in range(1, 51)]
    vector = [_result(i, 1.0) for i in range(26, 76)]

    def fake_lexical(*args):
        calls.append("lexical")
        return lexical

    def fake_vector(*args):
        calls.append("vector")
        return vector

    monkeypatch.setattr(search, "search_lexical", fake_lexical)
    monkeypatch.setattr(search, "search_vector", fake_vector)
    trace = SearchTrace()
    observed = search.search_hybrid(None, None, None, "query", limit=10, trace=trace)
    assert calls == ["lexical", "vector"]
    assert len(trace.lexical) == len(trace.vector) == 50
    assert len(trace.fused_pool) == 75
    assert trace.count_relation == "gte"
    assert observed == list(trace.fused_pool[:10])
    assert [(r.doc_id, r.score) for r in observed] == [
        (r.doc_id, r.score) for r in search.rrf_fuse(
            [("lexical", search.LEXICAL_WEIGHT, lexical),
             ("vector", search.VECTOR_WEIGHT, vector)], 10)]


def test_trace_tied_fused_scores_keep_path_order(monkeypatch):
    monkeypatch.setattr(search, "search_lexical", lambda *args: [_result(2, 1.0)])
    monkeypatch.setattr(search, "search_vector", lambda *args: [_result(1, 1.0)])
    trace = SearchTrace()
    observed = search.search_hybrid(None, None, None, "query", limit=1, trace=trace)
    assert [r.relpath for r in trace.fused_pool] == ["1.md", "2.md"]
    assert [r.relpath for r in observed] == ["1.md"]


def test_fts_vocabulary_uses_porter_df_phrase_terms_and_legacy_read_only_db(tmp_path):
    path = tmp_path / "legacy.db"
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("running quickly deployment")
    (root / "b.md").write_text("runs deployment")
    (root / "c.md").write_text("deployment pasta")
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "docs", root)
    indexer.sync_collection(conn, vector_store.NullStore(), "docs", indexer.IndexStats())
    conn.close()
    conn = db.connect(path, create=False, read_only=True)
    try:
        coverage = guidance.term_coverage(conn, '"running quickly" deployment unknown')
        assert coverage.matched == ("running", "quickly", "deployment")
        assert coverage.unmatched == ("unknown",)
        assert coverage.rare == ("running", "quickly")
        assert coverage.term_df == {"running": 2, "quickly": 1, "deployment": 3,
                                    "unknown": 0}
        assert coverage.ratio == 0.75
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO meta(key, value) VALUES ('probe', 'no')")
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'fidx_vocab'").fetchone()
    finally:
        conn.close()


def test_term_coverage_counts_only_active_documents_in_requested_collections(tmp_path):
    path = tmp_path / "scoped.db"
    alpha, beta = tmp_path / "alpha", tmp_path / "beta"
    alpha.mkdir(); beta.mkdir()
    (alpha / "a.md").write_text("running running quickly")
    (alpha / "b.md").write_text("running twice")
    (alpha / "old.md").write_text("retired retired")
    (beta / "c.md").write_text("deployment running")
    conn = db.connect(path)
    db.init_schema(conn)
    for name, root in (("alpha", alpha), ("beta", beta)):
        indexer.add_collection(conn, name, root)
        indexer.sync_collection(conn, vector_store.NullStore(), name, indexer.IndexStats())
    conn.execute("UPDATE documents SET active = 0 WHERE collection = 'alpha' AND relpath = 'old.md'")
    conn.commit()
    conn.close()
    conn = db.connect(path, create=False, read_only=True)
    try:
        scoped = guidance.term_coverage(conn, "runs deployment retired", collections=["alpha"])
        assert scoped.term_df == {"runs": 2, "deployment": 0, "retired": 0}
        assert scoped.matched == ("runs",)
        assert scoped.unmatched == ("deployment", "retired")
        assert scoped.ratio == pytest.approx(1 / 3)
        global_coverage = guidance.term_coverage(conn, "runs deployment retired")
        assert global_coverage.term_df == {"runs": 3, "deployment": 1, "retired": 0}
        unknown = guidance.term_coverage(conn, "runs", collections=["missing"])
        assert unknown.term_df == {"runs": 0} and unknown.ratio == 0
        mixed = guidance.term_coverage(conn, "runs", collections=["alpha", "missing"])
        assert mixed.term_df == {"runs": 2}
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        conn.close()


def test_qpp_score_shapes_empty_single_flat_cliff_and_ties():
    empty = guidance.score_shape([])
    assert empty.nqc_like == empty.wig_like == empty.top_gap == empty.head_concentration == 0
    single = guidance.score_shape([0.9])
    assert single.nqc_like == single.wig_like == single.top_gap == 0
    assert single.head_concentration == 1
    flat = guidance.score_shape([0.4, 0.4, 0.4])
    assert flat.nqc_like == flat.wig_like == flat.top_gap == 0
    assert flat.head_concentration == pytest.approx(1 / 3)
    cliff = guidance.score_shape([1.0, 0.0, 0.0])
    assert cliff.nqc_like == pytest.approx(math.sqrt(2))
    assert cliff.wig_like == pytest.approx(2 / 3)
    assert cliff.top_gap == cliff.head_concentration == 1
    tied = guidance.score_shape([0.8, 0.8, 0.1])
    assert tied.top_gap == 0
    assert math.isfinite(tied.nqc_like)


def test_qpp_channels_agreement_and_parameterized_miss_score():
    trace = SearchTrace(
        lexical=(_result(1, 1.0), _result(2, 0.0)),
        vector=(_result(2, 0.8), _result(3, 0.2)),
        fused_pool=(_result(2, 0.4), _result(1, 0.2), _result(3, 0.1)),
        count_relation="gte",
    )
    qpp = guidance.compute_qpp(trace, term_coverage_ratio=0.5)
    assert qpp.lexical.top_gap == qpp.vector.top_gap == 1
    assert qpp.channel_agreement == pytest.approx(1 / 3)
    assert qpp.term_coverage == 0.5
    weights = {"flatness": 1.0, "weak_head": 1.0, "disagreement": 1.0,
               "uncovered_terms": 1.0}
    score = guidance.miss_score(qpp, weights)
    assert 0 <= score <= 1 and math.isfinite(score)
    assert guidance.miss_score(qpp, {"uncovered_terms": 1}) == 0.5
    with pytest.raises(ValueError):
        guidance.miss_score(qpp, {"uncovered_terms": -1})


def test_qpp_single_channel_mode_has_neutral_agreement():
    trace = SearchTrace(lexical=(_result(1, 0.9),), mode="lexical")
    qpp = guidance.compute_qpp(trace, term_coverage_ratio=1.0)
    assert qpp.channel_agreement == 1.0
    assert guidance.miss_score(qpp, {"disagreement": 1.0}) == 0.0
