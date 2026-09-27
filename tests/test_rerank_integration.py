# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fake-only M12b1 dark reranker retrieval/filter integration fixtures."""

from __future__ import annotations

import hashlib

import pytest

from fidx import daemon, search, truncate
from fidx.search import Result, SearchTrace


class Scores:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return iter(self.scores)


class Broken:
    def rerank(self, query, documents):
        yield 1.0
        raise RuntimeError("synthetic scorer failure")


def _results(conn, n=4):
    results = []
    for i in range(n):
        body = f"Body {i}"
        doc_id = conn.execute(
            "INSERT INTO documents(collection,relpath,title,body,hash) "
            "VALUES ('docs',?,?,?,?)",
            (f"{i}.md", f"Title {i}", body,
             hashlib.sha256(body.encode()).hexdigest())).lastrowid
        results.append(Result(doc_id, "docs", f"{i}.md", f"Title {i}",
                              f"{i:06x}", 0.1 + i / 10))
    return results


def _request(mode="lexical", **extra):
    return {"query": "Body", "mode": mode, "collections": [], "limit": 2,
            "min_score": None, "truncate": "off", **extra}


def test_success_overfetches_then_reranks_top_limit_before_filters(conn, monkeypatch):
    ranked = _results(conn)
    calls = []

    def lexical(_conn, _query, _collections, limit):
        calls.append(limit)
        return ranked[:limit]

    monkeypatch.setattr(search, "search_lexical", lexical)
    scorer = Scores([0.0, 5.0, 3.0, 2.0])
    payload = daemon.run_search(conn, None, None, _request(), reranker=scorer,
                                rerank_n=4)
    assert calls == [4]
    assert [row["relpath"] for row in payload["results"]] == ["1.md", "2.md"]
    assert all("rerank" in row["sources"] and "fused" not in row["sources"]
               for row in payload["results"])
    assert payload["summary"]["result_count"] == 2
    assert payload["diagnostics"]["filters"]["raw_count"] == 2


@pytest.mark.parametrize("mode", ["lexical", "vector"])
def test_stage_failure_retries_original_limit_and_matches_default(
        conn, monkeypatch, mode):
    ranked = _results(conn)
    calls = []

    def fake(*args, **kwargs):
        limit = args[-1]
        calls.append(limit)
        return ranked[:limit]

    monkeypatch.setattr(search, f"search_{mode}", fake)
    request = _request(mode)
    baseline = daemon.run_search(conn, None, None, request)
    trace = SearchTrace()
    failed = daemon.run_search(conn, None, None, request, trace=trace,
                               reranker=Broken(), rerank_n=4)
    assert calls == [2, 4, 2]
    assert failed["results"] == baseline["results"]
    assert failed["diagnostics"]["filters"] == baseline["diagnostics"]["filters"]
    assert failed["summary"]["top_score"] == baseline["summary"]["top_score"]
    assert [r.doc_id for r in trace.fused_pool] == [
        r.doc_id for r in ranked[:2]]


def test_source_floor_uses_original_single_arm_score_not_sigmoid(conn, monkeypatch):
    ranked = _results(conn, 3)
    ranked[0].score, ranked[1].score, ranked[2].score = 0.1, 0.1, 0.9
    monkeypatch.setattr(search, "search_lexical",
                        lambda _c, _q, _scope, limit: ranked[:limit])
    payload = daemon.run_search(
        conn, None, None, _request(truncate="source:0.5"),
        reranker=Scores([3.0, 2.0, 1.0]), rerank_n=3)
    # The third candidate clears the source floor but cannot backfill after
    # the reranked top-2 cut; sigmoid scores of the first two do not qualify.
    assert payload["results"] == []
    assert payload["diagnostics"]["filters"]["raw_count"] == 2
    direct = Result(1, "docs", "x", "x", "abcdef", 0.99,
                    sources={"lexical": 0.1, "rerank": 4.0})
    assert truncate.truncate([direct], "source:0.5", "lexical") == []


def test_calibrated_skips_stored_fused_floor_on_success(conn, monkeypatch):
    ranked = _results(conn, 2)
    conn.execute("INSERT INTO meta(key,value) VALUES ('truncate_floor','0.99')")
    monkeypatch.setattr(search, "search_lexical",
                        lambda _c, _q, _scope, limit: ranked[:limit])
    payload = daemon.run_search(
        conn, None, None, _request(truncate="calibrated"),
        reranker=Scores([0.0, 1.0]), rerank_n=2)
    assert [row["relpath"] for row in payload["results"]] == ["1.md", "0.md"]
    # Without reranking the stored floor still applies.
    baseline = daemon.run_search(conn, None, None, _request(truncate="calibrated"))
    assert baseline["results"] == []


def test_internal_rerank_window_is_bounded(conn, monkeypatch):
    _results(conn, 1)
    monkeypatch.setattr(search, "search_lexical", lambda *_args: [])
    for value in (0, 101, -1):
        with pytest.raises(ValueError):
            daemon.run_search(conn, None, None, _request(),
                              reranker=Scores([]), rerank_n=value)


def test_hybrid_trace_keeps_pre_rerank_pool_and_source_scores(conn, monkeypatch):
    ranked = _results(conn)
    for row in ranked:
        row.sources = {"lexical": row.score, "vector": row.score / 2}
    calls = []

    def hybrid(_conn, _store, _embedder, _query, _scope, limit, *, trace=None):
        calls.append(limit)
        found = ranked[:limit]
        if trace is not None:
            trace.lexical = tuple(found)
            trace.vector = tuple(found)
            trace.fused_pool = tuple(found)
        return found

    monkeypatch.setattr(search, "search_hybrid", hybrid)
    trace = SearchTrace()
    before = [(row.score, dict(row.sources), row.matched_chunk_id,
               row.line_start, row.line_end) for row in ranked]
    payload = daemon.run_search(
        conn, None, None, _request("hybrid"), trace=trace,
        reranker=Scores([0, 2, 1, -1]), rerank_n=4)
    assert calls == [4]
    assert [row["relpath"] for row in payload["results"]] == ["1.md", "2.md"]
    assert [(row.score, row.sources, row.matched_chunk_id,
             row.line_start, row.line_end) for row in trace.fused_pool] == before


def test_empty_index_skips_injected_scorer_and_embedding(conn):
    scorer = Scores([1.0])
    payload = daemon.run_search(
        conn, None, None, _request("hybrid"), reranker=scorer, rerank_n=4)
    assert payload["status"] == "empty_index"
    assert scorer.calls == []
