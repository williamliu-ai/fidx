# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fake-only M12b1 dark reranker retrieval/filter integration fixtures."""

from __future__ import annotations

import hashlib
import json

import pytest
from click.testing import CliRunner

from fidx import budget, daemon, guidance, guidance_diagnostics, search, truncate
from fidx.cli import main
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


def test_runtime_failure_warning_is_closed_and_budget_classified(conn, monkeypatch):
    ranked = _results(conn, 2)
    monkeypatch.setattr(search, "search_lexical",
                        lambda _c, _q, _scope, limit: ranked[:limit])
    req = _request()
    baseline = daemon.run_search(conn, None, None, req)
    failed = daemon.run_search(conn, None, None, req, reranker=Broken(), rerank_n=2)
    assert failed["results"] == baseline["results"]
    assert failed["diagnostics"]["warnings"] == [
        {"component": "rerank", "code": "runtime_failure", "fallback": "fused"}]
    assert "synthetic scorer failure" not in json.dumps(failed)
    assert budget.unclassified_fields(failed) == []

    full = daemon.run_search(conn, None, None, {**req, "budget_tokens": 10000},
                             reranker=Broken(), rerank_n=2)
    assert full["diagnostics"]["warnings"] == failed["diagnostics"]["warnings"]
    tiny = daemon.run_search(conn, None, None, {**req, "budget_tokens": 0},
                             reranker=Broken(), rerank_n=2)
    assert "diagnostics" not in tiny
    assert tiny["summary"]["budget"]["dropped"]["diagnostics"] is True


def test_successful_calibrated_rerank_reports_inapplicable_fused_floor(
        conn, monkeypatch):
    ranked = _results(conn, 2)
    conn.execute("INSERT INTO meta(key,value) VALUES ('truncate_floor','0.99')")
    monkeypatch.setattr(search, "search_lexical",
                        lambda _c, _q, _scope, limit: ranked[:limit])
    payload = daemon.run_search(
        conn, None, None, _request(truncate="calibrated"),
        reranker=Scores([0.0, 1.0]), rerank_n=2)
    assert payload["diagnostics"]["calibration"] == {
        "floor_available": True, "floor": 0.99, "floor_applicable": False}
    assert payload["diagnostics"]["warnings"] == [
        {"component": "rerank", "code": "fused_calibration_ignored", "fallback": "knee"}]
    advice = payload["summary"]["truncation_advice"]
    calibrated = next(option for option in advice["options"]
                      if option["intent"] == "use_calibrated_abstention")
    assert calibrated["applicable"] is False
    assert calibrated["recommended"] is False
    assert "inapplicable" in calibrated["reason"].lower()
    assert advice["recommendation"] != "calibrated"
    assert "no calibration floor is stored" not in advice["reason"].lower()
    baseline = daemon.run_search(conn, None, None, _request(truncate="calibrated"))
    assert baseline["diagnostics"]["calibration"] == {
        "floor_available": True, "floor": 0.99}
    assert "warnings" not in baseline["diagnostics"]


@pytest.mark.parametrize("stored_floor", [None, "abc"])
def test_calibrated_rerank_without_valid_floor_has_no_ignored_floor_warning(
        conn, monkeypatch, stored_floor):
    ranked = _results(conn, 2)
    if stored_floor is not None:
        conn.execute("INSERT INTO meta(key,value) VALUES ('truncate_floor',?)",
                     (stored_floor,))
    monkeypatch.setattr(search, "search_lexical",
                        lambda _c, _q, _scope, limit: ranked[:limit])
    payload = daemon.run_search(
        conn, None, None, _request(truncate="calibrated"),
        reranker=Scores([0.0, 1.0]), rerank_n=2)
    assert payload["diagnostics"]["calibration"]["floor_available"] is False
    assert "warnings" not in payload["diagnostics"]
    assert "no calibration floor is stored" in (
        payload["summary"]["truncation_advice"]["reason"].lower())


def test_guidance_trace_stays_raw_and_default_request_key_is_inert(conn, monkeypatch):
    ranked = _results(conn, 4)
    monkeypatch.setattr(search, "search_lexical",
                        lambda _c, _q, _scope, limit: ranked[:limit])
    req = _request(guidance=True, explain=True)
    default = daemon.run_search(conn, None, None, req)
    assert json.dumps(daemon.run_search(conn, None, None, req, reranker=None),
                      sort_keys=True) == json.dumps(default, sort_keys=True)
    assert json.dumps(daemon.run_search(conn, None, None, {**req, "rerank": True}),
                      sort_keys=True) == json.dumps(default, sort_keys=True)
    trace = SearchTrace()
    before = [(r.score, dict(r.sources), r.matched_chunk_id,
               r.line_start, r.line_end) for r in ranked]
    payload = daemon.run_search(conn, None, None, req, trace=trace,
                                reranker=Scores([0, 2, 1, -1]), rerank_n=4)
    assert [(r.score, r.sources, r.matched_chunk_id, r.line_start, r.line_end)
            for r in trace.fused_pool] == before
    coverage = guidance.term_coverage(conn, req["query"], collections=[])
    expected_qpp = guidance_diagnostics.qpp_projection(
        guidance.compute_qpp(trace, term_coverage_ratio=coverage.ratio))
    assert payload["diagnostics"]["qpp"] == expected_qpp
    assert "warnings" not in payload["diagnostics"]


def test_failure_with_guidance_keeps_original_limit_envelope_and_trace(
        conn, monkeypatch):
    ranked = _results(conn, 4)
    calls = []

    def lexical(_conn, _query, _scope, limit):
        calls.append(limit)
        return ranked[:limit]

    monkeypatch.setattr(search, "search_lexical", lexical)
    req = _request(guidance=True, explain=True)
    baseline = daemon.run_search(conn, None, None, req)
    trace = SearchTrace()
    failed = daemon.run_search(conn, None, None, req, trace=trace,
                               reranker=Broken(), rerank_n=4)
    assert calls == [2, 4, 2]
    assert failed["diagnostics"].pop("warnings") == [
        {"component": "rerank", "code": "runtime_failure", "fallback": "fused"}]
    assert failed == baseline
    assert [r.doc_id for r in trace.fused_pool] == [r.doc_id for r in ranked[:2]]


def test_search_help_does_not_activate_dark_reranker():
    result = CliRunner().invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "--rerank" not in result.output
