# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Dark, deterministic no-result diagnostics; no CLI or envelope activation."""

from __future__ import annotations

import pytest

from fidx import daemon, db, guidance, guidance_diagnostics as diagnostics, indexer, vector_store
from fidx.guidance import QPPSignals, ScoreShape, TermCoverage
from fidx.search import Result, SearchTrace


def _coverage(*, matched=(), unmatched=()):
    terms = (*matched, *unmatched)
    return TermCoverage(tuple(matched), tuple(unmatched), (),
                        {term: 1 if term in matched else 0 for term in terms},
                        len(matched) / len(terms) if terms else 0.0)


def _envelope():
    return {"query": "running", "status": "no_results", "request": {
        "collections": [], "limit": 10}, "summary": {"index": {"stale_hint": False}},
                        "diagnostics": {"index_empty": False, "unknown_collections": [],
                        "filters": {"raw_count": 0, "after_truncate": 0,
                                    "dropped_by_min_score": 0,
                                    "dropped_by_truncate": 0}}}


def _unknown_scope(envelope):
    envelope["request"]["collections"] = ["missing"]
    envelope["diagnostics"]["unknown_collections"] = ["missing"]


@pytest.mark.parametrize("mutate,coverage,evidence,expected", [
    (lambda x: x["diagnostics"].update(index_empty=True), _coverage(unmatched=("running",)),
     diagnostics.ExecutionEvidence(True, None), ("empty_index",)),
    (lambda x: x.update(query="!!!"), _coverage(),
     diagnostics.ExecutionEvidence(True, None), ("query_invalid",)),
    (lambda x: x["request"].update(limit=0), _coverage(matched=("running",)),
     diagnostics.ExecutionEvidence(True, None), ("query_invalid",)),
    (_unknown_scope,
     _coverage(unmatched=("running",)), diagnostics.ExecutionEvidence(True, None),
     ("unknown_collection",)),
    (lambda x: x["diagnostics"]["filters"].update(raw_count=3, dropped_by_min_score=3),
     _coverage(matched=("running",)), diagnostics.ExecutionEvidence(True, None),
     ("filters_dropped_all",)),
    (lambda x: None, _coverage(matched=("running",)),
     diagnostics.ExecutionEvidence(False, "deadline"), ("execution_partial",)),
    (lambda x: None, _coverage(matched=("running",)),
     diagnostics.ExecutionEvidence(True, None, semantic_unavailable=True),
     ("semantic_unavailable",)),
    (lambda x: x["summary"]["index"].update(stale_hint=True),
     _coverage(matched=("running",)), diagnostics.ExecutionEvidence(True, None),
     ("index_incomplete",)),
    (lambda x: None, _coverage(unmatched=("running",)),
     diagnostics.ExecutionEvidence(True, None), ("vocabulary_mismatch",)),
    (lambda x: None, _coverage(matched=("running",)),
     diagnostics.ExecutionEvidence(True, None), ("likely_absent",)),
])
def test_each_cause_has_its_exact_evidence(mutate, coverage, evidence, expected):
    envelope = _envelope()
    mutate(envelope)
    assert diagnostics.no_result_causes(envelope, coverage, evidence) == expected


def test_cause_order_is_fixed_and_likely_absent_exclusive():
    envelope = _envelope()
    envelope["diagnostics"]["unknown_collections"] = ["missing"]
    envelope["diagnostics"]["filters"].update(raw_count=4, dropped_by_truncate=4)
    envelope["summary"]["index"]["stale_hint"] = True
    causes = diagnostics.no_result_causes(
        envelope, _coverage(unmatched=("running",)),
        diagnostics.ExecutionEvidence(False, "deadline", semantic_unavailable=True))
    assert causes == ("unknown_collection", "filters_dropped_all", "execution_partial",
                      "semantic_unavailable", "index_incomplete")
    assert "likely_absent" not in causes


def test_unknown_only_scope_never_claims_vocabulary_or_absence():
    envelope = _envelope()
    envelope["request"]["collections"] = ["missing"]
    envelope["diagnostics"]["unknown_collections"] = ["missing"]
    assert diagnostics.no_result_causes(
        envelope, _coverage(unmatched=("running",)),
        diagnostics.ExecutionEvidence(True, None)) == ("unknown_collection",)


def test_vector_residual_is_not_absence_even_with_matched_vocabulary():
    envelope = _envelope()
    envelope["request"]["mode"] = "vector"
    assert diagnostics.no_result_causes(
        envelope, _coverage(matched=("running",)),
        diagnostics.ExecutionEvidence(True, None)) == ()
    envelope["request"]["mode"] = "lexical"
    assert diagnostics.no_result_causes(
        envelope, _coverage(matched=("running",)),
        diagnostics.ExecutionEvidence(True, None)) == ("likely_absent",)


def test_ok_result_and_nonpositive_limit_with_results_have_no_causes():
    envelope = _envelope()
    envelope["status"] = "ok"
    envelope["request"]["limit"] = -1
    assert diagnostics.no_result_causes(
        envelope, _coverage(unmatched=("running",)),
        diagnostics.ExecutionEvidence(True, None)) == ()


def test_inconsistent_execution_evidence_is_rejected():
    with pytest.raises(ValueError):
        diagnostics.ExecutionEvidence(True, "deadline")
    with pytest.raises(ValueError):
        diagnostics.ExecutionEvidence(False, None)


def test_real_lexical_envelopes_distinguish_empty_index_and_unknown_scope(tmp_path):
    conn = db.connect(tmp_path / "search.db")
    db.init_schema(conn)
    request = {"query": "running", "mode": "lexical", "collections": [], "limit": 10}
    empty = daemon.run_search(conn, vector_store.NullStore(), None, request)
    assert diagnostics.no_result_causes(
        empty, guidance.term_coverage(conn, "running"),
        diagnostics.ExecutionEvidence(True, None)) == ("empty_index",)
    root = tmp_path / "docs"; root.mkdir()
    (root / "a.md").write_text("running quickly")
    indexer.add_collection(conn, "docs", root)
    indexer.sync_collection(conn, vector_store.NullStore(), "docs", indexer.IndexStats())
    request["collections"] = ["missing"]
    unknown = daemon.run_search(conn, vector_store.NullStore(), None, request)
    assert diagnostics.no_result_causes(
        unknown, guidance.term_coverage(conn, "running", collections=["missing"]),
        diagnostics.ExecutionEvidence(True, None)) == ("unknown_collection",)
    request["collections"] = ["docs"]
    request["min_score"] = 2.0
    filtered = daemon.run_search(conn, vector_store.NullStore(), None, request)
    assert diagnostics.no_result_causes(
        filtered, guidance.term_coverage(conn, "running", collections=["docs"]),
        diagnostics.ExecutionEvidence(True, None)) == ("filters_dropped_all",)
    conn.close()


def test_real_vector_empty_due_to_unembedded_or_postfiltered_pool_has_no_absence(tmp_path):
    class FakeEmbedder:
        def embed_queries(self, _queries):
            return [object()]

    class OtherCollectionOnly:
        name = "fake"

        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

        def knn(self, _vector, _limit):
            return [(self.chunk_id, 0.1)]

    conn = db.connect(tmp_path / "vectors.db")
    db.init_schema(conn)
    for name in ("alpha", "beta"):
        root = tmp_path / name; root.mkdir()
        (root / "a.md").write_text("running quickly")
        indexer.add_collection(conn, name, root)
        indexer.sync_collection(conn, vector_store.NullStore(), name, indexer.IndexStats())
    request = {"query": "running", "mode": "vector", "collections": ["alpha"],
               "limit": 10}
    coverage = guidance.term_coverage(conn, "running", collections=["alpha"])
    assert coverage.matched == ("running",)
    beta_id = conn.execute("SELECT id FROM documents WHERE collection='beta'").fetchone()[0]
    other_chunk = conn.execute(
        "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, 0, 0, 10)",
        (beta_id,)).lastrowid
    for store in (vector_store.NullStore(), OtherCollectionOnly(other_chunk)):
        envelope = daemon.run_search(conn, store, FakeEmbedder(), request)
        assert envelope["status"] == "no_results"
        assert diagnostics.no_result_causes(
            envelope, coverage, diagnostics.ExecutionEvidence(True, None)) == ()
    conn.close()


def test_qpp_and_execution_projections_are_internal_pure_data():
    shape = ScoreShape(0.25, 0.5, 0.75, 0.8)
    signals = QPPSignals(shape, shape, shape, 0.4, 0.6)
    assert diagnostics.qpp_projection(signals) == {
        "lexical": {"nqc_like": 0.25, "wig_like": 0.5,
                    "top_gap": 0.75, "head_concentration": 0.8},
        "vector": {"nqc_like": 0.25, "wig_like": 0.5,
                   "top_gap": 0.75, "head_concentration": 0.8},
        "fused": {"nqc_like": 0.25, "wig_like": 0.5,
                  "top_gap": 0.75, "head_concentration": 0.8},
        "channel_agreement": 0.4, "term_coverage": 0.6}
    assert diagnostics.execution_projection(
        diagnostics.ExecutionEvidence(True, None)) == {
            "complete": True, "partial_reason": None}


def _result(doc_id, score, relpath=None, tier="or", collection="docs"):
    return Result(doc_id, collection, relpath or f"{doc_id}.md", "", str(doc_id),
                  score, tier=tier)


def _ok(trace, coverage=None, *, raw=None, dropped=0, final=None):
    envelope = _envelope()
    envelope["status"] = "ok"
    envelope["request"]["mode"] = trace.mode
    envelope["diagnostics"]["filters"].update(
        raw_count=len(trace.fused_pool) if raw is None else raw,
        dropped_by_truncate=dropped,
        after_truncate=len(trace.fused_pool) if final is None else final)
    return diagnostics.confidence_reasons(trace, coverage or _coverage(matched=("running",)),
                                          envelope)


def test_flat_scores_only_for_tied_multi_result_head():
    tied = SearchTrace(mode="hybrid", fused_pool=(_result(1, 0.4), _result(2, 0.4)))
    assert "flat_scores" in _ok(tied)
    assert "flat_scores" not in _ok(SearchTrace(mode="hybrid", fused_pool=(
        _result(1, 0.4), _result(2, 0.3))))
    assert "flat_scores" not in _ok(SearchTrace(mode="hybrid", fused_pool=(
        _result(1, 0.4),)))


def test_channel_agreement_requires_two_nonempty_disjoint_hybrid_sets():
    lex = _result(1, 0.9, tier="and")
    vec = _result(2, 0.8)
    disjoint = SearchTrace(mode="hybrid", lexical=(lex,), vector=(vec,),
                           fused_pool=(lex, vec))
    assert "low_channel_agreement" in _ok(disjoint)
    assert "low_channel_agreement" not in _ok(SearchTrace(
        mode="hybrid", lexical=(lex,), vector=(lex,), fused_pool=(lex,)))
    assert "low_channel_agreement" not in _ok(SearchTrace(
        mode="lexical", lexical=(lex,), fused_pool=(lex,)))


def test_low_term_coverage_uses_scoped_unmatched_surface():
    trace = SearchTrace(mode="vector", fused_pool=(_result(1, 0.9),))
    assert _ok(trace, _coverage(matched=("running",), unmatched=("deployment",))) == (
        "low_term_coverage",)
    assert _ok(trace, _coverage(matched=("running",))) == ()


def test_diffuse_directories_requires_three_results_without_majority():
    split = SearchTrace(mode="vector", fused_pool=(
        _result(1, 0.9, "src/a.py"), _result(2, 0.8, "src/b.py"),
        _result(3, 0.7, "tests/c.py"), _result(4, 0.6, "tests/d.py")))
    assert "diffuse_directories" in _ok(split)
    majority = SearchTrace(mode="vector", fused_pool=split.fused_pool[:3] + (
        _result(5, 0.6, "src/e.py"),))
    assert "diffuse_directories" not in _ok(majority)
    two = SearchTrace(mode="vector", fused_pool=(split.fused_pool[0], split.fused_pool[2]))
    assert "diffuse_directories" not in _ok(two)


def test_weak_lexical_anchor_uses_phrase_or_and_tiers_only_when_searched():
    or_hit = _result(1, 0.9, tier="or")
    and_hit = _result(2, 0.8, tier="and")
    assert "weak_lexical_anchor" in _ok(SearchTrace(
        mode="hybrid", lexical=(or_hit,), fused_pool=(or_hit,)))
    assert "weak_lexical_anchor" not in _ok(SearchTrace(
        mode="hybrid", lexical=(and_hit,), fused_pool=(and_hit,)))
    assert "weak_lexical_anchor" not in _ok(SearchTrace(
        mode="vector", fused_pool=(or_hit,)))


def test_truncated_head_is_short_only_when_truncation_dropped_results():
    trace = SearchTrace(mode="vector", fused_pool=(_result(1, 0.9), _result(2, 0.8)))
    assert "truncated_before_stable_head" in _ok(trace, raw=2, dropped=1, final=1)
    assert "truncated_before_stable_head" not in _ok(trace, raw=2, final=1)
    assert "truncated_before_stable_head" not in _ok(trace, raw=1, dropped=1, final=1)


def test_dark_projection_has_exact_result_count_and_explain_only_qpp():
    hit = _result(1, 0.9, tier="and")
    trace = SearchTrace(mode="lexical", lexical=(hit,), fused_pool=(hit,))
    envelope = _envelope(); envelope["status"] = "ok"
    envelope["request"]["mode"] = "lexical"
    envelope["results"] = [{"path": "docs/1.md"}]
    envelope["summary"]["result_count"] = 1
    projection = diagnostics.assemble(envelope, trace, _coverage(matched=("running",)),
                                      diagnostics.ExecutionEvidence(True, None))
    assert projection["summary"]["count_relation"] == "gte"
    assert projection["summary"]["confidence_reasons"] == []
    assert "qpp" not in projection["diagnostics"]
    explained = diagnostics.assemble(envelope, trace, _coverage(matched=("running",)),
                                     diagnostics.ExecutionEvidence(True, None), explain=True)
    assert "qpp" in explained["diagnostics"]
    assert envelope["summary"]["result_count"] == 1
    assert "confidence_reasons" not in envelope["summary"]
