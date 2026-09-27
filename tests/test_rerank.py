# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Offline fixtures for the dark, pure rerank stage."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from fidx.rerank import NullReranker, rerank_results
from fidx.search import Result


class Scores:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return iter(self.scores)


def _doc(conn, path, title, body):
    digest = hashlib.sha256(body.encode()).hexdigest()
    return conn.execute(
        "INSERT INTO documents(collection,relpath,title,body,hash) "
        "VALUES ('docs',?,?,?,?)", (path, title, body, digest)).lastrowid


def _result(doc_id, path, *, score=0.1, sources=None, matched_chunk_id=None,
            line_start=None, line_end=None, title=None):
    return Result(doc_id, "docs", path, path if title is None else title, "#abcdef", score,
                  sources={} if sources is None else sources,
                  matched_chunk_id=matched_chunk_id,
                  line_start=line_start, line_end=line_end)


def test_scored_text_vector_best_chunk_and_lexical_first_chunk_location(conn):
    body = "First chunk.\n" + "x" * 300 + "\nBest chunk full text."
    doc_id = _doc(conn, "one.md", "Title", body)
    first = conn.execute(
        "INSERT INTO chunks(doc_id,seq,pos,length,line_start,line_end) "
        "VALUES (?,0,0,12,1,1)", (doc_id,)).lastrowid
    best_pos = body.index("x")
    best = conn.execute(
        "INSERT INTO chunks(doc_id,seq,pos,length,line_start,line_end) "
        "VALUES (?,1,?,?,2,3)", (doc_id, best_pos, 300)).lastrowid
    vector = _result(doc_id, "one.md", score=0.7,
                     matched_chunk_id=best, line_start=99, line_end=99)
    scorer = Scores([4.0])
    outcome = rerank_results(conn, "needle", [vector], "vector", scorer)
    assert scorer.calls == [("needle", ["x" * 300])]
    assert outcome.failed is False
    assert outcome.results[0].matched_chunk_id == best
    assert (outcome.results[0].line_start, outcome.results[0].line_end) == (2, 3)
    assert (vector.line_start, vector.line_end) == (99, 99)

    lexical = _result(doc_id, "one.md", score=0.2, title="Title")
    scorer = Scores([0.0])
    outcome = rerank_results(conn, "needle", [lexical], "lexical", scorer)
    assert scorer.calls == [("needle", ["Title\nFirst chunk."])]
    assert outcome.results[0].matched_chunk_id == first
    assert (outcome.results[0].line_start, outcome.results[0].line_end) == (1, 1)


def test_success_uses_raw_logit_order_sigmoid_and_copies_every_source(conn):
    docs = [_doc(conn, f"{i}.md", f"Title {i}", "") for i in range(3)]
    original = [
        _result(docs[0], "0.md", score=0.03, sources={"lexical": 0.9}),
        _result(docs[1], "1.md", score=0.02, sources={"vector": 0.8}),
        _result(docs[2], "2.md", score=0.01, sources={"lexical": 0.2}),
    ]
    source_maps = [dict(r.sources) for r in original]
    outcome = rerank_results(conn, "query", original, "hybrid",
                             Scores([-1000.0, 1000.0, 1000.0]))
    assert outcome.failed is False
    assert [r.doc_id for r in outcome.results] == [docs[1], docs[2], docs[0]]
    assert [r.score for r in outcome.results] == [1.0, 1.0, 0.0]
    assert outcome.results[0].sources == {
        "vector": 0.8, "fused": 0.02, "rerank": 1000.0}
    assert outcome.results[1].sources == {
        "lexical": 0.2, "fused": 0.01, "rerank": 1000.0}
    assert [r.sources for r in original] == source_maps
    assert [r.score for r in original] == [0.03, 0.02, 0.01]
    assert all(new is not old for new, old in zip(
        sorted(outcome.results, key=lambda r: r.doc_id), original))
    outcome.results[0].sources["vector"] = -1
    assert original[1].sources["vector"] == 0.8


@pytest.mark.parametrize("mode", ["lexical", "vector"])
def test_single_arm_preserves_original_score_without_fused_source(conn, mode):
    doc_id = _doc(conn, "one.md", "Title", "")
    original = _result(doc_id, "one.md", score=0.03,
                       sources={"fused": 99.0, "unexpected": 2.0})
    outcome = rerank_results(conn, "query", [original], mode, Scores([-3.0]))
    assert outcome.results[0].sources == {mode: 0.03, "rerank": -3.0}
    assert "fused" not in outcome.results[0].sources
    assert 0 < outcome.results[0].score < 0.5
    assert original.sources == {"fused": 99.0, "unexpected": 2.0}
    assert original.score == 0.03


def test_no_chunk_and_null_lines_emit_no_location(conn):
    empty_id = _doc(conn, "empty.md", "Only title", "")
    body_id = _doc(conn, "body.md", "Other", "A chunk")
    conn.execute("INSERT INTO chunks(doc_id,seq,pos,length) VALUES (?,0,0,7)",
                 (body_id,))
    scorer = Scores([0.0, 1.0])
    outcome = rerank_results(
        conn, "q", [_result(empty_id, "empty.md", title="Only title"),
                    _result(body_id, "body.md", title="Other")],
        "lexical", scorer)
    assert scorer.calls == [("q", ["Only title", "Other\nA chunk"])]
    assert outcome.results[0].matched_chunk_id is not None
    assert outcome.results[0].line_start is None
    assert outcome.results[0].line_end is None
    assert outcome.results[1].matched_chunk_id is None
    assert outcome.results[1].line_start is None


def test_old_chunk_schema_is_read_only_and_omits_lines():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE documents(id INTEGER PRIMARY KEY, body TEXT);"
        "CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id INTEGER,"
        " seq INTEGER, pos INTEGER, length INTEGER);"
        "INSERT INTO documents(id,body) VALUES (1,'Old body');"
        "INSERT INTO chunks(id,doc_id,seq,pos,length) VALUES (7,1,0,0,8);")
    before = tuple(row["name"] for row in conn.execute("PRAGMA table_info(chunks)"))
    outcome = rerank_results(conn, "q", [_result(1, "old.md")],
                             "lexical", Scores([1.0]))
    assert outcome.failed is False
    assert outcome.results[0].matched_chunk_id == 7
    assert outcome.results[0].line_start is None
    assert before == tuple(row["name"] for row in conn.execute("PRAGMA table_info(chunks)"))
    conn.close()


@pytest.mark.parametrize("scores", [
    [], [0.1, 0.2], [float("nan")], [float("inf")], [float("-inf")], ["0.5"],
])
def test_invalid_scores_degrade_without_any_candidate_mutation(conn, scores):
    doc_id = _doc(conn, "one.md", "Title", "Body")
    original = _result(doc_id, "one.md", sources={"lexical": 0.4},
                       line_start=10, line_end=11)
    outcome = rerank_results(conn, "q", [original], "hybrid", Scores(scores))
    assert outcome.failed is True
    assert outcome.results == (original,)
    assert outcome.results[0] is original
    assert original.sources == {"lexical": 0.4}
    assert (original.line_start, original.line_end) == (10, 11)


def test_generator_error_after_partial_score_degrades_atomically(conn):
    doc_id = _doc(conn, "one.md", "Title", "Body")
    original = _result(doc_id, "one.md", score=0.5)
    class Broken:
        def rerank(self, query, documents):
            yield 1.0
            raise RuntimeError("bad model output")
    outcome = rerank_results(conn, "q", [original], "lexical", Broken())
    assert outcome.failed is True
    assert outcome.results[0] is original
    assert original.score == 0.5 and original.sources == {}


def test_null_reranker_preserves_fused_tie_order(conn):
    first = _doc(conn, "one.md", "One", "")
    second = _doc(conn, "two.md", "Two", "")
    results = [_result(first, "one.md"), _result(second, "two.md")]
    outcome = rerank_results(conn, "q", results, "hybrid", NullReranker())
    assert outcome.failed is False
    assert [r.doc_id for r in outcome.results] == [first, second]
