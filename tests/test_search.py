# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import pytest

from fidx import indexer, search
from fidx.search import Result, fts_match_expr, rrf_fuse


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
