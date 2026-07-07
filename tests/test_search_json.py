# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from fidx import daemon, db as dbmod, indexer, vector_store
from fidx.cli import main
from fidx.search import Result


@pytest.fixture()
def indexed(conn, store, embedder, corpus_dir):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return conn


def test_run_search_returns_agent_envelope_for_results(indexed, store, embedder):
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "production deployment rolling restarts",
        "mode": "hybrid",
        "collections": [],
        "limit": 5,
        "min_score": None,
        "truncate": None,
    })

    assert out["schema"] == "fidx.search.v2"
    assert out["status"] == "ok"
    assert out["query"] == "production deployment rolling restarts"
    assert out["request"]["mode"] == "hybrid"
    assert out["summary"]["result_count"] == len(out["results"])
    assert out["summary"]["source_mix"]["both"] >= 1
    assert out["results"][0]["rank"] == 1
    assert out["results"][0]["path"] == "notes/deploy.md"
    assert out["results"][0]["docid"].startswith("#")
    assert any(a["intent"] == "inspect_best_match" for a in out["next_actions"])


def test_run_search_explains_when_min_score_filters_all_hits(indexed, store, embedder):
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "carbonara",
        "mode": "hybrid",
        "collections": [],
        "limit": 5,
        "min_score": 999.0,
        "truncate": None,
    })

    assert out["status"] == "no_results"
    assert out["results"] == []
    assert out["diagnostics"]["filters"]["raw_count"] > 0
    assert out["diagnostics"]["filters"]["after_min_score"] == 0
    assert any(a["intent"] == "remove_min_score" for a in out["next_actions"])


def test_run_search_reports_unknown_collection_and_retry_command(indexed, store, embedder):
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "carbonara",
        "mode": "hybrid",
        "collections": ["missing"],
        "limit": 5,
        "min_score": None,
        "truncate": None,
    })

    assert out["status"] == "no_results"
    assert out["diagnostics"]["known_collections"] == ["notes"]
    assert out["diagnostics"]["unknown_collections"] == ["missing"]
    retry = next(a for a in out["next_actions"] if a["intent"] == "retry_without_scope")
    assert retry["command"] == ["fidx", "search", "carbonara", "--json", "-n", "5"]


def test_run_search_empty_index_tells_agent_to_add_and_index(conn, store, embedder):
    out = daemon.run_search(conn, store, embedder, {
        "cmd": "search",
        "query": "anything",
        "mode": "hybrid",
        "collections": [],
        "limit": 5,
        "min_score": None,
        "truncate": None,
    })

    assert out["status"] == "empty_index"
    assert out["diagnostics"]["active_docs"] == 0
    assert out["results"] == []
    assert [a["intent"] for a in out["next_actions"][:2]] == ["add_collection", "index"]


def test_run_search_explains_when_truncation_filters_all_hits(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.9),
            Result(3, "notes", "c.md", "C", "cccccc", 0.1),
            Result(4, "notes", "d.md", "D", "dddddd", 0.09),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "synthetic cliff",
        "mode": "hybrid",
        "collections": [],
        "limit": 10,
        "min_score": None,
        "truncate": "abs:2",
    })

    assert out["status"] == "no_results"
    assert out["diagnostics"]["filters"]["raw_count"] == 4
    assert out["diagnostics"]["filters"]["after_min_score"] == 4
    assert out["diagnostics"]["filters"]["after_truncate"] == 0
    assert out["diagnostics"]["filters"]["dropped_by_truncate"] == 4
    retry = next(a for a in out["next_actions"] if a["intent"] == "disable_truncation")
    assert retry["command"] == [
        "fidx", "search", "synthetic cliff", "--json", "-n", "10"
    ]


def test_cli_search_json_outputs_agent_envelope_for_lexical_mode(tmp_path, corpus_dir):
    db_path = tmp_path / "index.db"
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, vector_store.NullStore(), "notes", stats)
    conn.close()

    result = CliRunner().invoke(main, [
        "--db", str(db_path),
        "search", "carbonara",
        "--mode", "lexical",
        "--json",
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "fidx.search.v2"
    assert payload["request"]["mode"] == "lexical"
    assert payload["results"][0]["path"] == "notes/recipes.md"
