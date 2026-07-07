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
    assert out["diagnostics"]["index_empty"] is False
    assert "active_docs" not in out["diagnostics"]
    assert "known_collections" not in out["diagnostics"]
    assert any(a["intent"] == "inspect_best_match" for a in out["next_actions"])


def test_run_search_too_many_results_recommends_knee_without_calibrated_duplicate(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0,
                   sources={"lexical": 0.9, "vector": 0.8}),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8,
                   sources={"lexical": 0.7}),
            Result(3, "notes", "c.md", "C", "cccccc", 0.6,
                   sources={"vector": 0.6}),
            Result(4, "notes", "d.md", "D", "dddddd", 0.1,
                   sources={"vector": 0.1}),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "broad operations note",
        "mode": "hybrid",
        "collections": [],
        "limit": 4,
        "min_score": None,
        "truncate": None,
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    assert out["status"] == "ok"
    assert out["summary"]["result_count"] == 4
    assert out["summary"]["limit_reached"] is True
    assert out["summary"]["confidence"] == "strong"
    advice = out["summary"]["truncation_advice"]
    assert advice["recommendation"] == "knee"
    assert advice["primary_action"] == "clean_shortlist"
    assert advice["lean"] == "balanced"
    assert advice["score_profile"]["count"] == 4
    assert advice["score_profile"]["has_knee"] is True
    assert out["diagnostics"]["calibration"] == {
        "floor_available": False,
        "floor": None,
    }
    assert intents["clean_shortlist"]["command"] == [
        "fidx", "search", "broad operations note", "--json", "-n", "4",
        "--truncate", "knee",
    ]
    assert "use_calibrated_abstention" not in intents
    calibrated = next(o for o in advice["options"] if o["truncate"] == "calibrated")
    assert calibrated["applicable"] is False
    assert calibrated["recommended"] is False
    assert intents["increase_limit"]["command"] == [
        "fidx", "search", "broad operations note", "--json", "-n", "8",
    ]


def test_run_search_too_many_results_offers_calibrated_when_floor_exists(
    indexed, store, embedder, monkeypatch
):
    dbmod.set_meta(indexed, "truncate_floor", "0.42")

    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.6),
            Result(4, "notes", "d.md", "D", "dddddd", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "broad operations note",
        "mode": "hybrid",
        "collections": [],
        "limit": 4,
        "min_score": None,
        "truncate": None,
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    assert out["diagnostics"]["calibration"] == {
        "floor_available": True,
        "floor": 0.42,
    }
    assert advice["recommendation"] == "knee"
    calibrated = next(o for o in advice["options"] if o["truncate"] == "calibrated")
    assert calibrated["applicable"] is True
    assert calibrated["recommended"] is False
    assert calibrated["lean"] == "purity"
    assert intents["use_calibrated_abstention"]["command"] == [
        "fidx", "search", "broad operations note", "--json", "-n", "4",
        "--truncate", "calibrated",
    ]


def test_run_search_three_results_keeps_truncation_off_because_knee_noops(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "three candidates",
        "mode": "hybrid",
        "collections": [],
        "limit": 3,
        "min_score": None,
        "truncate": None,
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    assert advice["recommendation"] == "off"
    assert advice["primary_action"] == "keep_current"
    assert advice["lean"] == "recall"
    assert advice["score_profile"]["count"] == 3
    assert advice["score_profile"]["has_knee"] is False
    assert "clean_shortlist" not in intents
    assert "use_calibrated_abstention" not in intents
    knee = next(o for o in advice["options"] if o["truncate"] == "knee")
    assert knee["applicable"] is False
    assert "at least 4" in knee["reason"]


def test_run_search_too_few_results_suggests_broadening_not_noise_controls(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [Result(1, "notes", "a.md", "A", "aaaaaa", 0.5)]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "overly specific remembered wording",
        "mode": "hybrid",
        "collections": [],
        "limit": 5,
        "min_score": None,
        "truncate": None,
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    assert out["status"] == "ok"
    assert out["summary"]["result_count"] == 1
    assert out["summary"]["limit_reached"] is False
    assert out["summary"]["confidence"] == "narrow"
    advice = out["summary"]["truncation_advice"]
    assert advice["recommendation"] == "off"
    assert advice["primary_action"] == "keep_current"
    assert advice["lean"] == "recall"
    assert "clean_shortlist" not in intents
    assert "use_calibrated_abstention" not in intents
    assert "increase_limit" not in intents
    assert intents["broaden_query"]["command"] == [
        "fidx", "search", "<broader query>", "--json", "-n", "5",
    ]


def test_run_search_no_results_suggests_alternate_modes_and_broader_query(
    indexed, store, embedder
):
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "zzznevermatch",
        "mode": "lexical",
        "collections": [],
        "limit": 5,
        "min_score": None,
        "truncate": None,
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    assert out["status"] == "no_results"
    assert out["summary"]["result_count"] == 0
    assert out["summary"]["confidence"] == "none"
    assert out["summary"]["truncation_advice"]["recommendation"] is None
    assert out["summary"]["truncation_advice"]["primary_action"] is None
    assert out["diagnostics"]["filters"] == {
        "raw_count": 0,
        "after_min_score": 0,
        "after_truncate": 0,
        "dropped_by_min_score": 0,
        "dropped_by_truncate": 0,
    }
    assert "try_exact_terms" not in intents
    assert intents["try_hybrid"]["command"] == [
        "fidx", "search", "zzznevermatch", "--json", "-n", "5",
    ]
    assert intents["try_conceptual_terms"]["command"] == [
        "fidx", "search", "zzznevermatch", "--json", "--mode", "vector", "-n", "5",
    ]
    assert intents["broaden_query"]["command"] == [
        "fidx", "search", "<broader query>", "--json", "--mode", "lexical", "-n", "5",
    ]


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
    assert out["diagnostics"]["unknown_collections"] == ["missing"]
    assert "known_collections" not in out["diagnostics"]
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
    assert out["diagnostics"]["index_empty"] is True
    assert "active_docs" not in out["diagnostics"]
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
    advice = out["summary"]["truncation_advice"]
    assert advice["recommendation"] == "off"
    assert advice["primary_action"] == "disable_truncation"
    assert advice["lean"] == "recall"
    retry = next(a for a in out["next_actions"] if a["intent"] == "disable_truncation")
    assert retry["command"] == [
        "fidx", "search", "synthetic cliff", "--json", "-n", "10"
    ]


def test_run_search_already_knee_truncated_results_offer_loosen_not_repeat(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.6),
            Result(4, "notes", "d.md", "D", "dddddd", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "already knee",
        "mode": "hybrid",
        "collections": [],
        "limit": 4,
        "min_score": None,
        "truncate": "knee",
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    assert out["status"] == "ok"
    assert out["diagnostics"]["filters"]["dropped_by_truncate"] > 0
    assert advice["current"] == "knee"
    assert advice["recommendation"] == "knee"
    assert advice["primary_action"] == "keep_current"
    assert "clean_shortlist" not in intents
    assert intents["disable_truncation"]["command"] == [
        "fidx", "search", "already knee", "--json", "-n", "4"
    ]


def test_run_search_knee_noop_results_recommend_off_for_clarity(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "knee no op",
        "mode": "hybrid",
        "collections": [],
        "limit": 3,
        "min_score": None,
        "truncate": "knee",
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    assert out["status"] == "ok"
    assert out["diagnostics"]["filters"]["dropped_by_truncate"] == 0
    assert advice["score_profile"]["has_knee"] is False
    assert advice["recommendation"] == "off"
    assert advice["primary_action"] == "disable_truncation"
    assert intents["disable_truncation"]["command"] == [
        "fidx", "search", "knee no op", "--json", "-n", "3"
    ]


def test_run_search_already_calibrated_results_offer_knee_not_calibrated_repeat(
    indexed, store, embedder, monkeypatch
):
    dbmod.set_meta(indexed, "truncate_floor", "0.42")

    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.6),
            Result(4, "notes", "d.md", "D", "dddddd", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "already calibrated",
        "mode": "hybrid",
        "collections": [],
        "limit": 4,
        "min_score": None,
        "truncate": "calibrated",
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    assert out["status"] == "ok"
    assert out["diagnostics"]["filters"]["dropped_by_truncate"] > 0
    assert advice["current"] == "calibrated"
    assert advice["recommendation"] == "calibrated"
    assert advice["primary_action"] == "keep_current"
    assert "use_calibrated_abstention" not in intents
    assert intents["clean_shortlist"]["command"] == [
        "fidx", "search", "already calibrated", "--json", "-n", "4",
        "--truncate", "knee",
    ]
    assert intents["disable_truncation"]["command"] == [
        "fidx", "search", "already calibrated", "--json", "-n", "4"
    ]


def test_run_search_calibrated_abstains_without_knee_recommends_off(
    indexed, store, embedder, monkeypatch
):
    dbmod.set_meta(indexed, "truncate_floor", "0.5")

    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 0.4),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.3),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "weak calibrated",
        "mode": "hybrid",
        "collections": [],
        "limit": 5,
        "min_score": None,
        "truncate": "calibrated",
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    knee = next(o for o in advice["options"] if o["truncate"] == "knee")
    assert out["status"] == "no_results"
    assert out["diagnostics"]["filters"]["after_min_score"] == 2
    assert out["diagnostics"]["filters"]["after_truncate"] == 0
    assert advice["score_profile"]["count"] == 2
    assert advice["score_profile"]["has_knee"] is False
    assert advice["recommendation"] == "off"
    assert advice["primary_action"] == "disable_truncation"
    assert knee["applicable"] is False
    assert knee["recommended"] is False
    assert "clean_shortlist" not in intents
    assert intents["disable_truncation"]["command"] == [
        "fidx", "search", "weak calibrated", "--json", "-n", "5"
    ]


def test_run_search_calibrated_without_floor_recommends_knee_when_it_applies(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.6),
            Result(4, "notes", "d.md", "D", "dddddd", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "calibrated no floor",
        "mode": "hybrid",
        "collections": [],
        "limit": 4,
        "min_score": None,
        "truncate": "calibrated",
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    calibrated = next(o for o in advice["options"] if o["truncate"] == "calibrated")
    assert out["diagnostics"]["calibration"]["floor_available"] is False
    assert advice["current"] == "calibrated"
    assert advice["recommendation"] == "knee"
    assert advice["primary_action"] == "clean_shortlist"
    assert advice["lean"] == "balanced"
    assert calibrated["applicable"] is False
    assert calibrated["recommended"] is False
    assert [a["intent"] for a in out["next_actions"]].count("clean_shortlist") == 1
    assert intents["clean_shortlist"]["command"] == [
        "fidx", "search", "calibrated no floor", "--json", "-n", "4",
        "--truncate", "knee",
    ]
    assert "use_calibrated_abstention" not in intents


def test_run_search_calibrated_without_floor_and_no_knee_recommends_off(
    indexed, store, embedder, monkeypatch
):
    def fake_hybrid(conn, store, embedder, query, collections, limit):
        return [
            Result(1, "notes", "a.md", "A", "aaaaaa", 1.0),
            Result(2, "notes", "b.md", "B", "bbbbbb", 0.8),
            Result(3, "notes", "c.md", "C", "cccccc", 0.1),
        ]

    monkeypatch.setattr(daemon.searchmod, "search_hybrid", fake_hybrid)
    out = daemon.run_search(indexed, store, embedder, {
        "cmd": "search",
        "query": "calibrated no floor no knee",
        "mode": "hybrid",
        "collections": [],
        "limit": 3,
        "min_score": None,
        "truncate": "calibrated",
    })

    intents = {a["intent"]: a for a in out["next_actions"]}
    advice = out["summary"]["truncation_advice"]
    calibrated = next(o for o in advice["options"] if o["truncate"] == "calibrated")
    assert out["diagnostics"]["calibration"]["floor_available"] is False
    assert advice["recommendation"] == "off"
    assert advice["primary_action"] == "disable_truncation"
    assert advice["lean"] == "recall"
    assert advice["score_profile"]["has_knee"] is False
    assert calibrated["applicable"] is False
    assert calibrated["recommended"] is False
    assert "clean_shortlist" not in intents
    assert "use_calibrated_abstention" not in intents
    assert intents["disable_truncation"]["command"] == [
        "fidx", "search", "calibrated no floor no knee", "--json", "-n", "3"
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
