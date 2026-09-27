# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The search budget changes presentation without changing retrieval order."""

from __future__ import annotations

import copy
import json
import math

import pytest

from fidx.budget import apply_budget, estimate_tokens, unclassified_fields


def envelope(status: str = "ok") -> dict:
    results = [] if status != "ok" else [
        {"rank": rank, "path": f"notes/{rank}.md", "collection": "notes",
         "relpath": f"{rank}.md", "docid": f"#{rank:06x}", "score": 1 / rank,
         "title": f"Title {rank} " + "T" * 80,
         "snippet": "é" * 160 + str(rank),
         "sources": {"lexical": 1 / rank, "vector": 0.5 / rank},
         "locations": [{"line_start": rank, "line_end": rank + 1}]}
        for rank in range(1, 4)
    ]
    diagnostics = {
        "index_empty": status == "empty_index",
        "unknown_collections": [],
        "filters": {"raw_count": len(results), "after_min_score": len(results),
                    "after_truncate": len(results), "dropped_by_min_score": 0,
                    "dropped_by_truncate": 0},
        "calibration": {"floor_available": False, "floor": None},
        "execution": {"daemon": "cold"},
        "term_coverage": {"matched": ["é"], "unmatched": ["other"]},
    }
    if status != "ok":
        diagnostics["no_results"] = {"causes": [{"kind": "missing", "reason": "No indexed match."}]}
    return {
        "schema": "fidx.search.v2", "query": "é", "status": status,
        "request": {"mode": "hybrid", "collections": [], "limit": 3,
                    "min_score": None, "truncate": "off"},
        "summary": {"result_count": len(results), "confidence": "strong",
                    "limit_reached": bool(results), "top_score": 1.0 if results else None,
                    "source_mix": {"both": len(results), "lexical_only": 0,
                                   "vector_only": 0, "other": 0},
                    "truncation_advice": {"reason": "A" * 300},
                    "confidence_reasons": ["low_term_coverage"],
                    "count_relation": "few", "index": {"stale": False}},
        "results": results, "diagnostics": diagnostics,
        "next_actions": [{"intent": "retry", "reason": "R" * 180,
                          "command": ["fidx", "search", "other"]}],
        "query_suggestions": [
            {"query": f"alternative {n}", "reason": "S" * 160}
            for n in range(3)],
        "facets": {"directories": ["F" * 300]},
    }


def test_estimate_is_canonical_compact_utf8_bytes_over_four():
    payload = {"z": "é", "a": [1, 2]}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert estimate_tokens(payload) == math.ceil(len(canonical.encode("utf-8")) / 4)


def test_zero_budget_reaches_floor_without_mutating_input_or_ranking():
    original = envelope()
    snapshot = copy.deepcopy(original)
    out = apply_budget(original, 0)
    assert original == snapshot
    assert out["schema"] == original["schema"]
    assert out["status"] == "ok"
    assert out["request"] == original["request"]
    for key in ("confidence", "limit_reached", "top_score", "source_mix"):
        assert out["summary"][key] == original["summary"][key]
    assert out["summary"]["result_count"] == 3
    assert out["results"] == [{k: original["results"][0][k] for k in
                               ("rank", "path", "collection", "relpath", "docid", "score")}]
    assert out["summary"]["budget"]["over_minimum"] is True
    assert out["summary"]["budget"]["estimated_used"] == estimate_tokens(out)
    assert out["summary"]["budget"]["dropped"]["results"] == 2
    assert out["summary"]["budget"]["dropped"]["titles"] == 3
    assert out["summary"]["budget"]["dropped"]["snippets"] == 3
    assert "diagnostics" not in out
    assert "next_actions" not in out


@pytest.mark.parametrize("status", ["no_results", "empty_index"])
def test_tiny_non_ok_preserves_abstention_guidance(status):
    original = envelope(status)
    out = apply_budget(original, 0)
    assert out["status"] == status
    assert out["next_actions"] == original["next_actions"]
    assert out["diagnostics"] == {"no_results": original["diagnostics"]["no_results"]}
    assert out["summary"]["budget"]["over_minimum"] is True
    assert out["summary"]["budget"]["dropped"]["next_actions"] is False
    assert out["results"] == []


def test_ladder_never_reorders_or_alters_retained_results_across_all_budgets():
    original = envelope()
    full = apply_budget(original, 100_000)
    maximum = estimate_tokens(full)
    for requested in range(maximum + 1):
        out = apply_budget(original, requested)
        rows = out["results"]
        assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
        assert [row["path"] for row in rows] == [row["path"] for row in original["results"][:len(rows)]]
        assert [row["score"] for row in rows] == [row["score"] for row in original["results"][:len(rows)]]
        assert out["summary"]["budget"]["estimated_used"] == estimate_tokens(out)
        if not out["summary"]["budget"]["over_minimum"]:
            assert estimate_tokens(out) <= requested


def test_every_existing_schema_field_is_classified_and_new_fields_fail():
    payload = envelope()
    assert unclassified_fields(payload) == []
    for path, target in [
        ("top", payload), ("summary", payload["summary"]),
        ("result", payload["results"][0]),
        ("diagnostic", payload["diagnostics"]),
    ]:
        changed = copy.deepcopy(payload)
        if path == "top":
            changed["future_field"] = 1
        elif path == "summary":
            changed["summary"]["future_field"] = 1
        elif path == "result":
            changed["results"][0]["future_field"] = 1
        else:
            changed["diagnostics"]["future_field"] = 1
        assert "future_field" in str(unclassified_fields(changed))


def test_every_ladder_step_has_a_boundary_in_the_budget_sweep():
    original = envelope()
    maximum = estimate_tokens(apply_budget(original, 100_000))
    states = []
    for requested in range(maximum, -1, -1):
        dropped = apply_budget(original, requested)["summary"]["budget"]["dropped"]
        if not states or dropped != states[-1]:
            states.append(dropped)
    changes = []
    for before, after in zip(states, states[1:]):
        changed = [key for key in after if after[key] != before[key]]
        assert len(changed) == 1
        changes.append(changed[0])
    assert changes == [
        "facets", "suggestions", "term_coverage", "suggestions",
        "snippets", "snippets", "snippets", "titles", "titles", "titles",
        "advice", "next_actions", "diagnostics", "confidence_reasons",
        "count_relation", "index", "sources", "sources", "sources",
        "locations", "locations", "locations", "results", "results",
    ]

    # Dropping a per-result field starts at the tail, even before any result
    # itself is omitted. This makes the best hit the most informative one.
    one_snippet = next(apply_budget(original, amount) for amount in range(maximum + 1)
                       if apply_budget(original, amount)["summary"]["budget"]
                       ["dropped"]["snippets"] == 1)
    assert ["snippet" in row for row in one_snippet["results"]] == [True, True, False]
    one_title = next(apply_budget(original, amount) for amount in range(maximum + 1)
                     if apply_budget(original, amount)["summary"]["budget"]
                     ["dropped"]["titles"] == 1)
    assert ["title" in row for row in one_title["results"]] == [True, True, False]


@pytest.mark.parametrize("status", ["ok", "no_results", "empty_index"])
def test_status_dependent_floor_intermediate_bands(status):
    original = envelope(status)
    floor = apply_budget(original, 0)
    floor_used = estimate_tokens(floor)
    bare = copy.deepcopy(floor)
    del bare["request"]
    for field in ("confidence", "limit_reached", "top_score", "source_mix"):
        del bare["summary"][field]
    bare_used = estimate_tokens(bare)
    assert bare_used + 5 < floor_used
    for requested in range(bare_used + 5, floor_used - 5):
        out = apply_budget(original, requested)
        assert out["summary"]["budget"]["over_minimum"] is True
    if status != "ok":
        without_guidance = copy.deepcopy(floor)
        del without_guidance["next_actions"]
        del without_guidance["diagnostics"]
        lower = estimate_tokens(without_guidance)
        assert lower + 5 < floor_used
        for requested in range(lower + 5, floor_used - 5):
            out = apply_budget(original, requested)
            assert out["summary"]["budget"]["over_minimum"] is True
            assert out["next_actions"] == original["next_actions"]
            assert out["diagnostics"]["no_results"]["causes"] == (
                original["diagnostics"]["no_results"]["causes"])


def test_daemon_request_and_cli_cold_and_warm_return_same_budgeted_payload(
    tmp_path, corpus_dir, monkeypatch
):
    from click.testing import CliRunner
    from fidx import daemon, db as dbmod, indexer, vector_store
    from fidx.cli import main

    db_path = tmp_path / "index.db"
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, vector_store.NullStore(), "notes", stats)
    request = {"cmd": "search", "query": "carbonara", "mode": "lexical",
               "collections": [], "limit": 10, "min_score": None,
               "truncate": None, "budget_tokens": 250}
    direct = daemon.run_search(conn, None, None, request)
    assert "budget" in direct["summary"]
    cold = CliRunner().invoke(main, [
        "--db", str(db_path), "search", "carbonara", "--mode", "lexical",
        "--json", "--budget-tokens", "250", "--no-daemon",
    ])
    assert cold.exit_code == 0, cold.output
    assert json.loads(cold.output) == direct
    assert cold.output.strip() == json.dumps(
        direct, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert math.ceil(len(cold.output.strip().encode("utf-8")) / 4) == (
        direct["summary"]["budget"]["estimated_used"])

    raw = daemon.run_search(conn, None, None, {k: v for k, v in request.items()
                                                if k != "budget_tokens"})
    instance_id = dbmod.get_meta(conn, "instance_id")
    monkeypatch.setattr(daemon, "client_request", lambda *args, **kwargs: {
        "ok": True, "instance_id": instance_id, "results": raw,
    })
    warm = CliRunner().invoke(main, [
        "--db", str(db_path), "search", "carbonara", "--mode", "lexical",
        "--json", "--budget-tokens", "250",
    ])
    assert warm.exit_code == 0, warm.output
    assert json.loads(warm.output) == direct
    conn.close()


def test_cli_budget_requires_json_and_rejects_negative(tmp_path):
    from click.testing import CliRunner
    from fidx.cli import main

    missing_json = CliRunner().invoke(main, [
        "--db", str(tmp_path / "index.db"), "search", "x", "--budget-tokens", "10",
    ])
    assert missing_json.exit_code != 0
    assert "--json" in missing_json.output
    negative = CliRunner().invoke(main, [
        "--db", str(tmp_path / "index.db"), "search", "x", "--json",
        "--budget-tokens", "-1",
    ])
    assert negative.exit_code != 0


def test_real_emitted_envelopes_are_fully_classified_and_preserve_empty_guidance(
    conn, corpus_dir
):
    from fidx import daemon, indexer, vector_store

    empty = daemon.run_search(conn, None, None, {
        "query": "absent", "mode": "lexical", "limit": 10,
    })
    assert empty["status"] == "empty_index"
    assert unclassified_fields(empty) == []
    tiny_empty = apply_budget(empty, 0)
    assert tiny_empty["next_actions"] == empty["next_actions"]
    assert "budget" not in empty["summary"]
    assert "facets" not in tiny_empty["summary"]["budget"]["dropped"]
    assert "index" not in tiny_empty["summary"]["budget"]["dropped"]

    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, vector_store.NullStore(), "notes", indexer.IndexStats())
    for query, status in (("carbonara", "ok"), ("unfindablexyz", "no_results")):
        emitted = daemon.run_search(conn, None, None, {
            "query": query, "mode": "lexical", "limit": 10,
        })
        assert emitted["status"] == status
        assert unclassified_fields(emitted) == []
        tiny = apply_budget(emitted, 0)
        if status != "ok":
            assert tiny["next_actions"] == emitted["next_actions"]
