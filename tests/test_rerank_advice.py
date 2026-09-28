# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fake-only rerank advice, budget and human-output contract fixtures."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from fidx import budget, cli, daemon, db, indexer, rerank_models, search as searchmod, vector_store
from fidx.search import Result


MODEL = "jinaai/jina-reranker-v1-tiny-en"


class Scores:
    def __init__(self, *, broken=False):
        self.broken = broken

    def rerank(self, _query, documents):
        if self.broken:
            raise RuntimeError("fake scorer failure")
        return iter(float(i) for i, _ in enumerate(documents))


def _ranked(conn, count=4):
    rows = []
    for i in range(count):
        body = f"alpha body number {i}\n"
        cur = conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash) "
            "VALUES ('docs', ?, ?, ?, ?)",
            (f"{i}.md", f"Title {i}", body, f"hash{i}"))
        doc_id = cur.lastrowid
        conn.execute(
            "INSERT INTO chunks(doc_id, seq, pos, length, line_start, line_end) "
            "VALUES (?, 0, 0, ?, 1, 1)", (doc_id, len(body)))
        rows.append(Result(doc_id, "docs", f"{i}.md", f"Title {i}",
                           f"{i:06x}", 0.1 + i / 10))
    conn.commit()
    return rows


def _search_commands(payload):
    commands = [a["command"] for a in payload.get("next_actions", [])
                if a.get("command", [])[:2] == ["fidx", "search"]]
    advice = payload.get("summary", {}).get("truncation_advice", {})
    commands.extend(option["command"] for option in advice.get("options", [])
                    if option.get("command", [])[:2] == ["fidx", "search"])
    return commands


def _direct(conn, monkeypatch, *, rerank_state, broken=False,
            no_results=False, budget_tokens=None):
    ranked = [] if no_results else _ranked(conn)
    monkeypatch.setattr(searchmod, "search_lexical",
                        lambda _conn, _q, _scope, limit: ranked[:limit])
    req = {"query": "alpha", "mode": "lexical", "collections": ["ghost"] if no_results else [],
           "limit": 4, "min_score": None, "truncate": None}
    if rerank_state == "on":
        req["_rerank_model"] = MODEL
    elif rerank_state == "off_env":
        req["_rerank_advice_off"] = True
    if budget_tokens is not None:
        req["budget_tokens"] = budget_tokens
    scorer = Scores(broken=broken) if rerank_state == "on" else None
    return daemon.run_search(conn, None, None, req, reranker=scorer)


@pytest.mark.parametrize("state,expected", [
    ("on", "--rerank=" + MODEL), ("off_env", "--no-rerank"),
    ("default", None),
])
@pytest.mark.parametrize("no_results", [False, True])
def test_every_generated_search_command_preserves_effective_state(
        conn, monkeypatch, state, expected, no_results):
    payload = _direct(conn, monkeypatch, rerank_state=state, no_results=no_results)
    commands = _search_commands(payload)
    assert commands
    for command in commands:
        flags = [arg for arg in command if arg == "--no-rerank"
                 or arg.startswith("--rerank=")]
        assert flags == ([expected] if expected else []), command
    assert budget.unclassified_fields(payload) == []
    assert payload["request"].get("rerank") == ({"model": MODEL} if state == "on" else None)


def test_runtime_fallback_keeps_on_flag_in_advice_and_warning(conn, monkeypatch):
    payload = _direct(conn, monkeypatch, rerank_state="on", broken=True)
    assert payload["request"]["rerank"] == {"model": MODEL}
    assert payload["diagnostics"]["warnings"] == [
        {"component": "rerank", "code": "runtime_failure", "fallback": "fused"}]
    assert all(command.count("--rerank=" + MODEL) == 1
               for command in _search_commands(payload))


def test_budget_request_is_atomic_and_tiny_status_guidance_survives(
        conn, monkeypatch, tmp_path):
    full = _direct(conn, monkeypatch, rerank_state="on", broken=True)
    tiny = budget.apply_budget(full, 0)
    assert tiny["request"] == full["request"]
    assert tiny["results"][0]["score"] == full["results"][0]["score"]
    assert tiny["results"][0]["path"] == full["results"][0]["path"]
    assert tiny["summary"]["result_count"] == full["summary"]["result_count"]
    assert tiny["summary"]["budget"]["estimated_used"] == budget.estimate_tokens(tiny)
    assert budget.unclassified_fields(tiny) == []
    assert full["diagnostics"]["warnings"]
    assert tiny["summary"]["budget"]["dropped"]["diagnostics"] is True

    other = _direct(conn, monkeypatch, rerank_state="on", no_results=True)
    tiny_other = budget.apply_budget(other, 0)
    assert tiny_other["status"] == "no_results"
    assert tiny_other["request"]["rerank"] == {"model": MODEL}
    assert tiny_other["next_actions"] == other["next_actions"]
    assert tiny_other["summary"]["budget"]["dropped"]["next_actions"] is False

    empty_conn = db.connect(tmp_path / "empty.db")
    db.init_schema(empty_conn)
    empty = daemon.run_search(
        empty_conn, None, None,
        {"query": "alpha", "mode": "lexical", "collections": [], "limit": 4,
         "_rerank_model": MODEL, "budget_tokens": 0})
    assert empty["status"] == "empty_index"
    assert empty["request"]["rerank"] == {"model": MODEL}
    assert [a["intent"] for a in empty["next_actions"]] == ["add_collection", "index"]
    empty_conn.close()


def test_guidance_and_explain_keep_generated_rerank_commands(conn, monkeypatch):
    ranked = _ranked(conn)
    monkeypatch.setattr(searchmod, "search_lexical",
                        lambda _conn, _q, _scope, limit: ranked[:limit])
    payload = daemon.run_search(
        conn, None, None,
        {"query": "alpha", "mode": "lexical", "collections": [], "limit": 4,
         "guidance": True, "explain": True, "_rerank_model": MODEL},
        reranker=Scores())
    assert payload["diagnostics"]["qpp"]
    assert all(command.count("--rerank=" + MODEL) == 1
               for command in _search_commands(payload))


def _indexed(tmp_path):
    path = tmp_path / "index.db"
    root = tmp_path / "notes"
    root.mkdir()
    (root / "one.md").write_text("# Search guide\nsearchable text here\n", encoding="utf-8")
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", root)
    indexer.sync_collection(conn, vector_store.NullStore(), "notes", indexer.IndexStats())
    row = conn.execute("SELECT id, body FROM documents WHERE relpath = 'one.md'").fetchone()
    conn.execute(
        "INSERT INTO chunks(doc_id, seq, pos, length, line_start, line_end) "
        "VALUES (?, 0, 0, ?, 1, 2)", (row["id"], len(row["body"])))
    conn.commit()
    conn.close()
    return path


def _cli(path, *args):
    return CliRunner().invoke(cli.main, ["--db", str(path), "search", "searchable",
                                         "--mode", "lexical", "--no-daemon", *args])


def test_env_present_explicit_off_preserves_flag_even_if_env_is_invalid(
        tmp_path, monkeypatch):
    path = _indexed(tmp_path)
    monkeypatch.setenv("FIDX_RERANK", "not-a-model")
    result = _cli(path, "--no-rerank", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "rerank" not in payload["request"]
    assert all(command.count("--no-rerank") == 1
               for command in _search_commands(payload))


def test_human_score_is_normalized_json_score_and_files_only_is_path(
        tmp_path, monkeypatch):
    path = _indexed(tmp_path)
    monkeypatch.delenv("FIDX_RERANK", raising=False)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: (Scores(), None))
    machine = _cli(path, "--rerank", "--json")
    human = _cli(path, "--rerank")
    files = _cli(path, "--rerank", "--files")
    assert machine.exit_code == human.exit_code == files.exit_code == 0
    row = json.loads(machine.stdout)["results"][0]
    assert row["sources"]["rerank"] == 0.0
    assert human.stdout.split()[0] == f"{row['score']:.4f}"
    assert files.stdout.strip() == row["path"]


def test_cli_budget_zero_preserves_selected_model_and_result_score(
        tmp_path, monkeypatch):
    path = _indexed(tmp_path)
    monkeypatch.delenv("FIDX_RERANK", raising=False)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: (Scores(), None))
    full = _cli(path, "--rerank", "--json")
    tiny = _cli(path, "--rerank", "--json", "--budget-tokens", "0")
    assert full.exit_code == tiny.exit_code == 0
    before, after = json.loads(full.stdout), json.loads(tiny.stdout)
    assert after["request"] == before["request"]
    assert after["results"][0]["score"] == before["results"][0]["score"]
    assert after["summary"]["budget"]["estimated_used"] == budget.estimate_tokens(after)
