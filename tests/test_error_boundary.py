# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""CLI execution errors, Click usage errors, and doctor report precedence."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest
from click.testing import CliRunner

from fidx import cli, db, diagnostics, indexer


@pytest.fixture()
def index_db(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    return path


@pytest.mark.parametrize("args", [
    ["search", "--json"],
    ["search", "topic", "--json", "--limit", "not-a-number"],
    ["--json", "search"],
])
def test_click_usage_errors_keep_exit_two_and_no_json(args):
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr
    assert "fidx.error.v1" not in result.stderr


@pytest.mark.parametrize(("command", "message_part"), [
    (["index", "--profile", "nomic-768"], "pinned to profile"),
    (["index"], "no collections registered"),
    (["get", "notes/absent.md"], "not found"),
])
def test_real_typed_failures_use_error_envelope(index_db, command, message_part):
    if "--profile" in command:
        conn = db.connect(index_db)
        db.set_meta(conn, "embed_profile", "bge-384")
        conn.close()
    result = CliRunner().invoke(
        cli.main, ["--json", "--db", str(index_db), *command]
    )
    assert result.exit_code == 1, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert payload["status"] == "error"
    assert message_part in payload["message"]
    assert isinstance(payload["next_actions"], list)


def test_unexpected_failure_is_json_and_human_one_liner(index_db, monkeypatch):
    def broken_open_index(*args, **kwargs):
        raise RuntimeError("injected execution failure\nsecond line")

    monkeypatch.setattr(cli, "open_index", broken_open_index)
    args = ["--db", str(index_db), "get", "notes/absent.md"]
    machine = CliRunner().invoke(cli.main, ["--json", *args])
    assert machine.exit_code == 1
    assert machine.stderr == ""
    payload = json.loads(machine.stdout)
    assert payload == {
        "schema": "fidx.error.v1",
        "status": "error",
        "message": "injected execution failure second line",
        "next_actions": [],
    }

    human = CliRunner().invoke(cli.main, args)
    assert human.exit_code == 1
    assert human.stdout == ""
    assert human.stderr == "Error: injected execution failure second line\n"
    assert "Traceback" not in human.stderr


def test_json_error_discards_prior_command_stdout(index_db, tmp_path, monkeypatch):
    root = tmp_path / "notes"
    root.mkdir()
    conn = db.connect(index_db)
    indexer.add_collection(conn, "notes", root)
    conn.close()

    def fail_after_index_output(*args, **kwargs):
        raise RuntimeError("calibration failed after progress output")

    monkeypatch.setattr("fidx.calibrate.should_recalibrate", fail_after_index_output)
    result = CliRunner().invoke(
        cli.main, ["--json", "--db", str(index_db), "index"]
    )
    assert result.exit_code == 1, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["message"] == "calibration failed after progress output"


def test_locked_database_maps_to_distinct_exit(index_db, monkeypatch):
    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cli, "open_index", locked)
    result = CliRunner().invoke(
        cli.main, ["--json", "--db", str(index_db), "get", "notes/absent.md"]
    )
    assert result.exit_code == 4
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "db_locked"


def test_reserved_index_and_ambiguity_statuses_have_pinned_codes(index_db, monkeypatch):
    from fidx.errors import AmbiguousReference, IndexInvalid, IndexMissing

    for failure, status, code in [
        (IndexMissing("missing index"), "index_missing", 3),
        (IndexInvalid("invalid index"), "index_invalid", 3),
        (AmbiguousReference("multiple matches", candidates=[
            {"docid": "#abc", "collection": "notes", "relpath": "a.md"},
            {"docid": "#abd", "collection": "notes", "relpath": "b.md"},
        ]),
         "ambiguous_reference", 6),
    ]:
        def fail(*args, **kwargs):
            raise failure

        monkeypatch.setattr(cli, "open_index", fail)
        result = CliRunner().invoke(
            cli.main, ["--json", "--db", str(index_db), "get", "#ab"]
        )
        assert result.exit_code == code
        payload = json.loads(result.stdout)
        assert payload["status"] == status
        assert ("candidates" in payload) == (status == "ambiguous_reference")


def test_cold_model_gets_status_five_and_fetch_action(index_db, tmp_path, monkeypatch):
    conn = db.connect(index_db)
    db.set_meta(conn, "embed_profile", "bge-384")
    conn.execute(
        "INSERT INTO documents(collection, relpath, title, body, hash) "
        "VALUES ('notes', 'topic.md', 'Topic', 'searchable topic content', 'abc123')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "cold-models"))

    result = CliRunner().invoke(
        cli.main, ["--db", str(index_db), "search", "topic", "--json", "--no-daemon"]
    )
    assert result.exit_code == 5, result.output
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "model_missing"
    assert payload["next_actions"] == [{
        "intent": "fetch_model",
        "reason": "Download the selected embedding model before searching offline.",
        "command": ["fidx", "models", "fetch", "embed", "--profile", "bge-384"],
    }]


def test_doctor_json_hard_failure_keeps_own_report(monkeypatch):
    monkeypatch.setattr(
        diagnostics, "capabilities",
        lambda: [diagnostics.Check("fts5", False, "unavailable", hard=True)],
    )
    for args in (["doctor", "--json"], ["--json", "doctor"]):
        result = CliRunner().invoke(cli.main, args)
        assert result.exit_code == 1
        assert result.stderr == ""
        assert json.loads(result.stdout) == {
            "schema": "fidx.doctor.v1", "ok": False,
            "checks": [{
                "name": "fts5", "ok": False, "detail": "unavailable",
                "remediation": "", "hard": True,
            }],
        }


def test_doctor_unexpected_failure_still_uses_doctor_report(monkeypatch):
    def broken_capabilities():
        raise RuntimeError("diagnostic probe failed")

    monkeypatch.setattr(diagnostics, "capabilities", broken_capabilities)
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 1
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["schema"] == "fidx.doctor.v1"
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "doctor"
    assert report["checks"][0]["hard"] is True
    assert "fidx.error.v1" not in result.stdout


def test_module_entrypoint_debug_restores_traceback(index_db):
    conn = db.connect(index_db)
    conn.execute(
        "INSERT INTO documents(collection, relpath, title, body, hash) "
        "VALUES ('notes', 'topic.md', 'Topic', 'searchable topic content', 'abc123')"
    )
    conn.commit()
    conn.close()
    env = os.environ.copy()
    env["FIDX_THREADS"] = "not-an-integer"
    command = [sys.executable, "-m", "fidx.cli", "--db", str(index_db),
               "search", "topic", "--json", "--no-daemon"]

    safe = subprocess.run(command, env=env, capture_output=True, text=True)
    assert safe.returncode == 1
    assert safe.stderr == ""
    assert json.loads(safe.stdout)["status"] == "error"

    env["FIDX_DEBUG"] = "1"
    debug = subprocess.run(command, env=env, capture_output=True, text=True)
    assert debug.returncode == 1
    assert debug.stdout == ""
    assert "Traceback (most recent call last)" in debug.stderr
    assert "not-an-integer" in debug.stderr
