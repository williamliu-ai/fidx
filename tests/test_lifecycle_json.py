# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Versioned lifecycle results and stdout/error separation through the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from fidx import cli, db, diagnostics, indexer
from fidx.embedder import HashEmbedder


def _index_fixture(tmp_path, monkeypatch, *, docs=("one.md",)):
    root = tmp_path / "notes"
    root.mkdir(parents=True)
    for name in docs:
        (root / name).write_text(f"# {name}\n\nA useful document.", encoding="utf-8")
    db_path = tmp_path / "index.db"
    conn = db.connect(db_path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", root)
    conn.close()
    monkeypatch.setattr(cli, "FastEmbedder", lambda profile, **kw: HashEmbedder(profile))
    return db_path, root


def _invoke(path, *args):
    return CliRunner().invoke(cli.main, ["--db", str(path), *args])


def _single_json(result):
    assert result.exit_code == 0, result.output
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(result.stdout)
    assert result.stdout[end:].strip() == ""
    return payload


def test_index_json_and_human_counts_match_with_calibration_disabled(tmp_path, monkeypatch):
    path, _ = _index_fixture(tmp_path / "machine", monkeypatch, docs=("one.md", "two.md"))
    result = _invoke(path, "index", "--json", "--no-calibrate")
    payload = _single_json(result)
    assert set(payload) == {"schema", "added", "updated", "removed", "embedded_chunks",
                            "seconds", "calibration", "warnings"}
    assert payload["schema"] == "fidx.index.v1"
    assert (payload["added"], payload["updated"], payload["removed"]) == (2, 0, 0)
    assert payload["embedded_chunks"] > 0
    assert isinstance(payload["seconds"], (int, float)) and payload["seconds"] >= 0
    assert payload["calibration"] == "skipped"
    assert payload["warnings"] == []
    assert "embedded 2/2 docs" in result.stderr
    assert "indexed in" not in result.stdout
    status = _single_json(_invoke(path, "status", "--json"))
    assert status["documents"] == 2
    assert status["chunks"] == status["embedded_chunks"] == payload["embedded_chunks"]

    human_path, _ = _index_fixture(tmp_path / "human", monkeypatch, docs=("one.md", "two.md"))
    human = _invoke(human_path, "index", "--no-calibrate")
    assert human.exit_code == 0, human.output
    assert "+2 added, ~0 updated, -0 removed" in human.stdout
    assert f"{payload['embedded_chunks']} chunks embedded" in human.stdout
    assert "calibration" not in human.stdout


@pytest.mark.parametrize(("calibration_result", "expected"), [
    ({"n_pos": 1, "floor": 0.5, "neg_rejected_at_floor": 0.7,
      "pos_retained_at_floor": 0.8}, "applied"),
    ({"n_pos": 0}, "deferred"),
])
def test_index_json_calibration_branches(tmp_path, monkeypatch, calibration_result, expected):
    path, _ = _index_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("fidx.calibrate.calibrate", lambda *a, **kw: calibration_result)
    result = _invoke(path, "index", "--json")
    assert _single_json(result)["calibration"] == expected
    assert "calibrated truncate_floor=" not in result.stdout
    assert "calibration deferred (" not in result.stdout


def test_index_json_skips_calibration_below_drift_threshold(tmp_path, monkeypatch):
    path, _ = _index_fixture(tmp_path, monkeypatch)
    conn = db.connect(path)
    db.set_meta(conn, "truncate_floor", "0.4")
    conn.close()
    result = _invoke(path, "index", "--json")
    assert _single_json(result)["calibration"] == "skipped"


def test_index_warning_is_stderr_and_mirrored_in_json(tmp_path, monkeypatch):
    path, root = _index_fixture(tmp_path, monkeypatch, docs=("good.md", "bad.md"))
    bad = root / "bad.md"
    original = Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == bad:
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    result = _invoke(path, "index", "--json", "--no-calibrate")
    payload = _single_json(result)
    assert payload["added"] == 1
    assert len(payload["warnings"]) == 1
    assert "notes/bad.md" in payload["warnings"][0]
    assert "denied" in payload["warnings"][0]
    assert result.stderr.count("warning: ") == 1
    assert payload["warnings"][0] in result.stderr


def test_index_warning_then_fatal_emits_only_error_document(tmp_path, monkeypatch):
    path, root = _index_fixture(tmp_path, monkeypatch, docs=("good.md", "bad.md"))
    bad = root / "bad.md"
    original = Path.read_text

    def unreadable(self, *args, **kwargs):
        if self == bad:
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)

    def fatal(*args, **kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr("fidx.calibrate.should_recalibrate", fatal)
    result = _invoke(path, "index", "--json")
    assert result.exit_code == 1
    assert result.stderr.count("warning: ") == 1
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert payload["message"] == "calibration failed"


def test_status_json_empty_and_populated_collections(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    for name in ("z", "a"):
        root = tmp_path / name
        root.mkdir()
        indexer.add_collection(conn, name, root)
    empty = _invoke(path, "status", "--json")
    payload = _single_json(empty)
    assert set(payload) == {"schema", "db_path", "profile", "backend", "collections",
                            "documents", "chunks", "embedded_chunks", "subtokens"}
    assert payload["schema"] == "fidx.status.v1"
    assert payload["db_path"] == str(path)
    assert isinstance(payload["profile"], str) and payload["profile"]
    assert payload["backend"] == "none"
    assert payload["subtokens"] is False
    assert payload["collections"] == [
        {"name": name, "root": str(tmp_path / name), "documents": 0}
        for name in ("a", "z")
    ]
    assert (payload["documents"], payload["chunks"], payload["embedded_chunks"]) == (0, 0, 0)
    assert empty.stderr == ""

    conn.execute("INSERT INTO documents(collection, relpath, title, body, hash) "
                 "VALUES ('z', 'one.md', 'One', 'body', 'abc123')")
    conn.commit()
    conn.close()
    populated = _single_json(_invoke(path, "status", "--json"))
    assert populated["documents"] == 1
    assert populated["collections"][0]["documents"] == 0
    assert populated["collections"][1]["documents"] == 1
    human = _invoke(path, "status")
    assert human.exit_code == 0
    assert "documents: 1  chunks: 0  vectors: 0" in human.stdout


@pytest.mark.parametrize(("checks", "exit_code", "ok"), [
    ([diagnostics.Check("model", False, "not cached", "fetch it")], 0, True),
    ([diagnostics.Check("fts5", False, "missing", "install sqlite", hard=True)], 1, False),
])
def test_doctor_json_hard_soft_and_remediation(monkeypatch, checks, exit_code, ok):
    monkeypatch.setattr(diagnostics, "capabilities", lambda: checks)
    for args in (("doctor", "--json"), ("--json", "doctor")):
        result = CliRunner().invoke(cli.main, list(args))
        assert result.exit_code == exit_code
        assert result.stderr == ""
        assert json.loads(result.stdout) == {
            "schema": "fidx.doctor.v1", "ok": ok,
            "checks": [check.__dict__ for check in checks],
        }


def test_doctor_probe_exception_keeps_versioned_doctor_report(monkeypatch):
    def broken():
        raise RuntimeError("diagnostic probe failed")

    monkeypatch.setattr(diagnostics, "capabilities", broken)
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 1
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["schema"] == "fidx.doctor.v1"
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "doctor"
    assert report["checks"][0]["hard"] is True
    assert report["checks"][0]["remediation"]


def test_changelog_declares_doctor_json_shape_change():
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert "**Breaking:** `doctor --json`" in changelog
    assert "`fidx.doctor.v1`" in changelog
