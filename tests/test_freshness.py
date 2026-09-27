# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""A bounded sample hints at index drift without changing search ranking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

import click
from click.testing import CliRunner

from fidx import cli, daemon, db, freshness, indexer, vector_store
from fidx.budget import apply_budget, estimate_tokens, unclassified_fields
from fidx.embedder import HashEmbedder


STAMP = "2026-09-27T10:00:00+00:00"
NOW = datetime(2026, 9, 27, 10, 0, 10, tzinfo=timezone.utc)


def _sample_index(tmp_path):
    db_path = tmp_path / "index.db"
    conn = db.connect(db_path)
    db.init_schema(conn)
    roots = {}
    paths = {}
    # Deliberately insert in reverse key order; SQL must choose the sample.
    keys = [("a", f"{n:02d}.md") for n in range(10)] + [
        ("b", f"{n:02d}.md") for n in range(11)]
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        indexer.add_collection(conn, name, root)
        roots[name] = root
    for collection, relpath in reversed(keys):
        path = roots[collection] / relpath
        path.write_text("common search text")
        paths[(collection, relpath)] = path
        conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash, mtime) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (collection, relpath, relpath, "common search text", "hash",
             path.stat().st_mtime),
        )
    conn.commit()
    db.set_meta(conn, "last_index_completed_at", STAMP)
    return db_path, conn, paths


def test_sample_is_first_twenty_globally_sorted_and_unsampled_change_is_not_claimed(
    tmp_path, monkeypatch
):
    _, conn, paths = _sample_index(tmp_path)
    monkeypatch.setattr(freshness, "_utc_now", lambda: NOW)
    original_stat = Path.stat
    seen = []

    def recording_stat(path, *args, **kwargs):
        if path in paths.values():
            seen.append(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", recording_stat)
    assert freshness.index_summary(conn) == {"age_seconds": 10, "stale_hint": False}
    assert seen == [paths[key] for key in sorted(paths)[:20]]
    assert paths[("b", "10.md")] not in seen

    future_mtime = max(path.stat().st_mtime for path in paths.values()) + 1000
    os.utime(paths[("b", "10.md")], (future_mtime, future_mtime))
    assert freshness.index_summary(conn)["stale_hint"] is False
    os.utime(paths[("a", "00.md")], (future_mtime, future_mtime))
    assert freshness.index_summary(conn)["stale_hint"] is True
    conn.close()


def test_missing_sampled_file_is_stale_but_other_stat_error_omits_hint(
    tmp_path, monkeypatch
):
    _, conn, paths = _sample_index(tmp_path)
    monkeypatch.setattr(freshness, "_utc_now", lambda: NOW)
    sampled = paths[("a", "00.md")]
    sampled.unlink()
    assert freshness.index_summary(conn) == {"age_seconds": 10, "stale_hint": True}
    sampled.write_text("common search text")
    original_stat = Path.stat

    def denied_stat(path, *args, **kwargs):
        if path == sampled:
            raise PermissionError("denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)
    assert freshness.index_summary(conn) == {"age_seconds": 10}
    conn.close()


def test_legacy_meta_absence_omits_whole_object_and_budget_counter(tmp_path, monkeypatch):
    _, conn, _ = _sample_index(tmp_path)
    conn.execute("DELETE FROM meta WHERE key = 'last_index_completed_at'")
    conn.commit()
    monkeypatch.setattr(freshness, "_utc_now", lambda: NOW)
    assert freshness.index_summary(conn) is None
    envelope = daemon.run_search(conn, None, None, {
        "query": "common", "mode": "lexical", "limit": 3,
    })
    assert "index" not in envelope["summary"]
    assert unclassified_fields(envelope) == []
    assert "index" not in apply_budget(envelope, 0)["summary"]["budget"]["dropped"]
    conn.close()


def test_v1_index_without_completion_key_remains_readable_and_unchanged(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema_version', '1');
        CREATE TABLE collections(name TEXT PRIMARY KEY, root TEXT NOT NULL,
            globs TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE documents(id INTEGER PRIMARY KEY, collection TEXT NOT NULL,
            relpath TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,
            hash TEXT NOT NULL, mtime REAL NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1, UNIQUE(collection, relpath));
        CREATE VIRTUAL TABLE documents_fts USING fts5(title, body, relpath,
            content='documents', content_rowid='id', tokenize='porter unicode61');
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
            seq INTEGER NOT NULL, pos INTEGER NOT NULL, length INTEGER NOT NULL,
            UNIQUE(doc_id, seq));
    """)
    conn.close()
    original = path.read_bytes()
    result = CliRunner().invoke(cli.main, [
        "--db", str(path), "search", "absent", "--mode", "lexical",
        "--json", "--no-daemon",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "empty_index"
    assert "index" not in json.loads(result.output)["summary"]
    assert path.read_bytes() == original


def test_real_envelope_index_is_additive_and_budget_step_four_drops_it(
    tmp_path, monkeypatch
):
    _, conn, _ = _sample_index(tmp_path)
    monkeypatch.setattr(freshness, "_utc_now", lambda: NOW)
    response = daemon.run_search(conn, None, None, {
        "query": "common", "mode": "lexical", "limit": 3,
    })
    assert response["summary"]["index"] == {"age_seconds": 10, "stale_hint": False}
    assert unclassified_fields(response) == []
    maximum = estimate_tokens(apply_budget(response, 100_000))
    stage = next(apply_budget(response, amount) for amount in range(maximum + 1)
                 if apply_budget(response, amount)["summary"]["budget"]["dropped"]
                 ["index"] is True
                 and apply_budget(response, amount)["summary"]["budget"]["dropped"]
                 ["sources"] == 0)
    assert "index" not in stage["summary"]
    assert stage["results"][0]["path"] == response["results"][0]["path"]
    assert stage["results"][0]["score"] == response["results"][0]["score"]
    conn.close()


def test_cli_index_completion_warning_zero_change_and_abort(tmp_path, monkeypatch):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# A\nCommon text")
    db_path = tmp_path / "index.db"
    conn = db.connect(db_path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", root)
    conn.close()
    monkeypatch.setattr(cli, "FastEmbedder", lambda profile, **kwargs: HashEmbedder(profile))
    stamps = iter([
        "2026-09-27T10:00:00+00:00",
        "2026-09-27T10:01:00+00:00",
        "2026-09-27T10:02:00+00:00",
    ])
    monkeypatch.setattr(freshness, "completed_at", lambda: next(stamps))

    def run():
        return CliRunner().invoke(cli.main, [
            "--db", str(db_path), "index", "--no-calibrate",
        ])

    first = run()
    assert first.exit_code == 0, first.output
    conn = db.connect(db_path, create=False)
    assert db.get_meta(conn, "last_index_completed_at") == "2026-09-27T10:00:00+00:00"
    second = run()
    assert second.exit_code == 0, second.output
    assert "0 chunks embedded" in second.output
    assert db.get_meta(conn, "last_index_completed_at") == "2026-09-27T10:01:00+00:00"

    (root / "warning.md").write_text("warning fixture")
    original_read_text = Path.read_text

    def warning_read_text(path, *args, **kwargs):
        if path == root / "warning.md":
            raise OSError("injected read failure")
        return original_read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", warning_read_text)
        third = run()
    assert third.exit_code == 0, third.output
    assert "warning:" in third.output
    assert db.get_meta(conn, "last_index_completed_at") == "2026-09-27T10:02:00+00:00"

    def abort(*args, **kwargs):
        raise click.Abort()

    with monkeypatch.context() as patch:
        patch.setattr(indexer, "embed_pending", abort)
        aborted = run()
    assert aborted.exit_code != 0
    assert db.get_meta(conn, "last_index_completed_at") == "2026-09-27T10:02:00+00:00"
    conn.close()


def test_cold_and_warm_cli_match_direct_envelope(tmp_path, monkeypatch):
    db_path, conn, _ = _sample_index(tmp_path)
    monkeypatch.setattr(freshness, "_utc_now", lambda: NOW)
    raw = daemon.run_search(conn, None, None, {
        "query": "common", "mode": "lexical", "limit": 3,
    })
    assert raw["summary"]["index"] == {"age_seconds": 10, "stale_hint": False}
    args = ["--db", str(db_path), "search", "common", "--mode", "lexical",
            "-n", "3", "--json"]
    cold = CliRunner().invoke(cli.main, [*args, "--no-daemon"])
    assert cold.exit_code == 0, cold.output
    assert json.loads(cold.output) == raw
    instance_id = db.get_meta(conn, "instance_id")
    monkeypatch.setattr(daemon, "client_request", lambda *a, **kw: {
        "ok": True, "instance_id": instance_id, "results": raw,
    })
    warm = CliRunner().invoke(cli.main, args)
    assert warm.exit_code == 0, warm.output
    assert json.loads(warm.output) == raw
    conn.close()
