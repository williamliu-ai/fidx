# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Existing-only index opening, instance identity, and document ambiguity."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from fidx import daemon, db, indexer
from fidx.cli import main


OPEN_COMMANDS = [
    ["search", "topic", "--mode", "lexical", "--no-daemon"],
    ["get", "notes/topic.md"],
    ["status"],
    ["serve"],
    ["calibrate", "--store"],
    ["collection", "list"],
    ["collection", "remove", "notes"],
]
INVALID_CASES = [
    ("missing_file", "index_missing"),
    ("missing_parent", "index_missing"),
    ("zero_byte", "index_invalid"),
    ("corrupt", "index_invalid"),
    ("foreign", "index_invalid"),
    ("missing_meta", "index_invalid"),
    ("missing_version", "index_invalid"),
    ("unsupported_version", "index_invalid"),
    ("missing_tables", "index_invalid"),
]


def _prepare_invalid(path: Path, case: str) -> None:
    if case == "missing_parent":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if case == "missing_file":
        return
    if case == "zero_byte":
        path.touch()
        return
    if case == "corrupt":
        path.write_bytes(b"not a SQLite database")
        return
    if case == "foreign":
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE unrelated(value TEXT)")
        conn.commit()
        conn.close()
        return

    conn = db.connect(path)
    db.init_schema(conn)
    if case == "missing_meta":
        conn.execute("DROP TABLE meta")
    elif case == "missing_version":
        conn.execute("DELETE FROM meta WHERE key = 'schema_version'")
    elif case == "unsupported_version":
        conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    elif case == "missing_tables":
        conn.execute("DROP TABLE chunks")
    else:
        raise AssertionError(case)
    conn.commit()
    conn.close()


def _schema(path: Path) -> list[tuple[str, str]] | None:
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        rows = conn.execute("SELECT type, name FROM sqlite_master ORDER BY type, name").fetchall()
        conn.close()
        return rows
    except sqlite3.DatabaseError:
        return None


@pytest.mark.parametrize("command", OPEN_COMMANDS, ids=lambda x: "-".join(x[:2]))
@pytest.mark.parametrize(("case", "status"), INVALID_CASES)
def test_open_existing_matrix_never_creates_or_changes_invalid_index(
    tmp_path, command, case, status
):
    path = tmp_path / "missing-parent" / "index.db" if case == "missing_parent" \
        else tmp_path / "index.db"
    _prepare_invalid(path, case)
    parent_existed = path.parent.exists()
    files_before = set(path.parent.iterdir()) if parent_existed else None
    bytes_before = path.read_bytes() if path.is_file() else None
    schema_before = _schema(path)

    result = CliRunner().invoke(main, ["--json", "--db", str(path), *command])

    assert result.exit_code == 3, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert payload["status"] == status
    assert payload["next_actions"] == [
        {"intent": "add_collection", "reason": "Register a source directory for a new index.",
         "command": ["fidx", "collection", "add", "<path>", "--name", "<collection>"]},
        {"intent": "index", "reason": "Build the index after registering a collection.",
         "command": ["fidx", "index"]},
    ]
    assert path.parent.exists() == parent_existed
    assert (set(path.parent.iterdir()) if parent_existed else None) == files_before
    assert (path.read_bytes() if path.is_file() else None) == bytes_before
    assert _schema(path) == schema_before


@pytest.mark.parametrize("command", [
    ["collection", "add"], ["index"],
])
def test_create_commands_initialize_missing_index(tmp_path, command):
    path = tmp_path / "new-parent" / "index.db"
    if command[0] == "collection":
        source = tmp_path / "notes"
        source.mkdir()
        command = [*command, str(source), "--name", "notes"]

    result = CliRunner().invoke(main, ["--db", str(path), *command])

    assert result.exit_code == (0 if command[0] == "collection" else 1), result.output
    assert path.is_file()
    conn = db.connect(path)
    assert db.get_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
    assert db.get_meta(conn, "instance_id")
    conn.close()


def test_ambiguous_hash_prefix_lists_both_paths_but_unique_hash_resolves(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    for collection, relpath, content_hash, body in [
        ("notes", "a.md", "abc123" + "0" * 58, "first duplicate"),
        ("other", "b.md", "abc123" + "0" * 58, "second duplicate"),
        ("notes", "c.md", "def456" + "1" * 58, "unique document"),
    ]:
        conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (collection, relpath, relpath, body, content_hash),
        )
    conn.commit()
    conn.close()

    ambiguous = CliRunner().invoke(main, ["--db", str(path), "get", "#abc", "--json"])
    assert ambiguous.exit_code == 2  # get has no command-level --json
    ambiguous = CliRunner().invoke(main, ["--json", "--db", str(path), "get", "#abc"])
    assert ambiguous.exit_code == 6, ambiguous.output
    payload = json.loads(ambiguous.stdout)
    assert payload["status"] == "ambiguous_reference"
    assert payload["candidates"] == [
        {"docid": "#abc123", "collection": "notes", "relpath": "a.md"},
        {"docid": "#abc123", "collection": "other", "relpath": "b.md"},
    ]
    for target in ("#def", "#" + "def456" + "1" * 58):
        unique = CliRunner().invoke(main, ["--db", str(path), "get", target])
        assert unique.exit_code == 0, unique.output
        assert unique.stdout.strip() == "unique document"
    by_path = CliRunner().invoke(main, ["--db", str(path), "get", "other/b.md"])
    assert by_path.exit_code == 0, by_path.output
    assert by_path.stdout.strip() == "second duplicate"


def test_ambiguous_hash_candidates_are_bounded(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    for n in range(12):
        conn.execute(
            "INSERT INTO documents(collection, relpath, title, body, hash) "
            "VALUES ('notes', ?, '', '', ?)",
            (f"{n:02}.md", f"abc{n:03x}" + "0" * 58),
        )
    conn.commit()
    conn.close()
    result = CliRunner().invoke(main, ["--json", "--db", str(path), "get", "#abc"])
    assert result.exit_code == 6, result.output
    assert [row["relpath"] for row in json.loads(result.stdout)["candidates"]] == [
        f"{n:02}.md" for n in range(10)
    ]


def _document_index(path: Path, relpath: str) -> None:
    conn = db.connect(path)
    db.init_schema(conn)
    conn.execute(
        "INSERT INTO documents(collection, relpath, title, body, hash) "
        "VALUES ('notes', ?, ?, 'shared searchable content', ?)",
        (relpath, relpath, relpath.replace(".", "") + "0" * 32),
    )
    conn.commit()
    conn.close()


@contextmanager
def _live_daemon(path: Path, tmp_path: Path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    env = os.environ.copy()
    source = """
import sys
from pathlib import Path
from fidx import daemon, db, vector_store
from fidx.config import EmbedProfile
from fidx.embedder import HashEmbedder
path = Path(sys.argv[1])
conn = db.connect(path)
store = vector_store.open_store(conn, path)
daemon.serve(path, conn, store, HashEmbedder(EmbedProfile('test', 'hash', 64)))
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", source, str(path)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    socket_path = daemon.socket_path_for(path)
    try:
        for _ in range(100):
            if proc.poll() is not None:
                raise AssertionError(f"daemon exited early: {proc.stderr.read()}")
            if socket_path.exists() and daemon.client_request(path, {"cmd": "ping"}):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("daemon did not start within 5 seconds")
        yield
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=3)
        socket_path.unlink(missing_ok=True)


@pytest.mark.skipif(os.name != "posix", reason="live daemon uses a Unix socket")
@pytest.mark.parametrize("replacement", ["deleted", "foreign", "valid"])
def test_live_daemon_cannot_answer_for_deleted_or_replaced_db(
    tmp_path, monkeypatch, replacement
):
    path = tmp_path / "index.db"
    _document_index(path, "old.md")
    with _live_daemon(path, tmp_path, monkeypatch):
        if replacement == "deleted":
            path.unlink()
        elif replacement == "foreign":
            foreign = tmp_path / "foreign.db"
            conn = sqlite3.connect(foreign)
            conn.execute("CREATE TABLE unrelated(value TEXT)")
            conn.close()
            os.replace(foreign, path)
        else:
            new = tmp_path / "new.db"
            _document_index(new, "new.md")
            os.replace(new, path)

        result = CliRunner().invoke(main, [
            "--db", str(path), "search", "shared", "--mode", "lexical", "--json",
        ])
        if replacement == "valid":
            assert result.exit_code == 0, result.output
            payload = json.loads(result.stdout)
            assert [row["path"] for row in payload["results"]] == ["notes/new.md"]
            assert payload["diagnostics"]["execution"]["daemon"] == "stale_instance"
            assert "restart" in payload["diagnostics"]["execution"]["message"].lower()
        else:
            assert result.exit_code == 3, result.output
            payload = json.loads(result.stdout)
            assert payload["status"] == (
                "index_missing" if replacement == "deleted" else "index_invalid"
            )


@pytest.mark.skipif(os.name != "posix", reason="live daemon uses a Unix socket")
def test_legacy_null_identity_uses_daemon_and_index_backfills_once(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    source = tmp_path / "notes"
    source.mkdir()
    indexer.add_collection(conn, "notes", source)
    conn.execute("DELETE FROM meta WHERE key = 'instance_id'")
    conn.commit()
    conn.close()

    with _live_daemon(path, tmp_path, monkeypatch):
        def forbidden_cold_search(*args, **kwargs):
            raise AssertionError("legacy null identity should accept the live daemon")

        monkeypatch.setattr(daemon, "run_search", forbidden_cold_search)
        result = CliRunner().invoke(main, [
            "--db", str(path), "search", "anything", "--mode", "lexical", "--json",
        ])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["status"] == "empty_index"

    first = CliRunner().invoke(main, ["--db", str(path), "index", "--no-calibrate"])
    assert first.exit_code == 0, first.output
    conn = db.connect(path)
    identity = db.get_meta(conn, "instance_id")
    assert identity
    conn.close()
    second = CliRunner().invoke(main, ["--db", str(path), "index", "--no-calibrate"])
    assert second.exit_code == 0, second.output
    conn = db.connect(path)
    assert db.get_meta(conn, "instance_id") == identity
    conn.close()


def test_v1_open_existing_does_not_migrate_until_index(tmp_path):
    path = tmp_path / "old.db"
    source = tmp_path / "notes"
    source.mkdir()
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
    conn.execute("INSERT INTO collections(name, root, globs) VALUES (?, ?, ?)",
                 ("notes", str(source), json.dumps(["**/*.md"])))
    conn.commit()
    conn.close()

    before = _schema(path)
    result = CliRunner().invoke(main, ["--db", str(path), "collection", "list"])
    assert result.exit_code == 0, result.output
    assert "notes" in result.stdout
    assert _schema(path) == before
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "1"
    assert conn.execute("SELECT value FROM meta WHERE key='instance_id'").fetchone() is None
    conn.close()

    indexed = CliRunner().invoke(main, ["--db", str(path), "index", "--no-calibrate"])
    assert indexed.exit_code == 0, indexed.output
    conn = db.connect(path)
    assert db.get_meta(conn, "schema_version") == "2"
    assert db.get_meta(conn, "instance_id")
    assert db.has_chunk_line_columns(conn)
    conn.close()
