# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Explicit model acquisition and offline query-path regressions."""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest
from click.testing import CliRunner

from fidx import cli, db
from fidx.config import get_profile


class _OrtNoSuchFile(Exception):
    """Model-load failure that does not inherit from OSError."""


def _partial_cache_fastembed(monkeypatch, tmp_path):
    snap = tmp_path / "cache" / "snap"
    snap.mkdir(parents=True)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "cache"))

    class Desc:
        model_file = "model_optimized.onnx"

    class Inner:
        _model_dir = snap
        model_description = Desc()
        tokenizer = None

    class FakeTextEmbedding:
        def __init__(self, *, local_files_only, lazy_load=False, **kwargs):
            assert local_files_only
            if not lazy_load:
                raise _OrtNoSuchFile("[ONNXRuntimeError] : 3 : NO_SUCHFILE")
            self.model = Inner()

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)


@pytest.fixture()
def cold_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    attempts = []

    def forbidden_connect(self, address):
        attempts.append(address)
        raise AssertionError(f"network attempted: {address}")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    return cache, attempts


@pytest.fixture()
def populated_db(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    db.set_meta(conn, "embed_profile", "bge-384")
    conn.execute(
        "INSERT INTO documents(collection, relpath, title, body, hash) "
        "VALUES (?, ?, ?, ?, ?)",
        ("notes", "topic.md", "Topic",
         "A topic with several searchable words and detailed context for retrieval.",
         "abc123"),
    )
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize("args", [
    ["search", "topic", "--json", "--no-daemon"],
    ["serve"],
    ["calibrate"],
])
def test_cold_query_commands_fail_closed_with_fetch_hint(
    cold_cache, populated_db, args
):
    cache, attempts = cold_cache
    result = CliRunner().invoke(cli.main, ["--db", str(populated_db), *args])
    assert result.exit_code == 5, result.output
    message = (
        "embedding model 'BAAI/bge-small-en-v1.5' for profile "
        "'bge-384' is unavailable locally; run "
        "`fidx models fetch embed --profile bge-384`"
    )
    if "--json" in args:
        payload = json.loads(result.stdout)
        assert payload["status"] == "model_missing"
        assert payload["message"] == message
    else:
        assert result.stderr.strip().splitlines()[-1] == f"Error: {message}"
    assert not attempts
    assert not any(cache.rglob("*.onnx"))


def test_empty_index_search_skips_embedder_construction(cold_cache, tmp_path, monkeypatch):
    path = tmp_path / "empty.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()

    def forbidden_embedder(*args, **kwargs):
        raise AssertionError("empty search constructed an embedder")

    monkeypatch.setattr(cli, "FastEmbedder", forbidden_embedder)
    result = CliRunner().invoke(
        cli.main, ["--db", str(path), "search", "topic", "--json", "--no-daemon"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "empty_index"


def test_lexical_search_never_constructs_embedder(populated_db, monkeypatch):
    def forbidden_embedder(*args, **kwargs):
        raise AssertionError("lexical search constructed an embedder")

    monkeypatch.setattr(cli, "FastEmbedder", forbidden_embedder)
    result = CliRunner().invoke(
        cli.main, ["--db", str(populated_db), "search", "topic", "--mode", "lexical",
                   "--json", "--no-daemon"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "ok"


def test_fetch_records_cache_path_and_warm_search_uses_local_files_only(
    tmp_path, populated_db, monkeypatch
):
    cache = tmp_path / "model-cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    calls = []

    class FakeModel:
        tokenizer = None

    class FakeTextEmbedding:
        def __init__(self, *, model_name, cache_dir, local_files_only, **kwargs):
            calls.append((model_name, cache_dir, local_files_only))
            if local_files_only and not (cache / "ready").exists():
                raise ValueError("model absent")
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "ready").touch()
            self.model = FakeModel()

        def embed(self, texts, **kwargs):
            return iter([np.ones(384, dtype=np.float32) for _ in texts])

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)
    fetched = CliRunner().invoke(
        cli.main, ["models", "fetch", "embed", "--profile", "bge-384"]
    )
    assert fetched.exit_code == 0, fetched.output
    assert str(cache) in fetched.output
    assert calls == [(get_profile("bge-384").model, str(cache), False)]

    result = CliRunner().invoke(
        cli.main, ["--db", str(populated_db), "search", "topic", "--json", "--no-daemon"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "ok"
    assert calls[-1] == (get_profile("bge-384").model, str(cache), True)


def test_rerank_fetch_is_dark_and_does_not_load_model(monkeypatch):
    def forbidden_embedder(*args, **kwargs):
        raise AssertionError("rerank fetch loaded an embedding model")

    monkeypatch.setattr(cli, "FastEmbedder", forbidden_embedder)
    result = CliRunner().invoke(cli.main, ["models", "fetch", "rerank"])
    assert result.exit_code != 0
    assert "rerank" in result.output.lower()


def test_doctor_json_reports_embedding_model_fetch_hint(cold_cache):
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["name"]: row for row in json.loads(result.output)}
    assert "embedding_model" in rows
    assert rows["embedding_model"]["ok"] is False
    assert "fidx models fetch embed" in rows["embedding_model"]["remediation"]
    assert not cold_cache[1]


def test_doctor_reports_invalid_profile_without_crashing(monkeypatch):
    monkeypatch.setattr("fidx.config.DEFAULT_PROFILE", "unknown-profile")
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["name"]: row for row in json.loads(result.output)}
    assert rows["embedding_model"]["ok"] is False
    assert "unknown-profile" in rows["embedding_model"]["detail"]


@pytest.mark.parametrize("args", [
    ["search", "topic", "--json", "--no-daemon"],
    ["serve"],
    ["calibrate"],
])
def test_partial_cache_fails_closed_with_fetch_hint(
    tmp_path, populated_db, monkeypatch, args
):
    _partial_cache_fastembed(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli.main, ["--db", str(populated_db), *args])
    assert result.exit_code == 5
    assert not isinstance(result.exception, _OrtNoSuchFile)
    assert "`fidx models fetch embed --profile bge-384`" in result.output


def test_doctor_flags_partial_cache_unavailable(tmp_path, monkeypatch):
    _partial_cache_fastembed(monkeypatch, tmp_path)
    monkeypatch.setattr("fidx.config.DEFAULT_PROFILE", "bge-384")
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    rows = {r["name"]: r for r in json.loads(result.output)}
    assert rows["embedding_model"]["ok"] is False
    assert "fidx models fetch embed --profile bge-384" in rows["embedding_model"]["remediation"]


def test_real_fastembed_partial_cache_fails_closed(cold_cache, populated_db, monkeypatch):
    cache, attempts = cold_cache
    partial = cache / "bge-small-en-v1.5"
    partial.mkdir(parents=True)
    (partial / "incomplete.txt").write_text("interrupted model acquisition")

    result = CliRunner().invoke(
        cli.main, ["--db", str(populated_db), "search", "topic", "--no-daemon"]
    )
    assert result.exit_code == 5, result.output
    assert "`fidx models fetch embed --profile bge-384`" in result.output
    assert not attempts

    monkeypatch.setattr("fidx.config.DEFAULT_PROFILE", "bge-384")
    report = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert report.exit_code == 0, report.output
    rows = {r["name"]: r for r in json.loads(report.output)}
    assert rows["embedding_model"]["ok"] is False
