# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Code-profile contracts, using fake models only."""

import json
import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from fidx import cli, config, db, indexer
from fidx.embedder import HashEmbedder
from fidx.vector_store import SqliteVecStore

PROFILE = "jina-code-768"
MODEL = "jinaai/jina-embeddings-v2-base-code"


def test_profile_contract():
    profile = config.get_profile(PROFILE)
    assert (profile.name, profile.model, profile.dim,
            profile.query_prefix, profile.doc_prefix,
            profile.hf_source, profile.model_file) == (
                PROFILE, MODEL, 768, "", "", None, None)


def test_profile_fingerprint_is_distinct():
    fingerprint = config.profile_fingerprint(config.get_profile(PROFILE))
    assert all(fingerprint != config.profile_fingerprint(other)
               for name, other in config.PROFILES.items() if name != PROFILE)


def _build_index(path, root, profile_name):
    profile = config.get_profile(profile_name)
    conn = db.connect(path)
    db.init_schema(conn)
    store = SqliteVecStore(conn)
    db.ensure_vectors(conn, store, profile.dim, config.profile_fingerprint(profile),
                      profile.model, profile.name)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", root)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, HashEmbedder(profile), stats)
    return conn, store


@pytest.mark.parametrize("stored,requested", [
    ("nomic-768-q", PROFILE), (PROFILE, "nomic-768-q"),
])
def test_profile_mismatch_is_hard_error(tmp_path, corpus_dir, stored, requested):
    path = tmp_path / "index.db"
    conn, store = _build_index(path, corpus_dir, stored)
    try:
        before = dict(conn.execute("SELECT key, value FROM meta"))
        profile = config.get_profile(requested)
        with pytest.raises(RuntimeError, match="pinned to embedding profile"):
            db.ensure_vectors(conn, store, profile.dim, config.profile_fingerprint(profile),
                              profile.model, profile.name)
        assert dict(conn.execute("SELECT key, value FROM meta")) == before
        result = CliRunner().invoke(cli.main, [
            "--db", str(path), "index", "--profile", requested, "--no-calibrate"])
        assert result.exit_code == 1, result.output
        assert f"pinned to profile {stored!r}" in result.output
        assert f"re-index into a fresh --db to use {requested!r}" in result.output
        assert db.get_meta(conn, "embed_fingerprint") == config.profile_fingerprint(
            config.get_profile(stored))
    finally:
        conn.close()


@pytest.mark.parametrize("selection", ["flag", "env"])
def test_index_accepts_profile(tmp_path, corpus_dir, selection):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", corpus_dir)
    conn.close()
    env = dict(os.environ, FIDX_PROFILE=PROFILE if selection == "env" else "nomic-768-q")
    # Import config in a fresh process to exercise FIDX_PROFILE at startup.
    script = ("from fidx import cli; from fidx.embedder import HashEmbedder; "
              "cli.FastEmbedder = lambda profile, **kwargs: HashEmbedder(profile); "
              "cli.main()")
    args = [sys.executable, "-c", script, "--db", str(path), "index", "--no-calibrate"]
    if selection == "flag":
        args += ["--profile", PROFILE]
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    conn = db.connect(path)
    try:
        assert db.get_meta(conn, "embed_profile") == PROFILE
        assert db.get_meta(conn, "embed_model") == MODEL
        assert db.get_meta(conn, "embed_dim") == "768"
        assert db.get_meta(conn, "embed_fingerprint") == config.profile_fingerprint(
            config.get_profile(PROFILE))
        assert SqliteVecStore(conn).count() > 0
    finally:
        conn.close()


def test_cold_search_never_downloads(tmp_path, corpus_dir, monkeypatch):
    path = tmp_path / "index.db"
    conn, _ = _build_index(path, corpus_dir, PROFILE)
    conn.close()
    cache = tmp_path / "empty-cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    calls = []

    def missing_model(*, model_name, local_files_only, **kwargs):
        calls.append((model_name, local_files_only))
        assert local_files_only, "search allowed model downloads"
        raise ValueError("model absent from cold cache")

    monkeypatch.setattr("fastembed.TextEmbedding", missing_model)
    result = CliRunner().invoke(cli.main, [
        "--db", str(path), "search", "OAuth", "--json", "--no-daemon"])
    assert calls == [(MODEL, True)]
    assert result.exit_code == 5, result.output
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert payload["status"] == "model_missing"
    assert f"fidx models fetch embed --profile {PROFILE}" in payload["message"]
    assert any(action.get("command") == [
        "fidx", "models", "fetch", "embed", "--profile", PROFILE]
        for action in payload["next_actions"])
    assert not cache.exists()


def test_fetch_routes_to_code_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "cache"))
    calls = []

    def fake_model(*, model_name, local_files_only, lazy_load, **kwargs):
        calls.append((model_name, local_files_only, lazy_load))

    monkeypatch.setattr("fastembed.TextEmbedding", fake_model)
    result = CliRunner().invoke(cli.main, [
        "models", "fetch", "embed", "--profile", PROFILE])
    assert result.exit_code == 0, result.output
    assert calls == [(MODEL, False, True)]
    assert MODEL in result.output
    assert PROFILE in result.output


def test_doctor_reports_selected_code_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_PROFILE", PROFILE)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "empty-cache"))
    calls = []

    def missing_model(*, model_name, local_files_only, lazy_load, **kwargs):
        calls.append((model_name, local_files_only, lazy_load))
        assert local_files_only, "doctor allowed model downloads"
        raise ValueError("model absent from cold cache")

    monkeypatch.setattr("fastembed.TextEmbedding", missing_model)
    result = CliRunner().invoke(cli.main, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    rows = {row["name"]: row for row in json.loads(result.stdout)["checks"]}
    row = rows["embedding_model"]
    assert calls == [(MODEL, True, True)]
    assert row["ok"] is False
    assert row["detail"] == f"{PROFILE} ({MODEL}) unavailable locally"
    assert f"fidx models fetch embed --profile {PROFILE}" in row["remediation"]
