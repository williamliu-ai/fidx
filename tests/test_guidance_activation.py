# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Internal engine/daemon activation, before CLI flags become public."""

from __future__ import annotations

import json
import os
import queue
import threading
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

from fidx import daemon, db, guidance, indexer, vector_store
from fidx.cli import main
from fidx.config import EmbedProfile
from fidx.embedder import HashEmbedder


@pytest.fixture()
def scoped_index(tmp_path):
    conn = db.connect(tmp_path / "index.db")
    db.init_schema(conn)
    for name, body in (("alpha", "running quickly"),
                       ("beta", "deployment quickly")):
        root = tmp_path / name
        root.mkdir()
        (root / "a.md").write_text(body)
        indexer.add_collection(conn, name, root)
        indexer.sync_collection(conn, vector_store.NullStore(), name, indexer.IndexStats())
    yield conn
    conn.close()


def _request():
    return {"cmd": "search", "query": "running deployment", "mode": "lexical",
            "collections": ["alpha"], "limit": 10, "min_score": None, "truncate": None}


@pytest.mark.parametrize("guidance_flag,explain", [
    ("absent", False), ("absent", True), (False, False),
    (False, True), (True, False), (True, True),
])
def test_internal_six_state_guidance_explain_matrix(scoped_index, guidance_flag, explain):
    req = _request()
    if guidance_flag != "absent":
        req["guidance"] = guidance_flag
    if explain:
        req["explain"] = True
    result = daemon.run_search(scoped_index, vector_store.NullStore(), None, req)
    assert result["status"] == "ok"
    assert result["request"].get("guidance", "absent") == guidance_flag
    active = guidance_flag is True
    assert ("confidence_reasons" in result["summary"]) is active
    assert ("count_relation" in result["summary"]) is active
    assert ("term_coverage" in result["diagnostics"]) is active
    assert ("execution" in result["diagnostics"]) is active
    assert ("qpp" in result["diagnostics"]) is explain
    if active:
        assert result["diagnostics"]["term_coverage"]["unmatched"] == ["deployment"]
        assert "low_term_coverage" in result["summary"]["confidence_reasons"]
        assert result["summary"]["count_relation"] == "gte"
        assert result["diagnostics"]["execution"] == {
            "complete": True, "partial_reason": None}
    if explain:
        assert result["diagnostics"]["qpp"]["term_coverage"] == 0.5


def test_absent_default_off_does_not_compute_or_change_payload(scoped_index, monkeypatch):
    req = _request()
    baseline = daemon.run_search(scoped_index, vector_store.NullStore(), None, req)
    monkeypatch.setattr(guidance, "term_coverage", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("default-off guidance unexpectedly computed")))
    observed = daemon.run_search(scoped_index, vector_store.NullStore(), None, req)
    assert json.dumps(observed, sort_keys=True) == json.dumps(baseline, sort_keys=True)
    assert "guidance" not in observed["request"]


def test_internal_default_on_echo_and_explicit_off_override(scoped_index, monkeypatch):
    monkeypatch.setattr(daemon, "GUIDANCE_DEFAULT", True)
    automatic = daemon.run_search(scoped_index, vector_store.NullStore(), None, _request())
    assert automatic["request"]["guidance"] is True
    assert "term_coverage" in automatic["diagnostics"]
    disabled = daemon.run_search(scoped_index, vector_store.NullStore(), None,
                                 dict(_request(), guidance=False))
    assert disabled["request"]["guidance"] is False
    assert "term_coverage" not in disabled["diagnostics"]


def test_scoped_coverage_and_guidance_are_attached_before_budget(scoped_index):
    req = dict(_request(), guidance=True, budget_tokens=0)
    budgeted = daemon.run_search(scoped_index, vector_store.NullStore(), None, req)
    dropped = budgeted["summary"]["budget"]["dropped"]
    assert dropped["term_coverage"] is True
    assert dropped["confidence_reasons"] is True
    assert dropped["count_relation"] is True
    absent = daemon.run_search(scoped_index, vector_store.NullStore(), None,
                               dict(_request(), query="unmatched", guidance=True,
                                    budget_tokens=0))
    assert absent["status"] == "no_results"
    assert absent["diagnostics"]["no_results"]["causes"] == ["vocabulary_mismatch"]


def test_explicit_internal_flags_require_boolean_values(scoped_index):
    with pytest.raises(ValueError):
        daemon.run_search(scoped_index, vector_store.NullStore(), None,
                          dict(_request(), guidance="true"))
    with pytest.raises(ValueError):
        daemon.run_search(scoped_index, vector_store.NullStore(), None,
                          dict(_request(), explain=1))


@contextmanager
def _live_server(path, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    ready = queue.Queue()
    failures = []
    original_bind = daemon._bind_private_unix_server

    def capture_server(sock_path, handler):
        server = original_bind(sock_path, handler)
        ready.put(server)
        return server

    monkeypatch.setattr(daemon, "_bind_private_unix_server", capture_server)

    def run():
        try:
            conn = db.connect(path, create=False)
            try:
                daemon.serve(path, conn, vector_store.NullStore(),
                             HashEmbedder(EmbedProfile("guidance-test", "hash", 64)))
            finally:
                conn.close()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        server = ready.get(timeout=5)
        yield
    finally:
        if "server" in locals():
            server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not failures, failures


@pytest.mark.skipif(os.name != "posix", reason="production daemon uses Unix sockets")
def test_real_daemon_ack_only_after_protocol_one_guidance_or_explain(
    scoped_index, tmp_path, monkeypatch,
):
    path = tmp_path / "index.db"
    with _live_server(path, tmp_path, monkeypatch):
        legacy = daemon.client_request(path, _request())
        assert legacy["ok"] is True
        assert "guidance_protocol" not in legacy
        assert "guidance" not in legacy["results"]["request"]

        guided = daemon.client_request_outcome(
            path, dict(_request(), guidance=True),
            expected_instance_id=legacy["instance_id"], require_guidance_protocol=True)
        assert guided.path == "daemon"
        assert guided.response["guidance_protocol"] == 1
        assert guided.response["results"]["request"]["guidance"] is True
        assert guided.response["results"]["diagnostics"]["term_coverage"][
            "unmatched"] == ["deployment"]

        disabled = daemon.client_request(path, dict(_request(), guidance=False,
                                                     guidance_protocol=1))
        assert disabled["guidance_protocol"] == 1
        assert disabled["results"]["request"]["guidance"] is False
        assert "term_coverage" not in disabled["results"]["diagnostics"]

        explained = daemon.client_request(path, dict(_request(), explain=True,
                                                      guidance_protocol=1))
        assert explained["guidance_protocol"] == 1
        assert "qpp" in explained["results"]["diagnostics"]
        assert "confidence_reasons" not in explained["results"]["summary"]

        rejected = daemon.client_request(path, dict(_request(), guidance_protocol=2))
        assert rejected["ok"] is False
        assert "guidance_protocol" not in rejected


def _cli_search(path, *options):
    result = CliRunner().invoke(main, [
        "--db", str(path), "search", "running deployment", "--mode", "lexical",
        "-c", "alpha", "--json", *options,
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "fidx.search.v2"
    assert payload["status"] == "ok"
    assert payload["results"]
    return payload


@pytest.mark.skipif(os.name != "posix", reason="production daemon uses Unix sockets")
@pytest.mark.parametrize("option", ["--guidance", "--explain", "--no-guidance", None])
def test_legacy_index_protocol_searches_cold_with_real_daemon(
    scoped_index, tmp_path, monkeypatch, option,
):
    scoped_index.execute("DELETE FROM meta WHERE key = 'instance_id'")
    scoped_index.commit()
    monkeypatch.setattr(daemon, "GUIDANCE_DEFAULT", option is None)
    options = [option] if option else []
    path = tmp_path / "index.db"
    with _live_server(path, tmp_path, monkeypatch):
        cold = _cli_search(path, *options, "--no-daemon")
        warm = _cli_search(path, *options)
    assert warm == cold
    active = option in ("--guidance", None)
    assert ("confidence_reasons" in warm["summary"]) is active
    assert ("count_relation" in warm["summary"]) is active
    assert ("term_coverage" in warm["diagnostics"]) is active
    assert ("qpp" in warm["diagnostics"]) is (option == "--explain")
    if active:
        assert warm["diagnostics"]["execution"] == {
            "complete": True, "partial_reason": None, "path": "cold"}
    assert db.get_meta(scoped_index, "instance_id") is None


@pytest.mark.skipif(os.name != "posix", reason="production daemon uses Unix sockets")
@pytest.mark.parametrize("legacy", [True, False])
def test_real_daemon_cli_preserves_legacy_and_current_guidance_routes(
    scoped_index, tmp_path, monkeypatch, legacy,
):
    if legacy:
        scoped_index.execute("DELETE FROM meta WHERE key = 'instance_id'")
        scoped_index.commit()
    else:
        assert db.get_meta(scoped_index, "instance_id")
    monkeypatch.setattr(daemon, "GUIDANCE_DEFAULT", False)
    path = tmp_path / "index.db"
    cli_thread = threading.current_thread()
    original_open_store = vector_store.open_store

    def daemon_store(*args, **kwargs):
        assert threading.current_thread() is not cli_thread, "CLI opened a cold store"
        return original_open_store(*args, **kwargs)

    with _live_server(path, tmp_path, monkeypatch):
        monkeypatch.setattr(vector_store, "open_store", daemon_store)
        payload = _cli_search(path, *([] if legacy else ["--guidance"]))
    if legacy:
        assert "guidance" not in payload["request"]
        assert "execution" not in payload["diagnostics"]
    else:
        assert payload["diagnostics"]["execution"]["path"] == "daemon"
