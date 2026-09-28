# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fake-only warm reranking protocol and daemon lifecycle regressions."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

from fidx import cli, daemon, db, indexer, rerank_models, vector_store
from fidx.config import get_profile
from fidx.embedder import HashEmbedder


MODEL = rerank_models.DEFAULT_RERANK_MODEL
OTHER = "jinaai/jina-reranker-v1-tiny-en"
ON = {"enabled": True, "model": MODEL}
OFF = {"enabled": False, "advice_flag": "--no-rerank"}


def _index(tmp_path):
    path = tmp_path / "index.db"
    root = tmp_path / "notes"
    root.mkdir()
    (root / "one.md").write_text("# Search guide\nsearchable text here\n", encoding="utf-8")
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", root)
    indexer.sync_collection(conn, vector_store.NullStore(), "notes", indexer.IndexStats())
    row = conn.execute("SELECT id, body FROM documents WHERE relpath = 'one.md'").fetchone()
    conn.execute("INSERT INTO chunks(doc_id, seq, pos, length, line_start, line_end) "
                 "VALUES (?, 0, 0, ?, 1, 2)", (row["id"], len(row["body"])))
    conn.commit()
    conn.close()
    return path


def _empty_index(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    return path


def _request(**extra):
    return {"cmd": "search", "query": "searchable", "mode": "lexical",
            "collections": ["notes"], "limit": 3, "min_score": None,
            "truncate": None, **extra}


def _commands(payload):
    actions = [a["command"] for a in payload["next_actions"]
               if a.get("command", [])[:2] == ["fidx", "search"]]
    options = [a["command"] for a in payload["summary"]["truncation_advice"]["options"]
               if a.get("command", [])[:2] == ["fidx", "search"]]
    return actions + options


class Scores:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return iter([2.0] * len(self.calls[-1][1]))


@contextmanager
def _live_server(path, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    ready = queue.Queue()
    failures = []
    original = daemon._bind_private_unix_server

    def bind(sock_path, handler):
        server = original(sock_path, handler)
        ready.put(server)
        return server

    monkeypatch.setattr(daemon, "_bind_private_unix_server", bind)

    def run():
        try:
            conn = db.connect(path, create=False)
            daemon.serve(path, conn, vector_store.NullStore(),
                         HashEmbedder(get_profile()))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        server = ready.get(timeout=5)
        assert daemon.client_request(path, {"cmd": "ping"}, timeout=5)["ok"]
        yield
    finally:
        if "server" in locals():
            server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not failures, failures


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_socket_rejects_private_markers_and_requires_exact_effective_state(
        tmp_path, monkeypatch):
    path = _index(tmp_path)
    scorer = Scores()
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object())
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda model, *, local_files_only: (scorer, None))
    with _live_server(path, tmp_path, monkeypatch):
        forged = daemon.client_request(path, _request(_rerank_model=MODEL))
        assert forged["ok"] is False
        forged_off = daemon.client_request(path, _request(_rerank_advice_off=True))
        assert forged_off["ok"] is False
        for bad in (True, "1", 2):
            assert daemon.client_request(path, _request(
                rerank_protocol=bad, rerank_effective=ON))["ok"] is False
        for bad in ({"enabled": 1, "model": MODEL},
                    {"enabled": True, "model": MODEL.lower()},
                    {"enabled": True, "model": MODEL, "advice_flag": "--no-rerank"},
                    {"enabled": False, "model": MODEL}):
            assert daemon.client_request(path, _request(
                rerank_protocol=1, rerank_effective=bad))["ok"] is False
        assert scorer.calls == []


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_warm_on_off_echo_and_advice_with_one_lazy_local_scorer(tmp_path, monkeypatch):
    path = _index(tmp_path)
    selected = []
    scorers = {}
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object())

    def resolve(model, *, local_files_only):
        assert local_files_only is True
        selected.append(model)
        scorer = Scores()
        scorers[model] = scorer
        return scorer, None

    monkeypatch.setattr(rerank_models, "resolve_reranker_model", resolve)
    with _live_server(path, tmp_path, monkeypatch):
        legacy = daemon.client_request(path, _request())
        assert legacy["ok"] and "rerank_protocol" not in legacy
        assert "rerank" not in legacy["results"]["request"]

        for state, expected in ((ON, "--rerank=" + MODEL),
                                (OFF, "--no-rerank"),
                                ({"enabled": True, "model": OTHER}, "--rerank=" + OTHER)):
            wire = daemon.client_request(path, _request(
                rerank_protocol=1, rerank_effective=state, guidance_protocol=1,
                guidance=False))
            assert wire["ok"], wire
            assert wire["rerank_protocol"] == 1
            assert wire["rerank_effective"] == state
            assert wire["guidance_protocol"] == 1
            payload = wire["results"]
            assert payload["request"].get("rerank") == (
                {"model": state["model"]} if state["enabled"] else None)
            assert _commands(payload)
            assert all(command.count(expected) == 1 for command in _commands(payload))
        assert selected == [MODEL, OTHER]
        assert scorers[MODEL].calls and scorers[OTHER].calls


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_warm_empty_index_probes_capability_but_never_resolves_scorer(
        tmp_path, monkeypatch):
    path = _empty_index(tmp_path)
    probes = []
    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: probes.append(True))
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: pytest.fail("empty index resolved scorer"))
    with _live_server(path, tmp_path, monkeypatch):
        wire = daemon.client_request(path, _request(
            rerank_protocol=1, rerank_effective=ON))
    assert wire["ok"] is True and wire["rerank_effective"] == ON
    assert wire["results"]["status"] == "empty_index"
    assert wire["results"]["request"]["rerank"] == {"model": MODEL}
    assert probes == [True]


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_warm_capability_failure_precedes_scorer_resolution(tmp_path, monkeypatch):
    path = _index(tmp_path)

    def absent():
        raise rerank_models.RerankCapabilityError("fake missing capability")

    monkeypatch.setattr(rerank_models, "probe_rerank_capability", absent)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: pytest.fail("missing capability resolved scorer"))
    with _live_server(path, tmp_path, monkeypatch):
        wire = daemon.client_request(path, _request(
            rerank_protocol=1, rerank_effective=ON))
    assert wire["ok"] is False
    assert "fmdidx[rerank]" in wire["error"]
    assert "rerank_protocol" not in wire


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_lazy_generator_scoring_is_serialized_across_requests(tmp_path, monkeypatch):
    path = _index(tmp_path)
    active = 0
    peak = 0
    lock = threading.Lock()

    class SlowScores:
        def rerank(self, _query, documents):
            def scores():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(active, peak)
                time.sleep(0.08)
                for _ in documents:
                    yield 1.0
                with lock:
                    active -= 1
            return scores()

    selected = []
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object())

    def resolve(model, *, local_files_only):
        assert local_files_only
        selected.append(model)
        return SlowScores(), None

    monkeypatch.setattr(rerank_models, "resolve_reranker_model", resolve)
    with _live_server(path, tmp_path, monkeypatch):
        req = _request(rerank_protocol=1, rerank_effective=ON)
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: daemon.client_request(path, req), range(4)))
        assert all(reply["ok"] for reply in replies)
        assert peak == 1
        assert selected == [MODEL]


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_cli_warm_and_old_daemon_fallback_preserve_effective_off(tmp_path, monkeypatch):
    path = _index(tmp_path)
    monkeypatch.setenv("FIDX_RERANK", "on")
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object())
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda model, *, local_files_only: (Scores(), None))

    def invoke(*args):
        result = CliRunner().invoke(cli.main, ["--db", str(path), "search", "searchable",
                                               "--mode", "lexical", "--json", *args])
        assert result.exit_code == 0, result.output
        return json.loads(result.output)

    with _live_server(path, tmp_path, monkeypatch):
        on = invoke("--rerank", "--guidance")
        assert on["diagnostics"]["execution"]["path"] == "daemon"
        assert on["request"]["rerank"] == {"model": MODEL}
        assert all(command.count("--rerank=" + MODEL) == 1 for command in _commands(on))
        off = invoke("--no-rerank", "--guidance")
        assert off["diagnostics"]["execution"]["path"] == "daemon"
        assert "rerank" not in off["request"]
        assert all(command.count("--no-rerank") == 1 for command in _commands(off))

    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("cold", fallback_cause="malformed_response"))
    fallback = invoke("--no-rerank", "--guidance")
    assert fallback["diagnostics"]["execution"]["fallback_cause"] == "malformed_response"
    assert "rerank" not in fallback["request"]
    assert all(command.count("--no-rerank") == 1 for command in _commands(fallback))


@pytest.mark.parametrize("damage", ["wrong_model", "missing_advice", "missing_flag",
                                     "missing_option_command", "missing_option"])
def test_cli_rejects_acknowledged_but_malformed_advice_before_budget(
        tmp_path, monkeypatch, damage):
    path = _index(tmp_path)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object())
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda model, *, local_files_only: (Scores(), None))
    conn = db.connect(path, create=False)
    req = _request(_rerank_model=MODEL, guidance=True)
    candidate = daemon.run_search(conn, vector_store.NullStore(), None, req,
                                  reranker=Scores())
    conn.close()
    candidate = deepcopy(candidate)
    if damage == "wrong_model":
        candidate["request"]["rerank"] = {"model": OTHER}
    elif damage == "missing_advice":
        del candidate["summary"]["truncation_advice"]
    elif damage == "missing_flag":
        command = next(a["command"] for a in candidate["next_actions"]
                       if a["command"][:2] == ["fidx", "search"])
        command.remove("--rerank=" + MODEL)
    elif damage == "missing_option":
        candidate["summary"]["truncation_advice"]["options"].pop()
    else:
        del candidate["summary"]["truncation_advice"]["options"][0]["command"]
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("daemon", response={"results": candidate}))
    original = daemon.run_search
    cold_calls = []

    def cold(*args, **kwargs):
        cold_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(daemon, "run_search", cold)
    result = CliRunner().invoke(cli.main, [
        "--db", str(path), "search", "searchable", "--mode", "lexical", "-c", "notes",
        "-n", "3", "--json", "--rerank", "--guidance", "--budget-tokens", "0",
    ])
    assert result.exit_code == 0, result.output
    assert cold_calls == [True]
    payload = json.loads(result.output)
    assert payload["request"]["rerank"] == {"model": MODEL}
    assert payload["summary"]["budget"]["requested"] == 0


@pytest.mark.parametrize("env,args", [
    ("off", ()), ("0", ()), ("on", ()), (None, ("--rerank",)),
])
def test_legacy_index_without_instance_id_searches_cold_without_writing_meta(
        tmp_path, monkeypatch, env, args):
    path = _index(tmp_path)
    conn = db.connect(path, create=False)
    conn.execute("DELETE FROM meta WHERE key = 'instance_id'")
    conn.commit()
    conn.close()
    if env is None:
        monkeypatch.delenv("FIDX_RERANK", raising=False)
    else:
        monkeypatch.setenv("FIDX_RERANK", env)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object())
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda model, *, local_files_only: (Scores(), None))
    monkeypatch.setattr(daemon, "client_request_outcome",
                        lambda *_a, **_k: pytest.fail("legacy index used daemon outcome"))
    result = CliRunner().invoke(cli.main, [
        "--db", str(path), "search", "searchable", "--mode", "lexical", "--json",
        *args,
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["request"].get("rerank") == (
        {"model": MODEL} if env == "on" or args else None)
    conn = db.connect(path, create=False, read_only=True)
    assert db.get_meta(conn, "instance_id") is None
    conn.close()
