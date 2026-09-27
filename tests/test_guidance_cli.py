# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Guidance opt-in routing and byte-compatible default-off CLI behavior."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from fidx import budget, daemon, db, indexer, vector_store
from fidx.cli import main


@pytest.fixture()
def indexed(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    root = tmp_path / "alpha"; root.mkdir()
    (root / "a.md").write_text("running quickly")
    indexer.add_collection(conn, "alpha", root)
    indexer.sync_collection(conn, vector_store.NullStore(), "alpha", indexer.IndexStats())
    instance = db.get_meta(conn, "instance_id")
    conn.close()
    return path, instance


def _run(path, *extra, query="running deployment"):
    result = CliRunner().invoke(main, [
        "--db", str(path), "search", query, "--mode", "lexical", "-c", "alpha",
        "--json", *extra,
    ])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.mark.parametrize("flag,explain", [
    (None, False), (None, True), ("--no-guidance", False),
    ("--no-guidance", True), ("--guidance", False), ("--guidance", True),
])
def test_cli_six_state_matrix_cold(indexed, flag, explain):
    path, _ = indexed
    extra = ["--no-daemon"]
    if flag:
        extra.append(flag)
    if explain:
        extra.append("--explain")
    payload = _run(path, *extra)
    active = flag == "--guidance"
    expected_echo = (True if active else False if flag == "--no-guidance" else "absent")
    assert payload["request"].get("guidance", "absent") == expected_echo
    assert ("confidence_reasons" in payload["summary"]) is active
    assert ("term_coverage" in payload["diagnostics"]) is active
    assert ("qpp" in payload["diagnostics"]) is explain
    assert ("execution" in payload["diagnostics"]) is active
    if active:
        assert payload["diagnostics"]["execution"]["path"] == "cold"
        assert payload["diagnostics"]["execution"]["complete"] is True
        assert "fallback_cause" not in payload["diagnostics"]["execution"]


@pytest.mark.parametrize("flag,explain", [
    (None, False), (None, True), ("--no-guidance", False),
    ("--no-guidance", True), ("--guidance", False), ("--guidance", True),
])
def test_cli_six_state_matrix_daemon(indexed, monkeypatch, flag, explain):
    path, instance = indexed
    req = {"query": "running deployment", "mode": "lexical", "collections": ["alpha"],
           "limit": 10, "min_score": None, "truncate": None}
    if flag:
        req["guidance"] = flag == "--guidance"
    if explain:
        req["explain"] = True
    conn = db.connect(path, create=False)
    remote = daemon.run_search(conn, vector_store.NullStore(), None, req)
    conn.close()
    calls = []

    def legacy(_path, _req):
        calls.append("legacy")
        return {"ok": True, "instance_id": instance, "results": remote}

    def capability(_path, _req, **kwargs):
        calls.append("capability")
        assert kwargs["require_guidance_protocol"] is True
        return daemon.ClientOutcome("daemon", response={
            "ok": True, "instance_id": instance, "guidance_protocol": 1,
            "results": remote})

    monkeypatch.setattr(daemon, "client_request", legacy)
    monkeypatch.setattr(daemon, "client_request_outcome", capability)
    options = [flag] if flag else []
    if explain:
        options.append("--explain")
    payload = _run(path, *options)
    assert calls == ["legacy" if flag is None and not explain else "capability"]
    assert ("execution" in payload["diagnostics"]) is (flag == "--guidance")
    if flag == "--guidance":
        assert payload["diagnostics"]["execution"]["path"] == "daemon"
    assert ("qpp" in payload["diagnostics"]) is explain


def test_absent_default_off_old_daemon_and_stale_payloads_remain_exact(indexed, monkeypatch):
    path, instance = indexed
    cold = _run(path, "--no-daemon")
    monkeypatch.setattr(daemon, "client_request", lambda *_a, **_k: {
        "ok": True, "instance_id": instance, "results": cold})
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("default-off path should use legacy client")))
    assert _run(path) == cold
    monkeypatch.setattr(daemon, "client_request", lambda *_a, **_k: {
        "ok": True, "instance_id": "old", "results": cold})
    stale = _run(path)
    assert stale["diagnostics"]["execution"] == {
        "daemon": "stale_instance",
        "message": "Restart fidx serve to use the current index database.",
    }
    assert "guidance" not in stale["request"]


def test_guided_daemon_path_and_old_daemon_protocol_fallback(indexed, monkeypatch):
    path, instance = indexed
    conn = db.connect(path, create=False)
    request = {"query": "running deployment", "mode": "lexical", "collections": ["alpha"],
               "limit": 10, "min_score": None, "truncate": None, "guidance": True}
    remote = daemon.run_search(conn, vector_store.NullStore(), None, request)
    conn.close()
    calls = []

    def accepted(_path, req, **kwargs):
        calls.append((dict(req), kwargs))
        return daemon.ClientOutcome("daemon", response={
            "ok": True, "instance_id": instance, "guidance_protocol": 1,
            "results": remote})

    monkeypatch.setattr(daemon, "client_request_outcome", accepted)
    warm = _run(path, "--guidance")
    assert warm["diagnostics"]["execution"]["path"] == "daemon"
    assert calls[0][0]["guidance"] is True
    assert calls[0][1]["require_guidance_protocol"] is True
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("cold", fallback_cause="malformed_response"))
    fallback = _run(path, "--guidance")
    assert fallback["diagnostics"]["execution"]["path"] == "cold"
    assert fallback["diagnostics"]["execution"]["fallback_cause"] == "malformed_response"
    assert fallback["results"] == warm["results"]


def test_ack_without_requested_guidance_fields_falls_back(indexed, monkeypatch):
    path, instance = indexed
    ordinary = _run(path, "--no-daemon")
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("daemon", response={
                            "ok": True, "instance_id": instance,
                            "guidance_protocol": 1, "results": ordinary}))
    payload = _run(path, "--guidance")
    assert payload["diagnostics"]["execution"]["path"] == "cold"
    assert payload["diagnostics"]["execution"]["fallback_cause"] == "malformed_response"


def test_ack_with_different_query_falls_back(indexed, monkeypatch):
    path, instance = indexed
    conn = db.connect(path, create=False)
    remote = daemon.run_search(conn, vector_store.NullStore(), None, {
        "query": "different", "mode": "lexical", "collections": ["alpha"],
        "limit": 10, "guidance": True})
    conn.close()
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("daemon", response={
                            "ok": True, "instance_id": instance,
                            "guidance_protocol": 1, "results": remote}))
    payload = _run(path, "--guidance")
    assert payload["query"] == "running deployment"
    assert payload["diagnostics"]["execution"]["fallback_cause"] == "malformed_response"


def test_explain_only_malformed_daemon_diagnostics_falls_back(indexed, monkeypatch):
    path, instance = indexed
    ordinary = _run(path, "--no-daemon")
    ordinary["diagnostics"]["no_results"] = None
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("daemon", response={
                            "ok": True, "instance_id": instance,
                            "guidance_protocol": 1, "results": ordinary}))
    payload = _run(path, "--explain")
    assert isinstance(payload["diagnostics"]["qpp"], dict)
    assert "no_results" not in payload["diagnostics"]


def test_internal_default_on_uses_protocol_and_echoes_guidance(indexed, monkeypatch):
    path, _ = indexed
    monkeypatch.setattr(daemon, "GUIDANCE_DEFAULT", True)
    calls = []

    def unavailable(_path, req, **kwargs):
        calls.append((req, kwargs))
        return daemon.ClientOutcome("cold")

    monkeypatch.setattr(daemon, "client_request_outcome", unavailable)
    payload = _run(path)
    assert payload["request"]["guidance"] is True
    assert payload["diagnostics"]["execution"]["path"] == "cold"
    assert calls[0][0]["guidance"] is True
    assert calls[0][1]["require_guidance_protocol"] is True


def test_guided_cli_applies_budget_once_after_path_provenance(indexed, monkeypatch):
    path, _ = indexed
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("cold", fallback_cause="timeout"))
    original = budget.apply_budget
    seen = []

    def counted(payload, requested):
        seen.append((payload["diagnostics"]["execution"].copy(), requested))
        return original(payload, requested)

    monkeypatch.setattr(budget, "apply_budget", counted)
    payload = _run(path, "--guidance", "--budget-tokens", "5000")
    assert seen == [({"complete": True, "partial_reason": None,
                     "path": "cold", "fallback_cause": "timeout"}, 5000)]
    assert payload["summary"]["budget"]["requested"] == 5000


def test_guidance_stale_and_daemon_error_keep_truthful_provenance(indexed, monkeypatch):
    path, _ = indexed
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("cold", stale_instance=True))
    stale = _run(path, "--guidance")
    assert stale["diagnostics"]["execution"] == {
        "daemon": "stale_instance",
        "message": "Restart fidx serve to use the current index database.",
        "complete": True, "partial_reason": None, "path": "cold",
    }
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("cold", fallback_cause="daemon_error",
                                             daemon_error="exact daemon failure"))
    failed = _run(path, "--guidance")
    assert failed["diagnostics"]["execution"]["fallback_cause"] == "daemon_error"
    assert failed["diagnostics"]["execution"]["error"] == "exact daemon failure"


@pytest.mark.parametrize("query,flag", [
    ("running", "--guidance"), ("running", "--explain"),
    ("missingword", "--guidance"), ("missingword", "--explain"),
])
def test_explain_or_guidance_budget_zero_is_classified_and_abstains(
    indexed, query, flag,
):
    path, _ = indexed
    payload = _run(path, "--no-daemon", flag, "--budget-tokens", "0", query=query)
    assert budget.unclassified_fields(payload) == []
    assert payload["summary"]["budget"]["dropped"]["diagnostics"] is True
    assert "qpp" not in payload.get("diagnostics", {})
    if query == "missingword" and flag == "--guidance":
        assert payload["diagnostics"]["no_results"]["causes"] == ["vocabulary_mismatch"]
        assert payload["next_actions"]


def test_explicit_off_and_explain_do_not_report_guidance_provenance(indexed, monkeypatch):
    path, _ = indexed
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *_a, **_k:
                        daemon.ClientOutcome("cold", fallback_cause="timeout"))
    for flag in ("--no-guidance", "--explain"):
        payload = _run(path, flag)
        assert "execution" not in payload["diagnostics"]


def test_guidance_lexical_cold_path_never_loads_a_model(indexed, monkeypatch):
    from fidx import cli
    path, _ = indexed
    monkeypatch.setattr(cli, "FastEmbedder", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("lexical guidance must not load a model")))
    payload = _run(path, "--no-daemon", "--guidance")
    assert payload["status"] == "ok"
