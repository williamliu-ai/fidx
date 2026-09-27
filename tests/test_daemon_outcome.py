# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Internal daemon routing evidence; the public search envelope stays unchanged."""

from __future__ import annotations

import errno
import json

import pytest

from fidx import daemon


class FakeSocket:
    def __init__(self, chunks=(), *, connect_error=None, recv_error=None):
        self.chunks = list(chunks)
        self.connect_error = connect_error
        self.recv_error = recv_error
        self.sent = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def settimeout(self, _timeout):
        pass

    def connect(self, _path):
        if self.connect_error is not None:
            raise self.connect_error

    def sendall(self, data):
        self.sent = data

    def recv(self, _size):
        if self.recv_error is not None:
            raise self.recv_error
        return self.chunks.pop(0) if self.chunks else b""


def _request(monkeypatch, tmp_path, response=None, *, chunks=None, connect_error=None,
             recv_error=None, socket_exists=True):
    # These tests exercise a simulated supported Unix-socket transport even on
    # Windows, where neither platform symbol need exist. The unsupported case
    # below deliberately omits this fixture.
    monkeypatch.setattr(daemon.socketserver, "UnixStreamServer", object(), raising=False)
    monkeypatch.setattr(daemon.socket, "AF_UNIX", object(), raising=False)
    path = tmp_path / "daemon.sock"
    if socket_exists:
        path.touch()
    fake = FakeSocket(
        [json.dumps(response).encode() + b"\n"] if chunks is None else chunks,
        connect_error=connect_error, recv_error=recv_error,
    )
    monkeypatch.setattr(daemon, "socket_path_for", lambda _db: path)
    monkeypatch.setattr(daemon.socket, "socket", lambda *_args: fake)
    return fake


def _response(*, instance="current", schema="fidx.search.v2", ack=None):
    response = {"ok": True, "instance_id": instance, "results": {"schema": schema}}
    if ack is not None:
        response["guidance_protocol"] = ack
    return response


def _outcome(tmp_path, *, require=False):
    req = {"cmd": "search", "query": "topic"}
    return daemon.client_request_outcome(
        tmp_path / "index.db", req, expected_instance_id="current",
        require_guidance_protocol=require,
    )


def test_successful_old_daemon_is_accepted_only_without_new_protocol(monkeypatch, tmp_path):
    fake = _request(monkeypatch, tmp_path, _response())
    old = _outcome(tmp_path)
    assert old.path == "daemon"
    assert old.response == _response()
    assert old.fallback_cause is None
    assert json.loads(fake.sent) == {"cmd": "search", "query": "topic"}

    fake = _request(monkeypatch, tmp_path, _response())
    incompatible = _outcome(tmp_path, require=True)
    assert incompatible.path == "cold"
    assert incompatible.fallback_cause == "malformed_response"
    assert incompatible.response is None
    assert json.loads(fake.sent)["guidance_protocol"] == 1


def test_protocol_ack_is_internal_and_does_not_mutate_request(monkeypatch, tmp_path):
    fake = _request(monkeypatch, tmp_path, _response(ack=1))
    req = {"cmd": "search", "query": "topic"}
    result = daemon.client_request_outcome(
        tmp_path / "index.db", req, expected_instance_id="current",
        require_guidance_protocol=True,
    )
    assert result.path == "daemon"
    assert result.response == _response(ack=1)
    assert req == {"cmd": "search", "query": "topic"}
    assert json.loads(fake.sent)["guidance_protocol"] == 1


@pytest.mark.parametrize("ack", [True, "1", 2])
def test_protocol_ack_requires_exact_integer_version(monkeypatch, tmp_path, ack):
    _request(monkeypatch, tmp_path, _response(ack=ack))
    assert _outcome(tmp_path, require=True).fallback_cause == "malformed_response"


def test_search_outcome_requires_expected_instance_id(monkeypatch, tmp_path):
    _request(monkeypatch, tmp_path, _response(instance="old"))
    with pytest.raises(ValueError, match="expected_instance_id"):
        daemon.client_request_outcome(tmp_path / "index.db", {"cmd": "search"})


def test_stale_instance_precedes_schema_and_capability_checks(monkeypatch, tmp_path):
    _request(monkeypatch, tmp_path, _response(instance="old", schema="wrong"))
    result = _outcome(tmp_path, require=True)
    assert result.path == "cold"
    assert result.stale_instance is True
    assert result.fallback_cause is None
    assert result.daemon_error is None


@pytest.mark.parametrize("response", [
    {"ok": False, "error": "daemon failed", "instance_id": "old"},
    {"ok": False, "error": "daemon failed", "instance_id": "current"},
])
def test_daemon_error_is_preserved_even_without_ack(monkeypatch, tmp_path, response):
    _request(monkeypatch, tmp_path, response)
    result = _outcome(tmp_path, require=True)
    assert result.path == "cold"
    assert result.fallback_cause == "daemon_error"
    assert result.daemon_error == "daemon failed"


@pytest.mark.parametrize("response", [
    {"ok": False, "instance_id": "current"},
    {"ok": False, "error": 42, "instance_id": "current"},
    {"ok": True, "instance_id": "current", "results": {}},
    _response(schema="wrong"),
    ["not", "a", "response"],
    {"ok": "true", "instance_id": "current", "results": {"schema": "fidx.search.v2"}},
])
def test_malformed_rpc_response_falls_back(monkeypatch, tmp_path, response):
    _request(monkeypatch, tmp_path, response)
    result = _outcome(tmp_path)
    assert result.path == "cold"
    assert result.fallback_cause == "malformed_response"
    assert result.daemon_error is None


@pytest.mark.parametrize("chunks", [
    [], [b"{"], [b"{}\n"], [b"\xff\n"],
])
def test_empty_truncated_invalid_json_or_utf8_is_malformed(monkeypatch, tmp_path, chunks):
    _request(monkeypatch, tmp_path, chunks=chunks)
    result = _outcome(tmp_path)
    assert result.path == "cold"
    assert result.fallback_cause == "malformed_response"


@pytest.mark.parametrize("error,cause", [
    (ConnectionRefusedError(errno.ECONNREFUSED, "refused"), "connect_refused"),
    (TimeoutError("timed out"), "timeout"),
    (ConnectionResetError(errno.ECONNRESET, "reset"), "transport_error"),
    (PermissionError(errno.EACCES, "denied"), "transport_error"),
    (FileNotFoundError(errno.ENOENT, "vanished"), "transport_error"),
])
def test_transport_errors_keep_their_distinct_causes(monkeypatch, tmp_path, error, cause):
    _request(monkeypatch, tmp_path, connect_error=error)
    result = _outcome(tmp_path)
    assert result.path == "cold"
    assert result.fallback_cause == cause
    assert result.daemon_error is None


def test_receive_timeout_is_not_malformed(monkeypatch, tmp_path):
    _request(monkeypatch, tmp_path, recv_error=TimeoutError("timed out"))
    assert _outcome(tmp_path).fallback_cause == "timeout"


def test_missing_socket_is_cold_without_fallback_cause(monkeypatch, tmp_path):
    _request(monkeypatch, tmp_path, socket_exists=False)
    result = _outcome(tmp_path)
    assert result.path == "cold"
    assert result.fallback_cause is None
    assert result.stale_instance is False


def test_fake_socket_can_exercise_supported_path_without_platform_unix_types(monkeypatch, tmp_path):
    monkeypatch.delattr(daemon.socketserver, "UnixStreamServer", raising=False)
    monkeypatch.delattr(daemon.socket, "AF_UNIX", raising=False)
    _request(monkeypatch, tmp_path, _response())
    assert _outcome(tmp_path).path == "daemon"


def test_unsupported_unix_socket_is_cold_without_fallback_cause(monkeypatch, tmp_path):
    monkeypatch.delattr(daemon.socketserver, "UnixStreamServer", raising=False)
    monkeypatch.delattr(daemon.socket, "AF_UNIX", raising=False)
    result = _outcome(tmp_path)
    assert result.path == "cold"
    assert result.fallback_cause is None
