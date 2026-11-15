# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Bounded daemon admission, ingress, lifetime, and failure cleanup."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import pytest

from fidx import daemon, db, vector_store
from fidx.config import EmbedProfile
from fidx.embedder import HashEmbedder

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Unix sockets required")
PROFILE = EmbedProfile("limits-test", "hash", 64)


def request(query="alpha", **options):
    return {"cmd": "search", "query": query, "mode": "lexical", **options}


@contextmanager
def running(tmp_path, monkeypatch, *, workers=1):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path.parent / ("r" + hashlib.sha256(tmp_path.name.encode()).hexdigest()[:8])))
    monkeypatch.setenv("FIDX_DAEMON_WORKERS", str(workers))
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    ready = queue.Queue()
    failures = []
    bind = daemon._bind_private_unix_server

    def capture(sock_path, handler):
        server = bind(sock_path, handler)
        ready.put(server)
        return server

    monkeypatch.setattr(daemon, "_bind_private_unix_server", capture)

    def serve():
        conn = db.connect(path, create=False)
        store = vector_store.open_store(conn, path)
        try:
            daemon.serve(path, conn, store, HashEmbedder(PROFILE))
        except BaseException as exc:
            failures.append(exc)
        finally:
            if hasattr(store, "close"):
                store.close()
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    server = None
    try:
        try:
            server = ready.get(timeout=3)
        except queue.Empty:
            pytest.fail(f"daemon did not start: {failures}")
        assert daemon.client_request(path, {"cmd": "ping"}, timeout=1)["ok"]
        yield path, server
    finally:
        if server is not None:
            server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive(), "daemon shutdown hung"
        assert not failures, failures


def test_busy_when_worker_and_two_waiting_slots_are_full(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocked(*args, **kwargs):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        return {"schema": daemon.SEARCH_SCHEMA}

    monkeypatch.setattr(daemon, "run_search", blocked)
    with running(tmp_path, monkeypatch) as (path, _):
        with ThreadPoolExecutor(max_workers=8) as clients:
            pending = [clients.submit(daemon.client_request, path, request(), 2)
                       for _ in range(8)]
            try:
                assert entered.wait(1)
                # Five refusals must arrive while the worker remains blocked.
                done = as_completed(pending, timeout=1)
                early = [next(done).result() for _ in range(5)]
                assert all(reply and reply.get("error_code") == "busy" for reply in early)
                assert len(calls) == 1
            finally:
                release.set()
            replies = [future.result(timeout=3) for future in pending]
        assert sum(reply["ok"] for reply in replies) == 3
        assert sum(reply.get("error_code") == "busy" for reply in replies) == 5


def test_ping_bypasses_saturated_search_workers(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return {"schema": daemon.SEARCH_SCHEMA}

    monkeypatch.setattr(daemon, "run_search", blocked)
    with running(tmp_path, monkeypatch) as (path, _):
        with ThreadPoolExecutor(max_workers=3) as clients:
            pending = [clients.submit(daemon.client_request, path, request(), 2)
                       for _ in range(3)]
            try:
                assert entered.wait(1)
                started = time.monotonic()
                reply = daemon.client_request(path, {"cmd": "ping"}, timeout=0.5)
                assert reply and reply.get("pong") is True
                assert time.monotonic() - started < 0.5
            finally:
                release.set()
            for future in pending:
                assert future.result(timeout=3) is not None


def test_request_line_size_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "MAX_REQUEST_BYTES", 256, raising=False)
    called = []
    monkeypatch.setattr(daemon, "run_search", lambda *a, **k: called.append(True) or {})
    with running(tmp_path, monkeypatch) as (path, _):
        reply = daemon.client_request(path, request("x" * 512), timeout=1)
        assert reply and reply["ok"] is False
        assert not called


def test_incomplete_request_line_read_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "REQUEST_READ_TIMEOUT", 0.1, raising=False)
    with running(tmp_path, monkeypatch) as (path, _):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.7)
            client.connect(str(daemon.socket_path_for(path)))
            client.sendall(b'{"cmd":')
            try:
                data = client.recv(4096)
            except TimeoutError:
                pytest.fail("incomplete request pinned a connection beyond the read deadline")
            assert not data or json.loads(data)["ok"] is False
        assert daemon.client_request(path, {"cmd": "ping"}, timeout=1)["ok"]


def test_stale_instance_rejected_before_search(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(daemon, "run_search", lambda *a, **k: called.append(True) or {})
    with running(tmp_path, monkeypatch) as (path, _):
        reply = daemon.client_request(path, request(instance_id="different-index"), timeout=1)
        assert reply and reply.get("error_code") == "stale_instance"
        assert not called


def test_live_socket_is_never_replaced(tmp_path, monkeypatch):
    with running(tmp_path, monkeypatch) as (path, _):
        sock_path = daemon.socket_path_for(path)
        inode = sock_path.stat().st_ino

        def refuse_rebind(*args):
            raise AssertionError("live socket replacement attempted")

        monkeypatch.setattr(daemon, "_bind_private_unix_server", refuse_rebind)
        conn = db.connect(path, create=False)
        store = vector_store.open_store(conn, path)
        try:
            with pytest.raises(RuntimeError, match="live|already") as caught:
                daemon.serve(path, conn, store, HashEmbedder(PROFILE))
            assert str(sock_path) in str(caught.value)
        finally:
            if hasattr(store, "close"):
                store.close()
            conn.close()
        assert sock_path.stat().st_ino == inode
        assert daemon.client_request(path, {"cmd": "ping"}, timeout=1)["ok"]


def test_dead_socket_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path.parent / ("r" + hashlib.sha256(tmp_path.name.encode()).hexdigest()[:8])))
    path = tmp_path / "index.db"
    sock_path = daemon._prepare_socket_path(path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(sock_path))
    assert sock_path.exists()
    with running(tmp_path, monkeypatch) as (path, _):
        assert daemon.client_request(path, {"cmd": "ping"}, timeout=1)["ok"]
    assert not sock_path.exists()


def test_timeout_closes_request_connection_and_store(tmp_path, monkeypatch):
    opened = []
    closed_stores = []
    connect = db.connect
    open_store = vector_store.open_store

    class TrackedConnection:
        def __init__(self, conn):
            self.conn = conn
            self.closed = False
            opened.append(self)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def close(self):
            self.closed = True
            self.conn.close()

    def tracked_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        return TrackedConnection(conn) if kwargs.get("read_only") else conn

    class TrackedStore:
        def __init__(self, store):
            self.store = store

        def __getattr__(self, name):
            return getattr(self.store, name)

        def close(self):
            closed_stores.append(self)
            if hasattr(self.store, "close"):
                self.store.close()

    def tracked_store(conn, *args, **kwargs):
        result = open_store(conn, *args, **kwargs)
        return TrackedStore(result) if kwargs.get("read_only") else result

    def expire(*args, **kwargs):
        from fidx.errors import SearchTimeout
        raise SearchTimeout("search timed out")

    monkeypatch.setattr(db, "connect", tracked_connect)
    monkeypatch.setattr(vector_store, "open_store", tracked_store)
    monkeypatch.setattr(daemon, "run_search", expire)
    with running(tmp_path, monkeypatch) as (path, _):
        reply = daemon.client_request(path, request(), timeout=1)
        assert reply and reply.get("error_code") == "timeout"
        # Response publication follows request cleanup.
        assert len(opened) == 1 and all(conn.closed for conn in opened)
        assert len(closed_stores) == 1
        assert daemon.client_request(path, {"cmd": "ping"}, timeout=1)["ok"]


def test_16_clients_25_mixed_requests_finish_with_health_checks(tmp_path, monkeypatch):
    calls = 0
    guard = threading.Lock()

    def mixed(conn, store, embedder, req, **kwargs):
        nonlocal calls
        with guard:
            calls += 1
        if req["query"] == "expire":
            from fidx.errors import SearchTimeout
            raise SearchTimeout("search timed out")
        return {"schema": daemon.SEARCH_SCHEMA}

    monkeypatch.setattr(daemon, "run_search", mixed)
    with running(tmp_path, monkeypatch, workers=2) as (path, _):
        stopped = threading.Event()
        pings = []

        def health():
            while not stopped.is_set():
                pings.append(daemon.client_request(path, {"cmd": "ping"}, timeout=1))
                stopped.wait(0.01)

        monitor = threading.Thread(target=health, daemon=True)
        monitor.start()
        started = time.monotonic()
        try:
            def client(client_id):
                return [daemon.client_request(
                    path, request("expire" if (client_id + n) % 3 == 0 else "alpha"),
                    timeout=2) for n in range(25)]

            with ThreadPoolExecutor(max_workers=16) as pool:
                pending = [pool.submit(client, n) for n in range(16)]
                responses = [reply for future in pending
                             for reply in future.result(timeout=10)]
        finally:
            stopped.set()
            monitor.join(timeout=2)
        assert time.monotonic() - started < 10
        assert len(responses) == 400
        assert all(reply and (reply.get("ok") is True or
                              reply.get("error_code") in {"timeout", "busy"})
                   for reply in responses)
        assert any(reply.get("error_code") == "timeout" for reply in responses)
        assert pings and all(reply and reply.get("pong") for reply in pings)


def test_daemon_sql_deadline_interrupts_work(tmp_path, monkeypatch):
    def slow_sql(conn, *args, **kwargs):
        conn.create_function('pause', 1, lambda n: (time.sleep(.002), n)[1])
        conn.execute('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000) SELECT sum(pause(x)) FROM n').fetchone()
        return {}

    monkeypatch.setattr(daemon, '_run_search', slow_sql)
    with running(tmp_path, monkeypatch) as (path, _):
        started = time.monotonic()
        reply = daemon.client_request(path, request(timeout=.03), timeout=1)
        assert reply and reply.get('error_code') == 'timeout'
        assert time.monotonic() - started < .3
        assert daemon.client_request(path, {'cmd': 'ping'}, timeout=1)['pong']
