# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Regression checks from independent daemon lifecycle review."""
import json
import os
import socket
import threading

import pytest
from click.testing import CliRunner

from fidx import daemon, db, errors
from fidx.cli import main
from test_daemon_limits import running

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Unix sockets required")


def test_deep_json_cannot_stop_accept_loop(tmp_path, monkeypatch):
    with running(tmp_path, monkeypatch) as (path, _):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            client.connect(str(daemon.socket_path_for(path)))
            client.sendall(b'[' * 20000 + b'0' + b']' * 20000 + b'\n')
            try:
                client.recv(4096)
            except ConnectionResetError:
                pass
        assert daemon.client_request(path, {'cmd': 'ping'}, timeout=1)['pong']


@pytest.mark.parametrize('wire', [b'', b'{"ok":false', b'{"ok":false,"error_code":"timeout"}\n'])
def test_protocol_transport_failure_does_not_repeat_cold(tmp_path, monkeypatch, wire):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path.parent / 'review-runtime'))
    path = tmp_path / 'index.db'
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    sock_path = daemon._prepare_socket_path(path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen()
    calls = []

    def peer():
        with listener:
            client, _ = listener.accept()
            with client:
                client.recv(65536)
                client.sendall(wire)

    def cold(*args, **kwargs):
        calls.append(True)
        raise errors.FidxError('unexpected cold repeat')

    monkeypatch.setattr(daemon, 'run_search', cold)
    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    result = CliRunner().invoke(main, ['--db', str(path), 'search', 'alpha', '--mode', 'lexical', '--guidance', '--json'])
    thread.join(2)
    assert not thread.is_alive()
    assert calls == [], result.output
    assert result.exit_code != 0
    assert json.loads(result.output)['schema'] == 'fidx.error.v1'


def test_expired_request_cleanup_does_not_wait_for_other_duckdb_search(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fidx import vector_store

    entered = threading.Event()
    release = threading.Event()
    open_store = vector_store.open_store

    class SlowStore:
        name = 'duckdb'

        def knn(self, *args):
            entered.set()
            assert release.wait(3)
            return []

        def close(self):
            pass

    def stores(conn, path, **kwargs):
        return SlowStore() if kwargs.get('read_only') else open_store(conn, path, **kwargs)

    def search(conn, store, embedder, req, **kwargs):
        if req['query'] == 'long':
            store.knn(None, 1)
            return {}
        raise errors.SearchTimeout('short request expired')

    monkeypatch.setattr(vector_store, 'open_store', stores)
    monkeypatch.setattr(daemon, 'run_search', search)
    with running(tmp_path, monkeypatch, workers=2) as (path, _):
        with ThreadPoolExecutor(max_workers=2) as pool:
            long = pool.submit(daemon.client_request, path, {'cmd':'search', 'query':'long', 'mode':'lexical'}, 4)
            assert entered.wait(1)
            short = pool.submit(daemon.client_request, path, {'cmd':'search', 'query':'short', 'mode':'lexical', 'timeout':.05}, 1)
            try:
                assert short.result(timeout=.3)['error_code'] == 'timeout'
            finally:
                release.set()
                long.result(timeout=2)


def test_sqlite_busy_wait_respects_search_budget(tmp_path):
    import sqlite3
    import time
    from fidx.search_limits import search_budget

    path = tmp_path / 'locked.db'
    writer = sqlite3.connect(path)
    reader = sqlite3.connect(path)
    try:
        writer.execute('CREATE TABLE t(x)')
        writer.commit()
        reader.execute('PRAGMA busy_timeout=200')
        writer.execute('BEGIN EXCLUSIVE')
        started = time.monotonic()
        with pytest.raises(errors.SearchTimeout):
            with search_budget(reader, {'timeout': .02}):
                reader.execute('SELECT * FROM t').fetchall()
        assert time.monotonic() - started < .15
        assert reader.execute('PRAGMA busy_timeout').fetchone()[0] == 200
    finally:
        writer.rollback()
        reader.close()
        writer.close()


@pytest.mark.parametrize('signal_name', ['SIGTERM', 'SIGINT'])
def test_signal_gracefully_removes_socket(tmp_path, monkeypatch, signal_name):
    import os
    import signal
    import subprocess
    import sys
    import time

    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path.parent / 'signal-runtime'))
    path = tmp_path / 'index.db'
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    code = '''
from pathlib import Path
import sys
from fidx import daemon, db, vector_store
from fidx.config import EmbedProfile
from fidx.embedder import HashEmbedder
path = Path(sys.argv[1])
conn = db.connect(path, create=False)
store = vector_store.open_store(conn, path)
try:
    daemon.serve(path, conn, store, HashEmbedder(EmbedProfile('signal-test', 'hash', 64)), workers=1)
finally:
    if hasattr(store, 'close'):
        store.close()
    conn.close()
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.communicate()[1].decode())
            reply = daemon.client_request(path, {'cmd':'ping'}, timeout=.1)
            if reply and reply.get('pong'):
                break
            time.sleep(.02)
        else:
            pytest.fail('daemon did not become healthy')
        os.kill(process.pid, getattr(signal, signal_name))
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0, stderr.decode()
        assert not daemon.socket_path_for(path).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=3)


def test_shutdown_preserves_replacement_socket(tmp_path, monkeypatch):
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with running(tmp_path, monkeypatch) as (path, server):
            sock_path = daemon.socket_path_for(path)
            close = server.server_close
            replacement_inode = []

            def replace_after_close():
                close()
                sock_path.unlink()
                replacement.bind(str(sock_path))
                replacement.listen()
                replacement_inode.append(sock_path.stat().st_ino)

            monkeypatch.setattr(server, 'server_close', replace_after_close)
        assert replacement_inode
        assert sock_path.exists(), 'old daemon removed replacement daemon socket'
        assert sock_path.stat().st_ino == replacement_inode[0]
    finally:
        replacement.close()


def test_simultaneous_startup_preserves_newly_bound_live_socket(tmp_path, monkeypatch):
    """A stale refusal must not authorize deleting a newly bound owner."""
    import time

    from fidx import vector_store
    from fidx.embedder import HashEmbedder
    from test_daemon_limits import PROFILE

    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path.parent / 'startup-run'))
    path = tmp_path / 'index.db'
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    sock_path = daemon._prepare_socket_path(path)
    real_socket = socket.socket
    with real_socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(sock_path))

    second_refusal = threading.Event()
    first_bound = threading.Event()
    start = threading.Barrier(3)
    condition = threading.Condition()
    refusals = []
    servers = []
    failures = []
    bind = daemon._bind_private_unix_server

    class CoordinatedSocket(real_socket):
        def connect(self, address):
            try:
                return super().connect(address)
            except ConnectionRefusedError:
                if address != str(sock_path):
                    raise
                with condition:
                    refusals.append(threading.get_ident())
                    refusal_number = len(refusals)
                if refusal_number == 1:
                    # A serialized lifecycle allows this bounded wait to expire.
                    second_refusal.wait(.5)
                else:
                    second_refusal.set()
                    assert first_bound.wait(2), 'first daemon never bound'
                raise

    def capture(address, handler):
        server = bind(address, handler)
        with condition:
            servers.append(server)
            condition.notify_all()
        first_bound.set()
        return server

    monkeypatch.setattr(socket, 'socket', CoordinatedSocket)
    monkeypatch.setattr(daemon, '_bind_private_unix_server', capture)

    def serve():
        conn = db.connect(path, create=False)
        store = vector_store.open_store(conn, path)
        try:
            start.wait(timeout=3)
            daemon.serve(path, conn, store, HashEmbedder(PROFILE))
        except BaseException as exc:
            with condition:
                failures.append(exc)
                condition.notify_all()
        finally:
            if hasattr(store, 'close'):
                store.close()
            conn.close()

    threads = [threading.Thread(target=serve, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=3)
    try:
        deadline = time.monotonic() + 3
        with condition:
            while len(servers) < 2 and not failures and time.monotonic() < deadline:
                condition.wait(max(0, deadline - time.monotonic()))
        assert len(servers) == 1, 'simultaneous startup stole a live daemon socket'
        assert len(failures) == 1 and isinstance(failures[0], RuntimeError), failures
        assert 'already' in str(failures[0]) or 'live' in str(failures[0])
        assert daemon.client_request(path, {'cmd': 'ping'}, timeout=1)['pong']
    finally:
        # Capture can occur just before serve_forever starts. Both accept loops
        # are running by the time the startup observation above completes.
        for server in servers:
            server.shutdown()
        for thread in threads:
            thread.join(timeout=3)
        assert not any(thread.is_alive() for thread in threads)
