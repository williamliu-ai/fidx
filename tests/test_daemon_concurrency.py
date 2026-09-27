# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Concurrent production Unix handlers keep SQLite and vector handles local."""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from fidx import daemon, db, indexer, vector_store
from fidx.config import EmbedProfile, profile_fingerprint
from fidx.embedder import HashEmbedder


pytestmark = pytest.mark.skipif(os.name != "posix", reason="production daemon uses Unix sockets")
PROFILE = EmbedProfile("concurrency-test", "hash", 64)


def _seed_index(tmp_path: Path, *, backend: str = "sqlite-vec") -> Path:
    path = tmp_path / "index.db"
    source = tmp_path / "notes"
    source.mkdir()
    (source / "alpha.md").write_text("alpha deployment rolling restart\n")
    (source / "beta.md").write_text("beta recipe eggs pasta\n")
    conn = db.connect(path)
    db.init_schema(conn)
    store = vector_store.make_store(backend, conn, path)
    db.ensure_vectors(conn, store, PROFILE.dim, profile_fingerprint(PROFILE),
                      PROFILE.model, PROFILE.name)
    indexer.add_collection(conn, "notes", source)
    stats = indexer.IndexStats()
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, HashEmbedder(PROFILE), stats)
    if hasattr(store, "close"):
        store.close()
    conn.close()
    return path


def _request(query: str, mode: str) -> dict:
    return {"cmd": "search", "query": query, "mode": mode,
            "collections": [], "limit": 5, "min_score": None, "truncate": None}


@contextmanager
def _live_server(path: Path, tmp_path: Path, monkeypatch, embedder=None):
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
            store = vector_store.open_store(conn, path)
            daemon.serve(path, conn, store, embedder or HashEmbedder(PROFILE))
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        try:
            server = ready.get(timeout=5)
        except queue.Empty as exc:
            raise AssertionError(f"daemon did not bind: {failures}") from exc
        ping = daemon.client_request(path, {"cmd": "ping"}, timeout=5)
        assert ping and ping["ok"], failures
        yield server
    finally:
        if "server" in locals():
            server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not failures, failures


def test_read_only_connection_rejects_writes(tmp_path):
    path = _seed_index(tmp_path)
    conn = db.connect(path, create=False, read_only=True)
    try:
        db.validate_index(conn)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO meta(key, value) VALUES ('probe', 'no')")
    finally:
        conn.close()


def test_concurrent_real_sqlite_vec_searches_match_cold_envelopes(tmp_path, monkeypatch):
    path = _seed_index(tmp_path)
    requests = [_request(q, mode) for _ in range(8) for q, mode in (
        ("alpha deployment", "lexical"), ("alpha restart", "hybrid"),
        ("beta pasta", "vector"),
    )]
    cold_conn = db.connect(path, create=False)
    cold_store = vector_store.open_store(cold_conn, path)
    expected = [daemon.run_search(cold_conn, cold_store, HashEmbedder(PROFILE), req)
                for req in requests]
    cold_conn.close()

    with _live_server(path, tmp_path, monkeypatch):
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda req: daemon.client_request(path, req), requests))

    assert all(resp and resp["ok"] for resp in responses), responses
    assert [resp["results"] for resp in responses] == expected


def test_burst_of_64_mixed_clients_is_accepted_in_three_rounds(tmp_path, monkeypatch):
    path = _seed_index(tmp_path)
    modes = ("lexical", "hybrid", "vector")
    with _live_server(path, tmp_path, monkeypatch) as server:
        assert server.request_queue_size >= 64
        with ThreadPoolExecutor(max_workers=64) as pool:
            for _ in range(3):
                gate = threading.Barrier(65)

                def send(i):
                    gate.wait(timeout=5)
                    return daemon.client_request(path, _request("alpha", modes[i % 3]))

                pending = [pool.submit(send, i) for i in range(64)]
                gate.wait(timeout=5)
                responses = [future.result(timeout=30) for future in pending]
                assert all(resp is not None and resp["ok"] for resp in responses), responses


class PausingEmbedder:
    def __init__(self):
        self.profile = PROFILE
        self.base = HashEmbedder(PROFILE)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    def embed_queries(self, texts):
        if texts[0] == "warmup":
            return self.base.embed_queries(texts)
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if "slow" in texts[0]:
                self.entered.set()
                assert self.release.wait(5), "slow embed was not released"
            return self.base.embed_queries(texts)
        finally:
            with self.guard:
                self.active -= 1


def test_fast_lexical_finishes_while_slow_vector_embedding_waits(tmp_path, monkeypatch):
    path = _seed_index(tmp_path)
    embedder = PausingEmbedder()
    with _live_server(path, tmp_path, monkeypatch, embedder):
        with ThreadPoolExecutor(max_workers=3) as pool:
            slow = pool.submit(daemon.client_request, path, _request("slow alpha", "vector"))
            assert embedder.entered.wait(3)
            try:
                fast = pool.submit(daemon.client_request, path, _request("beta", "lexical"))
                response = fast.result(timeout=1)
                assert response and response["ok"]
                assert response["results"]["results"][0]["path"] == "notes/beta.md"
                assert not slow.done()
            finally:
                embedder.release.set()
            assert slow.result(timeout=5)["ok"]


def test_shared_embedder_never_runs_two_queries_at_once(tmp_path, monkeypatch):
    path = _seed_index(tmp_path)
    embedder = PausingEmbedder()
    with _live_server(path, tmp_path, monkeypatch, embedder):
        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(daemon.client_request, path, _request("slow alpha", "vector"))
            assert embedder.entered.wait(3)
            fast = pool.submit(daemon.client_request, path, _request("beta", "hybrid"))
            try:
                with pytest.raises(TimeoutError):
                    fast.result(timeout=0.2)
                assert embedder.max_active == 1
            finally:
                embedder.release.set()
            assert slow.result(timeout=5)["ok"]
            assert fast.result(timeout=5)["ok"]
    assert embedder.max_active == 1


def test_handler_closes_request_connections_even_after_search_error(
    tmp_path, monkeypatch,
):
    path = _seed_index(tmp_path)
    real_connect = db.connect
    created = []

    class TrackedConnection:
        def __init__(self, conn):
            self.conn = conn
            self.owner = threading.get_ident()
            self.closed = False
            created.append(self)

        def __getattr__(self, name):
            assert threading.get_ident() == self.owner
            return getattr(self.conn, name)

        def close(self):
            assert threading.get_ident() == self.owner
            self.conn.close()
            self.closed = True

    def tracked_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        return TrackedConnection(conn) if kwargs.get("read_only") else conn

    monkeypatch.setattr(db, "connect", tracked_connect)
    with _live_server(path, tmp_path, monkeypatch):
        requests = [_request("alpha", "lexical") for _ in range(12)]
        requests += [_request("alpha", "bad-mode") for _ in range(4)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda req: daemon.client_request(path, req), requests))
    assert len(created) == len(requests)
    assert all(conn.closed for conn in created)
    assert all(response is not None for response in responses)
    assert all(response["ok"] for response in responses[:12])
    assert all(not response["ok"] for response in responses[12:])


def test_duckdb_handler_serializes_entire_request(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    path = _seed_index(tmp_path, backend="duckdb")
    closed = []
    real_close = vector_store.DuckDBStore.close

    def tracked_close(self):
        closed.append(threading.get_ident())
        real_close(self)

    monkeypatch.setattr(vector_store.DuckDBStore, "close", tracked_close)
    embedder = PausingEmbedder()
    with _live_server(path, tmp_path, monkeypatch, embedder):
        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(daemon.client_request, path, _request("slow alpha", "vector"))
            assert embedder.entered.wait(3)
            try:
                fast = pool.submit(daemon.client_request, path, _request("beta", "lexical"))
                with pytest.raises(TimeoutError):
                    fast.result(timeout=0.2)
            finally:
                embedder.release.set()
            assert slow.result(timeout=5)["ok"]
            response = fast.result(timeout=5)
            assert response and response["ok"]
            assert response["results"]["results"][0]["path"] == "notes/beta.md"
    assert len(closed) == 2


def test_duckdb_first_pinned_after_daemon_start_still_serializes(
    tmp_path, monkeypatch,
):
    pytest.importorskip("duckdb")
    path = tmp_path / "index.db"
    source = tmp_path / "notes"
    source.mkdir()
    (source / "alpha.md").write_text("alpha deployment rolling restart\n")
    (source / "beta.md").write_text("beta recipe eggs pasta\n")
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", source)
    assert vector_store.open_store(conn, path).name == "none"
    conn.close()

    embedder = PausingEmbedder()
    with _live_server(path, tmp_path, monkeypatch, embedder):
        writer = db.connect(path)
        duck = vector_store.make_store("duckdb", writer, path)
        try:
            db.ensure_vectors(writer, duck, PROFILE.dim, profile_fingerprint(PROFILE),
                              PROFILE.model, PROFILE.name)
            stats = indexer.IndexStats()
            indexer.sync_collection(writer, duck, "notes", stats)
            indexer.embed_pending(writer, duck, HashEmbedder(PROFILE), stats)
            assert vector_store.open_store(writer, path).name == "duckdb"
        finally:
            duck.close()
            writer.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(daemon.client_request, path, _request("slow alpha", "vector"))
            assert embedder.entered.wait(3)
            try:
                fast = pool.submit(daemon.client_request, path, _request("beta", "lexical"))
                with pytest.raises(TimeoutError):
                    fast.result(timeout=0.2)
            finally:
                embedder.release.set()
            assert slow.result(timeout=5)["ok"]
            response = fast.result(timeout=5)
            assert response and response["ok"]
            assert response["results"]["results"][0]["path"] == "notes/beta.md"


def test_first_index_with_different_profile_rejects_stale_daemon_vectors(
    tmp_path, monkeypatch,
):
    old_profile = EmbedProfile("old-2d", "old-fake", 2)
    new_profile = EmbedProfile("new-2d", "new-fake", 2)

    class DistinctEmbedder:
        def __init__(self, profile):
            self.profile = profile

        def embed_queries(self, texts):
            vector = [1.0, 0.0] if self.profile == old_profile else [0.0, 1.0]
            return np.array([vector for _ in texts], dtype=np.float32)

        def embed_docs(self, texts):
            return np.array([[0.0, 1.0] if "alpha" in text else [1.0, 0.0]
                             for text in texts], dtype=np.float32)

    path = tmp_path / "index.db"
    source = tmp_path / "notes"
    source.mkdir()
    (source / "alpha.md").write_text("alpha target")
    (source / "beta.md").write_text("beta target")
    setup = db.connect(path)
    db.init_schema(setup)
    indexer.add_collection(setup, "notes", source)
    instance_id = db.get_meta(setup, "instance_id")
    assert vector_store.open_store(setup, path).name == "none"
    setup.close()

    request = _request("target", "vector")
    with _live_server(path, tmp_path, monkeypatch, DistinctEmbedder(old_profile)):
        writer = db.connect(path)
        store = vector_store.make_store("sqlite-vec", writer, path)
        db.ensure_vectors(writer, store, 2, profile_fingerprint(new_profile),
                          new_profile.model, new_profile.name)
        stats = indexer.IndexStats()
        indexer.sync_collection(writer, store, "notes", stats)
        indexer.embed_pending(writer, store, DistinctEmbedder(new_profile), stats)
        assert db.get_meta(writer, "instance_id") == instance_id
        writer.close()

        cold_conn = db.connect(path, create=False)
        cold_store = vector_store.open_store(cold_conn, path)
        cold = daemon.run_search(cold_conn, cold_store, DistinctEmbedder(new_profile), request)
        cold_conn.close()
        assert [row["path"] for row in cold["results"]] == [
            "notes/alpha.md", "notes/beta.md"]

        for mode in ("vector", "hybrid"):
            response = daemon.client_request(path, _request("target", mode))
            assert response and not response["ok"]
            assert "embedding profile" in response["error"]
            assert "new-2d" in response["error"]
            assert "old-2d" in response["error"]
            assert "restart fidx serve" in response["error"]
        lexical = daemon.client_request(path, _request("target", "lexical"))
        assert lexical and lexical["ok"]

    with _live_server(path, tmp_path, monkeypatch, DistinctEmbedder(new_profile)):
        response = daemon.client_request(path, request)
        assert response and response["ok"]
        assert response["instance_id"] == instance_id
        assert response["results"] == cold
