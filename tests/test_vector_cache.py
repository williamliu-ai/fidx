# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import os
import numpy as np
import pytest

from fidx import db
from fidx.vector_cache import VectorCache
from fidx.vector_store import SqliteVecStore


def warm(cache, conn, store, query):
    with cache.snapshot(conn, store) as cached:
        cached.knn(query, 1)
    cache.loader.join(5)
    assert not cache.loader.is_alive()


def test_random_knn(conn, store, tmp_path):
    rng = np.random.default_rng(432)
    vectors = rng.normal(size=(180, 64)).astype(np.float32)
    store.add(list(range(1, 181)), vectors)
    cache = VectorCache(tmp_path / 'index.db')
    warm(cache, conn, store, vectors[0])
    for _ in range(50):
        query = rng.normal(size=64).astype(np.float32)
        with cache.snapshot(conn, store) as cached:
            actual = cached.knn(query, 25)
            expected = store.knn(query, 25)
            assert [r[0] for r in actual] == [r[0] for r in expected]
            np.testing.assert_allclose([r[1] for r in actual], [r[1] for r in expected], atol=1e-5)
    assert cache.backend == 'numpy-exact'
    cache.close()


def test_cap(conn, store, tmp_path, monkeypatch):
    store.add([1], np.ones((1, 64), dtype=np.float32))
    monkeypatch.setenv('FIDX_VECTOR_CACHE_MB', '0')
    cache = VectorCache(tmp_path / 'index.db')
    with cache.snapshot(conn, store) as cached:
        assert cached.knn(np.ones(64, dtype=np.float32), 1) == store.knn(np.ones(64, dtype=np.float32), 1)
    assert cache.backend == 'sqlite-vec'
    assert cache.matrix is None
    cache.close()


def test_external_reindex(conn, store, tmp_path):
    vector = np.ones((1, 64), dtype=np.float32)
    store.add([1], vector)
    cache = VectorCache(tmp_path / 'index.db')
    with cache.snapshot(conn, store) as cached:
        assert cached.knn(vector[0], 1)[0][0] == 1
    writer = db.connect(tmp_path / 'index.db', create=False)
    other = SqliteVecStore(writer)
    other.delete([1])
    other.add([2], vector)
    with cache.snapshot(conn, store) as cached:
        assert cached.knn(vector[0], 1)[0][0] == 2
    writer.close()
    cache.close()


def test_thread_pin(conn, store, tmp_path, monkeypatch):
    import fidx.vector_cache as module
    from contextlib import contextmanager
    calls = []
    @contextmanager
    def limits(**kwargs):
        calls.append(kwargs)
        yield
    monkeypatch.setattr(module, 'threadpool_limits', limits)
    store.add([1], np.ones((1, 64), dtype=np.float32))
    cache = VectorCache(tmp_path / 'index.db')
    warm(cache, conn, store, np.ones(64, dtype=np.float32))
    with cache.snapshot(conn, store) as cached:
        cached.knn(np.ones(64, dtype=np.float32), 1)
    assert calls and all(c == {'limits': 1, 'user_api': 'blas'} for c in calls)
    cache.close()


def test_ties_delegate(conn, store, tmp_path):
    store.add(list(range(1, 11)), np.ones((10, 64), dtype=np.float32))
    cache = VectorCache(tmp_path / 'index.db')
    with cache.snapshot(conn, store) as cached:
        assert cached.knn(np.ones(64, dtype=np.float32), 4) == store.knn(np.ones(64, dtype=np.float32), 4)
    cache.close()


def test_interleaved_snapshots_keep_their_own_generation(conn, store, tmp_path):
    old = np.ones((1, 64), dtype=np.float32)
    new = np.arange(1, 65, dtype=np.float32).reshape(1, -1)
    store.add([1], old)
    cache = VectorCache(tmp_path / 'index.db')
    reader = db.connect(tmp_path / 'index.db', create=False)
    reader_store = SqliteVecStore(reader)
    writer = db.connect(tmp_path / 'index.db', create=False)
    writer_store = SqliteVecStore(writer)
    try:
        with cache.snapshot(conn, store) as old_store:
            assert old_store.knn(old[0], 1)[0][0] == 1
            writer_store.delete([1])
            writer_store.add([2], new)
            with cache.snapshot(reader, reader_store) as new_store:
                assert new_store.knn(new[0], 1)[0][0] == 2
                assert old_store.knn(old[0], 1)[0][0] == 1
                assert new_store.knn(new[0], 1)[0][0] == 2
    finally:
        writer.close()
        reader.close()
        cache.close()


def test_lock_wait_respects_deadline(conn, store, tmp_path):
    import pytest
    from fidx import errors, search_limits
    cache = VectorCache(tmp_path / 'index.db')
    cache.lock.acquire()
    try:
        with pytest.raises(errors.SearchTimeout):
            with search_limits.search_budget(conn, {'timeout': .01}):
                with cache.snapshot(conn, store):
                    pass
    finally:
        cache.lock.release()
        cache.close()


def test_overflow_cap(tmp_path, monkeypatch):
    monkeypatch.setenv('FIDX_VECTOR_CACHE_MB', '1e308')
    cache = VectorCache(tmp_path / 'index.db')
    assert cache.cap == 0


def test_collection_filter_and_reindex(conn, store, embedder, corpus_dir, tmp_path):
    from fidx import indexer, search
    indexer.add_collection(conn, 'notes', corpus_dir)
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'one.md').write_text('OAuth authentication tokens')
    indexer.add_collection(conn, 'other', other)
    for collection in ['notes', 'other']:
        indexer.sync_collection(conn, store, collection, indexer.IndexStats())
    indexer.embed_pending(conn, store, embedder, indexer.IndexStats())
    cache = VectorCache(tmp_path / 'index.db')
    with cache.snapshot(conn, store) as cached:
        expected = search.search_vector(conn, store, embedder, 'authentication', ['other'], 10)
        actual = search.search_vector(conn, cached, embedder, 'authentication', ['other'], 10)
        assert [(r.collection, r.relpath) for r in actual] == [(r.collection, r.relpath) for r in expected]
        assert actual and {r.collection for r in actual} == {'other'}
    (other / 'one.md').unlink()
    (other / 'new.md').write_text('OAuth authentication tokens sessions')
    indexer.sync_collection(conn, store, 'other', indexer.IndexStats())
    indexer.embed_pending(conn, store, embedder, indexer.IndexStats())
    with cache.snapshot(conn, store) as cached:
        actual = search.search_vector(conn, cached, embedder, 'authentication', ['other'], 10)
        assert actual and {r.relpath for r in actual} == {'new.md'}
    cache.close()


def test_load_failure_falls_back(conn, store, tmp_path, monkeypatch):
    store.add([1], np.ones((1, 64), dtype=np.float32))
    cache = VectorCache(tmp_path / 'index.db')
    def fail(*args, **kwargs):
        raise MemoryError()
    monkeypatch.setattr(np, 'empty', fail)
    with cache.snapshot(conn, store) as cached:
        assert cached.knn(np.ones(64, dtype=np.float32), 1) == store.knn(np.ones(64, dtype=np.float32), 1)
    cache.loader.join(5)
    assert cache.matrix is None and cache.backend == 'sqlite-vec'
    cache.close()


@pytest.mark.skipif(os.name != "posix", reason="replacing an open database file is POSIX-only")
def test_replaced_database_does_not_cache_old_connection(conn, store, tmp_path):
    import os
    path = tmp_path / 'index.db'
    store.add([1], np.ones((1, 64), dtype=np.float32))
    original = path.stat()
    identity = (original.st_dev, original.st_ino)
    replacement = tmp_path / 'replacement.db'
    target = db.connect(replacement)
    conn.backup(target)
    replacement_store = SqliteVecStore(target)
    replacement_store.delete([1])
    replacement_store.add([2], np.ones((1, 64), dtype=np.float32))
    target.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    target.close()
    cache = VectorCache(path)
    os.replace(replacement, path)
    try:
        with cache.snapshot(conn, store, identity=identity) as cached:
            assert cached.key is None
            assert cached.knn(np.ones(64, dtype=np.float32), 1)[0][0] == 1
        assert cache.matrix is None
    finally:
        cache.close()


def test_slow_background_load_and_new_snapshot(conn, store, tmp_path, monkeypatch):
    import threading
    import time
    from fidx import search_limits
    vector = np.ones(64, dtype=np.float32)
    store.add([1], vector.reshape(1, -1))
    cache = VectorCache(tmp_path / 'index.db')
    entered, release = threading.Event(), threading.Event()
    original = cache._load
    attempts = []
    def slow(reader, key):
        attempts.append(key)
        entered.set()
        assert release.wait(5)
        return original(reader, key)
    monkeypatch.setattr(cache, '_load', slow)
    try:
        with cache.snapshot(conn, store) as cached:
            start = time.monotonic()
            with search_limits.search_budget(conn, {'timeout': .2}):
                assert cached.knn(vector, 1)[0][0] == 1
            assert time.monotonic() - start < .2
        assert entered.wait(2)
        writer = db.connect(tmp_path / 'index.db', create=False)
        try:
            other = SqliteVecStore(writer)
            other.delete([1])
            other.add([2], vector.reshape(1, -1))
            with search_limits.search_budget(conn, {'timeout': .2}):
                with cache.snapshot(conn, store) as cached:
                    assert cached.knn(vector, 1)[0][0] == 2
            assert len(attempts) == 1
            release.set()
            cache.loader.join(5)
            with cache.snapshot(conn, store) as cached:
                assert cached.knn(vector, 1)[0][0] == 2
            cache.loader.join(5)
            with cache.snapshot(conn, store) as cached:
                assert cached.key == cache.key
                assert cached.knn(vector, 1) == store.knn(vector, 1)
            assert cache.backend == 'numpy-exact'
        finally:
            writer.close()
    finally:
        release.set()
        cache.close()


@pytest.mark.parametrize('failure', ['memory', 'error', 'cap'])
def test_failed_background_load_does_not_retry(conn, store, tmp_path, monkeypatch, failure):
    vector = np.ones(64, dtype=np.float32)
    store.add([1], vector.reshape(1, -1))
    cache = VectorCache(tmp_path / 'index.db')
    attempts = []
    original = cache._load
    if failure == 'cap':
        cache.cap = 1
    def fail(reader, key):
        attempts.append(key)
        if failure == 'cap':
            return original(reader, key)
        if failure == 'memory':
            raise MemoryError()
        raise RuntimeError('load failed')
    monkeypatch.setattr(cache, '_load', fail)
    try:
        for _ in range(10):
            with cache.snapshot(conn, store) as cached:
                assert cached.knn(vector, 1) == store.knn(vector, 1)
            cache.loader.join(5)
        assert len(attempts) == 1
        assert cache.backend == 'sqlite-vec'
    finally:
        cache.close()


def test_internal_ties_refine_to_sqlite_distances(conn, store, tmp_path):
    vectors = np.zeros((3, 64), dtype=np.float32)
    vectors[0, 0] = vectors[1, 0] = vectors[2, 1] = 1
    store.add([2, 1, 3], vectors)
    cache = VectorCache(tmp_path / 'index.db')
    warm(cache, conn, store, vectors[0])
    query = vectors[0]
    with cache.snapshot(conn, store, path_ordered=True) as cached:
        actual = cached.knn(query, 2)
        expected = store.knn(query, 2)
        assert dict(actual) == dict(expected)
        assert cache.backend == 'numpy-exact'
    cache.close()


def test_internal_ties_keep_search_path_order(conn, store, embedder, tmp_path, monkeypatch):
    from fidx import indexer, search
    root = tmp_path / 'tied'
    root.mkdir()
    (root / 'z.md').write_text('shared authentication token')
    (root / 'a.md').write_text('shared authentication token')
    (root / 'other.md').write_text('gardening squash tomatoes')
    indexer.add_collection(conn, 'notes', root)
    indexer.sync_collection(conn, store, 'notes', indexer.IndexStats())
    indexer.embed_pending(conn, store, embedder, indexer.IndexStats())
    expected = search.search_vector(conn, store, embedder, 'shared authentication token', limit=10)
    cache = VectorCache(tmp_path / 'index.db')
    warm(cache, conn, store, embedder.embed_queries(['shared authentication token'])[0])
    def no_fallback(*args):
        raise AssertionError('internal ties should use refined cache results')
    monkeypatch.setattr(store, 'knn', no_fallback)
    with cache.snapshot(conn, store, path_ordered=True) as cached:
        actual = search.search_vector(conn, cached, embedder, 'shared authentication token', limit=10)
    assert [result.relpath for result in actual] == [result.relpath for result in expected]
    assert [result.relpath for result in actual][:2] == ['a.md', 'z.md']
    cache.close()


def test_failed_loader_records_its_own_newer_snapshot(conn, store, tmp_path, monkeypatch):
    import threading
    vector = np.ones(64, dtype=np.float32)
    store.add([1], vector.reshape(1, -1))
    cache = VectorCache(tmp_path / 'index.db')
    entered, release = threading.Event(), threading.Event()
    connect = db.connect
    attempts = []
    def delayed_connect(*args, **kwargs):
        if threading.current_thread().name == 'fidx-vector-loader':
            entered.set()
            assert release.wait(5)
        return connect(*args, **kwargs)
    def fail(reader, key):
        attempts.append(key)
        assert reader.execute('PRAGMA query_only').fetchone()[0] == 1
        assert reader is not conn
        raise MemoryError()
    monkeypatch.setattr(db, 'connect', delayed_connect)
    monkeypatch.setattr(cache, '_load', fail)
    try:
        with cache.snapshot(conn, store) as cached:
            cached.knn(vector, 1)
        assert entered.wait(2)
        writer = connect(tmp_path / 'index.db', create=False)
        try:
            SqliteVecStore(writer).add([2], vector.reshape(1, -1))
        finally:
            writer.close()
        release.set()
        cache.loader.join(5)
        for _ in range(5):
            with cache.snapshot(conn, store) as cached:
                assert cached.knn(vector, 1) == store.knn(vector, 1)
            cache.loader.join(5)
        assert len(attempts) == 1
    finally:
        release.set()
        cache.close()
