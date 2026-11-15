# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Bounded, daemon-owned exact cosine search over a SQLite snapshot."""
from contextlib import contextmanager
import math
import os
import sqlite3
import threading

import numpy as np
from threadpoolctl import threadpool_limits

from . import search_limits


class VectorCache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.observer = None
        self.identity = None
        self.key = None
        self.matrix = self.ids = self.norms = None
        self.backend = 'sqlite-vec'
        try:
            mb = float(os.environ.get('FIDX_VECTOR_CACHE_MB', '1024'))
            self.cap = int(mb * 1024 * 1024) if math.isfinite(mb) and mb >= 0 else 0
        except (ValueError, OverflowError):
            self.cap = 0

    def close(self):
        if self.observer is not None:
            self.observer.close()
        self.observer = None
        self.matrix = self.ids = self.norms = None

    def _version(self):
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or self.observer is None:
            self.close()
            self.identity = identity
            self.key = None
            self.observer = sqlite3.connect(self.path.absolute().as_uri() + '?mode=ro',
                                           uri=True, check_same_thread=False)
        return identity, self.observer.execute('PRAGMA data_version').fetchone()[0]

    @contextmanager
    def snapshot(self, conn, fallback, *, identity=None, path_ordered=False):
        # Bracket snapshot establishment with a persistent observer's version.
        # Request-local data_version values cannot be compared across connections.
        with search_limits.resource(self.lock):
            before = self._version()
            owned = not conn.in_transaction
            if owned:
                conn.execute('BEGIN')
            conn.execute('SELECT value FROM meta LIMIT 1').fetchone()
            after = self._version()
            # An existing transaction may predate both observations.
            key = before if owned and before == after and (identity is None or identity == before[0]) else None
        try:
            yield _SnapshotStore(self, conn, fallback, key, path_ordered)
        finally:
            if owned:
                conn.rollback()

    def _load(self, conn, key):
        # Discard the previous generation before allocating its replacement.
        self.matrix = self.ids = self.norms = None
        self.backend = 'sqlite-vec'
        self.key = key
        count, size = conn.execute(
            'SELECT count(*), length(embedding) FROM vectors').fetchone()
        if not count or not size or count * size > self.cap:
            return
        matrix = np.empty((count, size // 4), dtype=np.float32)
        ids = np.empty(count, dtype=np.int64)
        # Stream blobs, avoiding a second matrix-sized list of Python bytes.
        for i, row in enumerate(conn.execute('SELECT chunk_id, embedding FROM vectors')):
            if i % 1024 == 0:
                search_limits.remaining()
            ids[i] = row[0]
            matrix[i] = np.frombuffer(row[1], dtype=np.float32)
        norms = np.sqrt(np.einsum('ij,ij->i', matrix, matrix))
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            return
        self.matrix, self.ids, self.norms = matrix, ids, norms
        self.backend = 'numpy-exact'

    def knn(self, conn, fallback, key, query, k, path_ordered=False):
        if key is None:
            return fallback.knn(query, k)
        with search_limits.resource(self.lock):
            try:
                if key != self.key:
                    try:
                        self._load(conn, key)
                    except BaseException:
                        self.key = None
                        raise
                if self.matrix is None:
                    return fallback.knn(query, k)
                with threadpool_limits(limits=1, user_api='blas'):
                    norm = np.linalg.norm(query)
                    if not np.isfinite(norm) or norm == 0:
                        return fallback.knn(query, k)
                    distances = 1 - (self.matrix @ query) / (self.norms * norm)
                n = min(k + 1, len(distances))
                if n <= 0:
                    return []
                chosen = np.argpartition(distances, n - 1)[:n]
                chosen = chosen[np.argsort(distances[chosen], kind='stable')]
                # sqlite-vec's float accumulation and heap order govern ties.
                # Delegate ambiguous neighbours (including the cutoff) rather
                # than impose a different order on equal/near-equal distances.
                near = np.diff(distances[chosen]) < 2e-6
                if np.any(near):
                    if not path_ordered or (len(chosen) > k and near[k - 1]):
                        return fallback.knn(query, k)
                    # Internal ties are ordered by collection/path/sequence in
                    # search_vector, independent of the KNN heap order. Refine
                    # their distances with SQLite's arithmetic before that sort.
                    import sqlite_vec
                    positions = np.unique(np.concatenate((np.flatnonzero(near), np.flatnonzero(near) + 1)))
                    for pos in positions:
                        i = chosen[pos]
                        distances[i] = conn.execute(
                            'SELECT vec_distance_cosine(embedding, ?) FROM vectors WHERE chunk_id=?',
                            (sqlite_vec.serialize_float32(query), int(self.ids[i]))).fetchone()[0]
                    chosen = chosen[np.argsort(distances[chosen], kind='stable')]
                return [(int(self.ids[i]), float(distances[i])) for i in chosen[:k]]
            except (MemoryError, ValueError, sqlite3.Error):
                self.matrix = self.ids = self.norms = None
                self.backend = 'sqlite-vec'
                return fallback.knn(query, k)


class _SnapshotStore:
    # Preserve search_vector's exact-backend tie-cohort handling and cap.
    name = 'sqlite-vec'

    def __init__(self, cache, conn, fallback, key, path_ordered):
        self.cache, self.conn, self.fallback, self.key = cache, conn, fallback, key
        self.path_ordered = path_ordered
        self.KNN_MAX_K = fallback.KNN_MAX_K

    def knn(self, query, k):
        return self.cache.knn(self.conn, self.fallback, self.key, query, k, self.path_ordered)

    def scan(self, query):
        return self.fallback.scan(query)
