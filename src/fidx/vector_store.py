"""Pluggable vector backends.

Two implementations, selected at first embed and pinned in index meta:

- ``sqlite-vec`` (default): a vec0 virtual table inside the index file.
  Exact brute-force cosine KNN — zero extra files, zero recall loss.
- ``duckdb``: a DuckDB VSS/HNSW sidecar next to the index file
  (``<db>.vectors.duckdb``). ANN — faster KNN on very large corpora,
  approximate recall, extra storage.

The store only maps chunk_id <-> embedding. Documents, chunks and FTS stay
in SQLite regardless of backend; the sidecar is rebuildable from them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

BACKENDS = ("sqlite-vec", "duckdb")


def sidecar_path(db_path: Path) -> Path:
    return db_path.with_suffix(".vectors.duckdb")


class NullStore:
    """Pre-embed placeholder: no vectors exist yet, vector search is empty."""

    name = "none"

    def ensure(self, dim: int) -> None:  # pragma: no cover - never called
        raise RuntimeError("no vector backend pinned; run `fidx index`")

    def add(self, chunk_ids: list[int], vecs: np.ndarray) -> None:
        raise RuntimeError("no vector backend pinned; run `fidx index`")

    def delete(self, chunk_ids: list[int]) -> None:
        pass

    def knn(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        return []

    def all_ids(self) -> set[int]:
        return set()

    def count(self) -> int:
        return 0


class SqliteVecStore:
    name = "sqlite-vec"

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def ensure(self, dim: int) -> None:
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}] distance_metric=cosine
            )
            """
        )
        self.conn.commit()

    def _exists(self) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'vectors'").fetchone())

    def add(self, chunk_ids: list[int], vecs: np.ndarray) -> None:
        import sqlite_vec

        self.conn.executemany(
            "INSERT INTO vectors(chunk_id, embedding) VALUES (?, ?)",
            [(cid, sqlite_vec.serialize_float32(v)) for cid, v in zip(chunk_ids, vecs)],
        )
        self.conn.commit()

    def delete(self, chunk_ids: list[int]) -> None:
        if not self._exists() or not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        self.conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({placeholders})", chunk_ids)

    def knn(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        import sqlite_vec

        if not self._exists():
            return []
        rows = self.conn.execute(
            "SELECT chunk_id, distance FROM vectors WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(qvec), k),
        ).fetchall()
        return [(r["chunk_id"], r["distance"]) for r in rows]

    def all_ids(self) -> set[int]:
        if not self._exists():
            return set()
        return {r["chunk_id"] for r in self.conn.execute("SELECT chunk_id FROM vectors")}

    def count(self) -> int:
        if not self._exists():
            return 0
        return self.conn.execute("SELECT count(*) AS n FROM vectors").fetchone()["n"]


class DuckDBStore:
    name = "duckdb"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._conn = None
        self._dim: int | None = None

    def _connect(self):
        if self._conn is None:
            try:
                import duckdb
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "the duckdb backend requires the 'duckdb' extra: pip install fidx[duckdb]"
                ) from exc
            self._conn = duckdb.connect(str(self.path))
            self._conn.execute("INSTALL vss; LOAD vss")
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")
            row = self._conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'vectors'"
            ).fetchone()
            if row[0]:
                self._dim = self._conn.execute(
                    "SELECT len(embedding) FROM vectors LIMIT 1"
                ).fetchone()
                self._dim = self._dim[0] if self._dim else None
        return self._conn

    def ensure(self, dim: int) -> None:
        conn = self._connect()
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS vectors (chunk_id BIGINT PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS vectors_hnsw ON vectors "
            "USING HNSW (embedding) WITH (metric = 'cosine')"
        )
        self._dim = dim

    def add(self, chunk_ids: list[int], vecs: np.ndarray) -> None:
        conn = self._connect()
        conn.executemany(
            "INSERT INTO vectors VALUES (?, ?)",
            [(cid, [float(x) for x in v]) for cid, v in zip(chunk_ids, vecs)],
        )

    def delete(self, chunk_ids: list[int]) -> None:
        if not chunk_ids or not self.path.exists():
            return
        conn = self._connect()
        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({placeholders})", chunk_ids)

    def knn(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.path.exists():
            return []
        conn = self._connect()
        dim = self._dim or len(qvec)
        rows = conn.execute(
            f"SELECT chunk_id, array_cosine_distance(embedding, CAST(? AS FLOAT[{dim}])) AS d "
            "FROM vectors ORDER BY d LIMIT ?",
            ([float(x) for x in qvec], k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def all_ids(self) -> set[int]:
        if not self.path.exists():
            return set()
        return {int(r[0]) for r in self._connect().execute("SELECT chunk_id FROM vectors").fetchall()}

    def count(self) -> int:
        if not self.path.exists():
            return 0
        return int(self._connect().execute("SELECT count(*) FROM vectors").fetchone()[0])


def make_store(backend: str, conn: sqlite3.Connection, db_path: Path):
    if backend == "sqlite-vec":
        return SqliteVecStore(conn)
    if backend == "duckdb":
        return DuckDBStore(sidecar_path(db_path))
    raise ValueError(f"unknown vector backend {backend!r}; known: {', '.join(BACKENDS)}")


def open_store(conn: sqlite3.Connection, db_path: Path):
    """Store for an existing index: pinned backend, or NullStore pre-embed."""
    from . import db as dbmod

    backend = dbmod.get_meta(conn, "vector_backend")
    if backend is None:
        # Indexes created before backends existed used sqlite-vec implicitly.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'vectors'").fetchone():
            return SqliteVecStore(conn)
        return NullStore()
    return make_store(backend, conn, db_path)
