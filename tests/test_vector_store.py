"""Backend parity: sqlite-vec and the DuckDB HNSW sidecar must behave alike."""

import numpy as np
import pytest

from fidx import db as dbmod
from fidx import indexer, search
from fidx.config import profile_fingerprint
from fidx.embedder import HashEmbedder
from fidx.vector_store import DuckDBStore, SqliteVecStore, sidecar_path

from conftest import TEST_PROFILE

duckdb = pytest.importorskip("duckdb")


@pytest.fixture()
def duck_store(tmp_path, conn):
    store = DuckDBStore(sidecar_path(tmp_path / "index.db"))
    dbmod.ensure_vectors(conn, store, TEST_PROFILE.dim, profile_fingerprint(TEST_PROFILE),
                         TEST_PROFILE.model, TEST_PROFILE.name)
    return store


def test_duckdb_roundtrip(duck_store):
    vecs = np.eye(3, TEST_PROFILE.dim, dtype=np.float32)
    duck_store.add([10, 11, 12], vecs)
    assert duck_store.count() == 3
    hits = duck_store.knn(vecs[1], k=2)
    assert hits[0][0] == 11 and hits[0][1] == pytest.approx(0.0, abs=1e-5)
    duck_store.delete([11])
    assert duck_store.count() == 2
    assert duck_store.all_ids() == {10, 12}


def test_duckdb_backend_end_to_end(conn, duck_store, corpus_dir):
    embedder = HashEmbedder(TEST_PROFILE)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, duck_store, "notes", stats)
    indexer.embed_pending(conn, duck_store, embedder, stats)
    assert duck_store.count() > 0

    results = search.search_vector(conn, duck_store, embedder, "users log in with OAuth tokens")
    assert results and results[0].relpath == "auth.md"

    hybrid = search.search_hybrid(conn, duck_store, embedder, "carbonara recipe eggs")
    assert hybrid and hybrid[0].relpath == "recipes.md"

    indexer.remove_collection(conn, duck_store, "notes")
    assert duck_store.count() == 0


def test_backend_mismatch_is_hard_error(conn, store, tmp_path):
    duck = DuckDBStore(sidecar_path(tmp_path / "index.db"))
    with pytest.raises(RuntimeError, match="pinned to vector backend"):
        dbmod.ensure_vectors(conn, duck, TEST_PROFILE.dim, profile_fingerprint(TEST_PROFILE),
                             TEST_PROFILE.model, TEST_PROFILE.name)
