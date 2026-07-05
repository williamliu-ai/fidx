# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from fidx import indexer


def index_all(conn, store, embedder, root, name="notes"):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, name, root)
    indexer.sync_collection(conn, store, name, stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return stats


def counts(conn, store):
    docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
    chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return docs, chunks, store.count()


def test_initial_index(conn, store, embedder, corpus_dir):
    stats = index_all(conn, store, embedder, corpus_dir)
    assert stats.added == 3 and stats.removed == 0
    docs, chunks, vectors = counts(conn, store)
    assert docs == 3
    assert chunks == vectors > 0


def test_incremental_noop(conn, store, embedder, corpus_dir):
    index_all(conn, store, embedder, corpus_dir)
    stats = indexer.IndexStats()
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    assert stats.added == stats.updated == stats.removed == 0
    assert stats.unchanged == 3
    assert stats.embedded_chunks == 0


def test_update_and_delete(conn, store, embedder, corpus_dir):
    index_all(conn, store, embedder, corpus_dir)
    (corpus_dir / "auth.md").write_text("# Authentication\n\nNow with passkeys.")
    (corpus_dir / "recipes.md").unlink()

    stats = indexer.IndexStats()
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    assert stats.updated == 1 and stats.removed == 1

    docs, chunks, vectors = counts(conn, store)
    assert docs == 2
    assert chunks == vectors
    body = conn.execute("SELECT body FROM documents WHERE relpath='auth.md'").fetchone()["body"]
    assert "passkeys" in body


def test_remove_collection_cleans_everything(conn, store, embedder, corpus_dir):
    index_all(conn, store, embedder, corpus_dir)
    indexer.remove_collection(conn, store, "notes")
    assert counts(conn, store) == (0, 0, 0)


def test_remove_collection_before_any_embed(tmp_path, corpus_dir):
    # A db that has never run `index` has no vector store; remove must not crash.
    from fidx import db as dbmod
    from fidx.vector_store import NullStore

    conn = dbmod.connect(tmp_path / "fresh.db")
    dbmod.init_schema(conn)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, NullStore(), "notes", stats)
    indexer.remove_collection(conn, NullStore(), "notes")
    assert conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"] == 0


def test_repair_orphan_chunks(conn, store, embedder, corpus_dir):
    # Simulate an interrupted embed: chunks committed, vectors missing.
    index_all(conn, store, embedder, corpus_dir)
    victim = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()["id"]
    store.delete([victim])
    conn.commit()
    stats = indexer.IndexStats()
    indexer.embed_pending(conn, store, embedder, stats)
    docs, chunks, vectors = counts(conn, store)
    assert chunks == vectors
    assert stats.embedded_chunks == 1
