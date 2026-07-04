# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true

"""Indexing: scan collections, upsert documents, chunk and embed.

Incrementality contract: a document is re-embedded only when its content hash
or the embedding fingerprint changes. Deleted files are hard-deleted along
with their chunks and vectors.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_text, extract_title
from .config import DEFAULT_GLOBS
from .embedder import Embedder

EMBED_BATCH_CHUNKS = 256


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    embedded_chunks: int = 0
    errors: list[str] = field(default_factory=list)


def add_collection(conn: sqlite3.Connection, name: str, root: Path, globs: list[str] | None = None) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"collection root {root} is not a directory")
    conn.execute(
        "INSERT INTO collections(name, root, globs) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET root = excluded.root, globs = excluded.globs",
        (name, str(root), json.dumps(globs or DEFAULT_GLOBS)),
    )
    conn.commit()


def remove_collection(conn: sqlite3.Connection, store, name: str) -> None:
    doc_ids = [r["id"] for r in conn.execute("SELECT id FROM documents WHERE collection = ?", (name,))]
    _delete_documents(conn, store, doc_ids)
    conn.execute("DELETE FROM collections WHERE name = ?", (name,))
    conn.commit()


def _delete_documents(conn: sqlite3.Connection, store, doc_ids: list[int]) -> None:
    for doc_id in doc_ids:
        chunk_ids = [
            r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
        ]
        store.delete(chunk_ids)
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def _scan_files(root: Path, globs: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file():
                files[str(path.relative_to(root))] = path
    return files


def sync_collection(conn: sqlite3.Connection, store, name: str, stats: IndexStats) -> None:
    """Sync documents table (and FTS, via triggers) with the filesystem."""
    row = conn.execute("SELECT root, globs FROM collections WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"unknown collection {name!r}")
    root, globs = Path(row["root"]), json.loads(row["globs"])

    on_disk = _scan_files(root, globs)
    in_db = {
        r["relpath"]: r
        for r in conn.execute(
            "SELECT id, relpath, hash, mtime FROM documents WHERE collection = ?", (name,)
        )
    }

    stale_ids = [in_db[rel]["id"] for rel in in_db.keys() - on_disk.keys()]
    _delete_documents(conn, store, stale_ids)
    stats.removed += len(stale_ids)

    for rel, path in sorted(on_disk.items()):
        existing = in_db.get(rel)
        try:
            mtime = path.stat().st_mtime
            if existing is not None and existing["mtime"] == mtime:
                stats.unchanged += 1
                continue
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            stats.errors.append(f"{name}/{rel}: {exc}")
            continue
        digest = hashlib.sha256(body.encode()).hexdigest()
        title = extract_title(body, fallback=path.stem)
        if existing is None:
            conn.execute(
                "INSERT INTO documents(collection, relpath, title, body, hash, mtime) VALUES (?, ?, ?, ?, ?, ?)",
                (name, rel, title, body, digest, mtime),
            )
            stats.added += 1
        elif existing["hash"] != digest:
            _delete_documents(conn, store, [existing["id"]])
            conn.execute(
                "INSERT INTO documents(collection, relpath, title, body, hash, mtime) VALUES (?, ?, ?, ?, ?, ?)",
                (name, rel, title, body, digest, mtime),
            )
            stats.updated += 1
        else:
            conn.execute("UPDATE documents SET mtime = ? WHERE id = ?", (mtime, existing["id"]))
            stats.unchanged += 1
    conn.commit()


def embed_pending(conn: sqlite3.Connection, store, embedder: Embedder, stats: IndexStats,
                  progress: bool = False) -> None:
    """Chunk and embed every document that has no chunks yet."""
    # Ids only — bodies are fetched one document at a time so a 100k-doc
    # backlog doesn't materialize the whole corpus in memory.
    pending = [
        r["id"]
        for r in conn.execute(
            "SELECT d.id FROM documents d "
            "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.id) "
            "ORDER BY d.id"
        )
    ]

    batch_texts: list[str] = []
    batch_chunk_ids: list[int] = []

    def flush() -> None:
        if not batch_texts:
            return
        vecs = embedder.embed_docs(batch_texts)
        store.add(batch_chunk_ids, vecs)
        conn.commit()
        stats.embedded_chunks += len(batch_texts)
        batch_texts.clear()
        batch_chunk_ids.clear()

    for n, doc_id in enumerate(pending, 1):
        doc = conn.execute("SELECT body, title FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if doc is None:
            continue
        # Prepend the title so every chunk carries document identity.
        prefix = f"{doc['title']}\n" if doc["title"] else ""
        for chunk in chunk_text(doc["body"]):
            cur = conn.execute(
                "INSERT INTO chunks(doc_id, seq, pos, length) VALUES (?, ?, ?, ?)",
                (doc_id, chunk.seq, chunk.pos, chunk.length),
            )
            batch_texts.append(prefix + doc["body"][chunk.pos : chunk.pos + chunk.length])
            batch_chunk_ids.append(cur.lastrowid)
            if len(batch_texts) >= EMBED_BATCH_CHUNKS:
                flush()
        if progress and (n % 200 == 0 or n == len(pending)):
            print(f"  embedded {n}/{len(pending)} docs", file=sys.stderr)
    flush()

    # Repair pass: an interrupted earlier run can leave chunks whose vectors
    # were never committed; without this they would be skipped forever.
    chunk_rows = conn.execute(
        "SELECT c.id, c.doc_id, c.pos, c.length, d.title FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
    ).fetchall()
    embedded = store.all_ids()
    orphans = [r for r in chunk_rows if r["id"] not in embedded]
    for row in orphans:
        body = conn.execute("SELECT body FROM documents WHERE id = ?", (row["doc_id"],)).fetchone()["body"]
        prefix = f"{row['title']}\n" if row["title"] else ""
        batch_texts.append(prefix + body[row["pos"] : row["pos"] + row["length"]])
        batch_chunk_ids.append(row["id"])
        if len(batch_texts) >= EMBED_BATCH_CHUNKS:
            flush()
    flush()
