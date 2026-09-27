# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Indexing: scan collections, upsert documents, chunk and embed.

Incrementality contract: a document is re-embedded only when its content hash
or the embedding fingerprint changes. Deleted files are hard-deleted along
with their chunks and vectors.
"""

from __future__ import annotations

from bisect import bisect_left
from fnmatch import fnmatchcase
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_text, extract_title
from .config import DEFAULT_GLOBS
from .db import has_chunk_line_columns
from .embedder import Embedder

EMBED_BATCH_CHUNKS = 256


def _newlines(body: str) -> list[int]:
    return [i for i, char in enumerate(body) if char == "\n"]


def _chunk_lines(newlines: list[int], pos: int, length: int) -> tuple[int, int]:
    """Map a nonempty [pos, pos+length) span to inclusive, 1-based lines."""
    return 1 + bisect_left(newlines, pos), 1 + bisect_left(newlines, pos + length - 1)


def _backfill_chunk_lines(conn: sqlite3.Connection, collection: str) -> None:
    """Fill old chunks only when their collection is next indexed."""
    if not has_chunk_line_columns(conn):
        return
    doc_ids = [row["id"] for row in conn.execute(
        "SELECT DISTINCT d.id FROM documents d JOIN chunks c ON c.doc_id = d.id "
        "WHERE d.collection = ? AND (c.line_start IS NULL OR c.line_end IS NULL)",
        (collection,),
    )]
    for doc_id in doc_ids:
        body = conn.execute("SELECT body FROM documents WHERE id = ?", (doc_id,)).fetchone()["body"]
        newlines = _newlines(body)
        for chunk in conn.execute(
            "SELECT id, pos, length FROM chunks WHERE doc_id = ? "
            "AND (line_start IS NULL OR line_end IS NULL)", (doc_id,),
        ).fetchall():
            if chunk["length"] > 0:
                line_start, line_end = _chunk_lines(newlines, chunk["pos"], chunk["length"])
                conn.execute("UPDATE chunks SET line_start = ?, line_end = ? WHERE id = ?",
                             (line_start, line_end, chunk["id"]))


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    embedded_chunks: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CollectionGlobs:
    include: list[str]
    exclude: list[str]


def _glob_list(value, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if not isinstance(value, list) or any(not isinstance(pattern, str) for pattern in value):
        raise ValueError("collection globs must be lists of strings")
    return list(value)


def _decode_collection_globs(value) -> CollectionGlobs:
    if isinstance(value, list):
        include = _glob_list(value, list(DEFAULT_GLOBS)) or list(DEFAULT_GLOBS)
        return CollectionGlobs(include=include, exclude=[])
    if isinstance(value, dict):
        include = _glob_list(value.get("include"), list(DEFAULT_GLOBS)) or list(DEFAULT_GLOBS)
        exclude = _glob_list(value.get("exclude"), [])
        return CollectionGlobs(include=include, exclude=exclude)
    raise ValueError("collection globs must be a JSON list or object")


def _encode_collection_globs(globs: list[str] | None, exclude_globs: list[str] | None) -> str:
    return json.dumps({
        "include": list(globs or DEFAULT_GLOBS),
        "exclude": list(exclude_globs or []),
    })


def add_collection(
    conn: sqlite3.Connection,
    name: str,
    root: Path,
    globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"collection root {root} is not a directory")
    conn.execute(
        "INSERT INTO collections(name, root, globs) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET root = excluded.root, globs = excluded.globs",
        (name, str(root), _encode_collection_globs(globs, exclude_globs)),
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


def _normalize_glob_path(value: str) -> str:
    return value.replace("\\", "/")


def _path_glob_matches(relpath: str, pattern: str) -> bool:
    rel_parts = tuple(part for part in _normalize_glob_path(relpath).split("/") if part)
    pattern_parts = tuple(part for part in _normalize_glob_path(pattern).split("/") if part)

    def match(rel_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return rel_index == len(rel_parts)
        current = pattern_parts[pattern_index]
        if current == "**":
            if match(rel_index, pattern_index + 1):
                return True
            return rel_index < len(rel_parts) and match(rel_index + 1, pattern_index)
        return (
            rel_index < len(rel_parts)
            and fnmatchcase(rel_parts[rel_index], current)
            and match(rel_index + 1, pattern_index + 1)
        )

    return match(0, 0)


def _matches_any_glob(relpath: str, globs: list[str]) -> bool:
    return any(_path_glob_matches(relpath, pattern) for pattern in globs)


def _scan_files(root: Path, globs: list[str], exclude_globs: list[str] | None = None) -> dict[str, Path]:
    files: dict[str, Path] = {}
    excludes = exclude_globs or []
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                if not _matches_any_glob(rel, excludes):
                    files[rel] = path
    return files


def sync_collection(conn: sqlite3.Connection, store, name: str, stats: IndexStats) -> None:
    """Sync documents table (and FTS, via triggers) with the filesystem."""
    row = conn.execute("SELECT root, globs FROM collections WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"unknown collection {name!r}")
    root = Path(row["root"])
    globs = _decode_collection_globs(json.loads(row["globs"]))

    on_disk = _scan_files(root, globs.include, globs.exclude)
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
    _backfill_chunk_lines(conn, name)
    conn.commit()


def embed_pending(conn: sqlite3.Connection, store, embedder: Embedder, stats: IndexStats,
                  progress: bool = False) -> None:
    """Chunk and embed every document that has no chunks yet."""
    line_columns = has_chunk_line_columns(conn)
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
        newlines = _newlines(doc["body"]) if line_columns else []
        for chunk in chunk_text(doc["body"]):
            if line_columns:
                line_start, line_end = _chunk_lines(newlines, chunk.pos, chunk.length)
                cur = conn.execute(
                    "INSERT INTO chunks(doc_id, seq, pos, length, line_start, line_end) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, chunk.seq, chunk.pos, chunk.length, line_start, line_end),
                )
            else:
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
