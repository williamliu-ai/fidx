# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""SQLite storage: documents + FTS5 (BM25) + sqlite-vec (vectors), one file.

Schema invariants:
- `documents.body` is the only copy of file content; FTS5 is an external-content
  table over it and chunks store (pos, length) offsets, never text.
- `vectors` rows are keyed by chunk id; every chunk of an active document with
  embeddings has exactly one row. Vector dim is fixed at index creation by the
  embedding profile (meta key `embed_dim`).
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

# NOTE: `sqlite_vec` is imported lazily inside connect(), not at module load, so
# importing fidx (and running `fidx doctor` / `fidx --help`) never crashes on a
# host with a missing or wrong-arch sqlite-vec wheel. diagnostics.py probes it
# under guard instead.

SCHEMA_VERSION = 2


def has_chunk_line_columns(conn: sqlite3.Connection) -> bool:
    """Old read-only indexes can be searched before their schema is migrated."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    return {"line_start", "line_end"} <= columns


def sqlite_module():
    """Return a sqlite3-compatible module that supports loadable extensions.

    stdlib `sqlite3` is preferred. On hosts where it was built without extension
    loading (notably macOS system Python, compiled with
    SQLITE_OMIT_LOAD_EXTENSION), fall back to `pysqlite3` if importable (a wheel
    exists for linux-x86_64; elsewhere it needs a compiler). connect() raises a
    clear, doctor-pointing error if neither can load extensions.
    """
    if hasattr(sqlite3.Connection, "enable_load_extension"):
        return sqlite3
    try:
        import pysqlite3  # type: ignore
        return pysqlite3
    except ImportError:
        return sqlite3


def connect(db_path: Path, *, create: bool = True,
            read_only: bool = False) -> sqlite3.Connection:
    if create and read_only:
        raise ValueError("read_only requires create=False")
    sqlite = sqlite_module()
    db_path = Path(db_path)
    if create:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite.connect(str(db_path))
    else:
        from .errors import IndexInvalid, IndexMissing

        if not db_path.exists():
            raise IndexMissing(f"index {db_path} does not exist")
        if not db_path.is_file():
            raise IndexInvalid(f"index {db_path} is not a regular database file")
        with db_path.open("rb") as source:
            if source.read(16) != b"SQLite format 3\0":
                raise IndexInvalid(f"index {db_path} is not a SQLite database")
        # mode=rw permits the intentional writes in `collection remove` and
        # `calibrate --store`, but never creates a missing database path.
        uri = db_path.absolute().as_uri() + ("?mode=ro" if read_only else "?mode=rw")
        try:
            conn = sqlite.connect(uri, uri=True)
        except sqlite.DatabaseError as exc:
            raise IndexInvalid(f"index {db_path} cannot be opened: {exc}") from exc
    conn.row_factory = sqlite.Row
    if not hasattr(conn, "enable_load_extension"):
        raise RuntimeError(
            "this Python's sqlite3 was built without loadable-extension support "
            "(common on macOS system Python). Install fidx with `uv tool install` "
            "or use Homebrew Python. Run `fidx doctor` for a full diagnosis."
        )
    import sqlite_vec  # lazy: see module note
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout=5000")
    if read_only:
        conn.execute("PRAGMA query_only=ON")
    if create:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def validate_index(conn: sqlite3.Connection) -> None:
    """Reject foreign or incomplete indexes without changing their schema."""
    from .errors import IndexInvalid

    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            raise IndexInvalid("index has no schema_version metadata")
        try:
            version = int(row["value"])
        except (TypeError, ValueError) as exc:
            raise IndexInvalid("index has invalid schema_version metadata") from exc
        if version not in {1, SCHEMA_VERSION}:
            raise IndexInvalid(f"index schema_version {version} is unsupported")
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        required = {"meta", "collections", "documents", "documents_fts", "chunks"}
        missing = sorted(required - tables)
        if missing:
            raise IndexInvalid(f"index is missing required tables: {', '.join(missing)}")
    except sqlite3.DatabaseError as exc:
        raise IndexInvalid(f"index schema is invalid: {exc}") from exc


def init_schema(conn: sqlite3.Connection) -> None:
    fresh = not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collections (
            name TEXT PRIMARY KEY,
            root TEXT NOT NULL,
            globs TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            collection TEXT NOT NULL,
            relpath TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL,
            hash TEXT NOT NULL,
            mtime REAL NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            UNIQUE(collection, relpath)
        );
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);

        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            title, body, relpath,
            content='documents', content_rowid='id',
            tokenize='porter unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, title, body, relpath)
            VALUES (new.id, new.title, new.body, new.relpath);
        END;
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, body, relpath)
            VALUES ('delete', old.id, old.title, old.body, old.relpath);
        END;
        CREATE TRIGGER IF NOT EXISTS documents_au
        AFTER UPDATE OF title, body, relpath ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, body, relpath)
            VALUES ('delete', old.id, old.title, old.body, old.relpath);
            INSERT INTO documents_fts(rowid, title, body, relpath)
            VALUES (new.id, new.title, new.body, new.relpath);
        END;

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            pos INTEGER NOT NULL,
            length INTEGER NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            UNIQUE(doc_id, seq)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """
    )
    columns = {row["name"] for row in cur.execute("PRAGMA table_info(chunks)")}
    for column in ("line_start", "line_end"):
        if column not in columns:
            cur.execute(f"ALTER TABLE chunks ADD COLUMN {column} INTEGER")
    cur.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),))
    cur.execute("UPDATE meta SET value = ? WHERE key = 'schema_version' AND CAST(value AS INTEGER) < ?",
                (str(SCHEMA_VERSION), SCHEMA_VERSION))
    if fresh:
        cur.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('instance_id', ?)",
                    (uuid.uuid4().hex,))
    conn.commit()


def ensure_vectors(conn: sqlite3.Connection, store, embed_dim: int, embed_fingerprint: str,
                   embed_model: str, profile_name: str) -> None:
    """Create the vector store and pin the embedding profile + backend.

    Both are pinned by the first embedding run, not by whichever command
    happens to create the database; a later mismatch is an error, never a
    silent dim or backend conflict.
    """
    stored = get_meta(conn, "embed_profile")
    if stored is not None and stored != profile_name:
        raise RuntimeError(
            f"index is pinned to embedding profile {stored!r}; "
            f"re-index into a fresh database to use {profile_name!r}"
        )
    stored_backend = get_meta(conn, "vector_backend")
    if stored_backend is not None and stored_backend != store.name:
        raise RuntimeError(
            f"index is pinned to vector backend {stored_backend!r}; "
            f"re-index into a fresh database to use {store.name!r}"
        )
    store.ensure(embed_dim)
    for key, value in (
        ("embed_profile", profile_name),
        ("embed_dim", str(embed_dim)),
        ("embed_fingerprint", embed_fingerprint),
        ("embed_model", embed_model),
        ("vector_backend", store.name),
    ):
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def ensure_instance_id(conn: sqlite3.Connection) -> str:
    """Backfill identity once when indexing a legacy database."""
    instance_id = get_meta(conn, "instance_id")
    if instance_id is None:
        instance_id = uuid.uuid4().hex
        set_meta(conn, "instance_id", instance_id)
    return instance_id
