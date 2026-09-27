# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""A bounded, sample-only freshness signal for an existing index."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import stat

from .db import get_meta

SAMPLE_SIZE = 20


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def completed_at() -> str:
    """Timestamp recorded after a successful index command."""
    return _utc_now().isoformat()


def index_summary(conn: sqlite3.Connection) -> dict | None:
    """Return elapsed age and a bounded mtime hint, or None for old indexes."""
    value = get_meta(conn, "last_index_completed_at")
    if value is None:
        return None
    try:
        completed = datetime.fromisoformat(value)
        if completed.tzinfo is None:
            return None
        age = max(0, int((_utc_now() - completed).total_seconds()))
    except (ValueError, OverflowError):
        return None

    summary: dict[str, int | bool] = {"age_seconds": age}
    rows = conn.execute(
        "SELECT d.collection, d.relpath, d.mtime, c.root "
        "FROM documents d LEFT JOIN collections c ON c.name = d.collection "
        "ORDER BY d.collection, d.relpath LIMIT ?", (SAMPLE_SIZE,),
    ).fetchall()
    for row in rows:
        if row["root"] is None:
            return summary
        path = Path(row["root"]) / row["relpath"]
        try:
            current = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            summary["stale_hint"] = True
            return summary
        except OSError:
            return summary
        if not stat.S_ISREG(current.st_mode) or current.st_mtime > row["mtime"]:
            summary["stale_hint"] = True
            return summary
    summary["stale_hint"] = False
    return summary
