# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Public CLI error statuses, exit codes, and machine-readable envelope."""

from __future__ import annotations

import sqlite3
from enum import IntEnum

from .embedder import ModelUnavailableError


class ExitCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    USAGE = 2
    INDEX = 3
    DB_LOCKED = 4
    MODEL_MISSING = 5
    AMBIGUOUS_REFERENCE = 6


STATUS_CODES = {
    "error": ExitCode.ERROR,
    "index_missing": ExitCode.INDEX,
    "index_invalid": ExitCode.INDEX,
    "db_locked": ExitCode.DB_LOCKED,
    "model_missing": ExitCode.MODEL_MISSING,
    "ambiguous_reference": ExitCode.AMBIGUOUS_REFERENCE,
}


class FidxError(Exception):
    status = "error"

    def __init__(self, message: str, *, next_actions: list[dict] | None = None):
        super().__init__(message)
        self.next_actions = next_actions or []


class IndexMissing(FidxError):
    status = "index_missing"


class IndexInvalid(FidxError):
    status = "index_invalid"


class DatabaseLocked(FidxError):
    status = "db_locked"


class AmbiguousReference(FidxError):
    status = "ambiguous_reference"

    def __init__(self, message: str, *, candidates: list[dict],
                 next_actions: list[dict] | None = None):
        super().__init__(message, next_actions=next_actions)
        self.candidates = candidates[:10]


def one_line(exc: Exception) -> str:
    """Keep human diagnostics to one line without losing the exception message."""
    return " ".join(str(exc).splitlines()).strip() or type(exc).__name__


def classify(exc: Exception) -> tuple[str, int, list[dict], list[dict] | None]:
    if isinstance(exc, FidxError):
        candidates = exc.candidates if isinstance(exc, AmbiguousReference) else None
        return exc.status, int(STATUS_CODES[exc.status]), exc.next_actions, candidates
    if isinstance(exc, ModelUnavailableError):
        action = {
            "intent": "fetch_model",
            "reason": "Download the selected embedding model before searching offline.",
            "command": ["fidx", "models", "fetch", "embed", "--profile", exc.profile_name],
        }
        return "model_missing", int(ExitCode.MODEL_MISSING), [action], None
    if isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database table is locked" in str(exc).lower()
    ):
        return "db_locked", int(ExitCode.DB_LOCKED), [], None
    return "error", int(ExitCode.ERROR), [], None


def envelope(exc: Exception) -> tuple[dict, int]:
    status, code, next_actions, candidates = classify(exc)
    payload = {
        "schema": "fidx.error.v1",
        "status": status,
        "message": one_line(exc),
        "next_actions": next_actions,
    }
    if candidates is not None:
        payload["candidates"] = candidates
    return payload, code
