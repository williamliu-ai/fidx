# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Monotonic search budgets shared by SQL, embedding and resource waits."""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

from .errors import SearchTimeout

_DEADLINE = ContextVar('search_deadline', default=None)


def timeout_seconds(value=None):
    seconds = float(os.environ.get('FIDX_SEARCH_TIMEOUT', '10') if value is None else value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('search timeout must be a finite positive number of seconds')
    return seconds


def remaining():
    deadline = _DEADLINE.get()
    if deadline is None:
        return None
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise SearchTimeout('Search timed out; increase --timeout to allow more time.')
    return seconds


@contextmanager
def resource(lock):
    """Acquire one resource within budget; callers never nest resource locks."""
    seconds = remaining()
    acquired = lock.acquire() if seconds is None else lock.acquire(timeout=seconds)
    if not acquired:
        raise SearchTimeout('Search timed out waiting for a resource.')
    try:
        remaining()
        yield
        remaining()
    finally:
        lock.release()


@contextmanager
def search_budget(conn, req):
    deadline = req.get('_deadline')
    if deadline is None:
        deadline = time.monotonic() + timeout_seconds(req.get('timeout'))
    token = _DEADLINE.set(deadline)
    busy_timeout = None
    def expired():
        return time.monotonic() >= deadline
    try:
        seconds = remaining()
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        # SQLite's busy sleep does not invoke its progress callback. Bound
        # individual waits as well, including waits late in the search.
        wait_ms = min(busy_timeout, 50, max(1, math.ceil(seconds * 1000)))
        conn.execute(f"PRAGMA busy_timeout={wait_ms}")
        conn.set_progress_handler(expired, 100)
        try:
            yield
            remaining()
        except Exception as exc:
            if expired():
                raise SearchTimeout('Search timed out; increase --timeout to allow more time.') from exc
            raise
    finally:
        conn.set_progress_handler(None, 0)
        try:
            if busy_timeout is not None:
                conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
        finally:
            _DEADLINE.reset(token)
