# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Persistent advisory lock for the whole index command."""

from __future__ import annotations

import errno
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import DatabaseLocked


def _try_lock(fd: int) -> None:
    """Acquire a nonblocking kernel lock on the persistent sentinel."""
    if _is_windows():
        import msvcrt

        # msvcrt locks from the current file offset; all contenders use byte 0.
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno == errno.EACCES:
                raise BlockingIOError(exc.errno, str(exc)) from exc
            raise
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _is_windows() -> bool:
    return os.name == "nt"


def _holder_pid(lockfile) -> int | None:
    try:
        lockfile.seek(0)
        payload = json.loads(lockfile.read(256))
        pid = payload.get("pid")
        return pid if type(pid) is int and pid > 0 else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


@contextmanager
def index_lock(db_path: Path) -> Iterator[None]:
    """Exclude concurrent index runs; never unlink the lockfile sentinel."""
    lock_path = Path(f"{Path(db_path).expanduser().resolve()}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lockfile:
        for attempt in range(2):
            try:
                _try_lock(lockfile.fileno())
                break
            except BlockingIOError:
                if attempt == 0:
                    time.sleep(0.25)
        else:
            pid = _holder_pid(lockfile)
            holder = f"holder pid {pid}" if pid is not None else "another holder"
            raise DatabaseLocked(f"index {db_path} is locked by {holder}")

        lockfile.seek(0)
        lockfile.truncate()
        json.dump({"pid": os.getpid(), "start_time": time.time()}, lockfile)
        lockfile.flush()
        yield
