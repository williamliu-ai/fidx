# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Real-process tests for the persistent single-writer index lock."""

from __future__ import annotations

import json
import errno
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from fidx import db, indexer, locking
from fidx.errors import DatabaseLocked



def _seed_empty_index(tmp_path: Path) -> Path:
    path = tmp_path / "index.db"
    source = tmp_path / "notes"
    source.mkdir()
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", source)
    conn.close()
    return path


def _command(path: Path, *args: str) -> list[str]:
    return [sys.executable, "-m", "fidx.cli", "--json", "--db", str(path), *args]


@contextmanager
def _held_index(path: Path, tmp_path: Path, *, stage: str = "sync"):
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = """
import os
import sys
import time
from pathlib import Path
from fidx import calibrate, indexer
from fidx.cli import main
path, ready, release = map(Path, sys.argv[1:4])
stage = sys.argv[4]
def wait():
    ready.write_text(str(os.getpid()))
    while not release.exists():
        time.sleep(0.01)
if stage == "sync":
    original = indexer.sync_collection
    def paused(conn, store, name, stats):
        wait()
        return original(conn, store, name, stats)
    indexer.sync_collection = paused
else:
    original = calibrate.calibrate
    def paused(*args, **kwargs):
        wait()
        return original(*args, **kwargs)
    calibrate.calibrate = paused
args = ["--db", str(path), "index"]
if stage == "sync":
    args.append("--no-calibrate")
main(args)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(ready), str(release), stage],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2)
                raise AssertionError(f"index holder exited early: {out} {err}")
            if time.monotonic() > deadline:
                raise AssertionError("index holder did not reach sync barrier")
            time.sleep(0.01)
        yield proc, release
    finally:
        release.touch()
        if proc.poll() is None:
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate(timeout=5)


def _assert_locked(path: Path, *, holder_pid: int | None) -> dict:
    started = time.monotonic()
    loser = subprocess.run(
        _command(path, "index", "--no-calibrate"), capture_output=True,
        text=True, timeout=5,
    )
    elapsed = time.monotonic() - started
    assert loser.returncode == 4, loser.stdout + loser.stderr
    assert loser.stderr == ""
    payload = json.loads(loser.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert payload["status"] == "db_locked"
    assert "holder" in payload["message"].lower()
    if holder_pid is not None:
        assert str(holder_pid) in payload["message"]
    assert elapsed >= 0.20  # a second nonblocking attempt follows the 250 ms retry
    return payload


def test_two_process_race_loser_exits_four_winner_finishes(tmp_path):
    path = _seed_empty_index(tmp_path)
    lockfile = Path(f"{path}.lock")
    with _held_index(path, tmp_path) as (winner, release):
        assert lockfile.exists()
        _assert_locked(path, holder_pid=winner.pid if os.name == "posix" else None)
        assert winner.poll() is None
        release.touch()
        out, err = winner.communicate(timeout=5)
        assert winner.returncode == 0, out + err
    assert lockfile.exists()
    conn = db.connect(path, create=False)
    db.validate_index(conn)
    assert conn.execute("SELECT count(*) FROM collections").fetchone()[0] == 1
    conn.close()


def test_killed_holder_releases_kernel_lock_without_unlink(tmp_path):
    path = _seed_empty_index(tmp_path)
    lockfile = Path(f"{path}.lock")
    with _held_index(path, tmp_path) as (holder, _release):
        _assert_locked(path, holder_pid=holder.pid if os.name == "posix" else None)
        holder.kill()
        holder.communicate(timeout=5)
        assert holder.returncode != 0
    assert lockfile.exists()
    successor = subprocess.run(
        _command(path, "index", "--no-calibrate"), capture_output=True,
        text=True, timeout=5,
    )
    assert successor.returncode == 0, successor.stdout + successor.stderr
    assert lockfile.exists()


@pytest.mark.parametrize("contents", ["", "not-json\n%%garbled%%"])
@pytest.mark.skipif(os.name == "nt", reason="Windows byte-range lock denies a second writer")
def test_garbled_or_empty_payload_does_not_change_lock_exclusion(tmp_path, contents):
    path = _seed_empty_index(tmp_path)
    with _held_index(path, tmp_path):
        Path(f"{path}.lock").write_text(contents)
        payload = _assert_locked(path, holder_pid=None)
        assert "another holder" in payload["message"]


def test_lock_stays_held_through_calibration(tmp_path):
    path = _seed_empty_index(tmp_path)
    with _held_index(path, tmp_path, stage="calibration") as (holder, release):
        _assert_locked(path, holder_pid=holder.pid if os.name == "posix" else None)
        release.touch()
        out, err = holder.communicate(timeout=5)
        assert holder.returncode == 0, out + err


def test_query_during_index_returns_valid_wal_envelope(tmp_path):
    path = _seed_empty_index(tmp_path)
    with _held_index(path, tmp_path):
        query = subprocess.run(
            _command(path, "search", "topic", "--mode", "lexical", "--no-daemon"),
            capture_output=True, text=True, timeout=5,
        )
        assert query.returncode == 0, query.stdout + query.stderr
        payload = json.loads(query.stdout)
        assert payload["schema"] == "fidx.search.v2"
        assert payload["status"] == "empty_index"


def test_sqlite_connections_pin_five_second_busy_timeout(tmp_path):
    path = _seed_empty_index(tmp_path)
    for create in (False, True):
        conn = db.connect(path, create=create)
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        conn.close()


def test_windows_nonblocking_lock_uses_byte_zero_and_maps_only_contention(
    tmp_path, monkeypatch
):
    path = tmp_path / "sentinel.lock"
    path.write_text("old diagnostic payload")
    seen = []
    failure = None

    def fake_lock(fd, mode, nbytes):
        seen.append((mode, nbytes, os.lseek(fd, 0, os.SEEK_CUR)))
        if failure is not None:
            raise OSError(failure, "simulated failure")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=7, locking=fake_lock)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(locking, "_is_windows", lambda: True)
    with path.open("r+") as lockfile:
        lockfile.seek(8)
        locking._try_lock(lockfile.fileno())
        assert seen[-1] == (7, 1, 0)
        failure = errno.EACCES
        with pytest.raises(BlockingIOError):
            locking._try_lock(lockfile.fileno())
        failure = errno.EIO
        with pytest.raises(OSError) as exc:
            locking._try_lock(lockfile.fileno())
        assert exc.value.errno == errno.EIO


def test_unreadable_diagnostic_payload_keeps_generic_locked_error(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    def always_busy(_fd):
        raise BlockingIOError(errno.EACCES, "locked")
    monkeypatch.setattr(locking, "_try_lock", always_busy)
    monkeypatch.setattr(locking.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(locking, "_holder_pid", lambda _file: None)
    with pytest.raises(DatabaseLocked, match="another holder"):
        with locking.index_lock(path):
            pytest.fail("lock was never acquired")
    assert Path(f"{path}.lock").exists()
