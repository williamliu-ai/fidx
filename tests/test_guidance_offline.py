# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Search guidance stays offline in real processes with empty model caches."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fidx import db, indexer, vector_store


HOOK = r'''
import atexit, json, os, socket, sys
from pathlib import Path
log = Path(os.environ['FIDX_TEST_NETWORK_LOG'])
def blocked(*args, **kwargs):
    with log.open('a', encoding='utf-8') as out:
        out.write('network attempt\n')
    raise PermissionError('network denied by test')
def guard(original):
    def connect(self, address):
        frame = sys._getframe(1)
        while frame:
            if frame.f_code is getattr(getattr(socket, 'socketpair', None), '__code__', None):
                return original(self, address)
            frame = frame.f_back
        return blocked(self, address)
    return connect
socket.socket.connect = guard(socket.socket.connect)
socket.socket.connect_ex = guard(socket.socket.connect_ex)
socket.create_connection = blocked
socket.getaddrinfo = blocked
@atexit.register
def record_imports():
    Path(os.environ['FIDX_TEST_IMPORT_LOG']).write_text(
        json.dumps({'onnxruntime': 'onnxruntime' in sys.modules}), encoding='utf-8')
'''


@pytest.fixture()
def offline_child(tmp_path):
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(HOOK, encoding="utf-8")
    env = {key: os.environ[key] for key in
           ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT", "TEMP", "TMP")
           if key in os.environ}
    caches = []
    for key in ("FASTEMBED_CACHE_PATH", "HF_HOME", "XDG_CACHE_HOME"):
        cache = tmp_path / key.lower()
        cache.mkdir()
        env[key] = str(cache)
        caches.append(cache)
    network_log = tmp_path / "network.log"
    import_log = tmp_path / "imports.json"
    env.update(PYTHONPATH=os.pathsep.join((str(hook), str(
        Path(__file__).resolve().parents[1] / "src"))),
        PYTHONUTF8="1", PYTHONIOENCODING="utf-8", FIDX_PROFILE="bge-384",
        FIDX_TEST_NETWORK_LOG=str(network_log), FIDX_TEST_IMPORT_LOG=str(import_log))

    def run(args, *, lexical=False, network_expected=False):
        network_log.write_text("", encoding="utf-8")
        import_log.unlink(missing_ok=True)
        assert not [p for cache in caches for p in cache.rglob("*") if p.is_file()]
        result = subprocess.run([sys.executable, *args], env=env,
                                capture_output=True, encoding="utf-8", timeout=60)
        assert bool(network_log.read_text(encoding="utf-8")) is network_expected
        imports = json.loads(import_log.read_text(encoding="utf-8"))
        if lexical:
            assert imports["onnxruntime"] is False
        assert not [p for cache in caches for p in cache.rglob("*") if p.is_file()]
        return result

    return run


@pytest.fixture()
def lexical_index(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "topic.md").write_text("Running quickly through deployment notes.\n",
                                   encoding="utf-8")
    path = tmp_path / "index.db"
    conn = db.connect(path)
    try:
        db.init_schema(conn)
        db.set_meta(conn, "embed_profile", "bge-384")
        indexer.add_collection(conn, "notes", root)
        indexer.sync_collection(conn, vector_store.NullStore(), "notes", indexer.IndexStats())
    finally:
        conn.close()
    return path


def _search(path, query, *flags):
    return ["-m", "fidx.cli", "--db", str(path), "search", query,
            "--json", "--no-daemon", *flags]


@pytest.mark.parametrize("operation", [
    "socket.socket().connect(('127.0.0.1', 9))",
    "socket.socket().connect_ex(('127.0.0.1', 9))",
    "socket.create_connection(('127.0.0.1', 9))",
    "socket.getaddrinfo('localhost', 9)",
])
def test_network_guard_fires_in_child(offline_child, operation):
    result = offline_child(["-c", f"import socket; {operation}"], network_expected=True)
    assert result.returncode != 0
    assert "network denied by test" in result.stderr


@pytest.mark.parametrize("explain", [False, True])
def test_lexical_guidance_without_model_or_network(offline_child, lexical_index, explain):
    flags = ["--mode", "lexical", "--guidance"] + (["--explain"] if explain else [])
    result = offline_child(_search(lexical_index, "running", *flags), lexical=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.search.v2"
    assert payload["status"] == "ok"
    assert isinstance(payload["summary"]["confidence_reasons"], list)
    assert payload["diagnostics"]["term_coverage"]["matched"]
    assert ("qpp" in payload["diagnostics"]) is explain


def test_hybrid_cold_cache_failure_is_identical(offline_child, lexical_index):
    payloads = []
    for flags in ([], ["--guidance"], ["--no-guidance"]):
        result = offline_child(_search(lexical_index, "running", *flags))
        assert result.returncode == 5, result.stderr
        payload = json.loads(result.stdout)
        assert payload["schema"] == "fidx.error.v1"
        assert payload["status"] == "model_missing"
        payload.get("request", {}).pop("guidance", None)
        payloads.append(payload)
    assert payloads[0] == payloads[1] == payloads[2]


def test_empty_lexical_guidance_without_model_or_network(offline_child, lexical_index):
    result = offline_child(_search(lexical_index, "zzunmatchedzz", "--mode", "lexical",
                                   "--guidance"), lexical=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.search.v2"
    assert payload["status"] == "no_results"
    assert payload["diagnostics"]["no_results"]["causes"] == ["vocabulary_mismatch"]
