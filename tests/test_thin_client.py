# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The warm client must not import model or vector execution backends."""

import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest


def _run(code):
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_import_is_lightweight():
    _run('''
        import sys
        import fidx.cli
        forbidden = {'numpy', 'onnxruntime', 'fastembed', 'duckdb'}
        assert not forbidden.intersection(sys.modules), forbidden.intersection(sys.modules)
    ''')


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets required")
def test_daemon_search_is_lightweight(tmp_path):
    from fidx import db

    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    instance_id = db.get_meta(conn, "instance_id")
    conn.close()
    _run(f'''
        import json
        import socket
        import sys
        import threading
        from pathlib import Path
        from click.testing import CliRunner
        from fidx import cli, daemon

        path = Path({str(path)!r})
        payload = {{'schema': daemon.SEARCH_SCHEMA, 'results': [],
                   'summary': {{'result_count': 0}}, 'diagnostics': {{}}}}
        server = socket.socket(socket.AF_UNIX)
        daemon.socket_path_for(path).parent.mkdir(parents=True, exist_ok=True)
        server.bind(str(daemon.socket_path_for(path)))
        server.listen(1)
        requests = []
        def respond():
            with server.accept()[0] as peer:
                request = json.loads(peer.makefile('rb').readline())
                requests.append(request)
                peer.sendall((json.dumps({{'ok': True, 'instance_id': {instance_id!r},
                                          'results': payload}}) + '\\n').encode())
        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        try:
            result = CliRunner().invoke(cli.main, ['--db', str(path), 'search', '--json', 'query'])
            assert result.exit_code == 0, (result.output, result.exception)
            assert result.output == json.dumps(payload) + '\\n'
            thread.join(timeout=2)
            assert requests[0]['instance_id'] == {instance_id!r}
            forbidden = {{'numpy', 'onnxruntime', 'fastembed', 'duckdb', 'sqlite_vec'}}
            assert not forbidden.intersection(sys.modules), forbidden.intersection(sys.modules)
        finally:
            server.close()
            daemon.socket_path_for(path).unlink(missing_ok=True)
    ''')


@pytest.mark.skipif(os.name != "posix", reason="replacing an open database file is POSIX-only")
@pytest.mark.parametrize("mode", ["lexical", "vector"])
def test_cold_fallback_preserves_open_index_after_path_replacement(tmp_path, monkeypatch, mode):
    import os
    import sqlite3

    from click.testing import CliRunner
    from fidx import cli, daemon, db, rerank_models

    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("connection preservation requires loadable extensions")
    path = tmp_path / "index.db"
    replacement = tmp_path / "replacement.db"
    for candidate in (path, replacement):
        conn = db.connect(candidate)
        db.init_schema(conn)
        # Atomic replacement is safe here without retaining WAL sidecars.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        if candidate == replacement:
            conn.execute("INSERT INTO collections(name, root, globs) VALUES ('notes', '.', '[]')")
            conn.execute(
                "INSERT INTO documents(collection, relpath, title, body, hash) "
                "VALUES ('notes', 'new.md', 'alpha', 'alpha', 'newdocument1')")
            conn.commit()
        conn.close()

    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *a, **kw: pytest.fail("empty index loaded a reranker"))

    def replace_during_daemon_attempt(*args, **kwargs):
        os.replace(replacement, path)
        return daemon.ClientOutcome("cold", fallback_cause="connect_refused")

    monkeypatch.setattr(daemon, "client_request_outcome", replace_during_daemon_attempt)
    result = CliRunner().invoke(cli.main, [
        "--db", str(path), "search", "alpha", "--rerank", "--mode", mode, "--json",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "empty_index"
