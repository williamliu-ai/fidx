# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The cold CLI imports even where Python provides no Unix socket server."""

from __future__ import annotations

import os
import subprocess
import sys


def test_cli_import_and_cold_search_without_unix_stream_server(tmp_path):
    script = """
import json
import socketserver
import sys
from pathlib import Path

if hasattr(socketserver, 'UnixStreamServer'):
    del socketserver.UnixStreamServer

from click.testing import CliRunner
from fidx import daemon, db
from fidx.cli import main

path = Path(sys.argv[1])
conn = db.connect(path)
db.init_schema(conn)
conn.close()
socket_path = daemon.socket_path_for(path)
socket_path.parent.mkdir(parents=True)
socket_path.write_text('stale')
assert daemon.client_request(path, {'cmd': 'ping'}) is None
result = CliRunner().invoke(main, [
    '--db', str(path), 'search', 'topic', '--mode', 'lexical', '--json',
])
assert result.exit_code == 0, result.output
assert json.loads(result.stdout)['status'] == 'empty_index'
try:
    daemon._bind_private_unix_server(path.with_suffix('.sock'),
                                     socketserver.StreamRequestHandler)
except RuntimeError as exc:
    assert 'Unix' in str(exc)
else:
    raise AssertionError('unsupported daemon bind should fail clearly')
try:
    daemon.serve(path, None, None, None)
except RuntimeError as exc:
    assert 'Unix' in str(exc)
else:
    raise AssertionError('unsupported serve should fail before model warmup')
"""
    env = {**os.environ, "XDG_RUNTIME_DIR": str(tmp_path / "runtime")}
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "index.db")],
        text=True, capture_output=True, env=env, timeout=30,
    )
    assert result.returncode == 0, result.stderr
