# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from __future__ import annotations

import os
import socketserver
import stat

import pytest

from fidx import daemon


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Unix socket permissions")


def test_prepare_socket_path_uses_private_runtime_dir(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    sock_path = daemon._prepare_socket_path(tmp_path / "db" / "index.db")

    assert sock_path.parent == runtime / "fidx"
    assert sock_path.name.endswith(".sock")
    assert stat.S_IMODE(sock_path.parent.stat().st_mode) == 0o700


def test_bind_private_unix_server_creates_owner_only_socket(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    sock_path = daemon._prepare_socket_path(tmp_path / "db" / "index.db")

    class Handler(socketserver.StreamRequestHandler):
        pass

    server = daemon._bind_private_unix_server(sock_path, Handler)
    try:
        assert stat.S_IMODE(sock_path.stat().st_mode) == 0o600
    finally:
        server.server_close()
        sock_path.unlink(missing_ok=True)
