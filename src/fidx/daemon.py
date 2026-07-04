# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true

"""Warm-search daemon over a Unix socket.

`fidx serve` keeps the SQLite connection and ONNX embedding model loaded so
repeated searches cost milliseconds instead of paying Python/model startup.
The CLI transparently uses the socket when it exists (see cli.py); the
protocol is one JSON request line in, one JSON response line out.
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import sys
from pathlib import Path

from . import search as searchmod


def socket_path_for(db_path: Path) -> Path:
    return db_path.with_suffix(".sock")


def serialize_results(results: list[searchmod.Result]) -> list[dict]:
    return [
        {
            "path": f"{r.collection}/{r.relpath}",
            "collection": r.collection,
            "relpath": r.relpath,
            "title": r.title,
            "docid": f"#{r.docid}",
            "score": round(r.score, 5),
            "snippet": r.snippet,
            "sources": {k: round(v, 5) for k, v in r.sources.items()},
        }
        for r in results
    ]


def run_search(conn, store, embedder, req: dict) -> list[dict]:
    query = req["query"]
    mode = req.get("mode", "hybrid")
    collections = req.get("collections") or None
    limit = int(req.get("limit", 10))
    if mode == "lexical":
        results = searchmod.search_lexical(conn, query, collections, limit)
    elif mode == "vector":
        results = searchmod.search_vector(conn, store, embedder, query, collections, limit)
    elif mode == "hybrid":
        results = searchmod.search_hybrid(conn, store, embedder, query, collections, limit)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    min_score = req.get("min_score")
    if min_score is not None:
        results = [r for r in results if r.score >= float(min_score)]
    truncate = req.get("truncate")
    if truncate == "calibrated":
        # corpus-calibrated abstention floor (fidx calibrate --store) + knee tail
        from . import truncate as truncatemod
        from .db import get_meta
        floor = get_meta(conn, "truncate_floor")
        if floor is not None:
            results = truncatemod.truncate(results, f"abs:{float(floor)}", mode)
        results = truncatemod.truncate(results, "knee", mode)
    elif truncate:
        from . import truncate as truncatemod
        results = truncatemod.truncate(results, truncate, mode)
    return serialize_results(results)


def serve(db_path: Path, conn, store, embedder) -> None:
    sock_path = socket_path_for(db_path)
    if sock_path.exists():
        sock_path.unlink()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            if not line:
                return
            try:
                req = json.loads(line)
                if req.get("cmd") == "ping":
                    resp = {"ok": True, "pong": True}
                else:
                    resp = {"ok": True, "results": run_search(conn, store, embedder, req)}
            except Exception as exc:  # protocol boundary: report, don't crash server
                resp = {"ok": False, "error": str(exc)}
            self.wfile.write((json.dumps(resp) + "\n").encode())

    # Warm the model before accepting connections.
    embedder.embed_queries(["warmup"])
    server = socketserver.UnixStreamServer(str(sock_path), Handler)
    print(f"fidx daemon listening on {sock_path} (pid {os.getpid()})", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sock_path.unlink(missing_ok=True)


def client_request(db_path: Path, req: dict, timeout: float = 30.0) -> dict | None:
    """Send a request to a running daemon; None if no usable daemon."""
    sock_path = socket_path_for(db_path)
    if not sock_path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(sock_path))
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                data = s.recv(65536)
                if not data:
                    break
                buf += data
        return json.loads(buf) if buf else None
    except (OSError, json.JSONDecodeError):
        return None
