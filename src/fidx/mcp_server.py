# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Optional stdio adapter; retrieval and public envelopes belong to the CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import Field


def _failure(message: str) -> CallToolResult:
    payload = {"schema": "fidx.error.v1", "status": "error", "message": message,
               "next_actions": []}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))],
                          structuredContent=payload, isError=True)


def _cli(db_path: Path, schema: str, args: list[str]) -> CallToolResult:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "fidx.cli", "--db", str(db_path), *args],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return _failure("fidx CLI timed out")
    except OSError:
        return _failure("Could not launch fidx CLI")
    try:
        text = completed.stdout.decode("utf-8", errors="strict")
        payload = json.loads(text)
        if (not isinstance(payload, dict)
                or payload.get("schema") not in (schema, "fidx.error.v1")
                or (completed.returncode and payload["schema"] != "fidx.error.v1")):
            return _failure("fidx CLI returned an invalid response")
    except (UnicodeError, ValueError):
        return _failure("fidx CLI returned invalid UTF-8 JSON")
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structuredContent=payload,
                          isError=bool(completed.returncode) or payload["schema"] == "fidx.error.v1")


def build_server(db_path: Path) -> FastMCP:
    """Bind the three read tools to one explicitly selected index."""
    server = FastMCP("fidx")

    @server.tool(structured_output=False)
    def fidx_search(query: str, collection: str | None = None, limit: int = 10,
                    budget_tokens: Annotated[int | None, Field(ge=0)] = None,
                    guidance: bool | None = None) -> CallToolResult:
        """Search local files; return ranked results and next actions."""
        args = ["search", "--json", "-n", str(limit)]
        if collection is not None:
            args += ["-c", collection]
        if budget_tokens is not None:
            args += ["--budget-tokens", str(budget_tokens)]
        if guidance is not None:
            args += ["--guidance" if guidance else "--no-guidance"]
        return _cli(db_path, "fidx.search.v2", [*args, "--", query])

    @server.tool(structured_output=False)
    def fidx_get(docid: str, head: bool = False) -> CallToolResult:
        """Read a result document; optionally return its first 40 lines."""
        return _cli(db_path, "fidx.get.v1",
                    ["get", "--json", *(["--head"] if head else []), "--", docid])

    @server.tool(structured_output=False)
    def fidx_collections() -> CallToolResult:
        """List collection names, roots, globs and counts including inactive documents."""
        return _cli(db_path, "fidx.collections.v1", ["collection", "list", "--json"])

    return server
