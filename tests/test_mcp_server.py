# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import anyio
import pytest
from click.testing import CliRunner
from fidx import cli, db


def _module():
    pytest.importorskip("mcp", reason="install the optional MCP extra")
    assert importlib.util.find_spec("fidx.mcp_server"), "MCP bridge is missing"
    return importlib.import_module("fidx.mcp_server")


def test_optional_extra_and_guarded_command():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["optional-dependencies"].get("mcp") == ["mcp>=1.30,<2"]
    assert "mcp" in cli.main.commands


def test_core_cli_without_sdk():
    script = """import sys
class Block:
    def find_spec(self, name, *args):
        if name == 'mcp' or name.startswith('mcp.'):
            raise ModuleNotFoundError('SDK unavailable', name=name)
sys.meta_path.insert(0, Block())
from fidx.cli import main
main(args=sys.argv[1:])
"""
    help_result = subprocess.run([sys.executable, "-c", script, "--help"], capture_output=True)
    assert help_result.returncode == 0 and b"mcp" in help_result.stdout
    missing = subprocess.run([sys.executable, "-c", script, "mcp"], capture_output=True)
    assert missing.returncode == 1 and not missing.stdout
    assert b"python -m pip install 'fmdidx[mcp]'" in missing.stderr
    incompatible = subprocess.run([sys.executable, "-c", script, "--json", "mcp"], capture_output=True)
    assert incompatible.returncode == 2 and not incompatible.stdout
    assert b"--json is incompatible with MCP stdio" in incompatible.stderr


def test_exact_three_input_schemas():
    tools = anyio.run(_module().build_server(Path("index.db")).list_tools)
    schemas = {tool.name: tool.inputSchema for tool in tools}
    assert set(schemas) == {"fidx_search", "fidx_get", "fidx_collections"}
    assert set(schemas["fidx_search"]["properties"]) == {
        "query", "collection", "limit", "budget_tokens", "guidance"}
    assert schemas["fidx_search"]["required"] == ["query"]
    assert schemas["fidx_search"]["properties"]["limit"]["default"] == 10
    assert {"type": "integer", "minimum": 0} in schemas["fidx_search"]["properties"]["budget_tokens"]["anyOf"]
    assert set(schemas["fidx_get"]["properties"]) == {"docid", "head"}
    assert schemas["fidx_get"]["required"] == ["docid"]
    assert not schemas["fidx_collections"]["properties"]
    assert all(tool.outputSchema is None for tool in tools)


def test_root_json_mcp_refuses_before_server_start(monkeypatch):
    starts = []
    monkeypatch.setattr(_module(), "build_server", lambda path: starts.append(path))
    result = CliRunner().invoke(cli.main, ["--json", "mcp"])
    assert result.exit_code == 2
    assert not result.stdout and not starts
    assert "--json is incompatible with MCP stdio" in result.stderr


def test_negative_budget_refused_before_cli(monkeypatch):
    module, calls = _module(), []
    from mcp.server.fastmcp.exceptions import ToolError
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ToolError, match="greater than or equal to 0"):
        anyio.run(module.build_server(Path("missing")).call_tool,
                  "fidx_search", {"query": "test", "budget_tokens": -1})
    assert not calls


@pytest.mark.parametrize("guidance,flag", [(None, []), (True, ["--guidance"]),
                                          (False, ["--no-guidance"])])
def test_search_argv_environment_and_structured_result(monkeypatch, guidance, flag):
    module = _module()
    captured = {}
    payload = {"schema": "fidx.search.v2", "status": "ok", "query": "-é 東京"}
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    def run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, raw, b"")
    monkeypatch.setenv("FIDX_RERANK", "on")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/temporary/cache")
    monkeypatch.setattr(module.subprocess, "run", run)
    server = module.build_server(Path("index.db"))
    result = anyio.run(server.call_tool, "fidx_search", {
        "query": "-é 東京", "collection": "notes", "budget_tokens": 0,
        "guidance": guidance})
    assert captured["argv"] == [sys.executable, "-m", "fidx.cli", "--db", "index.db",
        "search", "--json", "-n", "10", "-c", "notes", "--budget-tokens", "0",
        *flag, "--", "-é 東京"]
    assert captured["env"] == {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    assert captured["stdin"] == subprocess.DEVNULL and captured["capture_output"] is True
    assert captured["timeout"] == 60 and not captured.get("text")
    assert result.structuredContent == json.loads(result.content[0].text) == payload
    assert not result.isError


@pytest.mark.parametrize("code,raw", [(3, b'{"schema":"fidx.error.v1","status":"index_missing","next_actions":[]}'),
                                    (0, b'not json'), (0, b'{"body":"\xff"}')])
def test_failures_are_structured_tool_errors(monkeypatch, code, raw):
    module = _module()
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kw:
                        subprocess.CompletedProcess(argv, code, raw, b"private stderr"))
    result = anyio.run(module.build_server(Path("missing")).call_tool, "fidx_collections", {})
    assert result.isError
    assert result.structuredContent == json.loads(result.content[0].text)
    assert result.structuredContent["schema"] == "fidx.error.v1"
    assert "private stderr" not in result.content[0].text
    if code:
        assert result.structuredContent == json.loads(raw)


@pytest.mark.parametrize("error", [OSError("private launch detail"),
                                    subprocess.TimeoutExpired("private command", 60)])
def test_launch_and_timeout_fail_closed(monkeypatch, error):
    module = _module()
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr(module.subprocess, "run", failed)
    result = anyio.run(module.build_server(Path("missing")).call_tool, "fidx_get", {"docid": "#abcdef"})
    assert result.isError and result.structuredContent["status"] == "error"
    assert "private" not in result.content[0].text


def test_timeout_reaps_direct_child(monkeypatch):
    module = _module()
    run, popen, children = subprocess.run, subprocess.Popen, []
    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child
    def timed(argv, **kwargs):
        return run([sys.executable, "-c", "import time; time.sleep(10)"],
                   **{**kwargs, "timeout": 0.1})
    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(module.subprocess, "run", timed)
    result = anyio.run(module.build_server(Path("missing")).call_tool, "fidx_collections", {})
    assert result.isError and "timed out" in result.content[0].text
    assert len(children) == 1 and children[0].returncode is not None


def test_real_stdio_cli_get_collections_empty_and_missing(tmp_path):
    _module()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.execute("INSERT INTO collections(name, root, globs) VALUES ('notes', ?, '[]')", (str(tmp_path),))
    body = "café 東京\n" * 12000
    conn.execute("INSERT INTO documents(collection, relpath, body, hash, active) "
                 "VALUES ('notes', 'é.md', ?, 'abcdef', 0)", (body,))
    conn.commit()
    conn.close()
    env = {**os.environ, "FASTEMBED_CACHE_PATH": str(tmp_path / "empty-cache"),
           "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1", "FIDX_RERANK": "off",
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    async def probe():
        params = StdioServerParameters(command=sys.executable, args=["-m", "fidx.cli", "--db", str(path), "mcp"], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert len((await session.list_tools()).tools) == 3
                async def call(name, args):
                    result = await session.call_tool(name, args)
                    assert result.structuredContent == json.loads(result.content[0].text)
                    return result
                collections = await call("fidx_collections", {})
                assert not collections.isError and collections.structuredContent["collections"][0]["doc_count"] == 1
                full = await call("fidx_get", {"docid": "notes/é.md"})
                assert not full.isError and full.structuredContent["truncated"] is True
                assert full.structuredContent["body"] == body.encode()[:65536].decode("utf-8", errors="ignore") + "\n… [truncated]"
                head = await call("fidx_get", {"docid": "#abcdef", "head": True})
                assert not head.isError and head.structuredContent["body"] == "\n".join(body.splitlines()[:40])
                empty = await call("fidx_search", {"query": "café 東京", "guidance": False})
                assert not empty.isError and empty.structuredContent["status"] == "empty_index"
                assert empty.structuredContent["query"].encode() == "café 東京".encode()
                assert empty.structuredContent["request"]["guidance"] is False
                path.unlink()
                missing = await call("fidx_collections", {})
                assert missing.isError and missing.structuredContent["status"] == "index_missing"
    anyio.run(probe)
    assert not any((tmp_path / "empty-cache").rglob("*.onnx"))
