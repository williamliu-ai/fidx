# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The JSON get surface caps raw UTF-8 content without changing human get."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from fidx import db, indexer, vector_store
from fidx.cli import main


MARKER = "\n… [truncated]"
DEFAULT_CAP = 65_536


def _index_doc(tmp_path, body: str, *, name: str = "note.md"):
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    (root / name).write_text(body, encoding="utf-8")
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "docs", root)
    indexer.sync_collection(conn, vector_store.NullStore(), "docs", indexer.IndexStats())
    row = conn.execute("SELECT hash FROM documents WHERE relpath = ?", (name,)).fetchone()
    conn.close()
    return path, "#" + row["hash"][:6]


def _get(path, target="docs/note.md", *flags):
    return CliRunner().invoke(main, ["--db", str(path), "get", target, *flags])


@pytest.mark.parametrize(("length", "truncated"), [
    (65_535, False), (65_536, False), (65_537, True),
])
def test_default_cap_exact_byte_boundaries(tmp_path, length, truncated):
    body = "a" * length
    path, docid = _index_doc(tmp_path, body)
    result = _get(path, "docs/note.md", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "schema": "fidx.get.v1", "docid": docid, "collection": "docs",
        "relpath": "note.md", "truncated": truncated,
        "body": body[:DEFAULT_CAP] + (MARKER if truncated else ""),
    }
    if truncated:
        assert len(payload["body"].removesuffix(MARKER).encode("utf-8")) == DEFAULT_CAP


def test_multibyte_codepoint_straddling_cap_is_not_split(tmp_path):
    body = "a" * (DEFAULT_CAP - 1) + "é" + "tail"
    path, _ = _index_doc(tmp_path, body)
    result = _get(path, "docs/note.md", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["truncated"] is True
    assert payload["body"] == "a" * (DEFAULT_CAP - 1) + MARKER
    assert "�" not in payload["body"]
    assert len(payload["body"].removesuffix(MARKER).encode("utf-8")) <= DEFAULT_CAP

    tiny = _get(path, "docs/note.md", "--json", "--max-bytes", "1")
    assert tiny.exit_code == 0, tiny.output
    assert json.loads(tiny.output)["body"] == "a" + MARKER


def test_escape_heavy_text_uses_raw_bytes_before_json_escaping(tmp_path):
    body = "\\" * DEFAULT_CAP
    path, _ = _index_doc(tmp_path, body)
    result = _get(path, "docs/note.md", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["truncated"] is False
    assert payload["body"] == body
    assert len(result.output.encode("utf-8")) > DEFAULT_CAP * 2


def test_zero_max_bytes_returns_full_body(tmp_path):
    body = "é" * (DEFAULT_CAP + 1)
    path, docid = _index_doc(tmp_path, body)
    result = _get(path, docid, "--json", "--max-bytes", "0")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["docid"] == docid
    assert payload["body"] == body
    assert payload["truncated"] is False


def test_head_is_applied_before_byte_cap(tmp_path):
    lines = [f"line-{n}" for n in range(1, 42)]
    body = "\n".join(lines)
    path, _ = _index_doc(tmp_path, body)
    head = "\n".join(lines[:40])
    generous = _get(path, "docs/note.md", "--json", "--head", "--max-bytes", "1000")
    assert generous.exit_code == 0, generous.output
    assert json.loads(generous.output)["body"] == head
    assert json.loads(generous.output)["truncated"] is False
    bounded = _get(path, "docs/note.md", "--json", "--head", "--max-bytes", "10")
    assert bounded.exit_code == 0, bounded.output
    assert json.loads(bounded.output)["body"] == head[:10] + MARKER
    assert json.loads(bounded.output)["truncated"] is True


def test_head_still_caps_an_oversized_single_line(tmp_path):
    path, _ = _index_doc(tmp_path, "z" * (DEFAULT_CAP + 5))
    result = _get(path, "docs/note.md", "--json", "--head")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["truncated"] is True
    assert payload["body"] == "z" * DEFAULT_CAP + MARKER


def test_human_get_and_full_are_byte_identical_and_uncapped(tmp_path):
    body = "a" * (DEFAULT_CAP + 1) + "\nlast\n"
    path, _ = _index_doc(tmp_path, body)
    plain = _get(path)
    full = _get(path, "docs/note.md", "--full")
    head = _get(path, "docs/note.md", "--head")
    assert plain.exit_code == full.exit_code == head.exit_code == 0
    # CliRunner's text stream writes platform line endings into stdout_bytes.
    expected_full = (body + "\n").replace("\n", os.linesep).encode("utf-8")
    expected_head = ("\n".join(body.splitlines()[:40]) + "\n")
    expected_head = expected_head.replace("\n", os.linesep).encode("utf-8")
    assert plain.stdout_bytes == full.stdout_bytes == expected_full
    assert head.stdout_bytes == expected_head


def test_negative_max_bytes_and_human_max_bytes_are_usage_errors(tmp_path):
    path, _ = _index_doc(tmp_path, "body")
    for flags in (("--json", "--max-bytes", "-1"), ("--max-bytes", "1")):
        result = _get(path, "docs/note.md", *flags)
        assert result.exit_code == 2
        assert "Usage:" in result.stderr
        assert result.stdout == ""


def test_json_get_errors_keep_error_schema_and_exit_codes(tmp_path):
    missing_db = _get(tmp_path / "missing.db", "docs/note.md", "--json")
    assert missing_db.exit_code == 3
    assert json.loads(missing_db.output)["schema"] == "fidx.error.v1"

    path, _ = _index_doc(tmp_path, "same body")
    missing_doc = _get(path, "docs/absent.md", "--json")
    assert missing_doc.exit_code == 1
    assert json.loads(missing_doc.output)["schema"] == "fidx.error.v1"

    root = tmp_path / "docs"
    (root / "second.md").write_text("same body")
    conn = db.connect(path)
    indexer.sync_collection(conn, vector_store.NullStore(), "docs", indexer.IndexStats())
    prefix = "#" + conn.execute(
        "SELECT hash FROM documents WHERE relpath = 'note.md'").fetchone()["hash"][:6]
    conn.close()
    ambiguous = _get(path, prefix, "--json")
    assert ambiguous.exit_code == 6
    error = json.loads(ambiguous.output)
    assert error["schema"] == "fidx.error.v1"
    assert error["status"] == "ambiguous_reference"
    assert len(error["candidates"]) == 2
