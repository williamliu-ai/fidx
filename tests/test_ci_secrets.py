# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The CI secret filter ignores only generated package integrity digests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


script = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.py"
spec = importlib.util.spec_from_file_location("check_secrets", script)
assert spec is not None and spec.loader is not None
check_secrets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_secrets)


def _finding(line: int) -> dict:
    return {"line_number": line, "type": "Synthetic Secret"}


def test_only_exact_pnpm_integrity_lines_are_exempt(tmp_path):
    lockfile = tmp_path / "obsidian-plugin" / "pnpm-lock.yaml"
    lockfile.parent.mkdir()
    digest = "sha512-" + "A" * 86 + "=="
    lockfile.write_text(
        f"  resolution: {{integrity: {digest}}}\n"
        "  api_token: SENSITIVE_VALUE\n"
        f"  resolution: {{integrity: {digest}}} api_token: SENSITIVE_VALUE\n"
    )
    other = tmp_path / "other.lock"
    other.write_text(f"resolution: {{integrity: {digest}}}\n")
    findings = {
        "obsidian-plugin/pnpm-lock.yaml": [_finding(1), _finding(2), _finding(3)],
        "other.lock": [_finding(1)],
    }

    kept = check_secrets.remaining_findings(findings, tmp_path)

    assert kept == {
        "obsidian-plugin/pnpm-lock.yaml": [_finding(2), _finding(3)],
        "other.lock": [_finding(1)],
    }


def test_cli_reports_remaining_findings_without_printing_values(tmp_path, monkeypatch, capsys):
    source = tmp_path / "obsidian-plugin" / "pnpm-lock.yaml"
    source.parent.mkdir()
    source.write_text("api_token: SENSITIVE_VALUE\n")
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({"results": {"obsidian-plugin/pnpm-lock.yaml": [_finding(1)]}}))
    monkeypatch.chdir(tmp_path)

    assert check_secrets.main([str(scan)]) == 1
    output = capsys.readouterr().out
    assert "obsidian-plugin/pnpm-lock.yaml:1 (Synthetic Secret)" in output
    assert "SENSITIVE_VALUE" not in output
