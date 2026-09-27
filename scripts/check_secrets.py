# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fail CI on secret findings except exact generated pnpm integrity lines."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PNPM_LOCK = "obsidian-plugin/pnpm-lock.yaml"
# SHA-512 is 64 bytes, encoded as 86 base64 characters plus two padding chars.
PNPM_INTEGRITY = re.compile(
    r"^\s*resolution:\s*\{integrity:\s*sha512-[A-Za-z0-9+/]{86}==\}\s*$"
)


def remaining_findings(results: dict, root: Path) -> dict:
    """Filter only findings on exact generated integrity-only lockfile lines."""
    kept = {}
    try:
        lock_lines = (root / PNPM_LOCK).read_text(encoding="utf-8").splitlines()
    except OSError:
        lock_lines = []
    for filename, findings in results.items():
        remaining = []
        for finding in findings:
            line_number = finding.get("line_number")
            generated_integrity = (
                filename == PNPM_LOCK
                and type(line_number) is int
                and 1 <= line_number <= len(lock_lines)
                and PNPM_INTEGRITY.fullmatch(lock_lines[line_number - 1]) is not None
            )
            if not generated_integrity:
                remaining.append(finding)
        if remaining:
            kept[filename] = remaining
    return kept


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_secrets.py SCAN_JSON", file=sys.stderr)
        return 2
    results = json.loads(Path(args[0]).read_text(encoding="utf-8"))["results"]
    findings = remaining_findings(results, Path.cwd())
    if findings:
        for filename, rows in findings.items():
            for row in rows:
                print(f"POTENTIAL SECRET: {filename}:{row['line_number']} ({row['type']})")
        return 1
    print("no secrets detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
