# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_mcp_installed_matrix():
    jobs = yaml.safe_load((ROOT / ".github/workflows/install-matrix.yml").read_text(encoding="utf-8"))["jobs"]
    job = jobs["install-mcp"]
    assert job["needs"] == "build"
    assert job["strategy"]["matrix"] == jobs["install"]["strategy"]["matrix"]
    steps = job["steps"]
    assert any("actions/download-artifact@" in s.get("uses", "")
               and s.get("with", {}).get("name") == "dist" for s in steps)
    assert not any("actions/cache@" in s.get("uses", "") for s in steps)
    commands = "\n".join(s.get("run", "") for s in steps)
    assert all(text in commands for text in (
        "pip install --only-binary=:all:", "[mcp]", "test_mcp_offline.py", "--installed",
        "brew --prefix", "-m venv", "Scripts/python", 'test "$#" -eq 1'))
    assert not any(text in commands for text in ("fidx index", "models fetch", "e2e_smoke"))
    assert "runner.temp" not in json.dumps(job.get("env", {}))


def test_mcp_documentation():
    for name in ("README.md", "docs/AGENT_GUIDE.md", "integrations/claude-code-skill/SKILL.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "fmdidx[mcp]" in text and "fidx --db ./index.db mcp" in text
    guide = (ROOT / "docs/AGENT_GUIDE.md").read_text(encoding="utf-8")
    assert all(word in guide for word in ("fidx_collections", "fidx.collections.v1", "inactive", "tri-state"))
