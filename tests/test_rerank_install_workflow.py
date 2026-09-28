# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The optional reranker install is proved on every supported native runner."""

import re
from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/install-matrix.yml"


def test_rerank_extra_uses_same_wheel_in_six_isolated_binary_only_cells():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    core = jobs["install"]
    extra = jobs["install-rerank"]

    assert extra["needs"] == "build"
    assert extra["strategy"]["matrix"] == core["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "windows-latest", "macos-14"],
        "python": ["3.11", "3.12"],
    }
    assert extra["env"]["HF_HUB_OFFLINE"] == "1"
    steps = extra["steps"]
    check_env = next(step for step in steps
                     if "rerank_capability" in step.get("run", ""))["env"]
    assert "runner.temp" in check_env["FASTEMBED_CACHE_PATH"]
    assert "runner.temp" in check_env["HF_HOME"]
    assert any("actions/download-artifact@" in step.get("uses", "")
               and step.get("with", {}).get("name") == "dist" for step in steps)
    assert not any("actions/cache@" in step.get("uses", "") for step in steps)

    commands = "\n".join(step.get("run", "") for step in steps)
    assert "pip install --only-binary=:all:" in commands
    assert "[rerank]" in commands
    assert "fidx doctor --json" in commands
    assert all(name in commands for name in (
        "rerank_capability", "rerank_model", "TextCrossEncoder", "sysconfig",
        "purelib", "platlib"))
    assert "models fetch" not in commands
    assert "fidx index" not in commands


JOB_ENV_CONTEXTS = {"github", "needs", "strategy", "matrix", "vars",
                    "secrets", "inputs"}


def test_job_level_env_uses_only_supported_contexts():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    bad = [(job_id, key, match.group(1))
           for job_id, job in jobs.items()
           for key, value in (job.get("env") or {}).items()
           for match in re.finditer(r"\$\{\{\s*([A-Za-z_][\w-]*)", str(value))
           if match.group(1) not in JOB_ENV_CONTEXTS]
    assert not bad, bad
