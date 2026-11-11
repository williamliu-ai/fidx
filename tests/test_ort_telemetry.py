# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from importlib.metadata import version
import os
import subprocess
import sys

import pytest


def _run(script, env):
    return subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True,
        text=True, check=True, timeout=30,
    )


def test_telemetry_disabled_by_default():
    env = os.environ.copy()
    env.pop("ORT_DISABLE_TELEMETRY", None)
    result = _run("import fidx, os; print(os.environ.get('ORT_DISABLE_TELEMETRY'))", env)
    assert result.stdout.strip() == "1"


def test_telemetry_user_override_preserved():
    env = dict(os.environ, ORT_DISABLE_TELEMETRY="0")
    result = _run("import fidx, os; print(os.environ.get('ORT_DISABLE_TELEMETRY'))", env)
    assert result.stdout.strip() == "0"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux cache layout")
def test_telemetry_writes_prevented(tmp_path):
    if tuple(map(int, version("onnxruntime").split(".")[:2])) < (1, 30):
        pytest.skip("ONNX Runtime telemetry client requires 1.30 or newer")
    env = os.environ.copy()
    env.pop("ORT_DISABLE_TELEMETRY", None)
    control_cache = tmp_path / "control"
    env["XDG_CACHE_HOME"] = str(control_cache)
    # Import alone initializes the telemetry client; no model or session is needed.
    _run("import onnxruntime", env)
    telemetry = "Microsoft/DeveloperTools/.onnxruntime"
    assert any(p.is_file() for p in (control_cache / telemetry).rglob("*"))

    protected_cache = tmp_path / "protected"
    env["XDG_CACHE_HOME"] = str(protected_cache)
    _run("import fidx, onnxruntime", env)
    assert not any(p.is_file() for p in (protected_cache / telemetry).rglob("*"))
