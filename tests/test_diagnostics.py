# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Tests for `fidx doctor` / diagnostics — including the crash-proof invariant.

The key property: `fidx doctor` must run and *report* a broken native dep, never
crash on import. We prove that with a subprocess that forces `import sqlite_vec`
to fail (faithful to a missing/wrong-arch wheel), not a post-import monkeypatch.
"""
from __future__ import annotations

import subprocess
import sys

from fidx import diagnostics


def test_capabilities_shape_and_hard_pass_on_this_host():
    checks = diagnostics.capabilities()
    names = {c.name for c in checks}
    assert {"python", "fidx", "sqlite", "load_extension", "fts5", "sqlite_vec",
            "model_cache"} <= names
    # The dev/test host is a supported triple, so hard checks must pass.
    assert diagnostics.hard_failures(checks) == []


_BLOCKER = r"""
import builtins
_real = builtins.__import__
def blocked(name, *a, **k):
    if name == "sqlite_vec" or name.startswith("sqlite_vec."):
        raise ImportError("blocked sqlite_vec for test")
    return _real(name, *a, **k)
builtins.__import__ = blocked
from click.testing import CliRunner
from fidx.cli import main
r = CliRunner().invoke(main, ["doctor"])
assert r.exit_code == 1, r.exit_code
assert "sqlite_vec" in r.output and "FAIL" in r.output, r.output
assert CliRunner().invoke(main, ["--help"]).exit_code == 0
print("OK")
"""


def test_doctor_runs_with_broken_sqlite_vec():
    proc = subprocess.run([sys.executable, "-c", _BLOCKER],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
