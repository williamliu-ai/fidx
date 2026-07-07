# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""The e2e corpus generator must be byte-deterministic across platforms.

The full install+search e2e lives in scripts/e2e_smoke.py and runs against the
*installed* wheel (Docker / CI); here we only guard that the generated corpus
matches the pinned manifest SHA, so a cross-platform drift (e.g. line endings)
fails fast in unit CI too.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "scripts")  # e2e_smoke is a standalone script, not packaged
import e2e_smoke  # noqa: E402


def test_corpus_manifest_matches_pinned_sha():
    pinned = Path("tests/fixtures/e2e-corpus.sha256").read_text().split()[0]
    dest = Path(tempfile.mkdtemp(prefix="e2e-gen-")) / "corpus"
    queries, sha = e2e_smoke.gen_corpus(dest, 1000, 40)
    assert sha == pinned, f"corpus drift: {sha} != pinned {pinned}"
    assert len(queries) == 40
    assert all(q["expected"].startswith("e2e/") for q in queries)


def test_search_results_accepts_agent_envelope():
    payload = {"schema": "fidx.search.v2", "results": [{"path": "e2e/doc0001.md"}]}
    assert e2e_smoke.search_results(json.dumps(payload)) == payload["results"]


def test_search_results_accepts_legacy_array():
    payload = [{"path": "e2e/doc0001.md"}]
    assert e2e_smoke.search_results(json.dumps(payload)) == payload
