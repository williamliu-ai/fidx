# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import pytest

from fidx import db as dbmod
from fidx.config import EmbedProfile, profile_fingerprint
from fidx.embedder import HashEmbedder
from fidx.vector_store import SqliteVecStore

TEST_PROFILE = EmbedProfile(name="test-hash", model="hash", dim=64)


@pytest.fixture()
def conn(tmp_path):
    conn = dbmod.connect(tmp_path / "index.db")
    dbmod.init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture()
def store(conn):
    store = SqliteVecStore(conn)
    dbmod.ensure_vectors(conn, store, TEST_PROFILE.dim, profile_fingerprint(TEST_PROFILE),
                         TEST_PROFILE.model, TEST_PROFILE.name)
    return store


@pytest.fixture()
def embedder():
    return HashEmbedder(TEST_PROFILE)


@pytest.fixture()
def corpus_dir(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "auth.md").write_text(
        "# Authentication middleware\n\nHow users log in with OAuth tokens and sessions."
    )
    (root / "deploy.md").write_text(
        "# Deployment guide\n\nShip the service to production with rolling restarts."
    )
    (root / "recipes.md").write_text(
        "# Pasta recipes\n\nCarbonara needs eggs, pecorino and guanciale."
    )
    return root
