# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from click.testing import CliRunner

from fidx import cli


def test_collection_add_passes_exclude_globs(monkeypatch, tmp_path):
    root = tmp_path / "vault"
    root.mkdir()
    captured = {}

    def fake_open_index(db_path):
        captured["db_path"] = db_path
        return object(), None

    def fake_add_collection(conn, name, collection_root, globs=None, exclude_globs=None):
        captured["name"] = name
        captured["root"] = collection_root
        captured["globs"] = globs
        captured["exclude_globs"] = exclude_globs

    monkeypatch.setattr(cli, "open_index", fake_open_index)
    monkeypatch.setattr(cli.indexer, "add_collection", fake_add_collection)

    result = CliRunner().invoke(
        cli.main,
        [
            "--db",
            str(tmp_path / "index.db"),
            "collection",
            "add",
            str(root),
            "--name",
            "vault",
            "--glob",
            "**/*.md",
            "--exclude-glob",
            ".obsidian/**",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["name"] == "vault"
    assert captured["root"] == root
    assert captured["globs"] == ["**/*.md"]
    assert captured["exclude_globs"] == [".obsidian/**"]
