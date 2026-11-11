# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from click.testing import CliRunner
import json

from fidx import cli, db


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


def test_collection_json_preserves_human_output_and_counts_all_documents(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    for name in ("zeta", "alpha"):
        conn.execute("INSERT INTO collections(name, root, globs) VALUES (?, ?, ?)",
                     (name, str(tmp_path / name), '["**/*.md"]'))
    conn.execute("INSERT INTO documents(collection, relpath, body, hash, active) "
                 "VALUES ('alpha', 'é.md', '東京', 'abcdef', 0)")
    conn.commit()
    conn.close()
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--db", str(path), "collection", "list", "--json"])
    assert result.exit_code == 0, result.output
    expected = {"schema": "fidx.collections.v1", "collections": [
        {"name": name, "root": str(tmp_path / name), "globs": ["**/*.md"],
         "doc_count": int(name == "alpha")} for name in ("alpha", "zeta")]}
    assert json.loads(result.stdout) == expected
    root_json = runner.invoke(cli.main, ["--db", str(path), "--json", "collection", "list"])
    assert json.loads(root_json.stdout) == expected
    human = runner.invoke(cli.main, ["--db", str(path), "collection", "list"])
    assert human.stdout == "".join(
        f"{row['name']}\t{row['root']}\t{row['doc_count']} docs\t['**/*.md']\n"
        for row in expected["collections"])


def test_collection_json_empty_and_missing_index(tmp_path):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.close()
    args = ["--db", str(path), "collection", "list", "--json"]
    assert json.loads(CliRunner().invoke(cli.main, args).stdout) == {
        "schema": "fidx.collections.v1", "collections": []}
    path.unlink()
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 3
    assert json.loads(result.stdout)["schema"] == "fidx.error.v1"
    assert not path.exists()
