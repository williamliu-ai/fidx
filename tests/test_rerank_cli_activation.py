# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fake-only CLI activation and precedence checks for offline reranking."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from fidx import cli, daemon, db, indexer, rerank_models, search as searchmod, vector_store
from fidx.embedder import HashEmbedder


class Scores:
    def __init__(self, *, fail: bool = False):
        self.calls = []
        self.fail = fail

    def rerank(self, query, documents):
        docs = list(documents)
        self.calls.append((query, docs))
        if self.fail:
            raise RuntimeError("fake scorer failed")
        return iter(float(i) for i in range(len(docs)))


def _index(tmp_path, *, active: bool):
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    if active:
        root = tmp_path / "notes"
        root.mkdir()
        (root / "one.md").write_text("# Search guide\nsearchable text here\n", encoding="utf-8")
        indexer.add_collection(conn, "notes", root)
        indexer.sync_collection(conn, vector_store.NullStore(), "notes", indexer.IndexStats())
        row = conn.execute("SELECT id, body FROM documents WHERE relpath = 'one.md'").fetchone()
        conn.execute(
            "INSERT INTO chunks(doc_id, seq, pos, length, line_start, line_end) "
            "VALUES (?, 0, 0, ?, 1, 2)", (row["id"], len(row["body"])))
        conn.commit()
    conn.close()
    return path


def _invoke(path, *args, global_json=False):
    return CliRunner().invoke(cli.main, [
        *(["--json"] if global_json else []), "--db", str(path),
        "search", *args,
    ])


def _fake_ready(monkeypatch, *, scorer=None):
    scorer = scorer or Scores()
    selected = []
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)

    def resolve(model, *, local_files_only):
        selected.append((model, local_files_only))
        return scorer, None

    monkeypatch.setattr(rerank_models, "resolve_reranker_model", resolve)
    monkeypatch.setattr(cli, "FastEmbedder",
                        lambda profile, **_kw: HashEmbedder(profile))
    return scorer, selected


@pytest.mark.parametrize("flag", [
    ("--rerank=unknown",), ("--rerank=",),
    ("--rerank", "--no-rerank"), ("--no-rerank", "--rerank"),
])
def test_bad_cli_options_fail_before_missing_index(tmp_path, flag, monkeypatch):
    monkeypatch.delenv("FIDX_RERANK", raising=False)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: pytest.fail("capability probed"))
    result = _invoke(tmp_path / "missing.db", "query", *flag, "--json")
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.parametrize("token", ["bogus", "2", "BAAI/bge-reranker-base"])
def test_invalid_env_only_is_usage_but_explicit_flag_overrides_it(
        tmp_path, monkeypatch, token):
    monkeypatch.setenv("FIDX_RERANK", token)
    missing = tmp_path / "missing.db"
    assert _invoke(missing, "query", "--json").exit_code == 2
    assert _invoke(missing, "query", "--rerank", "--json").exit_code == 3
    assert _invoke(missing, "query", "--no-rerank", "--json").exit_code == 3
    assert not missing.exists()


def test_optional_value_requires_equals_and_never_consumes_query(tmp_path):
    missing = tmp_path / "missing.db"
    assert _invoke(missing, "--rerank", "query", "--json").exit_code == 2
    assert _invoke(missing, "query", "--rerank", rerank_models.DEFAULT_RERANK_MODEL,
                   "--json").exit_code == 2
    assert not missing.exists()


@pytest.mark.parametrize("token,enabled", [
    ("", False), ("0", False), ("false", False), ("off", False), ("no", False),
    ("1", True), ("true", True), ("on", True), ("yes", True),
])
def test_env_boolean_grammar_on_valid_empty_index(tmp_path, monkeypatch, token, enabled):
    path = _index(tmp_path, active=False)
    monkeypatch.setenv("FIDX_RERANK", token)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: pytest.fail("empty index loaded scorer"))
    result = _invoke(path, "query", "--json", "--no-daemon")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "empty_index"
    assert payload["request"].get("rerank") == (
        {"model": rerank_models.DEFAULT_RERANK_MODEL} if enabled else None)


def test_index_precedes_capability_and_capability_precedes_empty(tmp_path, monkeypatch):
    missing = tmp_path / "missing.db"

    def no_capability():
        raise rerank_models.RerankCapabilityError("fake missing capability")

    monkeypatch.setattr(rerank_models, "probe_rerank_capability", no_capability)
    absent = _invoke(missing, "query", "--rerank", "--json")
    assert absent.exit_code == 3
    assert json.loads(absent.stdout)["status"] == "index_missing"
    empty = _index(tmp_path, active=False)
    unavailable = _invoke(empty, "query", "--rerank", global_json=True)
    assert unavailable.exit_code == 1
    error = json.loads(unavailable.stdout)
    assert error["schema"] == "fidx.error.v1" and error["status"] == "error"
    assert error["next_actions"][0]["intent"] == "install_rerank"


def test_invalid_existing_index_precedes_capability(tmp_path, monkeypatch):
    invalid = tmp_path / "invalid.db"
    invalid.write_bytes(b"not sqlite")
    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: pytest.fail("capability probed"))
    result = _invoke(invalid, "query", "--rerank", "--json")
    assert result.exit_code == 3
    assert json.loads(result.stdout)["status"] == "index_invalid"


def test_empty_index_skips_scorer_and_embedding_model(tmp_path, monkeypatch):
    path = _index(tmp_path, active=False)
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: pytest.fail("scorer constructed"))
    monkeypatch.setattr(cli, "FastEmbedder",
                        lambda *_a, **_k: pytest.fail("embedding loaded"))
    result = _invoke(path, "query", "--rerank", "--mode", "vector", "--json",
                     "--no-daemon")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "empty_index"
    assert payload["request"]["rerank"] == {
        "model": rerank_models.DEFAULT_RERANK_MODEL}


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_selected_cache_missing_before_retrieval_in_all_modes(
        tmp_path, monkeypatch, mode):
    path = _index(tmp_path, active=True)
    selected = "jinaai/jina-reranker-v1-tiny-en"
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)

    def unavailable(model, *, local_files_only):
        assert model == selected and local_files_only is True
        raise rerank_models.RerankModelUnavailableError(model)

    monkeypatch.setattr(rerank_models, "resolve_reranker_model", unavailable)
    monkeypatch.setattr(searchmod, "search_lexical",
                        lambda *_a, **_k: pytest.fail("lexical retrieval attempted"))
    monkeypatch.setattr(searchmod, "search_vector",
                        lambda *_a, **_k: pytest.fail("vector retrieval attempted"))
    monkeypatch.setattr(searchmod, "search_hybrid",
                        lambda *_a, **_k: pytest.fail("hybrid retrieval attempted"))
    result = _invoke(path, "searchable", "--mode", mode, "--json", "--no-daemon",
                     "--rerank=" + selected.upper())
    assert result.exit_code == 5, result.output
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert payload["status"] == "model_missing"
    assert payload["next_actions"][0]["command"] == [
        "fidx", "models", "fetch", "rerank", "--model", selected]


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_enabled_search_uses_cold_fake_scorer_and_echoes_canonical_model(
        tmp_path, monkeypatch, mode):
    path = _index(tmp_path, active=True)
    scorer, selected = _fake_ready(monkeypatch)
    if mode != "lexical":
        lexical = searchmod.search_lexical
        if mode == "vector":
            monkeypatch.setattr(searchmod, "search_vector",
                            lambda conn, _store, _embedder, query, collections, limit:
                            lexical(conn, query, collections, limit))
        else:
            monkeypatch.setattr(searchmod, "search_hybrid",
                            lambda conn, _store, _embedder, query, collections, limit:
                            lexical(conn, query, collections, limit))
    monkeypatch.setattr(daemon, "client_request",
                        lambda *_a, **_k: pytest.fail("legacy daemon consulted"))
    monkeypatch.setattr(daemon, "client_request_outcome",
                        lambda *_a, **_k: pytest.fail("protocol daemon consulted"))
    model = "jinaai/jina-reranker-v1-tiny-en"
    result = _invoke(path, "searchable", "--mode", mode, "--json",
                     "--rerank=" + model.upper())
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["request"]["rerank"] == {"model": model}
    assert selected == [(model, True)]
    assert scorer.calls and "rerank" in payload["results"][0]["sources"]
    assert payload["results"][0]["locations"][0]["line_start"] >= 1


def test_env_named_model_is_canonicalized_before_empty_response(tmp_path, monkeypatch):
    path = _index(tmp_path, active=False)
    model = "jinaai/jina-reranker-v1-turbo-en"
    monkeypatch.setenv("FIDX_RERANK", model.upper())
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    monkeypatch.setattr(rerank_models, "resolve_reranker_model",
                        lambda *_a, **_k: pytest.fail("empty index loaded scorer"))
    result = _invoke(path, "query", "--json", "--no-daemon")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["request"]["rerank"] == {"model": model}


def test_runtime_scorer_failure_keeps_original_results_and_request(
        tmp_path, monkeypatch):
    path = _index(tmp_path, active=True)
    _fake_ready(monkeypatch, scorer=Scores(fail=True))
    default = _invoke(path, "searchable", "--mode", "lexical", "--json",
                      "--no-daemon")
    failed = _invoke(path, "searchable", "--mode", "lexical", "--json",
                     "--no-daemon", "--rerank")
    assert default.exit_code == failed.exit_code == 0
    before, after = json.loads(default.stdout), json.loads(failed.stdout)
    assert after["results"] == before["results"]
    assert after["request"]["rerank"] == {
        "model": rerank_models.DEFAULT_RERANK_MODEL}
    assert after["diagnostics"]["warnings"] == [
        {"component": "rerank", "code": "runtime_failure", "fallback": "fused"}]


def test_default_off_and_explicit_off_keep_bytes_and_legacy_daemon(
        tmp_path, monkeypatch):
    path = _index(tmp_path, active=True)
    monkeypatch.delenv("FIDX_RERANK", raising=False)
    called = []
    monkeypatch.setattr(daemon, "client_request",
                        lambda *_a, **_k: called.append(True) or None)
    base = _invoke(path, "searchable", "--mode", "lexical", "--json")
    off = _invoke(path, "searchable", "--mode", "lexical", "--json",
                  "--no-rerank")
    assert base.exit_code == off.exit_code == 0
    assert base.stdout == off.stdout
    assert called == [True, True]
    assert "rerank" not in json.loads(base.stdout)["request"]


def test_explicit_off_with_env_on_cold_routes_and_omits_request_field(
        tmp_path, monkeypatch):
    path = _index(tmp_path, active=True)
    monkeypatch.setenv("FIDX_RERANK", "true")
    monkeypatch.setattr(daemon, "client_request",
                        lambda *_a, **_k: pytest.fail("legacy daemon consulted"))
    monkeypatch.setattr(daemon, "client_request_outcome",
                        lambda *_a, **_k: pytest.fail("protocol daemon consulted"))
    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: pytest.fail("capability probed"))
    result = _invoke(path, "searchable", "--mode", "lexical", "--json",
                     "--no-rerank")
    assert result.exit_code == 0, result.output
    assert "rerank" not in json.loads(result.stdout)["request"]
