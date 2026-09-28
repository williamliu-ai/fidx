# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Fake-only reranker acquisition, prefetch and soft doctor checks."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from fidx import cli, db, diagnostics, indexer, rerank_models
from fidx.embedder import HashEmbedder


_DEFAULT = rerank_models.DEFAULT_RERANK_MODEL
_TURBO = "jinaai/jina-reranker-v1-turbo-en"
_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json")


def _index_fixture(tmp_path, monkeypatch):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "one.md").write_text("A useful document about search.", encoding="utf-8")
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.init_schema(conn)
    indexer.add_collection(conn, "notes", root)
    conn.close()
    monkeypatch.setattr(cli, "FastEmbedder", lambda profile, **kw: HashEmbedder(profile))
    return path


def _invoke(*args):
    return CliRunner().invoke(cli.main, list(args))


def test_shared_resolver_uses_lazy_cross_encoder_and_exact_model_file(
        tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    calls = []

    class FakeCrossEncoder:
        def __init__(self, *, model_name, cache_dir, local_files_only, lazy_load):
            calls.append((model_name, cache_dir, local_files_only, lazy_load))
            assert lazy_load is True
            model_dir = cache / "models--jinaai--jina-reranker-v1-turbo-en" / "snapshots" / "one"
            model_file = model_dir / "onnx" / "model.onnx"
            if not local_files_only:
                model_file.parent.mkdir(parents=True, exist_ok=True)
                model_file.write_bytes(b"fake onnx")
                for name in _REQUIRED:
                    (model_dir / name).write_bytes(b"fake metadata")
            self.model = SimpleNamespace(_model_dir=model_dir,
                                         model_description=SimpleNamespace(
                                             model_file="onnx/model.onnx"))

    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: FakeCrossEncoder)
    assert rerank_models.reranker_available_locally(_TURBO) is False
    path = rerank_models.fetch_reranker_model(_TURBO.upper())
    assert path.is_file()
    assert rerank_models.reranker_available_locally(_TURBO) is True
    instance, checked = rerank_models.resolve_reranker_model(
        _TURBO, local_files_only=True)
    assert isinstance(instance, FakeCrossEncoder) and checked == path
    assert calls == [
        (_TURBO, str(cache), False, True),
        (_TURBO, str(cache), True, True),
        (_TURBO, str(cache), True, True),
    ]


def test_partial_or_missing_local_cache_never_reports_success(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    (cache / "models--Xenova--ms-marco-MiniLM-L-6-v2").mkdir(parents=True)

    class PartialCrossEncoder:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(
                _model_dir=cache / "missing-snapshot",
                model_description=SimpleNamespace(model_file="onnx/model.onnx"))

    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: PartialCrossEncoder)
    with pytest.raises(RuntimeError, match="incomplete"):
        rerank_models.fetch_reranker_model(_DEFAULT)
    assert rerank_models.reranker_available_locally(_DEFAULT) is False
    with pytest.raises(rerank_models.RerankModelUnavailableError):
        rerank_models.resolve_reranker_model(_DEFAULT, local_files_only=True)


@pytest.mark.parametrize("missing", ("onnx/model.onnx", *_REQUIRED, "onnx/model.onnx_data"))
def test_fetch_and_local_check_require_every_model_file(tmp_path, monkeypatch, missing):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    repo = cache / "models--Xenova--ms-marco-MiniLM-L-6-v2"
    snapshot = repo / "snapshots" / "one"
    required = ("onnx/model.onnx", *_REQUIRED, "onnx/model.onnx_data")
    for name in required:
        if name != missing:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    class FakeCrossEncoder:
        def __init__(self, **_kwargs):
            self.model = SimpleNamespace(
                _model_dir=snapshot,
                model_description=SimpleNamespace(
                    model_file="onnx/model.onnx",
                    additional_files=["onnx/model.onnx_data"],
                ),
            )

    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: FakeCrossEncoder)
    assert rerank_models.reranker_available_locally(_DEFAULT) is False
    with pytest.raises(RuntimeError, match="incomplete") as failure:
        rerank_models.fetch_reranker_model(_DEFAULT)
    assert str(repo) in str(failure.value)


def test_partial_snapshot_fetch_names_exact_repository_recovery_path(
        tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    repo = cache / "models--Xenova--ms-marco-MiniLM-L-6-v2"
    snapshot = repo / "snapshots" / "partial"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    class LocalFirstCrossEncoder:
        def __init__(self, **_kwargs):
            self.model = SimpleNamespace(
                _model_dir=snapshot,
                model_description=SimpleNamespace(model_file="onnx/model.onnx"),
            )

    monkeypatch.setattr(rerank_models, "probe_rerank_capability",
                        lambda: LocalFirstCrossEncoder)
    result = _invoke("--json", "models", "fetch", "rerank")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1"
    assert str(repo) in payload["message"]
    assert "fetched reranker" not in result.stdout


def test_models_fetch_rerank_exact_model_and_target_specific_usage(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rerank_models, "fetch_reranker_model",
                        lambda model: calls.append(model) or tmp_path / "ready.onnx")
    selected = _invoke("models", "fetch", "rerank", "--model", _TURBO.upper())
    assert selected.exit_code == 0 and _TURBO in selected.output
    bare = _invoke("models", "fetch", "rerank")
    assert bare.exit_code == 0 and _DEFAULT in bare.output
    assert calls == [_TURBO, _DEFAULT]
    for args in (("rerank", "--profile", "bge-384"),
                 ("embed", "--model", _TURBO),
                 ("rerank", "--model", "BAAI/bge-reranker-base")):
        result = _invoke("models", "fetch", *args)
        assert result.exit_code == 2, result.output
    assert calls == [_TURBO, _DEFAULT]


def test_failed_explicit_fetch_reports_error_without_success(monkeypatch):
    def fail(_model):
        raise RuntimeError("selected model file was not resolved")
    monkeypatch.setattr(rerank_models, "fetch_reranker_model", fail)
    result = _invoke("--json", "models", "fetch", "rerank", "--model", _TURBO)
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1" and payload["status"] == "error"
    assert "fetched reranker" not in result.stdout


def test_index_prefetch_is_locked_before_success_timestamp_and_output(
        tmp_path, monkeypatch):
    path = _index_fixture(tmp_path, monkeypatch)
    state = {"locked": False}
    calls = []

    @contextmanager
    def lock(_path):
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    def fetch(model):
        assert state["locked"]
        conn = db.connect(path)
        if not calls:
            assert db.get_meta(conn, "last_index_completed_at") is None
        assert conn.execute("SELECT count(*) FROM documents WHERE active = 1").fetchone()[0] == 1
        conn.close()
        calls.append(model)
        return tmp_path / "ready.onnx"

    monkeypatch.setattr(cli, "index_lock", lock)
    monkeypatch.setattr(rerank_models, "fetch_reranker_model", fetch)
    result = _invoke("--db", str(path), "index", "--no-calibrate",
                     "--rerank-prefetch=" + _TURBO.upper(), "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema"] == "fidx.index.v1"
    assert calls == [_TURBO]
    conn = db.connect(path)
    assert db.get_meta(conn, "last_index_completed_at") is not None
    conn.close()

    # An explicit zero-change pass still performs the requested prefetch.
    result = _invoke("--db", str(path), "index", "--no-calibrate",
                     "--rerank-prefetch", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["added"] == 0
    assert calls == [_TURBO, _DEFAULT]


def test_prefetch_failure_has_no_success_report_or_timestamp(tmp_path, monkeypatch):
    path = _index_fixture(tmp_path, monkeypatch)
    def fail(_model):
        raise RuntimeError("prefetch failed")
    monkeypatch.setattr(rerank_models, "fetch_reranker_model", fail)
    result = _invoke("--db", str(path), "index", "--no-calibrate",
                     "--rerank-prefetch", "--json")
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["schema"] == "fidx.error.v1" and payload["status"] == "error"
    assert "fidx.index.v1" not in result.stdout
    conn = db.connect(path)
    assert db.get_meta(conn, "last_index_completed_at") is None
    conn.close()
    human = _invoke("--db", str(path), "index", "--no-calibrate",
                    "--rerank-prefetch")
    assert human.exit_code == 1 and "indexed in" not in human.stdout

    # A later failed prefetch cannot erase or advance a prior success stamp.
    monkeypatch.setattr(rerank_models, "fetch_reranker_model",
                        lambda _model: tmp_path / "fake.onnx")
    success = _invoke("--db", str(path), "index", "--no-calibrate",
                      "--rerank-prefetch", "--json")
    assert success.exit_code == 0, success.output
    conn = db.connect(path)
    prior = db.get_meta(conn, "last_index_completed_at")
    conn.close()
    monkeypatch.setattr(rerank_models, "fetch_reranker_model", fail)
    failed_again = _invoke("--db", str(path), "index", "--no-calibrate",
                           "--rerank-prefetch", "--json")
    assert failed_again.exit_code == 1
    conn = db.connect(path)
    assert db.get_meta(conn, "last_index_completed_at") == prior
    conn.close()


def test_prefetch_success_line_waits_for_completion_stamp(tmp_path, monkeypatch):
    path = _index_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(rerank_models, "fetch_reranker_model",
                        lambda _model: tmp_path / "fake.onnx")
    original = db.set_meta

    def failing_stamp(conn, key, value):
        if key == "last_index_completed_at":
            raise RuntimeError("completion stamp failed")
        return original(conn, key, value)

    monkeypatch.setattr(db, "set_meta", failing_stamp)
    result = _invoke("--db", str(path), "index", "--no-calibrate",
                     "--rerank-prefetch")
    assert result.exit_code == 1
    assert "indexed in" not in result.stdout
    assert "completion stamp failed" in result.stderr


def test_prefetch_rejects_unsupported_and_separated_model_before_index(
        tmp_path, monkeypatch):
    path = _index_fixture(tmp_path, monkeypatch)
    for option in (("--rerank-prefetch=BAAI/bge-reranker-base",),
                   ("--rerank-prefetch", _TURBO)):
        result = _invoke("--db", str(path), "index", "--no-calibrate", *option)
        assert result.exit_code == 2, result.output
    conn = db.connect(path)
    assert db.get_meta(conn, "last_index_completed_at") is None
    assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    conn.close()


def test_doctor_reports_soft_rerank_capability_and_local_cache(
        monkeypatch, tmp_path):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", lambda: object)
    calls = []
    monkeypatch.setattr(rerank_models, "reranker_available_locally",
                        lambda model: calls.append(model) or False)
    checks = {check.name: check for check in diagnostics.capabilities()}
    assert checks["rerank_capability"].ok is True
    assert checks["rerank_model"].ok is False
    assert checks["rerank_model"].hard is False
    assert _DEFAULT in checks["rerank_model"].remediation
    assert calls == [_DEFAULT]

    def missing():
        raise rerank_models.RerankCapabilityError("missing")
    monkeypatch.setattr(rerank_models, "probe_rerank_capability", missing)
    checks = {check.name: check for check in diagnostics.capabilities()}
    assert checks["rerank_capability"].ok is False
    assert checks["rerank_capability"].hard is False
    assert checks["rerank_model"].ok is False
    assert diagnostics.hard_failures(list(checks.values())) == []
    assert calls == [_DEFAULT]

    doctor = _invoke("doctor", "--json")
    assert doctor.exit_code == 0, doctor.output
    payload = json.loads(doctor.stdout)
    assert payload["schema"] == "fidx.doctor.v1" and payload["ok"] is True
    rows = {row["name"]: row for row in payload["checks"]}
    assert rows["rerank_capability"]["hard"] is False
    assert rows["rerank_model"]["hard"] is False
