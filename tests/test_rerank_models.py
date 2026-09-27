# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Offline reranker catalogue, capability and error-boundary contracts."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from fidx import errors, rerank_models


_MODELS = (
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
)


def test_provisional_default_and_exact_four_model_allowlist():
    assert rerank_models.DEFAULT_RERANK_MODEL == _MODELS[0]
    assert rerank_models.RERANK_MODELS == _MODELS
    assert rerank_models.canonical_model(None) == _MODELS[0]
    for canonical in _MODELS:
        assert rerank_models.canonical_model(canonical) == canonical
        assert rerank_models.canonical_model(canonical.upper()) == canonical
    for invalid in ("BAAI/bge-reranker-base",
                    "jinaai/jina-reranker-v2-base-multilingual",
                    "Xenova/ms-marco-MiniLM-L-6-v2 ", "", "other"):
        with pytest.raises(ValueError):
            rerank_models.canonical_model(invalid)


def test_capability_probe_imports_class_without_constructing_or_downloading(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            raise AssertionError("cross-encoder constructed during capability probe")

        def rerank(self, query, documents):
            return []

    monkeypatch.setattr(rerank_models.metadata, "version",
                        lambda package: calls.append(("version", package)) or "0.8.0")
    monkeypatch.setattr(rerank_models.importlib, "import_module",
                        lambda module: calls.append(("import", module)) or
                        SimpleNamespace(TextCrossEncoder=FakeCrossEncoder))
    assert rerank_models.probe_rerank_capability() is FakeCrossEncoder
    assert calls == [("version", "fastembed"),
                     ("import", "fastembed.rerank.cross_encoder")]


@pytest.mark.parametrize("version", ["0.6.9", "0.7.9", "0.8.0rc1",
                                            "0.9.0", "broken"])
def test_capability_probe_rejects_unsupported_fastembed_without_import(
        monkeypatch, version):
    monkeypatch.setattr(rerank_models.metadata, "version", lambda _: version)
    monkeypatch.setattr(rerank_models.importlib, "import_module",
                        lambda _: pytest.fail("unsupported version imported model"))
    with pytest.raises(rerank_models.RerankCapabilityError):
        rerank_models.probe_rerank_capability()


def test_capability_probe_rejects_missing_distribution_or_class(monkeypatch):
    def missing(_):
        raise rerank_models.metadata.PackageNotFoundError("fastembed")

    monkeypatch.setattr(rerank_models.metadata, "version", missing)
    with pytest.raises(rerank_models.RerankCapabilityError):
        rerank_models.probe_rerank_capability()
    monkeypatch.setattr(rerank_models.metadata, "version", lambda _: "0.8.0")
    monkeypatch.setattr(rerank_models.importlib, "import_module",
                        lambda _: SimpleNamespace())
    with pytest.raises(rerank_models.RerankCapabilityError):
        rerank_models.probe_rerank_capability()
    def broken_binary(_):
        raise OSError("native dependency cannot load")
    monkeypatch.setattr(rerank_models.importlib, "import_module", broken_binary)
    with pytest.raises(rerank_models.RerankCapabilityError):
        rerank_models.probe_rerank_capability()


def test_typed_offline_model_and_capability_errors_have_distinct_envelopes():
    model = _MODELS[2]
    missing = rerank_models.RerankModelUnavailableError(model.upper())
    payload, code = errors.envelope(missing)
    assert (payload["schema"], payload["status"], code) == (
        "fidx.error.v1", "model_missing", 5)
    assert payload["next_actions"] == [{
        "intent": "fetch_model",
        "reason": "Download the selected reranker model before searching offline.",
        "command": ["fidx", "models", "fetch", "rerank", "--model", model],
    }]
    assert model in payload["message"]

    capability = rerank_models.RerankCapabilityError("FastEmbed cross-encoder unavailable")
    payload, code = errors.envelope(capability)
    assert (payload["schema"], payload["status"], code) == (
        "fidx.error.v1", "error", 1)
    assert payload["next_actions"] == [{
        "intent": "install_rerank",
        "reason": "Install a FastEmbed version with cross-encoder support.",
        "command": ["python", "-m", "pip", "install", "fmdidx[rerank]"],
    }]
    assert "fmdidx[rerank]" in payload["message"]


def test_rerank_extra_raises_fastembed_floor_without_changing_core():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "fastembed>=0.6,<0.9" in project["project"]["dependencies"]
    assert project["project"]["optional-dependencies"]["rerank"] == [
        "fastembed>=0.8,<0.9"]
    locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    package = next(row for row in locked["package"] if row["name"] == "fmdidx")
    assert {"name": "fastembed"} in package["optional-dependencies"]["rerank"]
    assert any(row["name"] == "fastembed"
               and row.get("marker") == "extra == 'rerank'"
               and row["specifier"] == ">=0.8,<0.9"
               for row in package["metadata"]["requires-dist"])
