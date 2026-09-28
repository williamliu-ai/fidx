# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Offline-only reranker catalogue and import-safe capability probe.

Model acquisition and search activation are separate, later public seams.
Importing this module never constructs a model or checks the network.
"""

from __future__ import annotations

import importlib
import re
from importlib import metadata
from pathlib import Path
from typing import Any

RERANK_MODELS = (
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
)
DEFAULT_RERANK_MODEL = RERANK_MODELS[0]
_BY_CASEFOLD = {model.casefold(): model for model in RERANK_MODELS}
_FASTEMBED_VERSION = re.compile(r"0\.8\.\d+(?:\+[A-Za-z0-9.]+)?\Z")
_TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json")


def canonical_model(name: str | None) -> str:
    """Select exactly one of the four approved, catalogue-cased models."""
    if name is None:
        return DEFAULT_RERANK_MODEL
    if not isinstance(name, str):
        raise ValueError("rerank model name must be text")
    model = _BY_CASEFOLD.get(name.casefold())
    if model is None:
        raise ValueError("rerank model is not in the supported catalogue")
    return model


class RerankCapabilityError(RuntimeError):
    """FastEmbed cross-encoder support is absent or outside the pinned line."""

    def __init__(self, reason: str):
        super().__init__(f"{reason}; install `fmdidx[rerank]`")


class RerankModelUnavailableError(RuntimeError):
    """The selected allowlisted reranker is not present in the local cache."""

    def __init__(self, model_name: str):
        self.model_name = canonical_model(model_name)
        super().__init__(
            f"reranker model {self.model_name!r} is unavailable locally; run "
            f"`fidx models fetch rerank --model {self.model_name}`"
        )


def probe_rerank_capability() -> type[Any]:
    """Read distribution metadata and import the class without model loading."""
    try:
        version = metadata.version("fastembed")
    except metadata.PackageNotFoundError as exc:
        raise RerankCapabilityError("FastEmbed is unavailable") from exc
    if not _FASTEMBED_VERSION.fullmatch(version):
        raise RerankCapabilityError(
            f"FastEmbed {version} lacks the supported 0.8 cross-encoder line")
    try:
        module = importlib.import_module("fastembed.rerank.cross_encoder")
        cross_encoder = module.TextCrossEncoder
        if not callable(getattr(cross_encoder, "rerank", None)):
            raise AttributeError("TextCrossEncoder.rerank is unavailable")
    except Exception as exc:
        raise RerankCapabilityError("FastEmbed cross-encoder is unavailable") from exc
    return cross_encoder


def _cache_path() -> Path:
    from .embedder import FastEmbedder
    return FastEmbedder.cache_path()


def _repo_cache_path(model: str) -> Path:
    return _cache_path() / f"models--{model.replace('/', '--')}"


def resolve_reranker_model(model_name: str | None, *, local_files_only: bool
                           ) -> tuple[Any, Path]:
    """Resolve a selected ONNX file without constructing an ONNX session."""
    model = canonical_model(model_name)
    cross_encoder = probe_rerank_capability()
    try:
        instance = cross_encoder(
            model_name=model, cache_dir=str(_cache_path()),
            local_files_only=local_files_only, lazy_load=True,
        )
        inner = instance.model
        description = inner.model_description
        model_file = description.model_file
        if not isinstance(model_file, str) or not model_file:
            raise ValueError("reranker model description lacks an ONNX file")
        additional_files = getattr(description, "additional_files", ()) or ()
        model_dir = Path(inner._model_dir)
        for name in (model_file, *additional_files, *_TOKENIZER_FILES):
            if not isinstance(name, str) or not name or not (model_dir / name).is_file():
                raise FileNotFoundError(f"required reranker file {name!r} is absent")
        resolved = model_dir / model_file
    except Exception as exc:
        if local_files_only:
            raise RerankModelUnavailableError(model) from exc
        repo_dir = _repo_cache_path(model)
        recovery = (f"; move the incomplete cache directory {repo_dir} aside and rerun "
                    f"`fidx models fetch rerank --model {model}`"
                    if repo_dir.is_dir() else "")
        raise RuntimeError(f"reranker fetch incomplete for {model!r}: {exc}{recovery}") from exc
    return instance, resolved


def fetch_reranker_model(model_name: str | None = None) -> Path:
    """Explicit acquisition path; allowed to download but never loads ONNX."""
    _instance, model_file = resolve_reranker_model(
        model_name, local_files_only=False)
    return model_file


def reranker_available_locally(model_name: str | None = None) -> bool:
    """Doctor's no-network, no-ONNX check for one selected model."""
    model = canonical_model(model_name)
    # The four allowlisted FastEmbed entries use Hugging Face cache snapshots.
    # Avoid even a lazy constructor when its selected repository is absent.
    candidate = _repo_cache_path(model)
    if not candidate.is_dir():
        return False
    try:
        resolve_reranker_model(model, local_files_only=True)
        return True
    except (RerankCapabilityError, RerankModelUnavailableError):
        return False
