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
