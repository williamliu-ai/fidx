# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Embedding backends. Production path is fastembed (ONNX, CPU-only).

`Embedder` is the seam for tests and benchmarks: anything with `embed_docs`,
`embed_queries` and a `profile` attribute works. Models are loaded lazily so
CLI commands that never embed (get, status, lexical search) pay no model cost.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from .config import EmbedProfile

# Hard cap on tokens per embedded sequence. Token-dense text (e.g. binary
# content read with errors="replace" — ~1 token/char) meets long-context
# models badly: attention memory grows with batch x seq^2, and one batch of
# such chunks can demand tens of GB from ONNX (nomic's catalogue limit is
# 8192 tokens). Chunks target ~450 tokens (chunker.py), so real text stays
# well under this cap; only token-dense junk is truncated.
MAX_EMBED_TOKENS = 768

# Sequences embedded per ONNX Runtime run. Attention activations scale with
# batch x seq^2; with the 768-token cap a 64-sequence batch made ONNX
# Runtime's CPU arena grow to ~19 GB and retain it. 16 sequences per run and
# arena off keep peak at a few GB with equal or better CPU throughput.
EMBED_BATCH_SIZE = 16


def _cap_tokenizer(tokenizer) -> None:
    """Clamp the tokenizer's truncation to MAX_EMBED_TOKENS — never raising a
    model's own limit (a 512-position model must stay at 512)."""
    current = (tokenizer.truncation or {}).get("max_length")
    cap = MAX_EMBED_TOKENS if current is None else min(current, MAX_EMBED_TOKENS)
    tokenizer.enable_truncation(max_length=cap)


def _truncate_text(tokenizer, text: str) -> str:
    """Cut text at the char boundary of the tokenizer's truncation cap."""
    enc = tokenizer.encode(text)
    cap = (tokenizer.truncation or {}).get("max_length")
    if cap is None or len(enc.ids) < cap:
        return text
    end = max((e for _, e in enc.offsets), default=len(text))
    return text[:end]


class Embedder(Protocol):
    profile: EmbedProfile

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class ModelUnavailableError(RuntimeError):
    """The selected embedding model cannot be loaded from local files."""

    def __init__(self, profile_name: str, model_name: str):
        self.profile_name = profile_name
        super().__init__(
            f"embedding model {model_name!r} for profile {profile_name!r} "
            f"is unavailable locally; run "
            f"`fidx models fetch embed --profile {profile_name}`"
        )


class FastEmbedder:
    """fastembed-backed embedder, with explicit local-only query loading."""

    def __init__(self, profile: EmbedProfile, threads: int | None = None,
                 parallel: int | None = None, local_files_only: bool = False):
        self.profile = profile
        if threads is None and os.environ.get("FIDX_THREADS"):
            threads = int(os.environ["FIDX_THREADS"])
        self._threads = threads
        # Data-parallel workers for bulk (indexing) embeds; 0 = all cores.
        # Queries always run in-process — worker spawn would dominate latency.
        self._parallel = parallel
        self._local_files_only = local_files_only
        self._model = None

    @staticmethod
    def cache_path() -> Path:
        cache = os.environ.get("FASTEMBED_CACHE_PATH")
        if cache:
            return Path(cache).expanduser()
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        return Path(xdg).expanduser() / "fidx" / "models"

    def _build_model(self, *, lazy_load: bool = False):
        from fastembed import TextEmbedding

        if self.profile.hf_source:
            from fastembed.common.model_description import ModelSource, PoolingType

            known = {m["model"] for m in TextEmbedding.list_supported_models()}
            if self.profile.model not in known:
                TextEmbedding.add_custom_model(
                    model=self.profile.model,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=self.profile.hf_source),
                    dim=self.profile.dim,
                    model_file=self.profile.model_file or "onnx/model.onnx",
                )

        return TextEmbedding(
            model_name=self.profile.model,
            cache_dir=str(self.cache_path()),
            threads=self._threads,
            local_files_only=self._local_files_only,
            lazy_load=lazy_load,
            enable_cpu_mem_arena=False,
        )

    def fetch(self) -> Path:
        """Download/cache the selected model without loading an ONNX session."""
        self._build_model(lazy_load=True)
        return self.cache_path()

    def available_locally(self) -> bool:
        """Check this profile's cache without downloading or loading ONNX."""
        if not self._local_files_only:
            raise ValueError("availability check requires local_files_only=True")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = self._build_model(lazy_load=True)
            inner = model.model
            return (Path(inner._model_dir) / inner.model_description.model_file).is_file()
        except Exception:  # doctor must report broken dependencies, not crash
            return False

    def _ensure_model(self):
        if self._model is None:
            try:
                self._model = self._build_model()
            except Exception as exc:
                if self._local_files_only:
                    raise ModelUnavailableError(self.profile.name, self.profile.model) from exc
                raise
            tokenizer = getattr(self._model.model, "tokenizer", None)
            if tokenizer is not None:
                _cap_tokenizer(tokenizer)
        return self._model

    def _embed(self, texts: Iterable[str], batch_size: int = EMBED_BATCH_SIZE,
               parallel: int | None = None) -> np.ndarray:
        model = self._ensure_model()
        texts = list(texts)
        if parallel is not None:
            # Data-parallel workers rebuild the model — and its uncapped
            # tokenizer — from the catalogue, so the cap must be cut into
            # the text itself before it is handed to fastembed.
            tokenizer = getattr(model.model, "tokenizer", None)
            if tokenizer is not None:
                texts = [_truncate_text(tokenizer, t) for t in texts]
        vecs = np.array(
            list(model.embed(texts, batch_size=batch_size, parallel=parallel)),
            dtype=np.float32,
        )
        # Normalize so cosine distance is well-defined regardless of model.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        # fastembed forks even for one worker on a full batch; inline avoids
        # that overhead. Zero still means all available workers.
        parallel = None if self._parallel == 1 else self._parallel
        return self._embed([self.profile.doc_prefix + t for t in texts],
                           parallel=parallel)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.profile.query_prefix + t for t in texts])


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests: token-hash bag vector.
    Texts sharing vocabulary get high cosine similarity. Never use in production."""

    def __init__(self, profile: EmbedProfile):
        self.profile = profile

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.profile.dim, dtype=np.float32)
        for token in text.lower().split():
            vec[hash(token) % self.profile.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.profile.dim), np.float32)

    embed_queries = embed_docs
