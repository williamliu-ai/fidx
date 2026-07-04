# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true

"""Embedding backends. Production path is fastembed (ONNX, CPU-only).

`Embedder` is the seam for tests and benchmarks: anything with `embed_docs`,
`embed_queries` and a `profile` attribute works. Models are loaded lazily so
CLI commands that never embed (get, status, lexical search) pay no model cost.
"""

from __future__ import annotations

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


class FastEmbedder:
    """fastembed-backed embedder. Model download happens on first use and is
    cached under ~/.cache/fidx/models (or FASTEMBED_CACHE_PATH)."""

    def __init__(self, profile: EmbedProfile, threads: int | None = None,
                 parallel: int | None = None):
        import os

        self.profile = profile
        if threads is None and os.environ.get("FIDX_THREADS"):
            threads = int(os.environ["FIDX_THREADS"])
        self._threads = threads
        # Data-parallel workers for bulk (indexing) embeds; 0 = all cores.
        # Queries always run in-process — worker spawn would dominate latency.
        self._parallel = parallel
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import os

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

            cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
            if not cache_dir:
                xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
                cache_dir = os.path.join(xdg, "fidx", "models")
            self._model = TextEmbedding(
                model_name=self.profile.model,
                cache_dir=cache_dir,
                threads=self._threads,
            )
            tokenizer = getattr(self._model.model, "tokenizer", None)
            if tokenizer is not None:
                _cap_tokenizer(tokenizer)
        return self._model

    def _embed(self, texts: Iterable[str], batch_size: int = 64,
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
        return self._embed([self.profile.doc_prefix + t for t in texts],
                           parallel=self._parallel)

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
