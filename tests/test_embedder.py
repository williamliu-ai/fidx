# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Embed-time token cap: token-dense text (e.g. binary content read with
errors="replace", ~1 token/char) must never reach the ONNX model untruncated —
attention memory grows with batch x seq^2 and long-context models (nomic:
8192 tokens) can demand tens of GB for one batch."""

import pytest
import numpy as np

pytest.importorskip("tokenizers")

from fidx.config import EmbedProfile
from fidx.embedder import (
    EMBED_BATCH_SIZE,
    FastEmbedder,
    MAX_EMBED_TOKENS,
    _cap_tokenizer,
    _truncate_text,
)


def word_tokenizer(max_length: int | None = None):
    """One token per whitespace-separated word, with offsets."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"w{i}": i for i in range(100)}
    vocab["[UNK]"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if max_length is not None:
        tok.enable_truncation(max_length=max_length)
    return tok


def test_cap_lowers_long_context_models():
    tok = word_tokenizer(max_length=8192)
    _cap_tokenizer(tok)
    assert tok.truncation["max_length"] == MAX_EMBED_TOKENS


def test_cap_never_raises_a_models_own_limit():
    tok = word_tokenizer(max_length=512)
    _cap_tokenizer(tok)
    assert tok.truncation["max_length"] == 512


def test_cap_applies_when_truncation_unset():
    tok = word_tokenizer()
    _cap_tokenizer(tok)
    assert tok.truncation["max_length"] == MAX_EMBED_TOKENS


def test_truncate_text_cuts_at_token_cap():
    tok = word_tokenizer(max_length=5)
    text = " ".join(f"w{i}" for i in range(10))
    out = _truncate_text(tok, text)
    assert out == "w0 w1 w2 w3 w4"
    assert len(tok.encode(out).ids) <= 5


def test_truncate_text_keeps_short_text_unchanged():
    tok = word_tokenizer(max_length=5)
    text = "w0 w1 w2"
    assert _truncate_text(tok, text) is text


@pytest.mark.parametrize(("parallel", "expected"), [
    (1, None),
    (0, 0),
    (3, 3),
])
def test_embed_docs_routes_worker_count_without_forking_one_worker(
    monkeypatch, parallel, expected
):
    profile = EmbedProfile(name="test", model="test", dim=2)
    embedder = FastEmbedder(profile, parallel=parallel)
    seen = []

    class FakeModel:
        tokenizer = None

        def embed(self, texts, *, parallel, **kwargs):
            seen.append(parallel)
            return iter([np.array([1.0, 0.0], dtype=np.float32) for _ in texts])

    class Wrapper:
        model = FakeModel()

        def embed(self, *args, **kwargs):
            return self.model.embed(*args, **kwargs)

    monkeypatch.setattr(embedder, "_ensure_model", lambda: Wrapper())
    vectors = embedder.embed_docs(["doc"] * 64)
    assert vectors.shape == (64, 2)
    assert seen == [expected]


def test_build_model_disables_cpu_mem_arena(monkeypatch):
    import fastembed

    seen = {}

    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models():
            return [{"model": "test"}]

        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    profile = EmbedProfile(name="test", model="test", dim=2)
    FastEmbedder(profile)._build_model()
    assert seen.get("enable_cpu_mem_arena") is False
    seen.clear()
    FastEmbedder(profile)._build_model(lazy_load=True)
    assert seen.get("enable_cpu_mem_arena") is False


def test_embed_batch_size_bounds_attention_memory():
    # 64-sequence batches drove the ONNX arena to ~20 GB; keep runs small.
    assert EMBED_BATCH_SIZE <= 16


@pytest.mark.parametrize(
    ("parallel", "expected_parallel"),
    [(1, None), (0, 0)],
)
def test_embed_docs_uses_embed_batch_size(monkeypatch, parallel, expected_parallel):
    profile = EmbedProfile(name="test", model="test", dim=2)
    embedder = FastEmbedder(profile, parallel=parallel)
    seen = []

    class FakeModel:
        tokenizer = None

        def embed(self, texts, batch_size=None, parallel=None):
            seen.append((batch_size, parallel))
            return iter([np.array([1.0, 0.0], dtype=np.float32) for _ in texts])

    class Wrapper:
        model = FakeModel()

        def embed(self, *args, **kwargs):
            return self.model.embed(*args, **kwargs)

    monkeypatch.setattr(embedder, "_ensure_model", lambda: Wrapper())
    vectors = embedder.embed_docs(["doc"] * 4)
    assert vectors.shape == (4, 2)
    assert seen == [(EMBED_BATCH_SIZE, expected_parallel)]


def test_embed_queries_uses_embed_batch_size(monkeypatch):
    profile = EmbedProfile(name="test", model="test", dim=2)
    embedder = FastEmbedder(profile)
    seen = []

    class FakeModel:
        tokenizer = None

        def embed(self, texts, batch_size=None, parallel=None):
            seen.append((batch_size, parallel))
            return iter([np.array([1.0, 0.0], dtype=np.float32) for _ in texts])

    class Wrapper:
        model = FakeModel()

        def embed(self, *args, **kwargs):
            return self.model.embed(*args, **kwargs)

    monkeypatch.setattr(embedder, "_ensure_model", lambda: Wrapper())
    vectors = embedder.embed_queries(["query"] * 4)
    assert vectors.shape == (4, 2)
    assert seen == [(EMBED_BATCH_SIZE, None)]


@pytest.mark.parametrize("parallel", [0, 3])
def test_worker_modes_pretruncate_before_fastembed(monkeypatch, parallel):
    profile = EmbedProfile(name="test", model="test", dim=2)
    embedder = FastEmbedder(profile, parallel=parallel)
    tokenizer = word_tokenizer(max_length=5)
    seen = []

    class FakeModel:
        def __init__(self):
            self.model = type("Inner", (), {"tokenizer": tokenizer})()

        def embed(self, texts, *, parallel, **kwargs):
            seen.append((list(texts), parallel))
            return iter([np.array([1.0, 0.0], dtype=np.float32) for _ in texts])

    monkeypatch.setattr(embedder, "_ensure_model", FakeModel)
    embedder.embed_docs([" ".join(f"w{i}" for i in range(10))])
    assert seen == [(["w0 w1 w2 w3 w4"], parallel)]
