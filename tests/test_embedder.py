"""Embed-time token cap: token-dense text (e.g. binary content read with
errors="replace", ~1 token/char) must never reach the ONNX model untruncated —
attention memory grows with batch x seq^2 and long-context models (nomic:
8192 tokens) can demand tens of GB for one batch."""

import pytest

pytest.importorskip("tokenizers")

from fidx.embedder import MAX_EMBED_TOKENS, _cap_tokenizer, _truncate_text


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
