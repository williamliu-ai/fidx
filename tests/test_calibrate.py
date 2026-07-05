# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Tests for fidx.calibrate (corpus-adaptive truncation calibration).

Covers the pure pseudo-query helper plus an integration run over a tiny indexed
corpus (determinism, floor sanity, empty-index guard).
"""
import random

import pytest

from fidx import calibrate as C
from fidx import indexer


@pytest.fixture()
def indexed(conn, store, embedder, corpus_dir):
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return conn


# ----------------------------------------------------------- _pseudo_query

def test_pseudo_query_drops_terms_and_has_no_phrase_quotes():
    body = "Authentication middleware lets users sign in with OAuth tokens safely"
    rng = random.Random(0)
    q = C._pseudo_query(body, rng, nwords=6, dropout=2)
    words = q.split()
    assert len(words) == 4                      # 6 picked minus 2 dropped
    assert '"' not in q                         # not a verbatim phrase query
    assert all(w in body for w in words)        # words come from the doc


def test_pseudo_query_returns_none_when_too_few_words():
    assert C._pseudo_query("tiny doc", random.Random(0), nwords=6, dropout=2) is None
    assert C._pseudo_query("", random.Random(0), nwords=6, dropout=2) is None


def test_pseudo_query_dropout_ge_nwords_keeps_one():
    body = "alpha bravo charlie delta echo foxtrot golf hotel"
    q = C._pseudo_query(body, random.Random(1), nwords=6, dropout=9)
    assert len(q.split()) == 1


def test_pseudo_query_deterministic_for_seed():
    body = "alpha bravo charlie delta echo foxtrot golf hotel india"
    a = C._pseudo_query(body, random.Random(7), 6, 2)
    b = C._pseudo_query(body, random.Random(7), 6, 2)
    assert a == b


# ------------------------------------------------------- _gibberish_query

def test_gibberish_query_shape_and_determinism():
    q = C._gibberish_query(random.Random(3), nwords=4)
    words = q.split()
    assert len(words) == 4
    assert all(w.isalpha() and w.islower() for w in words)
    assert q == C._gibberish_query(random.Random(3), nwords=4)  # deterministic


# -------------------------------------------------------------- calibrate

def test_calibrate_returns_floor_separation_and_curve(indexed, store, embedder):
    res = C.calibrate(indexed, store, embedder, sample=50, neg_sample=20, seed=0)
    assert res["n_pos"] >= 1 and res["n_neg"] == 20
    assert isinstance(res["floor"], float) and res["floor"] >= 0.0
    assert "separation" in res and "floor_retention" in res and "floor_reject" in res
    assert 0.0 <= res["pos_retained_at_floor"] <= 1.0
    assert 0.0 <= res["neg_rejected_at_floor"] <= 1.0
    assert res["curve"] and all(0.0 <= keep <= 1.0 for _, keep in res["curve"])


def test_calibrate_is_deterministic(indexed, store, embedder):
    a = C.calibrate(indexed, store, embedder, sample=50, neg_sample=20, seed=0)
    b = C.calibrate(indexed, store, embedder, sample=50, neg_sample=20, seed=0)
    assert a == b


def test_calibrate_empty_index(conn, store, embedder):
    res = C.calibrate(conn, store, embedder, sample=10, seed=0)
    assert res["n_pos"] == 0 and res["floor"] == 0.0


# ----------------------------------------------- incremental recalibration

def test_recalibration_threshold_min_and_fraction():
    assert C.recalibration_threshold(100, 0.1) == 100      # floor of 100
    assert C.recalibration_threshold(50000, 0.1) == 5000   # 10% of corpus


def test_should_recalibrate_first_build_always():
    assert C.should_recalibrate(have_floor=False, pending_changes=0, n_now=10000, fraction=0.1)


def test_should_recalibrate_skips_small_updates():
    # 10000-doc corpus -> threshold 1000; a few changed docs must skip
    assert not C.should_recalibrate(True, pending_changes=5, n_now=10000, fraction=0.1)
    assert not C.should_recalibrate(True, pending_changes=999, n_now=10000, fraction=0.1)


def test_should_recalibrate_triggers_past_threshold():
    assert C.should_recalibrate(True, pending_changes=1000, n_now=10000, fraction=0.1)
