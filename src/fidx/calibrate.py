# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Self-supervised, corpus-adaptive truncation calibration.

Picks the truncation floor from the INDEXED CORPUS ITSELF, not a benchmark.
A benchmark constant does not transfer across corpora/embedders because the
score *scale* differs (e.g. e5's compressed cosine band, RRF magnitude). A
*policy* — "retain `target` of the corpus's own answers" — does transfer,
because the floor is recomputed from this corpus's score distribution.

Method (no labels, no benchmark, no LLM):
  1. Sample indexed documents.
  2. Derive a hard pseudo-query from each: scattered content words with term
     dropout, so it mimics a real keyword query rather than a verbatim span
     (a verbatim span matches its source trivially and over-estimates scores).
  3. Run hybrid search; the source document is the known answer.
  4. The distribution of the source doc's fused score across samples is the
     corpus's own "what a real match scores here" signal. The floor is the
     quantile of that distribution that retains `target` of sources.

Pseudo-queries are still easier than real user queries (perfect term overlap),
so the floor is calibrated CONSERVATIVELY with term dropout, and `calibrate`
also returns the full retention curve so the operator can pick the
recall/abstention operating point on THIS corpus. Re-run after re-indexing.
"""
from __future__ import annotations

import math
import random
import re

from .search import search_hybrid

_TOK = re.compile(r"[A-Za-z][A-Za-z']{3,}")
_CURVE_FLOORS = [round(0.01 * i, 4) for i in range(1, 16)]  # 0.01..0.15
_CONS, _VOWS = "bcdfghjklmnpqrstvwxz", "aeiou"


def _pseudo_query(body: str, rng: random.Random, nwords: int, dropout: int) -> str | None:
    """Scattered content words with `dropout` of them removed (harder query)."""
    words = _TOK.findall(body)
    if len(words) < nwords:
        return None
    picked = rng.sample(words, nwords)
    keep = max(1, nwords - dropout)
    return " ".join(picked[:keep])


def _gibberish_query(rng: random.Random, nwords: int = 4) -> str:
    """A made-up no-match query: pronounceable non-words that no document
    contains (lexical miss) and whose embedding is out-of-distribution. Its top
    result's score is a sample of the corpus's noise ceiling."""
    def word() -> str:
        n = rng.randint(5, 8)
        return "".join((rng.choice(_CONS) if i % 2 == 0 else rng.choice(_VOWS))
                       for i in range(n))
    return " ".join(word() for _ in range(nwords))


def recalibration_threshold(n_now: int, fraction: float) -> int:
    """Changed-doc count that triggers a recalibration: `fraction` of the corpus,
    but never fewer than 100 (no point recalibrating a corpus-wide stat for a
    handful of docs)."""
    return max(100, int(fraction * n_now))


def should_recalibrate(have_floor: bool, pending_changes: int, n_now: int,
                       fraction: float) -> bool:
    """Recalibrate only on first build (no floor yet) or once accumulated changes
    since the last calibration reach the threshold. Small incremental updates
    fall below it and skip the cost entirely."""
    return (not have_floor) or pending_changes >= recalibration_threshold(n_now, fraction)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, math.floor(q * len(sorted_vals))))
    return sorted_vals[idx]


def calibrate(conn, store, embedder, sample: int = 200, target: float = 0.85,
              reject: float = 0.9, nwords: int = 6, dropout: int = 2,
              neg_sample: int = 100, seed: int = 0) -> dict:
    """Calibrate a truncation floor from corpus self-retrieval + noise probes.

    Positives: source-doc fused scores from hard pseudo-queries (the doc is the
    known answer). Negatives: top-result fused scores from gibberish queries
    (no real answer) — the corpus's noise ceiling. The recommended `floor` is
    separation-anchored: the `reject` quantile of the negative scores, i.e. the
    level that rejects ~`reject` of noise while keeping anything that scores
    above what gibberish achieves. Also reports positive-retention at that floor,
    the positive-retention floor (`floor_retention`, the (1-target) quantile of
    positives), and the Youden-J optimal split, for transparency. Deterministic
    given `seed`.
    """
    rng = random.Random(seed)
    # ORDER BY id so the sampled set is reproducible for a fixed seed (SQLite
    # does not guarantee row order without it).
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM documents WHERE active = 1 ORDER BY id").fetchall()]
    if not ids:
        return {"floor": 0.0, "n_pos": 0, "n_neg": 0, "note": "empty index", "curve": []}

    pos: list[float] = []
    for did in rng.sample(ids, min(sample, len(ids))):
        row = conn.execute("SELECT body FROM documents WHERE id = ?", (did,)).fetchone()
        q = _pseudo_query((row[0] if row else "") or "", rng, nwords, dropout)
        if not q:
            continue
        res = search_hybrid(conn, store, embedder, q, None, 15)
        # match the source by doc id, not relpath (relpath is not unique across
        # collections, which would let a same-named other-collection doc count).
        match = next((r for r in res if r.doc_id == did), None)
        if match is not None:
            pos.append(match.score)
    if not pos:
        return {"floor": 0.0, "n_pos": 0, "n_neg": 0, "note": "no usable samples", "curve": []}

    # Negatives: gibberish queries have no real answer; their top score samples
    # the noise ceiling. An empty result counts as score 0 (correct abstention).
    neg: list[float] = []
    for _ in range(neg_sample):
        res = search_hybrid(conn, store, embedder, _gibberish_query(rng), None, 15)
        neg.append(res[0].score if res else 0.0)

    pos.sort(); neg.sort()
    np_, nn = len(pos), len(neg)

    def frac_ge(xs, t): return sum(1 for s in xs if s >= t) / len(xs) if xs else 0.0

    floor_retention = pos[max(0, min(np_ - 1, math.floor((1.0 - target) * np_)))]
    # Recommended floor = Youden-J optimal split (uses BOTH distributions: the
    # threshold maximizing kept-positives minus kept-negatives). With `>=` keep
    # semantics this correctly lands just ABOVE a degenerate negative spike, so
    # it rejects noise without the boundary bug a raw neg-quantile would hit.
    # Candidates include a hair above neg_max so "reject all noise" is reachable.
    cands = sorted(set(pos + neg + ([neg[-1] + 1e-4] if neg else [])))
    j_floor, j_best = cands[0], -2.0
    for t in cands:
        j = frac_ge(pos, t) - frac_ge(neg, t)
        if j > j_best:
            j_best, j_floor = j, t
    # reject-target floor: smallest threshold rejecting >= `reject` of negatives
    floor_reject = next((t for t in cands if nn and
                         sum(1 for s in neg if s < t) / nn >= reject),
                        (neg[-1] + 1e-4) if neg else j_floor)
    floor = j_floor

    # how many real matches collide with the noise ceiling (the abstention cost)
    pos_below_noise = round(frac_ge(pos, 0.0) - frac_ge(pos, _quantile(neg, 0.9) + 1e-9), 3) \
        if neg else 0.0
    curve = [(f, round(frac_ge(pos, f), 3)) for f in _CURVE_FLOORS]
    return {
        "floor": round(floor, 4),
        "floor_retention": round(floor_retention, 4),
        "floor_reject": round(floor_reject, 4),
        "n_pos": np_, "n_neg": nn, "target": target, "reject": reject,
        "pos_collides_with_noise": pos_below_noise,
        "pos_p5": round(_quantile(pos, 0.05), 4),
        "pos_median": round(_quantile(pos, 0.5), 4),
        "neg_p50": round(_quantile(neg, 0.5), 4),
        "neg_p90": round(_quantile(neg, 0.9), 4),
        "neg_max": round(neg[-1], 4) if neg else 0.0,
        "separation": round(_quantile(pos, 0.25) - _quantile(neg, 0.9), 4),
        "pos_retained_at_floor": round(frac_ge(pos, floor), 3),
        "neg_rejected_at_floor": round(1.0 - frac_ge(neg, floor), 3) if neg else 1.0,
        "curve": curve,
    }
