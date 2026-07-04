# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true

"""Deterministic, score-based truncation of a ranked result list.

No LLM and no learned model (DESIGN.md invariant) — every method here is
arithmetic on scores already attached by search.py. A truncation method is a
pure filter over an already-ranked list: it may drop results but never reorders
or adds them, so a kept result is always one the ranker already surfaced. (That
property is what lets the benchmark evaluate methods offline against existing
relevance judgments.)

The methods were selected by an offline survey against existing relevance
judgments and externally reviewed; the guards below defend the edge cases that
review surfaced (empty lists, single results, ties, all-equal scores).

Spec grammar (passed as the `truncate` search arg / `--truncate` flag):
    off                  no truncation (default)
    abs:<tau>            drop fused score < tau                     (control)
    ratio:<alpha>        drop fused score < alpha * top_score       (scale-free)
    gap:<g>              cut before the first drop score[i] < g*score[i-1]
    knee                 cut at the max-distance knee of the score curve
    mad:<k>              keep score >= median + k * MAD (robust outlier)
    source:<vmin>[,<lmin>] hybrid: keep if vector>=vmin OR lexical>=lmin
                         (a source absent from a result cannot satisfy its
                         floor); vector mode: keep score>=vmin; lexical mode:
                         keep score>=lmin (or vmin if only one param given)

All methods may return fewer results than the input, including zero (abstain) —
that is the point: a query with no confident match should yield an empty list,
matching how QMD's vsearch/search abstain.
"""
from __future__ import annotations

from math import isfinite
from statistics import median

from .search import Result

# Minimum positional-parameter count per method; methods absent here accept a
# default (gap, mad, knee). `source` needs >=1 here; hybrid mode needs 2 and is
# enforced in _source.
_REQUIRED = {"abs": 1, "ratio": 1, "source": 1}


def parse_spec(spec: str | None) -> tuple[str, list[float]]:
    """'ratio:0.5' -> ('ratio', [0.5]); 'knee' -> ('knee', []); None/'off' -> ('off', []).

    Raises ValueError on an unknown method or missing required parameters, so a
    bad spec fails cleanly rather than as an IndexError deep in a method.
    """
    if not spec or spec == "off":
        return "off", []
    name, sep, rest = spec.partition(":")
    parts = [p for p in rest.split(",") if p != ""] if sep else []
    params = [float(x) for x in parts]  # ValueError on non-numeric
    if name not in _METHODS:
        raise ValueError(f"unknown truncate method {name!r}")
    if len(params) < _REQUIRED.get(name, 0):
        raise ValueError(f"truncate {name!r} requires "
                         f"{_REQUIRED[name]} numeric parameter(s)")
    return name, params


def truncate(results: list[Result], spec: str | None, mode: str = "hybrid") -> list[Result]:
    """Filter an already-ranked (score-descending) list per `spec`.

    Methods run even on a single result (so an absolute-floor method can drop a
    lone weak hit); structural methods (gap/knee/mad) keep short lists intact
    because they cannot estimate a tail from too few points.
    """
    method, params = parse_spec(spec)
    if method == "off" or not results:
        return results
    return _METHODS[method](results, params, mode)


def _abs(results: list[Result], p: list[float], mode: str) -> list[Result]:
    tau = p[0]
    return [r for r in results if r.score >= tau]


def _ratio(results: list[Result], p: list[float], mode: str) -> list[Result]:
    cut = p[0] * results[0].score
    return [r for r in results if r.score >= cut]


def _gap(results: list[Result], p: list[float], mode: str) -> list[Result]:
    g = p[0] if p else 0.5
    for i in range(1, len(results)):
        prev = results[i - 1].score
        if prev > 0 and results[i].score < g * prev:
            return results[:i]
    return results


def _knee(results: list[Result], p: list[float], mode: str) -> list[Result]:
    # The score-vs-rank curve of a confident query falls steeply then flattens
    # into the noise tail. Find the point of maximum vertical distance below the
    # chord joining the first and last points and cut BEFORE it (that point is
    # the first tail item). Needs >=4 points; flat curves have no knee.
    n = len(results)
    if n < 4:
        return results
    scores = [r.score for r in results]
    y0, yn = scores[0], scores[-1]
    if y0 - yn <= 0:
        return results
    best_i, best_d = 1, -1.0
    for i in range(1, n - 1):
        chord = y0 + (yn - y0) * (i / (n - 1))
        d = chord - scores[i]
        if d > best_d:
            best_d, best_i = d, i
    return results[:max(best_i, 1)]


def _mad(results: list[Result], p: list[float], mode: str) -> list[Result]:
    k = p[0] if p else 3.0
    scores = [r.score for r in results]
    if len(scores) < 3 or not all(isfinite(s) for s in scores):
        return results  # too few points to estimate a tail; or non-finite scores
    med = median(scores)
    mad = median([abs(s - med) for s in scores])
    if mad <= 0:
        return results  # degenerate spread (e.g. all-equal) -> no outlier cut
    cut = med + k * mad
    return [r for r in results if r.score >= cut]  # may be empty -> abstain


def _source(results: list[Result], p: list[float], mode: str) -> list[Result]:
    if mode == "hybrid":
        if len(p) < 2:
            raise ValueError("truncate source (hybrid) requires vmin,lmin")
        vmin, lmin = p[0], p[1]
        # An absent source cannot satisfy its floor — only a present, clearing
        # arm keeps a result. This is what trims the weak vector-only tail.
        return [r for r in results
                if ("vector" in r.sources and r.sources["vector"] >= vmin)
                or ("lexical" in r.sources and r.sources["lexical"] >= lmin)]
    # single-arm modes: r.score IS that arm's raw score (cosine / BM25-norm)
    floor = p[1] if (mode == "lexical" and len(p) > 1) else p[0]
    return [r for r in results if r.score >= floor]


_METHODS = {
    "abs": _abs,
    "ratio": _ratio,
    "gap": _gap,
    "knee": _knee,
    "mad": _mad,
    "source": _source,
}
