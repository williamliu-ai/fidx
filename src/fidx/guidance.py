# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Internal search evidence and uncalibrated query-performance signals.

Nothing in this module changes search ranking or the public search envelope.
Weights and confidence thresholds belong to a later calibration step.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Sequence

from .db import sqlite_module
from .search import SearchTrace, _TOKEN_RE


@dataclass(frozen=True)
class TermCoverage:
    matched: tuple[str, ...]
    unmatched: tuple[str, ...]
    rare: tuple[str, ...]
    term_df: dict[str, int]
    ratio: float


@dataclass(frozen=True)
class ScoreShape:
    nqc_like: float
    wig_like: float
    top_gap: float
    head_concentration: float


@dataclass(frozen=True)
class QPPSignals:
    lexical: ScoreShape
    vector: ScoreShape
    fused: ScoreShape
    channel_agreement: float
    term_coverage: float


def _surface_stems(surfaces: list[str]) -> dict[str, tuple[str, ...]]:
    """Ask the same FTS5 tokenizer as the index to stem query surface terms."""
    if not surfaces:
        return {}
    tokenizer = sqlite_module().connect(":memory:")
    try:
        tokenizer.execute(
            "CREATE VIRTUAL TABLE tokens USING fts5(term, tokenize='porter unicode61')")
        tokenizer.executemany(
            "INSERT INTO tokens(rowid, term) VALUES (?, ?)", enumerate(surfaces, 1))
        tokenizer.execute(
            "CREATE VIRTUAL TABLE token_vocab USING fts5vocab(tokens, 'instance')")
        stems: dict[str, list[str]] = {surface: [] for surface in surfaces}
        for rowid, stem in tokenizer.execute(
            "SELECT doc, term FROM token_vocab ORDER BY doc, offset"):
            surface = surfaces[rowid - 1]
            if stem not in stems[surface]:
                stems[surface].append(stem)
        return {surface: tuple(values) for surface, values in stems.items()}
    finally:
        tokenizer.close()


def _ensure_vocab(conn: sqlite3.Connection) -> None:
    """Create only a request-local temp view; never migrate the main index."""
    if conn.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE name = 'fidx_guidance_vocab'").fetchone():
        return
    query_only = bool(conn.execute("PRAGMA query_only").fetchone()[0])
    try:
        if query_only:
            # The connection URI remains mode=ro. SQLite's query_only pragma
            # also blocks temp-schema DDL, so briefly lift only that pragma.
            conn.execute("PRAGMA query_only=OFF")
        conn.execute(
            "CREATE VIRTUAL TABLE temp.fidx_guidance_vocab "
            "USING fts5vocab(main, 'documents_fts', 'instance')")
    finally:
        if query_only:
            conn.execute("PRAGMA query_only=ON")


def term_coverage(conn: sqlite3.Connection, query: str,
                  collections: Sequence[str] | None = None) -> TermCoverage:
    """Report Porter-stemmed DF over active documents in the searched scope."""
    surfaces = list(dict.fromkeys(_TOKEN_RE.findall(query)))
    if not surfaces:
        return TermCoverage((), (), (), {}, 0.0)
    stems = _surface_stems(surfaces)
    all_stems = sorted({stem for values in stems.values() for stem in values})
    _ensure_vocab(conn)
    dfs: dict[str, int] = {}
    scope = list(collections or ())
    collection_sql = ""
    if scope:
        collection_sql = f" AND d.collection IN ({','.join('?' for _ in scope)})"
    if all_stems:
        marks = ",".join("?" for _ in all_stems)
        dfs = {row["term"]: row["df"] for row in conn.execute(
            f"SELECT v.term, count(DISTINCT v.doc) AS df "
            f"FROM temp.fidx_guidance_vocab AS v "
            f"JOIN documents AS d ON d.id = v.doc "
            f"WHERE v.term IN ({marks}) AND d.active = 1{collection_sql} "
            f"GROUP BY v.term", [*all_stems, *scope])}
    doc_count = conn.execute(
        f"SELECT count(*) FROM documents AS d WHERE d.active = 1{collection_sql}",
        scope).fetchone()[0]
    rare_cutoff = max(3, 0.00001 * doc_count)
    term_df = {
        surface: min((dfs.get(stem, 0) for stem in stems[surface]), default=0)
        for surface in surfaces
    }
    matched = tuple(surface for surface in surfaces if term_df[surface] > 0)
    unmatched = tuple(surface for surface in surfaces if term_df[surface] == 0)
    rare = tuple(surface for surface in matched if term_df[surface] < rare_cutoff)
    return TermCoverage(matched, unmatched, rare, term_df,
                        len(matched) / len(surfaces))


def score_shape(scores: Sequence[float]) -> ScoreShape:
    """Bounded shape features on one channel's ranked scores."""
    if not scores:
        return ScoreShape(0.0, 0.0, 0.0, 0.0)
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    if len(scores) == 1:
        return ScoreShape(0.0, 0.0, 0.0, 1.0)
    low, high = min(scores), max(scores)
    if high == low:
        return ScoreShape(0.0, 0.0, 0.0, 1.0 / len(scores))
    normalized = [(score - low) / (high - low) for score in scores]
    mean = sum(normalized) / len(normalized)
    variance = sum((score - mean) ** 2 for score in normalized) / len(normalized)
    nqc = math.sqrt(variance) / mean if mean else 0.0
    wig = normalized[0] - mean
    gap = max(0.0, normalized[0] - normalized[1])
    concentration = normalized[0] / sum(normalized)
    return ScoreShape(nqc, wig, gap, concentration)


def compute_qpp(trace: SearchTrace, *, term_coverage_ratio: float) -> QPPSignals:
    if not 0.0 <= term_coverage_ratio <= 1.0:
        raise ValueError("term coverage ratio must be between 0 and 1")
    lexical_ids = {r.doc_id for r in trace.lexical}
    vector_ids = {r.doc_id for r in trace.vector}
    union = lexical_ids | vector_ids
    agreement = (1.0 if trace.mode in ("lexical", "vector") else
                 len(lexical_ids & vector_ids) / len(union) if union else 0.0)
    return QPPSignals(
        score_shape([r.score for r in trace.lexical]),
        score_shape([r.score for r in trace.vector]),
        score_shape([r.score for r in trace.fused_pool]),
        agreement,
        term_coverage_ratio,
    )


def miss_score(signals: QPPSignals, weights: Mapping[str, float]) -> float:
    """Weighted risk scaffold; caller supplies all calibration weights."""
    channels = [signals.lexical, signals.vector]
    values = {
        "flatness": 1.0 - max(channel.top_gap for channel in channels),
        "weak_head": 1.0 - max(channel.wig_like for channel in channels),
        "disagreement": 1.0 - signals.channel_agreement,
        "uncovered_terms": 1.0 - signals.term_coverage,
    }
    if any(key not in values or not math.isfinite(weight) or weight < 0
           for key, weight in weights.items()):
        raise ValueError("unknown, negative, or non-finite miss-score weight")
    total = sum(weights.values())
    return sum(values[key] * weight for key, weight in weights.items()) / total if total else 0.0
