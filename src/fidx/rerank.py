# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Dark document rerank stage; model activation remains outside this module."""

from __future__ import annotations

import math
import numbers
import sqlite3
from dataclasses import dataclass, replace
from typing import Iterable, Protocol, Sequence

from .db import has_chunk_line_columns
from .search import Result


class Reranker(Protocol):
    """Matches FastEmbed's cross-encoder `rerank(query, documents)` seam."""

    def rerank(self, query: str, documents: list[str]) -> Iterable[float]: ...


class NullReranker:
    """Offline test seam: equal logits retain the incoming fused order."""

    def rerank(self, query: str, documents: list[str]) -> Iterable[float]:
        return (0.0 for _ in documents)


@dataclass(frozen=True)
class RerankOutcome:
    results: tuple[Result, ...]
    failed: bool


@dataclass(frozen=True)
class _ScoredChunk:
    text: str
    chunk_id: int | None
    line_start: int | None
    line_end: int | None


def _scored_chunk(conn: sqlite3.Connection, result: Result,
                  with_lines: bool) -> _ScoredChunk:
    """Fetch the full indexed body slice and its inclusive line location."""
    line_sql = ("c.line_start, c.line_end" if with_lines else "NULL, NULL")
    if result.matched_chunk_id is not None:
        row = conn.execute(
            f"SELECT d.body, c.id, c.pos, c.length, {line_sql} "
            "FROM chunks AS c JOIN documents AS d ON d.id = c.doc_id "
            "WHERE c.doc_id = ? AND c.id = ?",
            (result.doc_id, result.matched_chunk_id)).fetchone()
        if row is None:
            raise ValueError("scored vector chunk is missing")
    else:
        row = conn.execute(
            f"SELECT d.body, c.id, c.pos, c.length, {line_sql} "
            "FROM documents AS d LEFT JOIN chunks AS c ON c.doc_id = d.id "
            "WHERE d.id = ? ORDER BY c.seq, c.id LIMIT 1",
            (result.doc_id,)).fetchone()
        if row is None:
            raise ValueError("scored document is missing")
    body, chunk_id, pos, length, line_start, line_end = tuple(row)
    if not isinstance(body, str):
        raise ValueError("scored document body is invalid")
    if chunk_id is None:
        return _ScoredChunk(result.title, None, None, None)
    if (not isinstance(pos, int) or not isinstance(length, int)
            or pos < 0 or length < 0 or pos + length > len(body)):
        raise ValueError("scored chunk bounds are invalid")
    chunk_text = body[pos:pos + length]
    if result.matched_chunk_id is None:
        text = f"{result.title}\n{chunk_text}" if result.title else chunk_text
    else:
        text = chunk_text
    if (not isinstance(line_start, int) or not isinstance(line_end, int)
            or line_start < 1 or line_end < line_start):
        line_start = line_end = None
    return _ScoredChunk(text, chunk_id, line_start, line_end)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def rerank_results(conn: sqlite3.Connection, query: str,
                   candidates: Sequence[Result], mode: str,
                   scorer: Reranker) -> RerankOutcome:
    """Rerank atomically, returning original objects/order on stage failure.

    The caller owns retrieval at the final user limit if failure occurs after
    a wider over-fetch. No candidate or trace object is ever changed here.
    """
    if mode not in {"lexical", "vector", "hybrid"}:
        raise ValueError("invalid rerank mode")
    original = tuple(candidates)
    if not original:
        return RerankOutcome((), False)
    try:
        with_lines = has_chunk_line_columns(conn)
        chunks = [_scored_chunk(conn, result, with_lines) for result in original]
        scores = list(scorer.rerank(query, [chunk.text for chunk in chunks]))
        if len(scores) != len(original):
            raise ValueError("reranker score count differs from candidates")
        logits: list[float] = []
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, numbers.Real):
                raise ValueError("reranker score is not numeric")
            logit = float(score)
            if not math.isfinite(logit):
                raise ValueError("reranker score is not finite")
            logits.append(logit)
        ranked: list[tuple[float, int, Result]] = []
        for ordinal, (result, chunk, logit) in enumerate(zip(original, chunks, logits)):
            if mode == "hybrid":
                sources = dict(result.sources)
                sources["fused"] = result.score
            else:
                # Direct search has one arm and no fused aggregate.
                sources = {mode: result.score}
            sources["rerank"] = logit
            copied = replace(
                result, score=_sigmoid(logit), sources=sources,
                matched_chunk_id=chunk.chunk_id,
                line_start=chunk.line_start, line_end=chunk.line_end,
            )
            ranked.append((logit, ordinal, copied))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return RerankOutcome(tuple(row[2] for row in ranked), False)
    except Exception:
        # Failures may occur while iterating a lazy cross-encoder generator.
        # Neither the partial score list nor partial copies escape.
        return RerankOutcome(original, True)
