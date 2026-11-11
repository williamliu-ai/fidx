# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Search: BM25 (FTS5), vector (sqlite-vec KNN), and hybrid RRF fusion.

All modes return document-level results. The latency-critical invariant: no
LLM calls anywhere in this path — one query embedding is the only model work,
and lexical-only search does no model work at all.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from math import isfinite

from .db import has_chunk_line_columns, read_snapshot, subtokens_ready
from .subtokens import TOKEN_RE as _TOKEN_RE, expand_query, subtoken_match_expr
from .embedder import Embedder

RRF_K = 60
LEXICAL_WEIGHT = 1.0
# A document matching ALL query terms is high-precision evidence; when the
# lexical list leads with such a match, lexical outvotes the vector list.
LEXICAL_AND_WEIGHT = 2.0
VECTOR_WEIGHT = 1.0
# Per-source top-rank bonus keeps each source's confident leaders from being
# averaged away by the other list (same idea as QMD's RRF bonus).
TOP_RANK_BONUS = {0: 0.05, 1: 0.02, 2: 0.02}
CANDIDATES_PER_SOURCE = 50
SNIPPET_TOKENS = 16
SUBTOKEN_WEIGHT = 0.5


@dataclass
class Result:
    doc_id: int
    collection: str
    relpath: str
    title: str
    docid: str  # short content-hash id, stable across path moves
    score: float
    snippet: str = ""
    sources: dict[str, float] = field(default_factory=dict)
    tier: str = ""  # lexical: "and" (all terms matched) or "or" (fill)
    matched_chunk_id: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class SearchTrace:
    """Opt-in candidate evidence; fetched-pool counts are lower bounds."""

    mode: str = "hybrid"
    lexical: tuple[Result, ...] = ()
    vector: tuple[Result, ...] = ()
    fused_pool: tuple[Result, ...] = ()
    count_relation: str = "gte"


def fts_match_expr(query: str, conjunctive: bool = False) -> str | None:
    """Build a safe FTS5 MATCH expression from untrusted input.

    conjunctive=False: quoted tokens OR-joined (recall-oriented fill).
    conjunctive=True: implicit AND — every term must appear (precision tier).
    User phrases in double quotes are preserved as phrase queries either way."""
    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]+"', " ", query)
    # Inside FTS5 double quotes only token characters are safe; characters
    # like * : ^ would change query semantics or raise syntax errors.
    cleaned = [" ".join(_TOKEN_RE.findall(p)) for p in phrases]
    terms = [f'"{p}"' for p in cleaned if p]
    terms += [f'"{t}"' for t in _TOKEN_RE.findall(rest)]
    if not terms:
        return None
    return " ".join(terms) if conjunctive else " OR ".join(terms)


def _collection_clause(collections: list[str] | None) -> tuple[str, list[str]]:
    if not collections:
        return "", []
    placeholders = ",".join("?" for _ in collections)
    return f" AND d.collection IN ({placeholders})", list(collections)


def _fts_query(conn: sqlite3.Connection, expr: str, collections: list[str] | None,
               limit: int, tier: str) -> list[Result]:
    clause, params = _collection_clause(collections)
    rows = conn.execute(
        f"""
        SELECT d.id, d.collection, d.relpath, d.title, d.hash,
               bm25(documents_fts, 4.0, 1.0, 2.0) AS rank,
               snippet(documents_fts, 1, '', '', '…', {SNIPPET_TOKENS}) AS snip
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ? AND d.active = 1{clause}
        ORDER BY rank, d.collection, d.relpath LIMIT ?
        """,
        [expr, *params, limit],
    ).fetchall()
    results = []
    for r in rows:
        raw = -r["rank"]  # bm25() is negative; larger magnitude = better
        results.append(
            Result(
                doc_id=r["id"], collection=r["collection"], relpath=r["relpath"],
                title=r["title"], docid=r["hash"][:6],
                score=raw / (1.0 + raw) if raw > 0 else 0.0,
                snippet=r["snip"], tier=tier,
            )
        )
    return results


def _sub_fts_query(conn: sqlite3.Connection, expr: str, collections: list[str] | None,
                   limit: int, tier: str) -> list[Result]:
    clause, params = _collection_clause(collections)
    rows = conn.execute(
        f"""
        SELECT d.id, d.collection, d.relpath, d.title, d.hash,
               bm25(documents_sub_fts) AS rank
        FROM documents_sub_fts JOIN documents d ON d.id = documents_sub_fts.rowid
        WHERE documents_sub_fts MATCH ? AND d.active = 1{clause}
        ORDER BY rank, d.collection, d.relpath LIMIT ?
        """, [expr, *params, limit],
    ).fetchall()
    results = []
    for row in rows:
        raw = -row["rank"] * SUBTOKEN_WEIGHT
        results.append(Result(
            row["id"], row["collection"], row["relpath"], row["title"], row["hash"][:6],
            raw / (1 + raw) if raw > 0 else 0.0, tier=tier))
    return results


def _lexical_tier(conn, expr, sub_expr, collections, limit, tier):
    main = _fts_query(conn, expr, collections, limit, tier)
    if sub_expr is None:
        return main
    candidates = {r.doc_id: r for r in main}
    for sub in _sub_fts_query(conn, sub_expr, collections, limit, tier):
        if sub.doc_id in candidates:
            result = candidates[sub.doc_id]
            # Mapping raw/(1+raw) is monotone, so max commutes with it.
            result.score = max(result.score, sub.score)
        else:
            # Only original body snippets escape this module. An expanded
            # MATCH by rowid is cheap; a camel-only body may have no match.
            row = conn.execute(
                f"SELECT snippet(documents_fts, 1, '', '', '…', {SNIPPET_TOKENS}) "
                "FROM documents_fts WHERE rowid = ? AND documents_fts MATCH ?",
                (sub.doc_id, sub_expr)).fetchone()
            sub.snippet = row[0] if row else ""
            candidates[sub.doc_id] = sub
    return sorted(candidates.values(), key=lambda r: (-r.score, r.collection, r.relpath))


def search_lexical(conn: sqlite3.Connection, query: str, collections: list[str] | None = None,
                   limit: int = CANDIDATES_PER_SOURCE) -> list[Result]:
    """Union main and optional subtoken evidence within phrase → AND → OR tiers."""
    with read_snapshot(conn):
        return _search_lexical(conn, query, collections, limit)


def _search_lexical(conn: sqlite3.Connection, query: str, collections: list[str] | None,
                    limit: int) -> list[Result]:
    and_expr = fts_match_expr(query, conjunctive=True)
    if and_expr is None:
        return []
    enabled = subtokens_ready(conn)
    sub_and = subtoken_match_expr(query, True) if enabled else None
    results: list[Result] = []
    seen: set[int] = set()

    def take(expr: str, sub_expr: str | None, tier: str) -> None:
        for r in _lexical_tier(conn, expr, sub_expr, collections, limit, tier):
            if r.doc_id not in seen and len(results) < limit:
                seen.add(r.doc_id)
                results.append(r)

    tokens = _TOKEN_RE.findall(query)
    if len(tokens) > 1 and '"' not in query:
        sub_tokens = _TOKEN_RE.findall(expand_query(query)) if enabled else []
        take('"' + " ".join(tokens) + '"',
             '"' + " ".join(sub_tokens) + '"' if sub_tokens else None, "phrase")
    if len(results) < limit:
        take(and_expr, sub_and, "and")
    if len(results) < limit:
        or_expr = fts_match_expr(query)
        sub_or = subtoken_match_expr(query) if enabled else None
        if or_expr != and_expr or sub_or != sub_and:
            take(or_expr, sub_or, "or")
    return results


def search_vector(conn: sqlite3.Connection, store, embedder: Embedder, query: str,
                  collections: list[str] | None = None,
                  limit: int = CANDIDATES_PER_SOURCE) -> list[Result]:
    if limit <= 0:
        return []
    qvec = embedder.embed_queries([query])[0]
    # Keep the existing four-chunks-per-document candidate cap. Exact sqlite-vec
    # needs one lookahead row: a tied cutoff can otherwise select arbitrary
    # chunk IDs before their document paths are available for ordering.
    cap = limit * 4
    if getattr(store, "name", None) == "sqlite-vec":
        if cap + 1 > store.KNN_MAX_K:
            knn = store.scan(qvec)
        else:
            window = cap
            knn = store.knn(qvec, window + 1)
            if len(knn) > cap:
                boundary = knn[cap - 1][1]
                # Follow only the cohort that straddles the original cap.
                # A separate tie farther down need not be completed.
                while len(knn) > window and knn[window][1] == boundary:
                    window *= 2
                    if window + 1 > store.KNN_MAX_K:
                        knn = store.scan(qvec)
                        break
                    knn = store.knn(qvec, window + 1)
    else:
        # DuckDB's HNSW may hide tied neighbours behind farther candidates.
        # Its returned set is sorted below; membership remains approximate.
        knn = store.knn(qvec, cap)
    if not knn:
        return []  # not embedded yet (`fidx index`); hybrid degrades to lexical
    chunk_ids = [cid for cid, _ in knn]
    distance = dict(knn)
    meta = []
    line_fields = ("c.line_start, c.line_end" if has_chunk_line_columns(conn)
                   else "NULL AS line_start, NULL AS line_end")
    # A complete tie cohort can exceed SQLite's host-parameter limit.
    for start in range(0, len(chunk_ids), 900):
        batch = chunk_ids[start:start + 900]
        placeholders = ",".join("?" for _ in batch)
        meta.extend(conn.execute(
            f"""
            SELECT c.id AS chunk_id, c.seq, c.pos, c.length, d.id, d.collection,
                   d.relpath, d.title, d.hash, d.active, {line_fields}
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders})
            """,
            batch,
        ).fetchall())
    meta.sort(key=lambda r: (distance[r["chunk_id"]], r["collection"],
                             r["relpath"], r["seq"]))
    allowed = set(collections) if collections else None

    best: dict[int, Result] = {}
    chunk_pos: dict[int, tuple[int, int]] = {}
    for r in meta[:cap]:
        if not r["active"] or (allowed is not None and r["collection"] not in allowed):
            continue
        score = 1.0 - distance[r["chunk_id"]]  # cosine distance -> similarity
        if r["id"] not in best:
            best[r["id"]] = Result(
                doc_id=r["id"], collection=r["collection"], relpath=r["relpath"],
                title=r["title"], docid=r["hash"][:6], score=score,
                matched_chunk_id=r["chunk_id"], line_start=r["line_start"],
                line_end=r["line_end"],
            )
            chunk_pos[r["id"]] = (r["pos"], r["length"])
    results = sorted(best.values(), key=lambda r: (-r.score, r.collection, r.relpath))[:limit]
    _attach_chunk_snippets(conn, results, chunk_pos)
    return results


def _attach_chunk_snippets(conn: sqlite3.Connection, results: list[Result],
                           chunk_pos: dict[int, tuple[int, int]]) -> None:
    for res in results:
        pos, length = chunk_pos[res.doc_id]
        row = conn.execute("SELECT substr(body, ?, ?) AS s FROM documents WHERE id = ?",
                           (pos + 1, min(length, 240), res.doc_id)).fetchone()
        res.snippet = " ".join(row["s"].split()) if row else ""


def _normalized_weights(weights: tuple[float, float]) -> tuple[float, float]:
    if len(weights) != 2 or any(not isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("fusion requires two finite, nonnegative weights with a positive sum")
    if sum(weights) == 1.0:
        return weights
    # Scale first so even two very large finite weights cannot overflow.
    scale = max(weights)
    scaled = tuple(w / scale for w in weights)
    total = sum(scaled)
    left, right = scaled[0] / total, scaled[1] / total
    # Pin the larger component to the complement, retaining tiny weights and
    # making canonical strings stable across CLI/server parsing.
    return (left, 1.0 - left) if left < right else (1.0 - right, right)


def parse_fusion(spec: str) -> str:
    """Canonical rrf or weighted weights (shortest round-trip repr; 0/1 endpoints)."""
    if spec == "rrf":
        return spec
    if not isinstance(spec, str) or not spec.startswith("weighted:"):
        raise ValueError("fusion must be rrf or weighted:W_LEX,W_VEC")
    try:
        weights = tuple(float(w) for w in spec.removeprefix("weighted:").split(","))
        normalized = _normalized_weights(weights)
    except (ValueError, OverflowError) as exc:
        raise ValueError("fusion requires two finite, nonnegative weights with a positive sum") from exc

    def spelling(w: float) -> str:
        return str(int(w)) if w in (0, 1) else repr(w)
    return "weighted:" + ",".join(spelling(w) for w in normalized)


def weighted_fuse(ranked_lists: list[tuple[str, float, list[Result]]],
                  weights: tuple[float, float], limit: int) -> list[Result]:
    """Min-max source scores; retain RRF metadata and its deterministic tie order."""
    normalized_weights = dict(zip(("lexical", "vector"), _normalized_weights(weights)))
    scores: dict[int, float] = {}
    for source, _, results in ranked_lists:
        if not results:
            continue
        low, high = min(r.score for r in results), max(r.score for r in results)
        for res in results:
            normalized = (res.score - low) / (high - low) if high != low else 1.0
            scores[res.doc_id] = scores.get(res.doc_id, 0.0) + normalized_weights[source] * normalized
    # RRF allocates fresh Results and owns all snippet/chunk/source merging.
    fused = rrf_fuse(ranked_lists, len(scores))
    order = {res.doc_id: rank for rank, res in enumerate(fused)}
    for res in fused:
        res.score = scores[res.doc_id]
    return sorted(fused, key=lambda r: (-r.score, order[r.doc_id], r.collection, r.relpath))[:limit]


def search_hybrid(conn: sqlite3.Connection, store, embedder: Embedder, query: str,
                  collections: list[str] | None = None, limit: int = 10,
                  *, trace: SearchTrace | None = None, fusion: str = "rrf") -> list[Result]:
    fusion = parse_fusion(fusion)
    lex = search_lexical(conn, query, collections)
    vec = search_vector(conn, store, embedder, query, collections)
    lex_weight = LEXICAL_AND_WEIGHT if (lex and lex[0].tier in ("phrase", "and")) else LEXICAL_WEIGHT
    ranked_lists = [("lexical", lex_weight, lex), ("vector", VECTOR_WEIGHT, vec)]
    fuse = rrf_fuse
    if fusion != "rrf":
        weights = tuple(float(w) for w in fusion.removeprefix("weighted:").split(","))
        fuse = lambda lists, n: weighted_fuse(lists, weights, n)
    if trace is None:
        return fuse(ranked_lists, limit)
    trace.lexical = tuple(lex)
    trace.vector = tuple(vec)
    trace.fused_pool = tuple(fuse(ranked_lists, len({r.doc_id for r in lex + vec})))
    return list(trace.fused_pool[:limit])


def rrf_fuse(ranked_lists: list[tuple[str, float, list[Result]]], limit: int) -> list[Result]:
    """Reciprocal-rank fusion across ranked result lists.

    score = sum over lists of weight / (RRF_K + rank + 1). Snippets prefer the
    lexical source (keyword-highlighted) and fall back to the vector chunk.
    """
    fused: dict[int, Result] = {}
    for source, weight, results in ranked_lists:
        for rank, res in enumerate(results):
            contribution = weight / (RRF_K + rank + 1) + TOP_RANK_BONUS.get(rank, 0.0)
            if res.doc_id in fused:
                f = fused[res.doc_id]
                f.score += contribution
                f.sources[source] = res.score
                if not f.snippet and res.snippet:
                    f.snippet = res.snippet
                if res.matched_chunk_id is not None:
                    f.matched_chunk_id = res.matched_chunk_id
                    f.line_start = res.line_start
                    f.line_end = res.line_end
            else:
                fused[res.doc_id] = Result(
                    doc_id=res.doc_id, collection=res.collection, relpath=res.relpath,
                    title=res.title, docid=res.docid, score=contribution,
                    snippet=res.snippet, sources={source: res.score},
                    matched_chunk_id=res.matched_chunk_id,
                    line_start=res.line_start, line_end=res.line_end,
                )
    return sorted(fused.values(), key=lambda r: (-r.score, r.collection, r.relpath))[:limit]
