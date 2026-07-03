"""Search: BM25 (FTS5), vector (sqlite-vec KNN), and hybrid RRF fusion.

All modes return document-level results. The latency-critical invariant: no
LLM calls anywhere in this path — one query embedding is the only model work,
and lexical-only search does no model work at all.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

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


_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


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
        ORDER BY rank LIMIT ?
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


def search_lexical(conn: sqlite3.Connection, query: str, collections: list[str] | None = None,
                   limit: int = CANDIDATES_PER_SOURCE) -> list[Result]:
    """Tiered BM25: exact phrase (terms adjacent, in order) ranks first, then
    documents matching ALL terms, then OR-matched documents fill (recall)."""
    and_expr = fts_match_expr(query, conjunctive=True)
    if and_expr is None:
        return []

    results: list[Result] = []
    seen: set[int] = set()

    def take(expr: str, tier: str) -> None:
        for r in _fts_query(conn, expr, collections, limit, tier=tier):
            if r.doc_id not in seen and len(results) < limit:
                seen.add(r.doc_id)
                results.append(r)

    tokens = _TOKEN_RE.findall(query)
    if len(tokens) > 1 and '"' not in query:
        take('"' + " ".join(tokens) + '"', tier="phrase")
    if len(results) < limit:
        take(and_expr, tier="and")
    if len(results) < limit:
        or_expr = fts_match_expr(query)
        if or_expr != and_expr:
            take(or_expr, tier="or")
    return results


def search_vector(conn: sqlite3.Connection, store, embedder: Embedder, query: str,
                  collections: list[str] | None = None,
                  limit: int = CANDIDATES_PER_SOURCE) -> list[Result]:
    qvec = embedder.embed_queries([query])[0]
    # Over-fetch chunks (docs have multiple chunks), then dedup to best per doc.
    knn = store.knn(qvec, limit * 4)
    if not knn:
        return []  # not embedded yet (`fidx index`); hybrid degrades to lexical
    chunk_ids = [cid for cid, _ in knn]
    distance = dict(knn)
    placeholders = ",".join("?" for _ in chunk_ids)
    clause, params = _collection_clause(collections)
    if not chunk_ids:
        return []
    meta = conn.execute(
        f"""
        SELECT c.id AS chunk_id, c.pos, c.length, d.id, d.collection, d.relpath, d.title, d.hash
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE c.id IN ({placeholders}) AND d.active = 1{clause}
        """,
        [*chunk_ids, *params],
    ).fetchall()

    best: dict[int, Result] = {}
    chunk_pos: dict[int, tuple[int, int]] = {}
    for r in meta:
        score = 1.0 - distance[r["chunk_id"]]  # cosine distance -> similarity
        if r["id"] not in best or score > best[r["id"]].score:
            best[r["id"]] = Result(
                doc_id=r["id"], collection=r["collection"], relpath=r["relpath"],
                title=r["title"], docid=r["hash"][:6], score=score,
            )
            chunk_pos[r["id"]] = (r["pos"], r["length"])
    results = sorted(best.values(), key=lambda x: x.score, reverse=True)[:limit]
    _attach_chunk_snippets(conn, results, chunk_pos)
    return results


def _attach_chunk_snippets(conn: sqlite3.Connection, results: list[Result],
                           chunk_pos: dict[int, tuple[int, int]]) -> None:
    for res in results:
        pos, length = chunk_pos[res.doc_id]
        row = conn.execute("SELECT substr(body, ?, ?) AS s FROM documents WHERE id = ?",
                           (pos + 1, min(length, 240), res.doc_id)).fetchone()
        res.snippet = " ".join(row["s"].split()) if row else ""


def search_hybrid(conn: sqlite3.Connection, store, embedder: Embedder, query: str,
                  collections: list[str] | None = None, limit: int = 10) -> list[Result]:
    lex = search_lexical(conn, query, collections)
    vec = search_vector(conn, store, embedder, query, collections)
    lex_weight = LEXICAL_AND_WEIGHT if (lex and lex[0].tier in ("phrase", "and")) else LEXICAL_WEIGHT
    return rrf_fuse([("lexical", lex_weight, lex), ("vector", VECTOR_WEIGHT, vec)], limit)


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
            else:
                fused[res.doc_id] = Result(
                    doc_id=res.doc_id, collection=res.collection, relpath=res.relpath,
                    title=res.title, docid=res.docid, score=contribution,
                    snippet=res.snippet, sources={source: res.score},
                )
    return sorted(fused.values(), key=lambda r: r.score, reverse=True)[:limit]
