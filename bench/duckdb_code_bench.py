# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Benchmark the duckdb-HNSW backend on the 92k-file code corpus by REUSING the
embeddings already in the sqlite-vec code index (no 6.6h re-embed).

This isolates the vector backend: identical e5-768 vectors and the identical
canonical hybrid search path (`search.search_hybrid`), only the vector store
differs (sqlite-vec brute force -> duckdb VSS/HNSW). It answers the open
question from docs/BENCHMARKS.md — does HNSW win past the ~30k-chunk crossover? —
at 460k vectors, where sqlite-vec measured p50 451.8 ms.

Recall is computed exactly as run_bench/bench_fidx does (path = collection/relpath,
hit_rank vs expected). Output JSON mirrors the existing duckdb result files.

Run with the fidx venv (needs sqlite_vec + duckdb + fastembed):
  .venv/bin/python bench/duckdb_code_bench.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
sys.path.insert(0, "src")
from run_bench import RESULTS, load_queries, hit_rank, percentile  # noqa: E402
from fidx import db as dbmod, config, search as S  # noqa: E402
from fidx.embedder import FastEmbedder  # noqa: E402
from fidx.vector_store import DuckDBStore, sidecar_path  # noqa: E402

CORPUS = "code"
PROFILE = "e5-768"
DIM = 768
SRC_DB = RESULTS / f"fidx-{PROFILE}-sqlite-vec-{CORPUS}.db"
DUCK_DB = RESULTS / f"fidx-{PROFILE}-duckdb-{CORPUS}.db"   # logical handle for sidecar path
SIDECAR = sidecar_path(DUCK_DB)


def build_sidecar(src_conn) -> tuple[float, float, int]:
    """Stream vec0 vectors from the sqlite-vec index into a fresh duckdb HNSW
    sidecar. Returns (insert_seconds, hnsw_build_seconds, n_vectors)."""
    import duckdb
    if SIDECAR.exists():
        SIDECAR.unlink()
    total = src_conn.execute("SELECT count(*) FROM vectors").fetchone()[0]
    d = duckdb.connect(str(SIDECAR))
    d.execute("INSTALL vss; LOAD vss")
    d.execute("SET hnsw_enable_experimental_persistence = true")
    d.execute(f"CREATE TABLE vectors (chunk_id BIGINT PRIMARY KEY, embedding FLOAT[{DIM}])")
    cur = src_conn.execute("SELECT chunk_id, embedding FROM vectors")
    t0 = time.time()
    n = 0
    B = 5000
    while True:
        rows = cur.fetchmany(B)
        if not rows:
            break
        batch = [(int(cid), np.frombuffer(emb, dtype=np.float32).tolist()) for cid, emb in rows]
        d.executemany("INSERT INTO vectors VALUES (?, ?)", batch)
        n += len(batch)
        if n % 50000 == 0 or n == total:
            print(f"  inserted {n}/{total}", flush=True)
    insert_s = time.time() - t0
    t1 = time.time()
    d.execute("CREATE INDEX vectors_hnsw ON vectors USING HNSW (embedding) WITH (metric = 'cosine')")
    d.close()
    hnsw_s = time.time() - t1
    print(f"  insert {insert_s:.0f}s | hnsw build {hnsw_s:.0f}s | {n} vectors", flush=True)
    return insert_s, hnsw_s, n


def main() -> None:
    if not SRC_DB.exists():
        sys.exit(f"missing {SRC_DB} — need the sqlite-vec code index to reuse embeddings")
    conn = dbmod.connect(SRC_DB)               # FTS + documents + chunks (+ unused vec0)
    insert_s, hnsw_s, nvec = build_sidecar(conn)

    prof = config.get_profile(PROFILE)
    emb = FastEmbedder(prof)
    store = DuckDBStore(SIDECAR)
    queries = load_queries(CORPUS, None)

    # warm-up (model load + first duckdb connect), untimed
    S.search_hybrid(conn, store, emb, queries[0]["query"], None, 10)

    rows, latencies = [], []
    for q in queries:
        t = time.perf_counter()
        res = S.search_hybrid(conn, store, emb, q["query"], None, 10)
        latencies.append((time.perf_counter() - t) * 1000)
        paths = [f"{r.collection}/{r.relpath}" for r in res]
        rows.append({"qid": q["qid"], "type": q.get("type", "?"),
                     "rank": hit_rank(paths, q["expected"])})

    def recall(k):
        return round(sum(1 for r in rows if r["rank"] is not None and r["rank"] < k) / len(rows), 4)

    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        by_type[t] = {f"recall@{k}": round(sum(1 for r in sub if r["rank"] is not None and r["rank"] < k) / len(sub), 4)
                      for k in (1, 3, 10)}

    report = {
        "engine": "fidx", "corpus": CORPUS, "mode": f"hybrid/{PROFILE}/duckdb",
        "vector_reuse": True,
        "note": "embeddings reused from the sqlite-vec code index (not re-embedded); "
                "index_seconds below is the HNSW build only — a from-scratch duckdb "
                "index would add the same ~6.6h e5-768 embed as sqlite-vec.",
        "hnsw_insert_seconds": round(insert_s, 1),
        "hnsw_build_seconds": round(hnsw_s, 1),
        "index_seconds": round(insert_s + hnsw_s, 1),
        "n_vectors": nvec,
        "storage_bytes_sidecar": SIDECAR.stat().st_size if SIDECAR.exists() else 0,
        "n": len(rows),
        "recall@1": recall(1), "recall@3": recall(3), "recall@10": recall(10),
        "latency_ms": {"p50": round(percentile(latencies, 50), 1),
                       "p99": round(percentile(latencies, 99), 1),
                       "mean": round(statistics.fmean(latencies), 1)},
        "by_type": by_type,
    }
    out = RESULTS / f"fidx-{PROFILE}-duckdb-{CORPUS}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nduckdb-HNSW code: R@1/3/10 = {report['recall@1']}/{report['recall@3']}/{report['recall@10']} "
          f"| p50 {report['latency_ms']['p50']} ms | sidecar {report['storage_bytes_sidecar']/1e9:.2f} GB")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
