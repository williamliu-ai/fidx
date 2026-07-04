"""Benchmark SymDex on a prepared corpus, mirroring run_bench.py's fidx schema.

SymDex (https://github.com/husnainpk/SymDex) indexes *symbols* (via tree-sitter)
plus local semantic embeddings (sentence-transformers, offline). For the code
corpus the queries are natural language -> expected *file*, so the comparable
retrieval modes are:

  - semantic : cosine similarity over symbol embeddings — the fair counterpart to
               fidx's vector arm / qmd `vsearch`. This is SymDex's only viable
               mode for NL queries.
  - text     : literal case-insensitive SUBSTRING match (NOT BM25). Reported for
               completeness; it returns ~nothing on NL queries by construction,
               which is itself the finding (SymDex has no tokenized lexical arm).

Recall is scored at FILE granularity (ranked symbols are de-duplicated to
distinct files, preserving order), identical to how run_bench scores fidx/qmd, so
the numbers are directly comparable. Per-corpus == per-language; run all ten
`code-<lang>` corpora for the per-language breakdown.

Resource usage captured (mirrors fidx JSON):
  index : index_seconds (wall), index_cpu_seconds (user+sys, rusage),
          storage_bytes (SQLite db size), memory_mb.index_peak, n_files,
          n_symbols, n_embeddings
  query : latency_ms {p50,p99,mean} (warm, in-process: embedding model loaded
          once), memory_mb.query_peak
  recall: recall@1/3/10 overall and by_type (phrase/unique/vague)

Usage:
  python bench/bench_symdex.py run --corpus code-node [--modes semantic,text]
  python bench/bench_symdex.py run --corpus code-node --reindex
Outputs bench/results/symdex-<corpus>.json (a list of per-mode report dicts).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bench import DATA, RESULTS, load_queries, hit_rank, summarize  # noqa: E402

# SymDex reads its state dir from $SYMDEX_STATE_DIR; set per-corpus so each
# language gets an isolated index/registry (mirrors run_bench's qmd cache).


def model_tag(model: str) -> str:
    """Short slug for an embedding model name; '' for the default MiniLM."""
    if model == "all-MiniLM-L6-v2":
        return ""
    base = model.split("/")[-1].lower()
    if "mpnet" in base:
        return "mpnet"
    return base.replace("_", "-")


def state_dir(corpus: str, tag: str = "") -> Path:
    suffix = f"-{tag}" if tag else ""
    return RESULTS / f"symdex-state{suffix}-{corpus}"


def db_path_for(corpus: str) -> str:
    from symdex.core.storage import get_db_path
    return get_db_path(corpus)


def dedup_files(symbols: list[dict], k: int) -> list[str]:
    """Ranked symbol dicts -> first `k` distinct files, preserving rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        f = s.get("file")
        if f and f not in seen:
            seen.add(f)
            out.append(f)
            if len(out) >= k:
                break
    return out


def do_index(corpus: str, sd: Path, reindex: bool) -> dict:
    """Index DATA/<corpus> with SymDex; return resource metrics."""
    from run_bench import run_measured
    db = db_path_for(corpus)
    if Path(db).exists() and not reindex:
        size = os.path.getsize(db)
        return {"index_seconds": None, "index_cpu_seconds": None,
                "storage_bytes": size, "index_peak_mb": None, "reused": True}
    sd.mkdir(parents=True, exist_ok=True)
    corpus_path = DATA / corpus
    cmd = ["symdex", "--state-dir", str(sd), "index", str(corpus_path),
           "--name", corpus, "--embed"]
    t0 = time.perf_counter()
    rc, _, rss, cpu = run_measured(cmd, env={**os.environ, "SYMDEX_STATE_DIR": str(sd)},
                                   capture=True, timeout=14400)
    wall = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"symdex index failed rc={rc} for {corpus}")
    size = os.path.getsize(db) if Path(db).exists() else 0
    return {"index_seconds": round(wall, 1), "index_cpu_seconds": round(cpu, 1),
            "storage_bytes": size, "index_peak_mb": round(rss / 1e6, 1), "reused": False}


def index_counts(corpus: str) -> dict:
    """Best-effort symbol/file/embedding counts from the SymDex SQLite db."""
    import sqlite3
    out = {"n_files": None, "n_symbols": None, "n_embeddings": None}
    try:
        conn = sqlite3.connect(db_path_for(corpus))
        cur = conn.cursor()
        out["n_symbols"] = cur.execute("SELECT count(*) FROM symbols").fetchone()[0]
        out["n_files"] = cur.execute("SELECT count(DISTINCT file) FROM symbols").fetchone()[0]
        try:
            out["n_embeddings"] = cur.execute(
                "SELECT count(*) FROM symbols WHERE embedding IS NOT NULL").fetchone()[0]
        except sqlite3.OperationalError:
            pass
        conn.close()
    except Exception:
        pass
    return out


def recall_ceiling(corpus: str, queries: list[dict]) -> dict:
    """Fraction of queries whose expected file has >=1 indexed symbol — the hard
    upper bound on SymDex semantic recall (symbol-less files are unreachable)."""
    import sqlite3
    out = {"reachable": None, "ceiling": None, "by_type": {}}
    try:
        c = sqlite3.connect(db_path_for(corpus))
        files = {r[0] for r in c.execute("SELECT DISTINCT file FROM symbols")}
        c.close()
    except Exception:
        return out

    def reachable(exp: str) -> bool:
        return any(f == exp or f.endswith("/" + exp) for f in files)
    hit = sum(1 for q in queries if reachable(q["expected"]))
    out["reachable"] = hit
    out["ceiling"] = round(hit / max(1, len(queries)), 4)
    for t in sorted({q.get("type", "?") for q in queries}):
        sub = [q for q in queries if q.get("type", "?") == t]
        out["by_type"][t] = round(sum(reachable(q["expected"]) for q in sub) / len(sub), 4)
    return out


def run_mode(corpus: str, queries: list[dict], mode: str, symbol_limit: int) -> dict:
    """Warm in-process query loop for one SymDex mode; returns a fidx-shaped report."""
    from symdex.core.storage import get_connection
    from symdex.search.semantic import search_semantic
    from symdex.search.text_search import search_text
    repo_root = str(DATA / corpus)
    conn = get_connection(db_path_for(corpus))

    def query_files(q: str) -> list[str]:
        if mode == "semantic":
            syms = search_semantic(conn, query=q, repo=corpus, limit=symbol_limit)
            return dedup_files(syms, 10)
        if mode == "text":
            try:
                hits = search_text(query=q, repo=corpus, repo_root=repo_root)
            except Exception:
                return []
            return dedup_files(hits, 10)
        raise ValueError(mode)

    # Warm-up: load embedding model / caches once, untimed.
    if queries:
        query_files(queries[0]["query"])

    rows, latencies = [], []
    for q in queries:
        t0 = time.perf_counter()
        paths = query_files(q["query"])
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({"qid": q["qid"], "type": q.get("type", "?"),
                     "rank": hit_rank(paths, q["expected"])})
    conn.close()
    rep = {"engine": "symdex", "corpus": corpus, "mode": mode,
           "symbol_limit": symbol_limit}
    rep.update(summarize(rows, latencies))
    rep["query_peak_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return rep


def emit_purity_lists(corpus: str, queries: list[dict], symbol_limit: int,
                      engine: str) -> int:
    """Append SymDex semantic top-10 file lists to the shared purity-lists.jsonl
    (engine=<engine>, mode=semantic) so judge_purity.py can judge them for the
    noise@10/clean@10 purity axis. Idempotent per (engine,corpus,qid)."""
    from symdex.core.storage import get_connection
    from symdex.search.semantic import search_semantic
    lists = RESULTS / "purity-lists.jsonl"
    have = set()
    if lists.exists():
        for line in lists.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("engine") == engine and r.get("corpus") == corpus:
                have.add(r["qid"])
    conn = get_connection(db_path_for(corpus))
    n = 0
    with lists.open("a") as fh:
        for q in queries:
            if q["qid"] in have:
                continue
            syms = search_semantic(conn, query=q["query"], repo=corpus, limit=symbol_limit)
            paths = dedup_files(syms, 10)
            fh.write(json.dumps({"corpus": corpus, "engine": engine,
                                 "mode": "semantic", "qid": q["qid"],
                                 "query": q["query"], "expected": q["expected"],
                                 "paths": paths}) + "\n")
            n += 1
    conn.close()
    return n


def cmd_run(args) -> None:
    corpus = args.corpus
    tag = model_tag(args.embed_model)
    sd = state_dir(corpus, tag)
    os.environ["SYMDEX_STATE_DIR"] = str(sd)
    if tag:  # non-default model
        os.environ["SYMDEX_EMBED_MODEL"] = args.embed_model
    engine = f"symdex-{tag}" if tag else "symdex"
    suffix = f"-{tag}" if tag else ""
    queries = load_queries(corpus, args.max_queries)
    idx = do_index(corpus, sd, args.reindex)
    idx["embed_model"] = args.embed_model
    idx.update(index_counts(corpus))
    ceiling = recall_ceiling(corpus, queries)
    modes = args.modes.split(",")
    reports = []
    for mode in modes:
        rep = run_mode(corpus, queries, mode, args.symbol_limit)
        rep["engine"] = engine
        rep["embed_model"] = args.embed_model
        rep["index"] = idx
        rep["recall_ceiling"] = ceiling
        reports.append(rep)
        s = rep
        print(f"[{corpus}] {engine} {mode:8} n={s['n']} "
              f"R@1/3/10={s['recall@1']:.3f}/{s['recall@3']:.3f}/{s['recall@10']:.3f} "
              f"p50={s['latency_ms']['p50']:.1f}ms")
    out = RESULTS / f"symdex{suffix}-{corpus}.json"
    out.write_text(json.dumps(reports, indent=1))
    print(f"wrote {out}  (model={args.embed_model}, index: {idx.get('index_seconds')}s wall, "
          f"{idx.get('storage_bytes', 0)/1e6:.0f} MB, {idx.get('n_symbols')} symbols)")
    if args.emit_purity:
        added = emit_purity_lists(corpus, queries, args.symbol_limit, engine)
        print(f"purity-lists: +{added} {engine}/semantic rows "
              f"(judge with: judge_purity.py judge; report with: judge_purity.py report)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="index + query one corpus")
    r.add_argument("--corpus", required=True, help="e.g. code-node")
    r.add_argument("--modes", default="semantic,text")
    r.add_argument("--symbol-limit", type=int, default=50,
                   help="symbols fetched per query before de-dup to 10 files")
    r.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                   help="local sentence-transformers model (e.g. all-mpnet-base-v2, 768-d)")
    r.add_argument("--max-queries", type=int, default=None)
    r.add_argument("--reindex", action="store_true")
    r.add_argument("--emit-purity", action="store_true",
                   help="append semantic top-10 lists to purity-lists.jsonl for judging")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
