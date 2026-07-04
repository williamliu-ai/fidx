"""Generalized offline knee-truncation eval + per-group (per-language) breakdown.

Re-queries the kept fidx e5-768/sqlite-vec index for scored top-10 lists, applies
`knee`, and scores recall@10 / noise@10 / clean@10 against existing judgments
(judge-free — knee only drops already-judged docs). With --by-group it also
stratifies by the expected doc's top-level dir (= programming language for the
`code` corpus; newsgroup for docs) and folds in qmd `query` from the saved
purity-lists for the same per-group comparison.

Usage: python bench/knee_eval.py <corpus> [--by-group]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "bench")
sys.path.insert(0, "src")
import judge_purity as J
from run_bench import RESULTS, load_queries
from fidx.cli import open_index
from fidx.embedder import FastEmbedder
from fidx import vector_store, search as S, truncate as T

LANG = {"golang": "Go", "home-assistant": "Python", "node": "JavaScript",
        "grafana": "TypeScript", "elasticsearch": "Java", "zephyr": "C",
        "clickhouse": "C++", "aspnetcore": "C#", "rust": "Rust", "symfony": "PHP"}

corpus = sys.argv[1]
by_group = "--by-group" in sys.argv
db = RESULTS / f"fidx-e5-768-sqlite-vec-{corpus}.db"
lab = {(r["corpus"], r["qid"], r["path"]): r["label"]
       for r in J.jsonl_read(J.JUDGMENTS)
       if r.get("qid") not in (None, "_") and "label" in r}

conn, prof = open_index(db)
store = vector_store.open_store(conn, db)
emb = FastEmbedder(prof)
qs = load_queries(corpus, None)
cap = [(q, S.search_hybrid(conn, store, emb, q["query"], None, 10)) for q in qs]
conn.close()


def score(items, kept_paths_fn):
    """items: list of (qid, expected, paths). kept_paths_fn already applied."""
    rec = clean = nq = 0
    noise_sum = 0.0
    for qid, expected, paths in items:
        nq += 1
        if any(J.is_expected(p, expected) for p in paths):
            rec += 1
        labs = [lab.get((corpus, qid, p)) for p in paths]
        labs = [x for x in labs if x is not None]
        nn = sum(1 for x in labs if x == "noise")
        if labs:
            noise_sum += nn / len(labs)
        if nn == 0:
            clean += 1
    return rec / nq, noise_sum / nq, clean / nq, nq


def fidx_items(spec):
    out = []
    for q, res in cap:
        kept = res[:10] if spec == "off" else T.truncate(list(res), spec, "hybrid")[:10]
        out.append((q["qid"], q["expected"], [k.relpath for k in kept]))
    return out


print(f"== {corpus}: knee truncation (fidx e5-768/sqlite-vec) ==")
for spec in ("off", "knee"):
    r, n, c, nq = score(fidx_items(spec), None)
    print(f"  fidx {spec:5} R@10={r:.3f} noise@10={n:.3f} clean@10={c:.3f}  (n={nq})")

if by_group:
    # qmd query from saved purity-lists (paths already resolved/judged)
    qmd_q = {r["qid"]: (r["expected"], r["paths"])
             for r in J.jsonl_read(RESULTS / "purity-lists.jsonl")
             if r.get("corpus") == corpus and r["engine"] == "qmd" and r["mode"] == "query"}
    grp_off = defaultdict(list); grp_knee = defaultdict(list); grp_qmd = defaultdict(list)
    for qid, exp, paths in fidx_items("off"):
        grp_off[exp.split("/")[0]].append((qid, exp, paths))
    for qid, exp, paths in fidx_items("knee"):
        grp_knee[exp.split("/")[0]].append((qid, exp, paths))
    for qid, (exp, paths) in qmd_q.items():
        grp_qmd[exp.split("/")[0]].append((qid, exp, paths))

    print(f"\n== {corpus}: per-group metrics (group = expected top-level dir) ==")
    print("group(lang)        n | fidx off R@10/noise/clean | fidx+knee R@10/noise/clean | qmd query R@10/noise/clean")
    for g in sorted(grp_off):
        o = score(grp_off[g], None)
        k = score(grp_knee[g], None)
        ql = grp_qmd.get(g)
        qs_ = score(ql, None) if ql else (0, 0, 0, 0)
        lang = LANG.get(g, g)
        print(f"  {lang:14} {o[3]:4} | {o[0]:.2f}/{o[1]:.2f}/{o[2]:.2f}        | "
              f"{k[0]:.2f}/{k[1]:.2f}/{k[2]:.2f}          | {qs_[0]:.2f}/{qs_[1]:.2f}/{qs_[2]:.2f}")
