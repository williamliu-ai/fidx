# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Offline sweep of fidx truncation methods — judge-free, reuses prior LLM labels.

Truncation only ever DROPS docs from fidx's existing top-10, and every doc in
that top-10 is already labelled in purity-judgments.jsonl (real queries) and
nomatch-judgments.jsonl (the 2026-06-15 unanswerable probe). So we can evaluate
any score-based truncation method offline:

  capture: re-query the kept fidx e5-768/sqlite-vec docs/chat indexes (fast, no
           re-index) for every real + no-match query, recording the top-10 with
           per-doc score and per-source scores -> truncation-capture.jsonl
  sweep:   apply each method/param to the captured lists, score against the
           reused judgments (recall@10, noise@10, clean@10, abstain%, rel-FP),
           and rank candidates.

Apple-to-apple framing: hybrid truncation is compared against QMD `query`;
vector truncation against QMD `vsearch` (numbers from the prior purity report).

Usage: .venv/bin/python bench/truncation_sweep.py [capture|sweep|all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import judge_purity as J
from run_bench import DATA, RESULTS, load_queries
from fidx.cli import open_index
from fidx.embedder import FastEmbedder
from fidx import vector_store, search as searchmod, truncate as truncatemod

CAPTURE = RESULTS / "truncation-capture.jsonl"
COMBO = ("e5-768", "sqlite-vec")          # the combo with full judgment coverage
CORPORA = ["docs", "chat"]
MODES = ["hybrid", "vector"]
TOPK = 10

# Parameter grid for the sweep (edit to iterate).
GRID = {
    "hybrid": [
        "off",
        "abs:0.02", "abs:0.03", "abs:0.05",
        "ratio:0.3", "ratio:0.5", "ratio:0.6", "ratio:0.7",
        "gap:0.4", "gap:0.5", "gap:0.55", "gap:0.6",
        "knee",
        "mad:2", "mad:2.5", "mad:3",
        "source:0.2,0.2", "source:0.3,0.2", "source:0.4,0.2", "source:0.5,0.2",
    ],
    "vector": [
        "off",
        "source:0.2", "source:0.3", "source:0.4", "source:0.5",
        "ratio:0.7", "ratio:0.85", "knee", "mad:2",
    ],
}


# ----------------------------------------------------------------- capture ---

def capture() -> None:
    CAPTURE.unlink(missing_ok=True)
    for corpus in CORPORA:
        db = RESULTS / f"fidx-{COMBO[0]}-{COMBO[1]}-{corpus}.db"
        if not db.exists():
            print(f"SKIP {corpus}: no index {db}", file=sys.stderr)
            continue
        conn, prof = open_index(db)
        store = vector_store.open_store(conn, db)
        embedder = FastEmbedder(prof)
        hmap = J.unhandelize_map(corpus)
        real = [dict(q, set="real") for q in load_queries(corpus, None)]
        nomatch = [dict(q, set="nomatch")
                   for q in J.jsonl_read(DATA / f"queries-{corpus}-nomatch.jsonl")]
        queries = real + nomatch
        for mode in MODES:
            for i, q in enumerate(queries, 1):
                if mode == "hybrid":
                    res = searchmod.search_hybrid(conn, store, embedder, q["query"], None, TOPK)
                else:
                    res = searchmod.search_vector(conn, store, embedder, q["query"], None, TOPK)
                rows = [{"path": J.real_path(f"{r.collection}/{r.relpath}", hmap, corpus)
                         or f"{r.collection}/{r.relpath}",
                         "score": round(r.score, 6), "sources": r.sources} for r in res]
                J.jsonl_append(CAPTURE, {"corpus": corpus, "mode": mode, "set": q["set"],
                                         "qid": q["qid"], "query": q["query"],
                                         "expected": q["expected"], "results": rows})
                if i % 200 == 0:
                    print(f"  {corpus}/{mode}: {i}/{len(queries)}", file=sys.stderr)
        conn.close()
    print(f"capture done -> {CAPTURE}", file=sys.stderr)


# ------------------------------------------------------------------- sweep ---

def _load_labels() -> tuple[dict, dict]:
    """real: (corpus,qid,path)->label.  nomatch: (corpus,query)->{path:label}."""
    real = {(r["corpus"], r["qid"], r["path"]): r["label"]
            for r in J.jsonl_read(J.JUDGMENTS) if "label" in r and r.get("qid") not in (None, "_")}
    nomatch: dict = {}
    for r in J.jsonl_read(RESULTS / "nomatch-judgments.jsonl"):
        if "labels" in r:
            nomatch[(r["corpus"], r["query"])] = r["labels"]
    return real, nomatch


def _result_to_truncatable(rows: list[dict]) -> list[searchmod.Result]:
    return [searchmod.Result(0, "", row["path"], "", "", row["score"],
                             sources=row.get("sources") or {}) for row in rows]


def sweep() -> None:
    cap = J.jsonl_read(CAPTURE)
    real_lab, nomatch_lab = _load_labels()
    # index capture rows by (corpus, mode, set)
    out_lines = []
    for corpus in CORPORA:
        for mode in MODES:
            rows_real = [c for c in cap if c["corpus"] == corpus and c["mode"] == mode and c["set"] == "real"]
            rows_nom = [c for c in cap if c["corpus"] == corpus and c["mode"] == mode and c["set"] == "nomatch"]
            if not rows_real:
                continue
            out_lines.append(f"\n### {corpus} / fidx {mode}  (vs QMD {'query' if mode=='hybrid' else 'vsearch'})")
            out_lines.append("| method | recall@10 | noise@10 | clean@10 | avg_len | nm-abstain% | nm-relFP | uncov |")
            out_lines.append("|---|---|---|---|---|---|---|---|")
            for spec in GRID[mode]:
                rec = noise_sum = clean = length = nq = uncov = 0
                for c in rows_real:
                    kept = truncatemod.truncate(_result_to_truncatable(c["results"]), spec, mode)[:TOPK]
                    nq += 1
                    length += len(kept)
                    if any(J.is_expected(k.relpath, c["expected"]) for k in kept):
                        rec += 1
                    labs = []
                    for k in kept:
                        lab = real_lab.get((corpus, c["qid"], k.relpath))
                        if lab is None:
                            uncov += 1
                        else:
                            labs.append(lab)
                    nz = sum(1 for x in labs if x == "noise")
                    if labs:
                        noise_sum += nz / len(labs)
                    if nz == 0:
                        clean += 1
                # no-match abstain + false-positive
                nm_n = abst = relfp = 0
                for c in rows_nom:
                    kept = truncatemod.truncate(_result_to_truncatable(c["results"]), spec, mode)[:TOPK]
                    nm_n += 1
                    if not kept:
                        abst += 1
                    labmap = nomatch_lab.get((corpus, c["query"]), {})
                    if any(labmap.get(k.relpath) == "relevant" for k in kept):
                        relfp += 1
                out_lines.append(
                    f"| {spec} | {rec/nq:.3f} | {noise_sum/nq:.3f} | {clean/nq:.3f} | "
                    f"{length/nq:.1f} | {100*abst/nm_n if nm_n else 0:.0f}% | {relfp} | {uncov} |")
    report = "\n".join(out_lines)
    (RESULTS / "truncation-sweep.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("capture", "all"):
        capture()
    if cmd in ("sweep", "all"):
        sweep()
