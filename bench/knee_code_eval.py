"""Evaluate `knee` truncation on the code-corpus purity, offline + judge-free.

Re-queries the kept fidx e5-768/sqlite-vec code index for scored top-10 lists,
applies knee truncation, and scores recall@10 / noise@10 / clean@10 against the
EXISTING code judgments (valid because knee only drops already-judged docs).
Self-reports the time / CPU / IO consumed, split into: model+index load, the
search capture (the real cost), and the knee step itself (the truncation math).
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "bench")
sys.path.insert(0, "src")
import judge_purity as J
from run_bench import RESULTS, load_queries
from fidx.cli import open_index
from fidx.embedder import FastEmbedder
from fidx import vector_store, search as S, truncate as T

CORPUS = "code"
DB = RESULTS / f"fidx-e5-768-sqlite-vec-{CORPUS}.db"
lab = {(r["corpus"], r["qid"], r["path"]): r["label"]
       for r in J.jsonl_read(J.JUDGMENTS)
       if r.get("qid") not in (None, "_") and "label" in r}

wall0, cpu0 = time.perf_counter(), time.process_time()
conn, prof = open_index(DB)
store = vector_store.open_store(conn, DB)
emb = FastEmbedder(prof)
load_wall = time.perf_counter() - wall0

qs = load_queries(CORPUS, None)
cap = []
t = time.perf_counter()
for q in qs:
    cap.append((q, S.search_hybrid(conn, store, emb, q["query"], None, 10)))
capture_wall = time.perf_counter() - t


def metrics(spec):
    rec = clean = nq = uncov = 0
    noise_sum = lens = 0.0
    ktime = 0.0
    for q, res in cap:
        nq += 1
        t0 = time.perf_counter()
        kept = res[:10] if spec == "off" else T.truncate(list(res), spec, "hybrid")[:10]
        ktime += time.perf_counter() - t0
        lens += len(kept)
        if any(J.is_expected(k.relpath, q["expected"]) for k in kept):
            rec += 1
        labs = []
        for k in kept:
            l = lab.get((CORPUS, q["qid"], k.relpath))
            uncov += l is None
            if l is not None:
                labs.append(l)
        nn = sum(1 for x in labs if x == "noise")
        if labs:
            noise_sum += nn / len(labs)
        if nn == 0:
            clean += 1
    return {"R@10": rec / nq, "noise@10": noise_sum / nq, "clean@10": clean / nq,
            "avg_len": lens / nq, "knee_time_s": ktime, "uncov": uncov, "n": nq}


off = metrics("off")
knee = metrics("knee")
ru = resource.getrusage(resource.RUSAGE_SELF)
total_wall = time.perf_counter() - wall0

print("== code-corpus knee truncation (fidx e5-768/sqlite-vec) ==")
for name, m in (("off", off), ("knee", knee)):
    print(f"  {name:5} R@10={m['R@10']:.3f} noise@10={m['noise@10']:.3f} "
          f"clean@10={m['clean@10']:.3f} avg_len={m['avg_len']:.1f} uncov={m['uncov']}")
print("\n== resource cost to create the knee result ==")
print(f"  total wall: {total_wall:.1f}s  (model+index load {load_wall:.1f}s, "
      f"search capture {capture_wall:.1f}s over {off['n']} queries = "
      f"{1000*capture_wall/off['n']:.0f} ms/query)")
print(f"  CPU: user {ru.ru_utime:.1f}s + sys {ru.ru_stime:.1f}s = {ru.ru_utime+ru.ru_stime:.1f}s")
print(f"  peak RSS: {ru.ru_maxrss/1e6:.2f} GB")
print(f"  IO: {ru.ru_inblock} blocks read, {ru.ru_oublock} blocks written (512B blocks)")
print(f"  KNEE STEP ITSELF: {knee['knee_time_s']*1e3:.1f} ms total over {off['n']} "
      f"queries = {knee['knee_time_s']*1e6/off['n']:.1f} us/query")
print(f"  LLM/judge calls to produce this: 0 (reused existing code judgments)")
conn.close()
