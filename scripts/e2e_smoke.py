#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Standalone end-to-end smoke test for an *installed* fidx.

Proves the published artifact works on a clean machine: generates a deterministic
~1,000-doc corpus, indexes it and runs known-item searches **through the installed
`fidx` CLI by subprocess only** (this script never imports the `fidx` package, and
refuses to run against the source checkout), then asserts recall@10 clears a
conservative floor. Used by the Linux Docker check and the CI install-matrix.

Usage:
  python scripts/e2e_smoke.py                # generate corpus, index, search, gate
  python scripts/e2e_smoke.py --docs 1000 --queries 40 --floor 0.6

Determinism: corpus bytes are written with `\\n` only (stable across OSes); a
manifest SHA-256 is printed and can be pinned via --expect-sha. Isolation: a temp
`--db` and `--no-daemon` searches, so no user state is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DRIVER = [sys.executable, "-m", "fidx.cli"]

_ADJ = ["amber", "boreal", "coral", "dusky", "ember", "frosted", "gilded",
        "hollow", "ivory", "jade", "kindled", "lunar", "marbled", "nimbus"]
_NOUN = ["lantern", "marsh", "atlas", "cipher", "delta", "ferry", "grove",
         "harbor", "isthmus", "junction", "kiln", "lattice", "meadow", "nexus"]
_TOPIC = ["irrigation", "metallurgy", "cartography", "beekeeping", "acoustics",
          "horology", "mycology", "glassblowing", "navigation", "seismology"]


def refuse_source_checkout() -> None:
    """Abort if `fidx` would import from this repo's src/ (not an install)."""
    repo_src = (Path(__file__).resolve().parent.parent / "src" / "fidx")
    probe = subprocess.run(
        [sys.executable, "-c", "import fidx, os; print(os.path.realpath(os.path.dirname(fidx.__file__)))"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(f"fidx is not importable by {sys.executable}; install the wheel first.\n{probe.stderr}")
    loc = Path(probe.stdout.strip())
    if repo_src.exists() and loc == repo_src.resolve():
        sys.exit(f"refusing to run e2e against the source checkout ({loc}); "
                 "install the built wheel into a fresh environment and run with that Python.")


def gen_corpus(dest: Path, n_docs: int, n_queries: int) -> tuple[list[dict], str]:
    dest.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    rels: list[str] = []
    for i in range(n_docs):
        adj = _ADJ[i % len(_ADJ)]
        noun = _NOUN[(i // len(_ADJ)) % len(_NOUN)]
        topic = _TOPIC[i % len(_TOPIC)]
        marker = f"{adj}-{noun}-{i:04d}"
        rel = f"doc{i:04d}.md"
        body = (
            f"# {topic.title()} field note {i:04d}\n\n"
            f"This note about {topic} concerns the {marker}. "
            f"The {adj} {noun} is studied here through the lens of {topic}, "
            f"with observations on its {topic} behaviour and seasonal change.\n"
        ).encode("utf-8")
        (dest / rel).write_bytes(body)
        h.update(rel.encode("utf-8"))
        h.update(body)
        rels.append(rel)
    # deterministic, evenly-spaced known-item queries
    step = max(1, n_docs // n_queries)
    queries = []
    for i in range(0, n_docs, step):
        if len(queries) >= n_queries:
            break
        adj = _ADJ[i % len(_ADJ)]
        noun = _NOUN[(i // len(_ADJ)) % len(_NOUN)]
        topic = _TOPIC[i % len(_TOPIC)]
        marker = f"{adj}-{noun}-{i:04d}"
        queries.append({"expected": f"e2e/{rels[i]}",
                        "query": f"{topic} field observations of the {marker}"})
    return queries, h.hexdigest()


def run(driver_args: list[str], **kw):
    return subprocess.run(DRIVER + driver_args, capture_output=True, text=True, **kw)


def norm(p: str) -> str:
    return p.replace("\\", "/")


def search_results(stdout: str) -> list[dict]:
    payload = json.loads(stdout or "[]")
    if isinstance(payload, dict):
        return payload.get("results", [])
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--floor", type=float, default=0.6)
    ap.add_argument("--p50-ceiling-ms", type=float, default=2000.0)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="use an existing corpus dir instead of generating")
    ap.add_argument("--expect-sha", default=None, help="assert corpus manifest sha256")
    args = ap.parse_args()

    refuse_source_checkout()
    ver = run(["--version"])
    print(f"installed: {ver.stdout.strip() or ver.stderr.strip()}")

    os.environ.setdefault("FIDX_PROFILE", "e5-768")
    tmp = Path(tempfile.mkdtemp(prefix="fidx-e2e-"))
    db = tmp / "e2e.db"
    corpus = args.corpus or (tmp / "corpus")

    if args.corpus is None:
        queries, sha = gen_corpus(corpus, args.docs, args.queries)
        print(f"corpus: {args.docs} docs at {corpus}  sha256={sha[:16]}…")
        if args.expect_sha and sha != args.expect_sha:
            print(f"FAIL: corpus sha {sha} != expected {args.expect_sha}", file=sys.stderr)
            return 1
    else:
        # external corpus: build queries from filenames is out of scope here
        sys.exit("--corpus provided but query generation for external corpora is not supported")

    db_arg = ["--db", str(db)]
    r = run(db_arg + ["collection", "add", str(corpus), "--name", "e2e"])
    if r.returncode != 0:
        print("FAIL collection add:\n", r.stderr, file=sys.stderr); return 1
    t0 = time.time()
    r = run(db_arg + ["index", "--no-calibrate", "--backend", "sqlite-vec"])
    if r.returncode != 0:
        print("FAIL index:\n", r.stderr, file=sys.stderr); return 1
    print(f"indexed in {time.time()-t0:.1f}s")

    # untimed prewarm (model load) so p50 reflects steady-state cold search
    run(db_arg + ["search", queries[0]["query"], "-c", "e2e", "--no-daemon",
                  "--json", "-n", "10"])

    hits = 0
    lat = []
    for q in queries:
        t = time.time()
        r = run(db_arg + ["search", q["query"], "-c", "e2e", "--no-daemon",
                          "--json", "-n", "10"])
        lat.append((time.time() - t) * 1000)
        if r.returncode != 0:
            print("FAIL search:\n", r.stderr, file=sys.stderr); return 1
        paths = [norm(x.get("path", "")) for x in search_results(r.stdout)]
        if any(p == norm(q["expected"]) or p.endswith("/" + norm(q["expected"]).split("/", 1)[-1])
               for p in paths):
            hits += 1
    recall = hits / len(queries)
    lat.sort()
    p50 = lat[len(lat) // 2] if lat else 0.0
    print(f"recall@10 = {recall:.3f}  ({hits}/{len(queries)})   "
          f"p50 cold search = {p50:.0f} ms")
    if p50 > args.p50_ceiling_ms:
        print(f"note: p50 {p50:.0f}ms > {args.p50_ceiling_ms:.0f}ms ceiling "
              "(informational — cold model load dominates)")
    if recall < args.floor:
        print(f"FAIL: recall@10 {recall:.3f} < floor {args.floor}", file=sys.stderr)
        return 1
    print("E2E PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
