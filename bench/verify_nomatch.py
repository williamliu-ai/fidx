"""Metric-verification harness: 100 deliberately unanswerable queries.

Purpose: validate the top-k purity metric (noise-rate@10 / clean@10; metric
definitions in bench/judge_purity.py and bench/README.md) by feeding queries
that have NO correct answer in either corpus. A correct
engine should either abstain (return nothing) or, if it returns results, those
results are pure noise by construction. This exposes the fill-to-10 vs.
threshold-abstain behaviour difference directly.

Reuses the kept docs/chat indexes (no re-indexing) and judge_purity's replay
helpers. Queries are coherent but reference fabricated entities, so they embed
well (a fair test of the vector path) yet match no real document.

Run: .venv/bin/python bench/verify_nomatch.py [replay|judge|report|all]
Artifacts: bench/results/nomatch-{lists,judgments}.jsonl, nomatch-report.md
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import judge_purity as J
from run_bench import DATA, RESULTS

# Redirect judge_purity's append targets to nomatch-specific files BEFORE any
# replay call, so the real purity artifacts are never touched.
J.LISTS = RESULTS / "nomatch-lists.jsonl"
J.JUDGMENTS = RESULTS / "nomatch-judgments.jsonl"

NOMATCH_Q = DATA / "queries-docs-nomatch.jsonl"   # same 100 queries reused for chat
SENTINEL = "__nomatch__"                            # expected: no real doc
CORPORA = ["docs", "chat"]
# One fidx combo (profiles/backends shown immaterial) + the three qmd modes.
FIDX_COMBO = ("e5-768", "sqlite-vec")
QMD_MODES = ["search", "vsearch", "query"]

# Fabricated entities — clearly non-words, so no FTS token can match; combined
# into coherent phrases so the embedding is in-distribution (finance/tech/etc).
ORGS = ["Vexiltang Dynamics", "Qophrium Industries", "Brennalox Corp",
        "Zudimar Group", "Plathberg Systems"]
PEOPLE = ["Doravyn Klesh", "Mirelda Tavok", "Sennforth Quill", "Ulgath Prenno",
          "Yvanic Sorrel"]
PRODUCTS = ["Vexiltang 9000", "Qophar X12", "Brennalox Pro", "Zudibox 7"]
CONCEPTS = ["hyperbolic glomsk theory", "quantum threpwork", "neural vexilology",
            "the doravyn equilibrium"]
CITIES = ["Glomsk", "Threpwall", "Yarruvex City", "New Plathberg"]
YEARS = ["2031", "2034", "2037", "2041"]
TEMPLATES = [
    lambda i: f"quarterly earnings report for {ORGS[i % len(ORGS)]} in {YEARS[i % len(YEARS)]}",
    lambda i: f"how to install the {PRODUCTS[i % len(PRODUCTS)]} firmware update",
    lambda i: f"{PEOPLE[i % len(PEOPLE)]}'s lecture on {CONCEPTS[i % len(CONCEPTS)]}",
    lambda i: f"the {['Qophrium Accord', 'Vexiltang Pact'][i % 2]} treaty signed in {CITIES[i % len(CITIES)]}",
    lambda i: f"troubleshooting the {PRODUCTS[i % len(PRODUCTS)]} glomsk valve error code {1000 + i}",
    lambda i: f"history of the {ORGS[i % len(ORGS)]} expedition to {CITIES[i % len(CITIES)]}",
    lambda i: f"{PEOPLE[i % len(PEOPLE)]} versus {PEOPLE[(i + 1) % len(PEOPLE)]} championship results {YEARS[i % len(YEARS)]}",
    lambda i: f"specifications of the {PRODUCTS[i % len(PRODUCTS)]} threpwork module",
]


def make_queries() -> list[dict]:
    qs = []
    for i in range(100):
        text = TEMPLATES[i % len(TEMPLATES)](i)
        qs.append({"qid": f"nomatch-{i:04d}", "type": "nomatch",
                   "query": text, "expected": SENTINEL})
    return qs


def write_query_files(qs: list[dict]) -> None:
    for corpus in CORPORA:
        path = DATA / f"queries-{corpus}-nomatch.jsonl"
        path.write_text("".join(json.dumps(q) + "\n" for q in qs))
        print(f"wrote {path} ({len(qs)} queries)", file=sys.stderr)


NOMATCH_SYSTEM = """\
You judge search results for a benchmark probe. The QUERY below was constructed
to have NO correct answer in the corpus — it references fabricated entities that
do not exist. Label each CANDIDATE document:

- relevant: the candidate genuinely and specifically answers the query (it
  actually contains the fabricated entity/fact the query asks for). For these
  probe queries this should essentially never happen.
- related: same broad topic or vocabulary as the query, but does not answer it.
- noise: unrelated to the query's information need.

Judge each candidate independently against the query text alone."""


def cmd_replay() -> None:
    qs = make_queries()
    write_query_files(qs)
    J.LISTS.unlink(missing_ok=True)
    for corpus in CORPORA:
        J.replay_fidx(corpus, *FIDX_COMBO, qs, done=set())
        for mode in QMD_MODES:
            J.replay_qmd(corpus, mode, qs, done=set())
    print(f"replay done -> {J.LISTS}", file=sys.stderr)


def judge_one(corpus: str, query: str, paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    cands = "\n\n".join(f"CANDIDATE {i}: {p}\n{(J.doc_text(corpus, p) or '')[:1000]}"
                        for i, p in enumerate(paths, 1))
    prompt = (f"{NOMATCH_SYSTEM}\n\nQUERY: {query}\n\n{cands}\n\n"
              'Respond with ONLY a JSON object {"labels": [{"path": "...", '
              '"label": "relevant|related|noise"}]} with one entry per candidate.')
    import os
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    for _ in (1, 2):
        with tempfile.TemporaryDirectory() as td:
            last, schema = Path(td) / "last.txt", Path(td) / "s.json"
            schema.write_text(json.dumps(J.JUDGE_SCHEMA))
            cmd = [J.CODEX_BIN, "exec", "-", "-C", str(J.REPO_ROOT),
                   "--dangerously-bypass-approvals-and-sandbox",
                   "--skip-git-repo-check", "--ephemeral",
                   "--output-schema", str(schema), "-o", str(last)]
            try:
                subprocess.run(cmd, input=prompt, env=env, capture_output=True,
                               text=True, timeout=600)
                t = last.read_text()
                t = t[t.index("{"): t.rindex("}") + 1]
                labels = {l["path"]: l["label"] for l in json.loads(t)["labels"]
                          if l.get("label") in J.LABELS}
                if labels:
                    return labels
            except (subprocess.TimeoutExpired, OSError, ValueError,
                    KeyError, TypeError, json.JSONDecodeError):
                pass
    return {}


def cmd_judge() -> None:
    rows = J.jsonl_read(J.LISTS)
    groups: dict[tuple, dict] = {}
    for r in rows:
        g = groups.setdefault((r["corpus"], r["qid"]),
                              {"corpus": r["corpus"], "query": r["query"], "paths": []})
        for p in r["paths"]:
            if p not in g["paths"]:
                g["paths"].append(p)
    J.JUDGMENTS.unlink(missing_ok=True)
    for n, g in enumerate(groups.values(), 1):
        labels = judge_one(g["corpus"], g["query"], g["paths"])
        for p in g["paths"]:
            J.jsonl_append(J.JUDGMENTS, {"corpus": g["corpus"], "qid": "_",
                                         "path": p, "label": labels.get(p, "unjudged")})
        # qid is per (corpus, path) union here; store query-level labels too
        J.jsonl_append(J.JUDGMENTS, {"corpus": g["corpus"], "query": g["query"],
                                     "labels": labels})
        if n % 25 == 0:
            print(f"  judged {n}/{len(groups)}", file=sys.stderr)
    print(f"judge done -> {J.JUDGMENTS}", file=sys.stderr)


def cmd_report() -> None:
    lists = J.jsonl_read(J.LISTS)
    # path -> label per corpus (union judging)
    label = {}
    for row in J.jsonl_read(J.JUDGMENTS):
        if "labels" in row:
            for p, lab in row["labels"].items():
                label[(row["corpus"], p)] = lab
    agg: dict[tuple, dict] = {}
    for r in lists:
        key = (r["corpus"], r["engine"], r["mode"])
        a = agg.setdefault(key, {"n": 0, "returned": 0, "lens": 0,
                                 "clean": 0, "noise_frac_sum": 0.0, "rel_q": 0})
        a["n"] += 1
        paths = r["paths"]
        a["lens"] += len(paths)
        if paths:
            a["returned"] += 1
            labs = [label.get((r["corpus"], p), "unjudged") for p in paths]
            noise = sum(1 for x in labs if x == "noise")
            a["noise_frac_sum"] += noise / len(paths)
            if noise == 0:
                a["clean"] += 1
            if any(x == "relevant" for x in labs):
                a["rel_q"] += 1
        else:
            a["clean"] += 1  # abstained -> vacuously clean
    out = ["| corpus | engine/mode | n | returned | abstain% | avg_len | noise@10 | clean@10 | rel-FP |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (corpus, engine, mode), a in sorted(agg.items()):
        ret = a["returned"]
        noise = a["noise_frac_sum"] / ret if ret else 0.0
        out.append(f"| {corpus} | {engine} {mode} | {a['n']} | {ret} | "
                   f"{100 * (a['n'] - ret) / a['n']:.0f}% | {a['lens'] / a['n']:.1f} | "
                   f"{noise:.3f} | {a['clean'] / a['n']:.3f} | {a['rel_q']} |")
    report = "\n".join(out)
    (RESULTS / "nomatch-report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("replay", "all"):
        cmd_replay()
    if cmd in ("judge", "all"):
        cmd_judge()
    if cmd in ("report", "all"):
        cmd_report()
