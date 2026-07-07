# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Top-k purity metrics for completed benchmark runs.

Purity = noise-rate@10 (fraction of returned docs judged noise) and clean@10
(fraction of queries whose top-10 contains no noise); always read alongside
recall, which it trades off against.
The benchmark JSONs persist only each query's hit rank, so `replay` re-runs the
queries against the kept indexes to recover the top-10 lists; `judge` labels
every retrieved doc relevant / related / noise with an LLM judge; `report`
prints noise-rate@10 and clean@10 per engine/mode/corpus.

Judging is per (corpus, query) over the UNION of all engines' top-10s, so every
engine's results are scored by the same judgment — engines never get judged by
different coin flips on the same document.

The judge runs through the Codex CLI (`codex exec`), so it uses the machine's
existing Codex auth — no Anthropic API key. Some hosts cannot create the user
namespaces Codex's bwrap sandbox needs, so the call passes
--dangerously-bypass-approvals-and-sandbox (the prompt is self-contained and
needs no tools). The shell ANTHROPIC_API_KEY is stripped from the child env
for hygiene. The final
labels JSON is read from `codex exec -o` (the last-message file) with its shape
pinned by `--output-schema`, so the noisy tool-trace stdout is never parsed.

Usage:
  python bench/judge_purity.py replay [--corpus docs-small] [--only fidx] [--limit N]
  python bench/judge_purity.py judge [--model MODEL] [--parallel 4]
  python bench/judge_purity.py report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))          # run_bench
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # fidx

from run_bench import (DATA, RESULTS, load_queries, qmd_env, qmd_handelize,
                       qmd_result_paths)

LISTS = RESULTS / "purity-lists.jsonl"
JUDGMENTS = RESULTS / "purity-judgments.jsonl"
EXCERPT_CHARS = 1000
REFERENCE_CHARS = 1200

# The combos that produced the published text-only results (qmd mcp shares the
# query-mode backend, so its lists are represented by `query`).
REPLAYS = {
    "docs-small": {
        "fidx": [("nomic-768-q", "sqlite-vec"), ("e5-768", "sqlite-vec"), ("e5-768", "duckdb")],
        "qmd": ["search", "vsearch", "query"],
        "max_queries": 150,
    },
    "docs": {
        "fidx": [("nomic-768-q", "sqlite-vec"), ("nomic-768-q", "duckdb"),
                 ("e5-768", "sqlite-vec"), ("e5-768", "duckdb")],
        "qmd": ["search", "vsearch", "query"],
        "max_queries": None,
    },
    "chat": {
        "fidx": [("nomic-768-q", "sqlite-vec"), ("nomic-768-q", "duckdb"),
                 ("e5-768", "sqlite-vec"), ("e5-768", "duckdb")],
        "qmd": ["search", "vsearch", "query"],
        "max_queries": None,
    },
    "code": {
        # only the combos actually indexed for the 92k code corpus (duckdb skipped)
        "fidx": [("e5-768", "sqlite-vec"), ("nomic-768-q", "sqlite-vec")],
        "qmd": ["search", "vsearch", "query"],
        "max_queries": None,
    },
}

LABELS = ("relevant", "related", "noise")

# Codex judge wiring. REPO_ROOT is codex's working root (-C).
# JUDGE_SCHEMA pins the final-message shape.
REPO_ROOT = Path(__file__).resolve().parent.parent
CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "label"],
                "properties": {
                    "path": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                },
            },
        },
    },
}

JUDGE_SYSTEM = """\
You judge search results for a known-item search benchmark over a document
corpus. For one query you receive the REFERENCE document (the known correct
answer for this query) and a list of CANDIDATE documents that search engines
returned. Label every candidate:

- relevant: the candidate contains the information the query seeks — it is the
  reference itself, a duplicate or quoted copy of it, or another document that
  genuinely answers the query on its own.
- related: same topic, thread, or vocabulary; an agent might glean context from
  it, but it does not itself contain the sought information.
- noise: unrelated to the query's information need — pure noise in the result
  list; an agent reading it gains nothing toward this query.

The corpus contains quoted reply chains and reposts: a near-duplicate of the
reference that carries the same content is relevant, not noise. Judge each
candidate independently. Return one label per candidate path, exactly the
paths given."""


def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl_append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def corpus_files(corpus: str) -> list[str]:
    root = DATA / corpus
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def doc_text(corpus: str, rel: str) -> str | None:
    p = DATA / corpus / rel
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def unhandelize_map(corpus: str) -> dict[str, str]:
    return {qmd_handelize(rel): rel for rel in corpus_files(corpus)}


def real_path(p: str, hmap: dict[str, str], corpus: str) -> str | None:
    """Map an engine result path back to a corpus relpath. fidx prefixes the
    collection name; qmd returns handelized display paths."""
    parts = p.split("/")
    for n in range(len(parts), 0, -1):
        cand = "/".join(parts[-n:])
        if (DATA / corpus / cand).is_file():
            return cand
    # Handelized (qmd) paths: try every suffix length, LONGEST first, so deep
    # nested paths (code corpus, 5-8 segments) resolve — not just the last 1-3
    # segments, which silently failed on code (qmd rewrites dots->dashes in dir
    # names, so the handelized full relpath is the only key that matches).
    for n in range(len(parts), 0, -1):
        cand = "/".join(parts[-n:])
        if cand in hmap:
            return hmap[cand]
    return None


# ---------------------------------------------------------------- replay ----


def replay_fidx(corpus: str, profile: str, backend: str, queries: list[dict],
                done: set) -> None:
    from fidx import daemon as fidx_daemon

    db = RESULTS / f"fidx-{profile}-{backend}-{corpus}.db"
    if not db.exists():
        print(f"SKIP fidx {profile}/{backend}/{corpus}: no index at {db}", file=sys.stderr)
        return
    mode = f"{profile}/{backend}"
    todo = [q for q in queries if (corpus, "fidx", mode, q["qid"]) not in done]
    if not todo:
        return
    env = os.environ.copy()
    env["FIDX_THREADS"] = "1"
    sock = fidx_daemon.socket_path_for(db)
    sock.unlink(missing_ok=True)
    cli = [sys.executable, "-m", "fidx.cli", "--db", str(db)]
    daemon_log = open(RESULTS / f"purity-daemon-{profile}-{backend}-{corpus}.log", "w")
    server = subprocess.Popen(cli + ["serve"], env=env, stderr=daemon_log)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if sock.exists() and fidx_daemon.client_request(db, {"cmd": "ping"}):
                break
            if server.poll() is not None:
                raise RuntimeError(f"fidx daemon for {db.name} exited rc={server.returncode}")
            time.sleep(0.2)
        else:
            raise RuntimeError(f"fidx daemon for {db.name} did not come up")
        hmap = unhandelize_map(corpus)
        failures = 0
        for q in todo:
            resp = fidx_daemon.client_request(
                db, {"cmd": "search", "query": q["query"], "mode": "hybrid", "limit": 10})
            if not resp or not resp.get("ok"):
                # Daemon hiccup — do NOT record, so a rerun retries this query.
                failures += 1
                if failures <= 3:
                    print(f"  WARN fidx {mode} {q['qid']}: "
                          f"{(resp or {}).get('error', 'no response')}", file=sys.stderr)
                continue
            results = resp["results"]["results"]
            paths = [real_path(r["path"], hmap, corpus) or r["path"]
                     for r in results]
            jsonl_append(LISTS, {"corpus": corpus, "engine": "fidx", "mode": mode,
                                 "qid": q["qid"], "query": q["query"],
                                 "expected": q["expected"], "paths": paths})
        if failures:
            print(f"  WARN fidx {mode} {corpus}: {failures} failed queries "
                  "not recorded (rerun replay to retry)", file=sys.stderr)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    print(f"replayed fidx {mode} {corpus}: {len(todo)} queries", file=sys.stderr)


def replay_qmd(corpus: str, mode: str, queries: list[dict], done: set,
               qmd_bin: str = "qmd") -> None:
    env = qmd_env(corpus, force_cpu=True)
    hmap = unhandelize_map(corpus)
    todo = [q for q in queries if (corpus, "qmd", mode, q["qid"]) not in done]
    if not todo:
        return
    for i, q in enumerate(todo, 1):
        proc = subprocess.run([qmd_bin, mode, q["query"], "--json", "-n", "10"],
                              env=env, capture_output=True, text=True, timeout=900)
        raw = qmd_result_paths(proc.stdout) if proc.returncode == 0 else []
        paths = [real_path(p, hmap, corpus) or p for p in raw]
        jsonl_append(LISTS, {"corpus": corpus, "engine": "qmd", "mode": mode,
                             "qid": q["qid"], "query": q["query"],
                             "expected": q["expected"], "paths": paths})
        if i % 25 == 0:
            print(f"  qmd {mode} {corpus}: {i}/{len(todo)}", file=sys.stderr)
    print(f"replayed qmd {mode} {corpus}: {len(todo)} queries", file=sys.stderr)


def replays_for(corpus: str) -> dict:
    """Replay spec for a corpus. Per-repo code corpora (code-<lang>) aren't listed
    individually — they share the standard single-profile + qmd-modes spec."""
    if corpus in REPLAYS:
        return REPLAYS[corpus]
    if corpus.startswith("code-"):
        return {"fidx": [("e5-768", "sqlite-vec")],
                "qmd": ["search", "vsearch", "query"], "max_queries": None}
    raise SystemExit(f"unknown corpus {corpus!r}")


def cmd_replay(args) -> None:
    done = {(r["corpus"], r["engine"], r["mode"], r["qid"]) for r in jsonl_read(LISTS)}
    corpora = [args.corpus] if args.corpus else list(REPLAYS)
    for corpus in corpora:
        spec = replays_for(corpus)
        queries = load_queries(corpus, spec["max_queries"])
        if args.limit:
            queries = queries[:args.limit]
        if not args.only or args.only == "fidx":
            for profile, backend in spec["fidx"]:
                replay_fidx(corpus, profile, backend, queries, done)
        if not args.only or args.only == "qmd":
            for mode in spec["qmd"]:
                replay_qmd(corpus, mode, queries, done)


# ----------------------------------------------------------------- judge ----


def is_expected(path: str, expected: str) -> bool:
    return path == expected or path.endswith("/" + expected)


def build_judge_groups() -> list[dict]:
    """One judging unit per (corpus, qid): query + union of all retrieved docs.
    Path-level incremental: docs a later replay adds to an already-judged
    query still get judged."""
    rows = jsonl_read(LISTS)
    judged = {(r["corpus"], r["qid"], r["path"]) for r in jsonl_read(JUDGMENTS)}
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r["corpus"], r["qid"])
        g = groups.setdefault(key, {"corpus": r["corpus"], "qid": r["qid"],
                                    "query": r["query"], "expected": r["expected"],
                                    "paths": []})
        for p in r["paths"]:
            if p not in g["paths"] and (r["corpus"], r["qid"], p) not in judged:
                g["paths"].append(p)
    out = []
    for g in groups.values():
        if not g["paths"]:
            continue
        # The known item needs no LLM call; judge the rest.
        g["auto"] = [p for p in g["paths"] if is_expected(p, g["expected"])]
        g["to_judge"] = []
        for p in g["paths"]:
            if p in g["auto"]:
                continue
            if doc_text(g["corpus"], p) is None:
                g.setdefault("unreadable", []).append(p)
                continue
            g["to_judge"].append(p)
        out.append(g)
    return out


def judge_prompt(g: dict) -> str:
    ref = (doc_text(g["corpus"], g["expected"]) or "")[:REFERENCE_CHARS]
    cands = "\n\n".join(
        f"CANDIDATE {i}: {p}\n{(doc_text(g['corpus'], p) or '')[:EXCERPT_CHARS]}"
        for i, p in enumerate(g["to_judge"], 1))
    return (f"{JUDGE_SYSTEM}\n\nQUERY: {g['query']}\n\n"
            f"REFERENCE ({g['expected']}):\n{ref}\n\n{cands}\n\n"
            'Respond with ONLY a JSON object, no prose, of the form '
            '{"labels": [{"path": "...", "label": "relevant|related|noise"}]} '
            "with exactly one entry per candidate.")


def record_judgment(g: dict, labels: dict[str, str]) -> None:
    for p in g["paths"]:
        if p in g["auto"]:
            label, source = "relevant", "known-item"
        elif p in g.get("unreadable", []):
            label, source = "unjudged", "unreadable"
        else:
            label = labels.get(p, "unjudged")
            source = "llm" if p in labels else "judge-omitted"
        jsonl_append(JUDGMENTS, {"corpus": g["corpus"], "qid": g["qid"],
                                 "path": p, "label": label, "source": source})


def judge_one(g: dict, model: str | None) -> dict[str, str]:
    """Label one query's candidates via the Codex CLI (`codex exec`), using the
    machine's Codex auth. The sandbox is bypassed (see module docstring) and
    the prompt is self-contained, so codex needs no tools. The final JSON is
    taken from the -o last-message file (schema-pinned), never from the
    tool-trace stdout. ANTHROPIC_API_KEY is stripped for hygiene."""
    prompt = judge_prompt(g)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    for attempt in (1, 2):
        with tempfile.TemporaryDirectory() as td:
            last = Path(td) / "last.txt"
            schema = Path(td) / "schema.json"
            schema.write_text(json.dumps(JUDGE_SCHEMA))
            cmd = [CODEX_BIN, "exec", "-", "-C", str(REPO_ROOT),
                   "--dangerously-bypass-approvals-and-sandbox",
                   "--skip-git-repo-check", "--ephemeral",
                   "--output-schema", str(schema), "-o", str(last)]
            if model:
                cmd += ["-m", model]
            try:
                subprocess.run(cmd, input=prompt, env=env, capture_output=True,
                               text=True, timeout=600)
                text = last.read_text()
                text = text[text.index("{"): text.rindex("}") + 1]
                labels = {l["path"]: l["label"] for l in json.loads(text)["labels"]
                          if l.get("label") in LABELS}
                if labels:
                    return labels
            except (subprocess.TimeoutExpired, OSError, FileNotFoundError,
                    json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
        print(f"  WARN judge attempt {attempt} failed for {g['corpus']}/{g['qid']}",
              file=sys.stderr)
    return {}


def cmd_judge(args) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    groups = build_judge_groups()
    if not groups:
        print("nothing to judge", file=sys.stderr)
        return
    n_docs = sum(len(g["to_judge"]) for g in groups)
    print(f"judging {n_docs} docs across {len(groups)} queries via codex exec "
          f"({args.model or 'codex default model'})", file=sys.stderr)
    lock = threading.Lock()
    done = 0

    def work(g: dict) -> None:
        nonlocal done
        labels = judge_one(g, args.model) if g["to_judge"] else {}
        with lock:
            record_judgment(g, labels)
            done += 1
            if done % 25 == 0:
                print(f"  judged {done}/{len(groups)}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(work, groups))
    print(f"judged {done}/{len(groups)} queries", file=sys.stderr)


# ---------------------------------------------------------------- report ----


def cmd_report(args) -> None:
    labels = {(r["corpus"], r["qid"], r["path"]): r["label"] for r in jsonl_read(JUDGMENTS)}
    combos: dict[tuple, dict] = {}
    for r in jsonl_read(LISTS):
        key = (r["corpus"], r["engine"], r["mode"])
        c = combos.setdefault(key, {"queries": 0, "results": 0, "noise": 0,
                                    "related": 0, "relevant": 0, "unjudged": 0,
                                    "clean": 0, "hit10": 0, "empty": 0})
        c["queries"] += 1
        if not r["paths"]:
            c["empty"] += 1
        if any(is_expected(p, r["expected"]) for p in r["paths"]):
            c["hit10"] += 1
        q_noise = 0
        for p in r["paths"]:
            c["results"] += 1
            label = labels.get((r["corpus"], r["qid"], p), "unjudged")
            c[label if label in ("noise", "related", "relevant") else "unjudged"] += 1
            q_noise += label == "noise"
        c["clean"] += q_noise == 0
    print("| corpus | engine/mode | n | R@10 (replay) | noise-rate@10 | clean@10 "
          "| related | relevant | unjudged |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (corpus, engine, mode), c in sorted(combos.items()):
        nr = c["noise"] / c["results"] if c["results"] else 0
        print(f"| {corpus} | {engine} {mode} | {c['queries']} "
              f"| {c['hit10'] / c['queries']:.3f} "
              f"| {nr:.3f} | {c['clean'] / c['queries']:.3f} "
              f"| {c['related'] / max(1, c['results']):.3f} "
              f"| {c['relevant'] / max(1, c['results']):.3f} "
              f"| {c['unjudged']} |")
        if c["empty"]:
            print(f"  WARN {corpus}/{engine}/{mode}: {c['empty']} queries returned "
                  "no results", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--corpus")  # any corpus incl. per-repo code-<lang>
    p.add_argument("--only", choices=["fidx", "qmd"])
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("judge")
    p.add_argument("--model", default=None,
                   help="codex -m model (default: codex's configured model)")
    p.add_argument("--parallel", type=int, default=4)
    p.set_defaults(func=cmd_judge)
    p = sub.add_parser("report")
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
