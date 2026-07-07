#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Run retrieval benchmarks for fidx and QMD on a prepared corpus.

Measures, per engine/mode:
  - R@1, R@3, R@10 against the query's source file (overall and per query type)
  - per-query latency median and p99 (warm; plus a cold-CLI sample)
  - index build time and on-disk index size
  - peak memory: index step and query path (fidx daemon VmHWM / per-process
    ru_maxrss for subprocess engines; Linux semantics)

Results land in bench/results/<engine>-<corpus>.json; render a comparison
with `python bench/run_bench.py report`.

Examples:
  python bench/run_bench.py run --engine fidx --corpus docs
  python bench/run_bench.py run --engine qmd --corpus docs --modes search,vsearch,query
  python bench/run_bench.py report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

BENCH = Path(__file__).parent
DATA = BENCH / "data"
RESULTS = BENCH / "results"
REPO = BENCH.parent

sys.path.insert(0, str(REPO / "src"))


def load_queries(corpus: str, limit: int | None) -> list[dict]:
    path = DATA / f"queries-{corpus}.jsonl"
    if not path.exists():
        sys.exit(f"{path} missing — run gen_queries.py first")
    queries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return queries[:limit] if limit else queries


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))
    return values[idx]


def hit_rank(result_paths: list[str], expected: str) -> int | None:
    """Rank (0-based) of the expected file in results, by path-suffix match."""
    for i, p in enumerate(result_paths):
        if p == expected or p.endswith("/" + expected):
            return i
    return None


def summarize(rows: list[dict], latencies_ms: list[float]) -> dict:
    out: dict = {"n": len(rows)}
    for k in (1, 3, 10):
        out[f"recall@{k}"] = round(
            sum(1 for r in rows if r["rank"] is not None and r["rank"] < k) / max(1, len(rows)), 4
        )
    out["latency_ms"] = {
        "p50": round(percentile(latencies_ms, 50), 1),
        "p99": round(percentile(latencies_ms, 99), 1),
        "mean": round(statistics.fmean(latencies_ms), 1) if latencies_ms else 0,
    }
    by_type: dict = {}
    for qtype in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == qtype]
        by_type[qtype] = {
            f"recall@{k}": round(
                sum(1 for r in sub if r["rank"] is not None and r["rank"] < k) / len(sub), 4
            )
            for k in (1, 3, 10)
        }
    out["by_type"] = by_type
    return out


def run_measured(cmd: list[str], env: dict | None = None, capture: bool = False,
                 check: bool = False, timeout: float | None = None
                 ) -> tuple[int, str | None, int, float]:
    """Run cmd to completion; return (returncode, stdout, peak_rss_bytes,
    cpu_seconds). CPU time is ru_utime + ru_stime — user+system across the
    child and its waited-for descendants, so it captures multi-core cost that
    wall-clock hides.

    Reaps via os.wait4 so ru_maxrss covers the child and its waited-for
    descendants. Linux reports ru_maxrss in KiB (macOS: bytes). A timeout
    kills the process (watchdog thread — wait4 has no native timeout) so one
    hung subprocess cannot stall an unattended run forever; the caller sees
    the kill as a non-zero returncode."""
    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.DEVNULL if capture else None,
                            text=capture)
    watchdog = None
    if timeout:
        watchdog = threading.Timer(timeout, proc.kill)
        watchdog.start()
    try:
        out = proc.stdout.read() if capture else None
        _, status, ru = os.wait4(proc.pid, 0)
    finally:
        if watchdog:
            watchdog.cancel()
    proc.returncode = os.waitstatus_to_exitcode(status)
    if capture:
        proc.stdout.close()
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    scale = 1 if sys.platform == "darwin" else 1024
    return proc.returncode, out, ru.ru_maxrss * scale, ru.ru_utime + ru.ru_stime


def run_resilient(cmd: list[str], env: dict | None = None, stall_timeout: float = 900,
                  max_restarts: int = 2) -> tuple[int, int, float]:
    """Run an *incremental* long task with stall detect/kill/resume.

    Index steps can wedge in native code with no way out — observed 2026-06-12:
    onnxruntime's intra-op thread pool livelocked inside one Run() call and
    spun a core for 11 h with zero syscalls. A wall-clock timeout cannot cover
    index steps (minutes on docs-small, hours on full corpora), so staleness
    is detected on *output silence*: both fidx and qmd print progress while
    indexing, and `stall_timeout` seconds without a line from a still-running
    process means it is stuck, not slow. The child is SIGKILLed (a livelocked
    process may never reach a Python signal handler) and the same command is
    re-run — fidx `index` and qmd `embed` are incremental, so a rerun resumes
    from the last committed batch instead of starting over.

    Only stalls are retried; a non-zero exit without a stall is a real error
    and raises immediately. Returns (stall_count, peak_rss_bytes,
    cpu_seconds) accumulated across attempts; raises RuntimeError when still
    stalling after max_restarts resumes. Caller should record the stall count
    in its report — a stalled-and-resumed run has inflated wall/CPU numbers.
    """
    stalls, peak_rss, total_cpu = 0, 0, 0.0
    scale = 1 if sys.platform == "darwin" else 1024
    for attempt in range(max_restarts + 1):
        proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
        last_output = time.monotonic()

        def pump(stream) -> None:
            nonlocal last_output
            for line in stream:
                last_output = time.monotonic()
                sys.stderr.write(line)
                sys.stderr.flush()

        reader = threading.Thread(target=pump, args=(proc.stderr,), daemon=True)
        reader.start()
        stalled = False
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.monotonic() - last_output > stall_timeout:
                stalled = True
                proc.kill()
                pid, status, ru = os.wait4(proc.pid, 0)
                break
            time.sleep(5)
        proc.returncode = os.waitstatus_to_exitcode(status)
        reader.join(timeout=10)
        proc.stderr.close()
        peak_rss = max(peak_rss, ru.ru_maxrss * scale)
        total_cpu += ru.ru_utime + ru.ru_stime
        if stalled:
            stalls += 1
            action = "resuming incrementally" if attempt < max_restarts else "giving up"
            print(f"  WARN: no output for {stall_timeout:.0f}s — killed stalled "
                  f"`{' '.join(cmd[-4:])}` (attempt {attempt + 1}/{max_restarts + 1}), "
                  f"{action}", file=sys.stderr)
            continue
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return stalls, peak_rss, total_cpu
    raise RuntimeError(f"`{' '.join(cmd)}` still stalling after {max_restarts + 1} attempts")


def proc_peak_rss(pid: int) -> int | None:
    """Lifetime peak resident set (VmHWM) of a live process, in bytes."""
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    m = re.search(r"^VmHWM:\s+(\d+)\s+kB", text, re.M)
    return int(m.group(1)) * 1024 if m else None


def proc_tree_peak_rss(pid: int) -> int | None:
    """Sum of VmHWM over a live process and its descendants — the `qmd mcp`
    launcher (~50 MB) delegates model hosting to a worker child (~GBs)."""
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += proc_peak_rss(p) or 0
        try:
            for task in Path(f"/proc/{p}/task").iterdir():
                stack.extend(int(c) for c in (task / "children").read_text().split())
        except OSError:
            pass
    return total or None


def mb(nbytes: int | None) -> float | None:
    return round(nbytes / 1e6, 1) if nbytes is not None else None


MAX_LOAD_FOR_LATENCY = 2.0


def require_quiet_system(ignore: bool, wait_seconds: int = 600) -> dict:
    """Latency numbers are only meaningful on an otherwise idle machine.

    Queries are replayed strictly sequentially (one in flight, ever); this
    gate additionally waits for the load average to decay (our own index step
    saturates the cores just before measurement) and aborts if the box is
    still busy after the timeout. Returns provenance recorded into the report.
    """
    deadline = time.time() + wait_seconds
    while True:
        load1 = os.getloadavg()[0]
        if load1 <= MAX_LOAD_FOR_LATENCY or ignore:
            break
        if time.time() > deadline:
            sys.exit(
                f"system load {load1:.1f} > {MAX_LOAD_FOR_LATENCY} after waiting "
                f"{wait_seconds}s — latency measurement needs an idle machine "
                "(override with --ignore-load)"
            )
        print(f"  waiting for load to settle: {load1:.1f} -> {MAX_LOAD_FOR_LATENCY}",
              file=sys.stderr)
        time.sleep(20)
    if load1 > MAX_LOAD_FOR_LATENCY:
        print(f"warning: measuring under load {load1:.1f}", file=sys.stderr)
    return {"loadavg_at_start": round(load1, 2), "cpus": os.cpu_count()}


# ----------------------------------------------------------------- fidx ----


def bench_fidx(corpus: str, queries: list[dict], profile: str, cold_sample: int,
               keep_index: bool, ignore_load: bool, query_threads: int = 1,
               backend: str = "sqlite-vec") -> dict:
    from fidx import daemon as fidx_daemon  # same venv
    from fidx.vector_store import sidecar_path

    corpus_dir = DATA / corpus
    db_path = RESULTS / f"fidx-{profile}-{backend}-{corpus}.db"
    if not keep_index:
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        sidecar_path(db_path).unlink(missing_ok=True)
    cli = [sys.executable, "-m", "fidx.cli", "--db", str(db_path)]

    subprocess.run(cli + ["collection", "add", str(corpus_dir), "--name", corpus,
                          "--glob", "**/*"], check=True, capture_output=True)
    t0 = time.perf_counter()
    # duckdb gets a longer leash: its HNSW build/checkpoint at the end of the
    # index step is one long silent native call at large corpus sizes.
    index_stalls, index_peak_rss, index_cpu = run_resilient(
        cli + ["index", "--profile", profile, "--backend", backend],
        stall_timeout=3600 if backend == "duckdb" else 900)
    index_seconds = time.perf_counter() - t0

    # Latency is measured conservatively: single ONNX thread, sequential
    # queries, idle machine (QMD keeps its own default threading).
    provenance = require_quiet_system(ignore_load)
    env = os.environ.copy()
    env["FIDX_THREADS"] = str(query_threads)
    sock = fidx_daemon.socket_path_for(db_path)
    sock.unlink(missing_ok=True)  # stale socket from a killed run must not answer the ping
    daemon_log = open(RESULTS / f"daemon-{profile}-{backend}-{corpus}.log", "w")
    server = subprocess.Popen(cli + ["serve"], stderr=daemon_log, env=env)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"fidx daemon exited rc={server.returncode} "
                                   f"(see {daemon_log.name})")
            if sock.exists() and fidx_daemon.client_request(db_path, {"cmd": "ping"}):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("fidx daemon did not come up")

        rows, latencies = [], []
        failures = 0
        for q in queries:
            t = time.perf_counter()
            resp = fidx_daemon.client_request(
                db_path, {"cmd": "search", "query": q["query"], "mode": "hybrid", "limit": 10}
            )
            elapsed_ms = (time.perf_counter() - t) * 1000
            ok = resp is not None and resp.get("ok")
            if ok:
                latencies.append(elapsed_ms)  # failed requests are counted, not timed
            else:
                failures += 1
                if failures <= 3 or failures % 100 == 0:
                    print(f"  WARN daemon request failed ({failures}) at {q['qid']}: "
                          f"{(resp or {}).get('error', 'no response')}", file=sys.stderr)
            results = resp["results"]["results"] if ok else []
            paths = [r["path"] for r in results]
            rows.append({"qid": q["qid"], "type": q["type"],
                         "rank": hit_rank(paths, q["expected"])})
        if failures:
            print(f"  WARN {failures}/{len(queries)} daemon requests failed — "
                  "recall/latency are NOT valid", file=sys.stderr)
        daemon_peak_rss = proc_peak_rss(server.pid)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        daemon_log.close()

    cold, cold_peak_rss, cold_failures = [], 0, 0
    for q in queries[:cold_sample]:
        t = time.perf_counter()
        code, _, rss, _ = run_measured(
            cli + ["search", q["query"], "--json", "-n", "10", "--no-daemon"],
            env=env, capture=True, timeout=600)
        if code:  # a flaky cold run must not destroy the measured warm report
            cold_failures += 1
            continue
        cold.append((time.perf_counter() - t) * 1000)
        cold_peak_rss = max(cold_peak_rss, rss)
    if cold_failures:
        print(f"  WARN {cold_failures} cold-CLI sample queries failed", file=sys.stderr)

    # Include live WAL/SHM: un-checkpointed pages are real on-disk cost.
    storage = sum(Path(str(db_path) + sfx).stat().st_size
                  for sfx in ("", "-wal", "-shm") if Path(str(db_path) + sfx).exists())
    if sidecar_path(db_path).exists():
        storage += sidecar_path(db_path).stat().st_size
    report = {"engine": "fidx", "corpus": corpus, "mode": f"hybrid/{profile}/{backend}",
              "daemon_failures": failures, "keep_index": keep_index,
              "query_threads": query_threads, **provenance,
              "index_seconds": round(index_seconds, 1),
              "index_cpu_seconds": round(index_cpu, 1),
              "index_stalls": index_stalls,
              "storage_bytes": storage,
              "memory_mb": {"index_peak": mb(index_peak_rss),
                            "daemon_peak": mb(daemon_peak_rss),
                            "cold_cli_peak": mb(cold_peak_rss or None)},
              **summarize(rows, latencies)}
    if cold_failures:
        report["cold_cli_failures"] = cold_failures
    if cold:
        # Small sample: report max, not a pseudo-p99.
        report["cold_cli_ms"] = {"p50": round(percentile(cold, 50), 1),
                                 "max": round(max(cold), 1)}
    return report


# ------------------------------------------------------------------ qmd ----


def qmd_env(corpus: str, force_cpu: bool) -> dict:
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(RESULTS / f"qmd-cache-{corpus}")
    # qmd registers collections in XDG_CONFIG_HOME/qmd/*.yml — left at the
    # real ~/.config, that registry outlives fresh_qmd_cache() and a later
    # `collection add` fails with "already exists". Keep config per corpus
    # beside the cache so a fresh run really starts fresh.
    env["XDG_CONFIG_HOME"] = str(RESULTS / f"qmd-cache-{corpus}" / "config")
    if force_cpu:
        env["QMD_FORCE_CPU"] = "1"
    return env


def qmd_handelize(path: str) -> str:
    """Replicate QMD's display-path sanitization (store.ts handelize): per
    segment keep letters/digits/$, dash-separate everything else (including
    dots), preserve the final extension. Needed to match QMD result URIs
    like qmd://coll/alt-atheism/49960.txt back to alt.atheism/49960.txt."""
    path = path.replace("___", "/")  # store.ts maps ___ to a path separator
    def clean(seg: str) -> str:
        out = "".join(ch if (ch.isalnum() or ch == "$") else "-" for ch in seg)
        out = re.sub("-+", "-", out).strip("-")
        return out

    segments = [s for s in path.split("/") if s]
    cleaned = []
    for i, seg in enumerate(segments):
        if i == len(segments) - 1:
            m = re.search(r"(\.[A-Za-z0-9]+)$", seg)
            ext = m.group(1) if m else ""
            cleaned.append(clean(seg[: len(seg) - len(ext)] if ext else seg) + ext)
        else:
            cleaned.append(clean(seg))
    return "/".join(c for c in cleaned if c)


def qmd_result_paths(stdout: str) -> list[str]:
    """Extract document paths from `qmd ... --json` output, tolerating shape drift."""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else payload.get("results", [])
    paths = []
    for item in items:
        if isinstance(item, dict):
            p = item.get("file") or item.get("path") or item.get("filepath") or item.get("display_path")
            if p:
                paths.append(str(p))
    return paths


class QmdMcp:
    """Minimal JSON-RPC client for `qmd mcp` (stdio): QMD's warm query path.

    Cold CLI invocations reload ~2 GB of GGUF models per query; the MCP
    server is how agents actually use QMD, with models resident — the number
    comparable to fidx's warm daemon. The server's `query` tool always runs
    the structured pipeline (RRF fusion + rerank), skipping LLM expansion."""

    def __init__(self, qmd_bin: str, env: dict, log):
        import queue

        self.proc = subprocess.Popen([qmd_bin, "mcp"], env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=log)
        self._lines: queue.Queue = queue.Queue()
        self._id = 0
        threading.Thread(target=self._pump, daemon=True).start()
        self._rpc({"method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "fidx-bench", "version": "0"}}}, timeout=60)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)  # EOF sentinel

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, obj: dict, timeout: float) -> dict:
        import queue

        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, **obj})
        deadline = time.time() + timeout
        while True:
            try:
                line = self._lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                self.close()
                raise RuntimeError(f"qmd mcp: no response within {timeout}s") from None
            if line is None:
                raise RuntimeError("qmd mcp: server exited")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == self._id:
                return resp

    def query(self, text: str, limit: int = 10, timeout: float = 900) -> list[str]:
        """Lex+vec typed sub-queries — the hybrid request an agent sends for
        one user query (QMD docs: capable LLMs do their own expansion)."""
        # Double quotes are phrase syntax in QMD's lex grammar; an unmatched
        # one is a hard error, so strip them from the lex variant only.
        resp = self._rpc({"method": "tools/call", "params": {
            "name": "query", "arguments": {
                "searches": [{"type": "lex", "query": text.replace('"', " ")},
                             {"type": "vec", "query": text}],
                "limit": limit}}}, timeout=timeout)
        result = resp.get("result", {})
        if result.get("isError") or "error" in resp:
            raise RuntimeError(str(resp)[:200])
        structured = result.get("structuredContent", {}).get("results", [])
        return [str(r["file"]) for r in structured if isinstance(r, dict) and r.get("file")]

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def bench_qmd_mcp(queries: list[dict], qmd_bin: str, env: dict, log) -> tuple[list, list, int, int]:
    """Warm-path measurement: one server, sequential queries, first (model-
    loading) query untimed. Returns (rows, latencies, failures, peak_rss)."""
    client = QmdMcp(qmd_bin, env, log)
    try:
        try:  # warmup: loads embed + rerank models, excluded from latency
            client.query(queries[0]["query"], timeout=1800)
        except RuntimeError as exc:
            print(f"  WARN qmd mcp warmup failed: {exc}", file=sys.stderr)
        rows, latencies = [], []
        failures = 0
        for i, q in enumerate(queries):
            t = time.perf_counter()
            try:
                paths = client.query(q["query"])
                latencies.append((time.perf_counter() - t) * 1000)
            except RuntimeError as exc:
                failures += 1
                paths = []
                if failures <= 3 or failures % 100 == 0:
                    print(f"  WARN qmd mcp failed ({failures}) at {q['qid']}: {exc}",
                          file=sys.stderr)
            rows.append({"qid": q["qid"], "type": q["type"],
                         "rank": hit_rank(paths, qmd_handelize(q["expected"]))})
            if (i + 1) % 25 == 0:
                print(f"  qmd mcp: {i + 1}/{len(queries)} queries", file=sys.stderr)
        peak_rss = proc_tree_peak_rss(client.proc.pid) or 0
    finally:
        client.close()
    return rows, latencies, failures, peak_rss


def fresh_qmd_cache(cache: Path) -> None:
    """Wipe the per-corpus qmd cache but keep (or seed from a sibling cache)
    the models directory, so model download time never lands inside the
    timed index window — fidx's model cache is likewise never deleted."""
    qmd_dir = cache / "qmd"
    if qmd_dir.exists():
        for child in qmd_dir.iterdir():
            if child.name == "models":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    config_qmd = cache / "config" / "qmd"  # collection registry, see qmd_env
    if config_qmd.exists():
        shutil.rmtree(config_qmd)
    models = qmd_dir / "models"
    if not models.exists():
        for donor in sorted(RESULTS.glob("qmd-cache-*/qmd/models")):
            if donor.is_dir() and donor.resolve() != models.resolve():
                qmd_dir.mkdir(parents=True, exist_ok=True)
                models.symlink_to(donor.resolve())
                break


def qmd_storage_bytes(cache: Path) -> int:
    """Index storage = everything under the qmd cache except the models
    subtree (robust whether models is a real dir or a seeded symlink)."""
    qmd_dir = cache / "qmd"
    if not qmd_dir.exists():
        return 0
    return sum(p.stat().st_size for p in qmd_dir.rglob("*")
               if p.is_file() and "models" not in p.relative_to(qmd_dir).parts)


def bench_qmd(corpus: str, queries: list[dict], modes: list[str], force_cpu: bool,
              keep_index: bool, qmd_bin: str, ignore_load: bool,
              dest: Path | None = None) -> list[dict]:
    corpus_dir = DATA / corpus
    env = qmd_env(corpus, force_cpu)
    cache = Path(env["XDG_CACHE_HOME"])
    if not keep_index:
        fresh_qmd_cache(cache)

    t0 = time.perf_counter()
    # With --keep-index the collection already exists; qmd exits non-zero then.
    # `collection add` ingests the files (unlike fidx's, which only registers
    # a path), so its memory/CPU belong to the index cost.
    _, _, add_peak_rss, add_cpu = run_measured(
        [qmd_bin, "collection", "add", str(corpus_dir), "--name", corpus,
         "--mask", "**/*"], env=env, check=not keep_index, capture=keep_index)
    index_stalls, embed_peak_rss, embed_cpu = run_resilient(
        [qmd_bin, "embed"], env=env, stall_timeout=1800)
    index_seconds = time.perf_counter() - t0
    index_peak_rss = max(add_peak_rss, embed_peak_rss)
    index_cpu = add_cpu + embed_cpu
    storage = qmd_storage_bytes(cache)

    provenance = require_quiet_system(ignore_load)
    reports = []
    for mode in modes:
        # QMD caches LLM expansions and rerank scores in its DB keyed by query
        # text; without this, mode order and --keep-index reruns silently
        # change what later modes measure (e.g. `query` inherits the
        # expansions `vsearch` cached for the same 500 queries).
        subprocess.run([qmd_bin, "cleanup"], env=env, capture_output=True)
        if mode == "mcp":
            mcp_log = open(RESULTS / f"qmd-mcp-{corpus}.log", "w")
            try:
                rows, latencies, failures, query_peak_rss = bench_qmd_mcp(
                    queries, qmd_bin, env, mcp_log)
            finally:
                mcp_log.close()
        else:
            rows, latencies = [], []
            query_peak_rss = 0
            failures = 0
            for i, q in enumerate(queries):
                t = time.perf_counter()
                code, stdout, rss, _ = run_measured([qmd_bin, mode, q["query"], "--json", "-n", "10"],
                                                    env=env, capture=True, timeout=900)
                elapsed_ms = (time.perf_counter() - t) * 1000
                query_peak_rss = max(query_peak_rss, rss)
                paths = qmd_result_paths(stdout) if code == 0 else []
                if code == 0:
                    latencies.append(elapsed_ms)  # failed requests are counted, not timed
                else:
                    failures += 1
                    if failures <= 3 or failures % 100 == 0:
                        print(f"  WARN qmd {mode} failed ({failures}) at {q['qid']}: rc={code}",
                              file=sys.stderr)
                rows.append({"qid": q["qid"], "type": q["type"],
                             "rank": hit_rank(paths, qmd_handelize(q["expected"]))})
                if (i + 1) % 25 == 0:
                    print(f"  qmd {mode}: {i + 1}/{len(queries)} queries", file=sys.stderr)
        if failures:
            print(f"  WARN {failures}/{len(queries)} qmd {mode} queries failed — "
                  "recall/latency are NOT valid", file=sys.stderr)
        reports.append({"engine": "qmd", "corpus": corpus, "mode": mode,
                        "query_failures": failures, "force_cpu": force_cpu,
                        "keep_index": keep_index,
                        **provenance,
                        "index_seconds": round(index_seconds, 1),
                        "index_cpu_seconds": round(index_cpu, 1),
                        "index_stalls": index_stalls,
                        "storage_bytes": storage,
                        "memory_mb": {"index_peak": mb(index_peak_rss),
                                      "query_peak": mb(query_peak_rss or None)},
                        **summarize(rows, latencies)})
        if dest is not None:  # persist after each mode: a crash in the multi-hour
            dest.write_text(json.dumps(reports, indent=2))  # LLM mode keeps earlier modes
    return reports


# ---------------------------------------------------------------- report ----


def render_report() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if ".run" in path.name or path.name.endswith(".bak.json"):
            continue  # preserved raw copies, not report rows
        data = json.loads(path.read_text())
        rows.extend(data if isinstance(data, list) else [data])
    if not rows:
        sys.exit("no results yet")
    header = f"{'corpus':<10} {'engine':<6} {'mode':<28} {'n':>4} {'R@1':>6} {'R@3':>6} {'R@10':>6} {'p50ms':>8} {'p99ms':>9} {'index_s':>8} {'cpu_s':>8} {'store_MB':>9} {'qry_MB':>7} {'idx_MB':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (x["corpus"], x["engine"], x["mode"])):
        mem = r.get("memory_mb", {})
        qry_mb = mem.get("daemon_peak") or mem.get("query_peak")
        idx_mb = mem.get("index_peak")
        cpu_s = r.get("index_cpu_seconds")  # absent in pre-2026-06-11 results
        print(f"{r['corpus']:<10} {r['engine']:<6} {r['mode']:<28} {r['n']:>4} "
              f"{r['recall@1']:>6.3f} {r['recall@3']:>6.3f} {r['recall@10']:>6.3f} "
              f"{r['latency_ms']['p50']:>8.1f} {r['latency_ms']['p99']:>9.1f} "
              f"{r['index_seconds']:>8.1f} "
              f"{cpu_s if cpu_s is not None else '-':>8} "
              f"{r['storage_bytes'] / 1e6:>9.1f} "
              f"{qry_mb if qry_mb is not None else '-':>7} "
              f"{idx_mb if idx_mb is not None else '-':>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--engine", choices=["fidx", "qmd"], required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--max-queries", type=int, default=None)
    run.add_argument("--profile", default="nomic-768-q", help="fidx embedding profile")
    run.add_argument("--backend", default="sqlite-vec", choices=["sqlite-vec", "duckdb"],
                     help="fidx vector backend")
    run.add_argument("--modes", default="search,vsearch,query,mcp",
                     help="qmd modes (comma-separated; mcp = warm MCP server, "
                          "the path comparable to fidx's daemon)")
    run.add_argument("--qmd-bin", default="qmd")
    run.add_argument("--qmd-gpu", action="store_true", help="let QMD use the GPU")
    run.add_argument("--cold-sample", type=int, default=20)
    run.add_argument("--keep-index", action="store_true",
                     help="reuse an existing index instead of rebuilding")
    run.add_argument("--query-threads", type=int, default=1,
                     help="ONNX threads for fidx query embedding (default 1: conservative)")
    run.add_argument("--ignore-load", action="store_true",
                     help="measure latency even if the system is busy")
    sub.add_parser("report")
    args = parser.parse_args()

    if args.cmd == "report":
        render_report()
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.corpus, args.max_queries)
    if args.engine == "fidx":
        out = bench_fidx(args.corpus, queries, args.profile, args.cold_sample,
                         args.keep_index, args.ignore_load, args.query_threads,
                         args.backend)
        dest = RESULTS / f"fidx-{args.profile}-{args.backend}-{args.corpus}.json"
    else:
        dest = RESULTS / f"qmd-{args.corpus}.json"
        out = bench_qmd(args.corpus, queries, args.modes.split(","), not args.qmd_gpu,
                        args.keep_index, args.qmd_bin, args.ignore_load, dest=dest)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
