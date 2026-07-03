# fmdidx benchmark harness

Reproducible comparison of fidx against [QMD](https://github.com/tobi/qmd)
on four corpora, per the project's acceptance criteria: recall and latency
must beat QMD on the same datasets. Current numbers: [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

## Corpora (all open data, ≥10k items each)

| name | source | contents |
|------|--------|----------|
| `docs` | 20 Newsgroups | 18,821 text documents |
| `docs-small` | subset of `docs` | 2,000 documents (3 groups) — smoke corpus |
| `chat` | Cornell Movie-Dialogs | ~8,000 WhatsApp-style chat export files (~30k messages, speaker names + timestamps) |
| `code` | one repo per top-10 language, pinned tags | golang (Go), home-assistant/core (Python), node (JS), grafana (TS), elasticsearch (Java), zephyr (C), clickhouse (C++), aspnetcore (C#), rust (Rust), symfony (PHP) — 92,294 files combined; also prepared per-repo (~10k each); per-language sources + .md copied |

The corpora are **text-only**: `is_text_document()` in `corpora.py` rejects
binary payloads at prepare time (uuencoded/BinHex newsgroup posts, files with
NUL bytes, minified bundles under source suffixes). Engines are never
benchmarked on binary content — 25 such posts are excluded from `docs`.

```sh
uv run python bench/corpora.py            # prepare all under bench/data/
```

## Queries

500 known-item queries per corpus, generated from randomly selected files
(seeded, reproducible), one of three types per the target use cases:

- **unique** — the file's rarest tokens (names, identifiers)
- **phrase** — a contiguous word span from the file
- **vague** — shuffled mid-frequency content words (conceptual recall;
  no rare tokens, no original word order)

```sh
uv run python bench/gen_queries.py bench/data/docs -n 500
```

Ground truth is the source file. The same queries are replayed against every
engine, so generation bias hits all engines equally.

## Running

```sh
# fidx (hybrid mode; defaults: nomic-768-q profile, sqlite-vec backend)
uv run python bench/run_bench.py run --engine fidx --corpus docs

# fidx variants: embedding profile x vector backend
uv run python bench/run_bench.py run --engine fidx --corpus docs --profile e5-768 --backend duckdb

# QMD (default modes: search, vsearch, query, mcp)
#   search  = cold CLI, BM25 only (no models loaded)
#   vsearch = cold CLI, LLM query expansion + vector (NOT vector-only: it runs
#             the 1.7B expansion model per query, then embeds every variant)
#   query   = cold CLI, full pipeline (expansion + rerank)
#   mcp     = warm `qmd mcp` server, lex+vec typed sub-queries with rerank —
#             QMD's agent-facing path and the number comparable to fidx's
#             warm daemon (cold CLI modes reload ~2 GB of GGUF per query)
bun install -g @tobilu/qmd
uv run python bench/run_bench.py run --engine qmd --corpus docs

# Render the comparison table from bench/results/*.json
uv run python bench/run_bench.py report
```

Measurement methodology (enforced by the harness):

- Queries are replayed **strictly sequentially** — one request in flight, ever.
- The run **refuses to measure latency if system load > 2.0** at the start
  (`--ignore-load` overrides; the load average and CPU count are recorded in
  every report for provenance).
- fidx query embedding runs with a **single ONNX thread** by default
  (`--query-threads`), a conservative lower bound that approximates laptop
  hardware; QMD keeps its own default threading.
- **Index cost** is recorded as wall time (`index_seconds`), CPU time
  (`index_cpu_seconds`: user+system via `os.wait4` rusage, summed across the
  index subprocesses, so multi-core cost is visible), and on-disk size
  (`storage_bytes`). For QMD the window covers `collection add` + `embed`
  (its `add` ingests files; fidx's only registers a path).
- **Peak memory** is recorded per phase (`memory_mb` in each report):
  the index step and QMD per-query processes via `os.wait4` `ru_maxrss`
  (covers waited-for descendants), the warm fidx daemon via `/proc/<pid>/status`
  `VmHWM`. Linux semantics; on a `--keep-index` run the index peak reflects
  only the incremental scan, not a fresh build.

Notes for fairness:

- QMD runs with `QMD_FORCE_CPU=1` by default (fidx is CPU-only; pass
  `--qmd-gpu` to lift this).
- fidx warm latency is measured through its daemon (the documented agentic
  mode); a cold-CLI sample is also reported. QMD gets the symmetric pair:
  cold CLI modes plus the warm `mcp` mode (one persistent server, first
  model-loading query untimed, memory via `VmHWM` like the fidx daemon).
- `qmd cleanup` runs before **every** mode: QMD caches LLM expansions and
  rerank scores in its DB keyed by query text, so without it mode order and
  `--keep-index` reruns contaminate later measurements (e.g. `query` would
  inherit the expansions `vsearch` cached for the same queries).
- Each engine gets its own isolated index/cache; model downloads are
  excluded from the storage metric.
- `--max-queries N` subsets for smoke runs; `--keep-index` skips rebuilds.

## Results

Canonical published numbers live in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).
`bench/data/` and `bench/results/` are gitignored — only the harness is
source; raw result JSONs are regenerated by the runs above.
