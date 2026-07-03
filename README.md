# fidx

[![CI](https://github.com/williamliu-ai/fidx/actions/workflows/ci.yml/badge.svg)](https://github.com/williamliu-ai/fidx/actions/workflows/ci.yml)
[![install-matrix](https://github.com/williamliu-ai/fidx/actions/workflows/install-matrix.yml/badge.svg)](https://github.com/williamliu-ai/fidx/actions/workflows/install-matrix.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Fast, local-only semantic search for markdown, text, chat exports and code.**

CPU-only. No cloud, no GPU, no API keys. One SQLite file holds the full index.
Built to be the retrieval layer for agentic workflows: millisecond warm
queries, JSON output, and collection scoping.

## Why

Local semantic search tools tend to buy recall with latency: LLM query
expansion and LLM reranking push a single query to ~10 seconds on CPU. fidx
takes a different trade — hybrid BM25 + 768-dim vector search fused with
reciprocal-rank fusion (RRF), and *no LLM calls in the query path*. One ONNX
embedding pass per query is the only model work.

- **Hybrid recall** — FTS5 BM25 catches exact names and identifiers; 768-dim
  embeddings catch "that doc that discussed the indexing project"; RRF fuses both.
- **Millisecond queries** — a warm daemon answers hybrid searches in ~5 ms
  (p50) on a ~20k-chunk index; cold CLI calls stay well under a second.
- **One file** — documents, BM25 index and vectors live in a single SQLite
  database (FTS5 + [sqlite-vec]). Copy it, back it up, delete it.
- **Scoped search** — group sources into named collections (`-c emails`)
  and search only what you mean.

[sqlite-vec]: https://github.com/asg017/sqlite-vec

## Requirements

- **Python 3.11 or 3.12** whose `sqlite3` supports **loadable extensions** and
  **FTS5** (fidx loads the `sqlite-vec` extension). Run `fidx doctor` to verify.
- Prebuilt wheels exist for the verified platforms below — **no compiler needed**.
- First `fidx index` downloads the embedding model once (then fully offline).

| Platform (triple) | Status | Notes |
|---|---|---|
| Linux x86_64 | ✅ verified (CI + Docker) | any Python 3.11/3.12 with extensions |
| macOS arm64 (Apple Silicon) | ✅ verified (CI) | **use Homebrew Python** (see install) |
| Windows x86_64 | ✅ verified (CI) | python.org / uv Python |
| macOS Intel, Linux/Windows arm64 | best-effort | depends on upstream wheel availability |

## Install

> **Why is the package named `fmdidx`?** The project and its command are
> **fidx**, but the PyPI name `fidx` was already taken by an unrelated
> package — so fidx is distributed as **`fmdidx`** ("fast markdown index").
> That is the only place the name differs: `uv tool install fmdidx` installs
> the `fidx` command. Until the first release, install from a built wheel or
> from source (below).

The recommended installer is [uv](https://docs.astral.sh/uv/), because a
uv-managed Python ships loadable sqlite extensions on Linux and Windows.

**Linux / Windows:**

```sh
uv tool install fmdidx          # or: pipx install fmdidx
fidx doctor                          # verify your host
```

**macOS:** uv's bundled Python (and the python.org build) ship a `sqlite3`
*without* loadable-extension support, so use **Homebrew Python**:

```sh
brew install python
uv tool install --python "$(brew --prefix python)/libexec/bin/python" fmdidx
# or: pipx install --python "$(brew --prefix python)/libexec/bin/python3" fmdidx
fidx doctor
```

**From a built wheel (works today, name-independent):**

```sh
uv build
pip install --only-binary=:all: dist/*.whl    # use Homebrew Python on macOS
fidx doctor
```

**From a checkout (development):**

```sh
uv sync && uv run fidx doctor
```

If `fidx doctor` reports a failure, it prints exactly what is missing and how to
fix it — see [Troubleshooting](#troubleshooting).

## Quick start

```sh
# Register directories as named collections
fidx collection add ~/notes --name notes
fidx collection add ~/mail/export --name emails --glob "**/*.txt"

# Scan + chunk + embed (incremental; first run downloads the ONNX model)
fidx index

# Search (hybrid BM25 + vector by default)
fidx search "the doc that discussed the most recent indexing project"
fidx search "Grace Hopper" --mode lexical      # exact-name lookup, no model load
fidx search "deployment checklist" -c notes    # scope to one collection

# Agent-friendly output
fidx search "error handling" --json -n 10
fidx search "auth" --files --min-score 0.02

# Fetch a document by path or docid
fidx get "notes/meeting.md"
fidx get "#a1b2c3"
```

### Warm daemon (recommended for agents)

```sh
fidx serve &          # keeps the model + index hot on a unix socket
fidx search "..."     # all searches now take milliseconds
```

The CLI uses the daemon automatically when it is running; `--no-daemon` opts out.

## Verifying your install

```sh
fidx doctor                          # host capability report (exit 0 = ready)

# Full end-to-end benchmark on a ~1,000-doc corpus against the installed CLI:
python scripts/e2e_smoke.py          # builds corpus, indexes, searches, gates recall

# Clean-machine proof in pristine Docker containers (Linux):
scripts/verify-install.sh            # builds the wheel, installs + runs e2e on 3.11 & 3.12
```

The same e2e runs in CI on Linux, macOS (arm64) and Windows × Python 3.11/3.12
(the `install-matrix` workflow) — installing the built wheel from scratch and
asserting `recall@10`.

## How it works

```
files ──> documents (SQLite) ──> FTS5 (BM25, porter)        ─┐
                │                                            ├─> RRF fusion ─> results
                └──> chunks ──> ONNX embeddings ─> sqlite-vec ┘
```

- **Chunking** splits at the best structural break (headings, code-fence
  boundaries, blank lines) near a ~1800-char target with 15% overlap, never
  inside a code fence. Chunks store offsets, not copies.
- **Embeddings** via [fastembed]/ONNX (CPU). The default profile is 768-dim;
  smaller profiles exist for small corpora.
- **Search** runs BM25 and vector KNN in parallel and fuses with RRF
  (k=60); results are document-level with best-chunk snippets.

[fastembed]: https://github.com/qdrant/fastembed

## Troubleshooting

- **`enable_load_extension` / "sqlite3 was built without loadable-extension
  support"** — your Python's sqlite cannot load `sqlite-vec`. This is the default
  on **macOS system Python** and uv/python.org macOS builds. Fix: install fidx
  with **Homebrew Python** (see macOS install above). `fidx doctor` confirms the
  fix.
- **`sqlite-vec failed to load` / wrong architecture** — ensure a `sqlite-vec`
  wheel exists for your platform: `pip install --only-binary=:all: sqlite-vec`.
- **First search is slow / offline use** — the embedding model downloads once on
  first index. Pre-seed `FASTEMBED_CACHE_PATH` to use fidx air-gapped.

## Benchmarks

`bench/` is a reproducible harness comparing fidx against **[QMD]** on four
corpora with known-item queries (CPU-only, warm engines, idle box). Result
quality has **two axes that trade off**: *recall* (is the right document in
the top-10) and *purity* — **noise@10** (share of returned results an LLM
judge rated irrelevant, lower is better) and **clean@10** (share of queries
whose results contain zero noise, higher is better). The headline:
**fidx beats QMD's hybrid on recall, noise, and clean on every corpus, at
~300–1000× lower latency.**

| Corpus (size) | Engine | R@10 ↑ | noise@10 ↓ | clean@10 ↑ | p50 latency |
|---|---|---|---|---|---|
| docs-small (2k) | **fidx** | 0.933 | **0.219** | **0.560** | **20 ms** |
| | QMD `query` (LLM hybrid) | 0.933 | 0.564 | 0.053 | 33 s |
| | QMD `search` (FTS) | 0.920 | n/m | n/m | 78 ms |
| docs (18.8k) | **fidx** | **0.962** | **0.250** | 0.482 | **49 ms** |
| | QMD `query` | 0.914 | 0.677 | 0.060 | 36 s |
| | QMD `search` | 0.896 | 0.353 | 0.818 | 87 ms |
| chat (8k) | **fidx** | 0.908 | 0.133 | 0.710 | **18 ms** |
| | QMD `query` | 0.912 | 0.472 | 0.186 | 18 s |
| | QMD `search` | 0.916 | 0.086 | 0.964 | 81 ms |
| code (92.3k) | **fidx** | **0.864** | **0.127** | 0.704 | 452 ms |
| | QMD `query` | 0.782 | 0.713 | 0.056 | 33 s |
| | QMD `search` | 0.784 | 0.256 | 0.868 | 121 ms |

fidx rows are measured with its built-in deterministic result truncation
enabled (`--truncate`; ships off by default — without it fidx trades purity
for recall, e.g. code R@10 0.900 at noise 0.297). "n/m" = not measured.

- **Hybrid vs hybrid** (fidx vs QMD `query`): fidx wins all three quality
  metrics on every corpus — e.g. code recall +8 pts with 5.6× less noise —
  with no LLM anywhere in its query path.
- **Where QMD wins:** its FTS `search` mode is the purity champion on chat
  (clean 0.964) and the latency champion on the big code corpus (121 ms vs
  fidx's 452 ms brute-force KNN over 92k vectors), but trails on recall
  where it matters (docs, code). QMD's pure-vector mode collapses to 0.048
  R@10 on code.
- fidx stays sub-second even on its weakest corpus and is ~65× faster than
  QMD's LLM modes there.

Full tables (R@1/R@3, untruncated numbers, per-language code results), the
SymDex comparison, conditions, and the honest threats-to-validity:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md); harness usage and methodology:
[bench/README.md](bench/README.md).

[QMD]: https://github.com/tobi/qmd

## Development

```sh
uv sync --extra dev
uv run pytest
scripts/verify-install.sh    # clean-machine install + e2e (Docker)
```

Architecture notes: [docs/DESIGN.md](docs/DESIGN.md). Contributing guide:
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
