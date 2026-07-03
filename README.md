# fmdidx

[![CI](https://github.com/williamliu-ai/fmdidx/actions/workflows/ci.yml/badge.svg)](https://github.com/williamliu-ai/fmdidx/actions/workflows/ci.yml)
[![install-matrix](https://github.com/williamliu-ai/fmdidx/actions/workflows/install-matrix.yml/badge.svg)](https://github.com/williamliu-ai/fmdidx/actions/workflows/install-matrix.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Fast, local-only semantic search for markdown, text, chat exports and code.**

CPU-only. No cloud, no GPU, no API keys. One SQLite file holds the full index.
Built to be the retrieval layer for agentic workflows: millisecond warm
queries, JSON output, and collection scoping.

## Why

Local semantic search tools tend to buy recall with latency: LLM query
expansion and LLM reranking push a single query to ~10 seconds on CPU. fmdidx
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
  **FTS5** (fmdidx loads the `sqlite-vec` extension). Run `fidx doctor` to verify.
- Prebuilt wheels exist for the verified platforms below — **no compiler needed**.
- First `fidx index` downloads the embedding model once (then fully offline).

| Platform (triple) | Status | Notes |
|---|---|---|
| Linux x86_64 | ✅ verified (CI + Docker) | any Python 3.11/3.12 with extensions |
| macOS arm64 (Apple Silicon) | ✅ verified (CI) | **use Homebrew Python** (see install) |
| Windows x86_64 | ✅ verified (CI) | python.org / uv Python |
| macOS Intel, Linux/Windows arm64 | best-effort | depends on upstream wheel availability |

## Install

> **Naming.** **fmdidx** ("fast markdown index") installs a CLI named
> **`fidx`** — short for typing, and distinct because the PyPI name `fidx`
> belongs to an unrelated project. Until the first release, install from a
> built wheel or from source (below). After release: `uv tool install fmdidx`.

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
  on **macOS system Python** and uv/python.org macOS builds. Fix: install fmdidx
  with **Homebrew Python** (see macOS install above). `fidx doctor` confirms the
  fix.
- **`sqlite-vec failed to load` / wrong architecture** — ensure a `sqlite-vec`
  wheel exists for your platform: `pip install --only-binary=:all: sqlite-vec`.
- **First search is slow / offline use** — the embedding model downloads once on
  first index. Pre-seed `FASTEMBED_CACHE_PATH` to use fmdidx air-gapped.

## Benchmarks

`bench/` is a reproducible harness comparing fmdidx against **[QMD]** on four
corpora with known-item queries (CPU-only, warm fmdidx daemon, idle box). The
headline: **fmdidx matches or beats QMD on recall@10 on every corpus, at
~300–1000× lower latency than QMD's LLM modes.**

recall@10 / median (p50) latency per query:

| Corpus (size) | fmdidx hybrid | QMD `query` (LLM hybrid) | QMD `search` (FTS) |
|---|---|---|---|
| docs-small (2k) | **1.000** / **20 ms** | 0.920 / 33 s | 0.920 / 78 ms |
| docs (18.8k) | **0.990** / **49 ms** | 0.916 / 36 s | 0.896 / 87 ms |
| chat (8k) | 0.914 / **18 ms** | 0.914 / 18 s | 0.916 / 81 ms |
| code (92.3k) | **0.900** / 452 ms | 0.786 / 33 s | 0.784 / 121 ms |

- **Code is the widest gap:** fmdidx R@10 **0.900 vs QMD 0.786 (+11 pts)**; QMD's
  pure-vector mode collapses to 0.048 there.
- fmdidx also leads **R@3** on every corpus (e.g. docs 0.962 vs 0.876).
- **Where QMD wins:** raw latency on the big code corpus (its FTS `search` 121 ms
  vs fmdidx 452 ms — fmdidx's brute-force vector KNN over 92k vectors), and chat R@1.
  Even then fmdidx stays sub-second and ~65× faster than QMD's LLM modes.

Full per-corpus tables, conditions, purity metrics, the SymDex comparison,
and the honest threats-to-validity: [docs/BENCHMARKS.md](docs/BENCHMARKS.md);
harness usage and methodology: [bench/README.md](bench/README.md).

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
