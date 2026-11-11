# Changelog

All notable changes to fidx are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and fidx adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-10

### Added
- Optional MCP server for agent hosts. Install `fmdidx[mcp]` and have the host
  launch `fidx --db PATH mcp`. It serves three read-only tools over stdio:
  `fidx_search`, `fidx_get` (body capped at 64 KiB) and `fidx_collections`.
  Each tool runs the CLI and returns its JSON envelope unchanged; CLI errors
  set the MCP tool error flag. The server opens no network port, and the base
  install is unchanged.
- `fidx collection list --json` returns a `fidx.collections.v1` object with
  each collection's root, globs and document count.
- A Claude Code skill under `integrations/claude-code-skill/` for searching an
  existing index and reading cited files before answering.
- `index --json` and `status --json` return versioned `fidx.index.v1` and
  `fidx.status.v1` objects. Index progress and warnings go to stderr; warnings
  are also included in the successful JSON result.
- `fidx.error.v1` JSON execution errors and stable exit codes for generic,
  usage, index, database-lock, missing-model, and ambiguous-reference failures.
  Use root `--json` for commands without their own JSON result option;
  `FIDX_DEBUG=1` restores tracebacks.
- `fidx models fetch embed [--profile NAME]` acquires an embedding model
  explicitly and prints its cache path. `fidx doctor` now reports whether the
  default embedding profile is available locally.
- Optional offline cross-encoder reranking with `search --rerank[=MODEL]` and
  `--no-rerank`, explicit `models fetch rerank [--model MODEL]` and
  `index --rerank-prefetch[=MODEL]`, plus soft doctor capability/cache checks.
  Search advice retains the effective rerank setting, including an explicit
  off override of `FIDX_RERANK`.

### Changed
- **Breaking:** `doctor --json` now wraps its diagnostic checks in a
  `fidx.doctor.v1` object with `ok` and `checks`, rather than returning a bare
  array. Each check retains its remediation text. Soft failures leave `ok`
  true; hard failures exit 1.
- Hybrid/vector search, daemon startup, and calibration now load embedding
  models from local files only. On a cold cache they exit with a `fidx models
  fetch embed` hint instead of implicitly downloading. A valid empty index
  returns `empty_index` without constructing an embedder.
- Search, get, status, serve, calibrate, and collection list/remove now require
  an existing valid fidx index. These commands no longer create an index from
  a missing path or add schema tables to an unrelated SQLite file; they exit
  3 with create-side actions instead. Search validates the database before
  using a daemon and ignores daemon results for a replaced database.
- `get "#prefix"` now exits 6 with matching paths when more than one document
  has the prefix, instead of silently returning the first match.
- Embedding runs 16 sequences per ONNX Runtime call (was 64) with the CPU
  memory arena disabled. Indexing peak memory drops from about 20 GB to under
  3 GB with the default profile, at the same or better throughput. Stored
  vectors shift slightly (batch padding affects the quantized model's output);
  existing indexes stay compatible.

## [0.1.2] - 2026-07-07

### Added
- `fidx doctor` — diagnoses host capabilities (sqlite loadable extensions, FTS5,
  sqlite-vec `vec0`, embedding-model cache) and exits non-zero on a hard failure.
  Safe to run anywhere: it never imports native deps eagerly, so it reports a
  broken/wrong-arch `sqlite-vec` instead of crashing.
- Packaging for publication: single-sourced dynamic version, project URLs,
  explicit Python classifiers, and lower+upper bounds on native dependencies
  (including a direct, bounded `onnxruntime`).
- `scripts/e2e_smoke.py` — a standalone end-to-end benchmark that installs-and-
  drives the published CLI on a deterministic ~1,000-doc corpus and gates
  recall@10.
- Clean-machine verification: `docker/Dockerfile.linux` + `scripts/verify-install.sh`
  (local Linux proof) and a CI install-matrix that installs the built wheel and
  runs the e2e on linux-x86_64, windows-x86_64, and macos-arm64 × Python 3.11/3.12.
- Agent-facing indexing/search guide for tool-discovery agents and CLI callers
  choosing how to use fidx.

### Changed
- **Breaking:** `fidx search --json` now returns an agent-oriented envelope
  instead of a bare result array. Results live under `results`; the envelope
  also includes `schema`, `status`, request echo, summary, diagnostics, and
  suggested `next_actions` so agents can decide whether to inspect, retry with
  different search parameters, relax filters, or index data.
- Search diagnostics avoid exposing collection inventory or exact active
  document counts. The envelope reports `index_empty` and only echoes requested
  scopes that are unknown.
- The search JSON summary now includes `truncation_advice`, which recommends
  `off`, `knee` or `calibrated` according to the result count, score profile
  and stored calibration floor.
- The warm daemon socket now lives under a per-user runtime/cache directory and
  is bound with owner-only permissions.
- `db.py` loads `sqlite_vec` lazily and falls back to `pysqlite3` where the
  stdlib `sqlite3` lacks extension loading (best-effort; Linux convenience).

## [0.1.0] - unreleased
- Initial hybrid (BM25 + sqlite-vec, RRF) local semantic-search engine, warm
  daemon, collections, deterministic truncation, and corpus calibration.
