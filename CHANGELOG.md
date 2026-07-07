# Changelog

All notable changes to fidx are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and fidx adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed
- **Breaking:** `fidx search --json` now returns an agent-oriented envelope
  instead of a bare result array. Results live under `results`; the envelope
  also includes `schema`, `status`, request echo, summary, diagnostics, and
  suggested `next_actions` so agents can decide whether to inspect, retry with
  different search parameters, relax filters, or index data.
- `db.py` loads `sqlite_vec` lazily and falls back to `pysqlite3` where the
  stdlib `sqlite3` lacks extension loading (best-effort; Linux convenience).

## [0.1.0] - unreleased
- Initial hybrid (BM25 + sqlite-vec, RRF) local semantic-search engine, warm
  daemon, collections, deterministic truncation, and corpus calibration.
