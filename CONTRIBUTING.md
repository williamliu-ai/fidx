# Contributing to fmdidx

Thanks for your interest! fmdidx is a CPU-only, local-first semantic-search CLI.

## Dev setup

fmdidx uses [uv](https://docs.astral.sh/uv/). A uv-managed Python is recommended
because it ships a `sqlite3` with loadable extensions on every OS (see the macOS
note in the README).

```sh
git clone https://github.com/williamliu-ai/fmdidx
cd fmdidx
uv sync --extra dev        # creates .venv with dev deps
uv run fidx doctor         # confirm your host can run fidx
uv run pytest              # unit tests
```

## Tests

- **Unit:** `uv run pytest` (fast; no network).
- **End-to-end (installed artifact):** builds the wheel, installs it into a
  throwaway venv, and runs the ~1k-doc benchmark:
  ```sh
  uv build
  python -m venv /tmp/fidx-e2e && /tmp/fidx-e2e/bin/pip install --only-binary=:all: dist/*.whl
  /tmp/fidx-e2e/bin/python scripts/e2e_smoke.py
  ```
- **Clean-machine (Docker, Linux):** `scripts/verify-install.sh` builds the wheel
  and runs `fidx doctor` + the e2e inside pristine `python:3.11/3.12-slim`
  containers. Requires Docker.
- **Cross-platform (Windows/macOS):** the `install-matrix` GitHub Actions
  workflow installs the built wheel and runs the e2e on
  ubuntu/windows/macos-arm64 × Python 3.11/3.12 on every push.

## Conventions

- Match the surrounding code style; keep changes surgical.
- Add or update tests for behavior changes; keep `fidx doctor` import-safe (it
  must never import native deps eagerly).
- Update `CHANGELOG.md` under `[Unreleased]`.
- Design rationale lives in `docs/DESIGN.md`; benchmark results in
  `docs/BENCHMARKS.md`, harness methodology in `bench/README.md`.

## Pull requests

Open a PR against `main`. CI runs unit tests (ubuntu/windows/macOS) and the
install-matrix; both must be green. Be ready to describe what user-facing
behavior changes and how you verified it.
