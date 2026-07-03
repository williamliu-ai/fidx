# Security Policy

## Supported versions

fidx is pre-1.0; security fixes are applied to the latest released version.

## Reporting a vulnerability

Please report security issues privately via GitHub's
["Report a vulnerability"](https://github.com/williamliu-ai/fidx/security/advisories/new)
(Security → Advisories) rather than a public issue. We aim to acknowledge within
a few days.

## Threat model notes

- fidx runs **entirely locally**: no network calls in the index or query path.
  The only outbound network is a **one-time embedding-model download** on first
  index (via `fastembed`/Hugging Face); after that it is fully offline. You can
  pre-seed the model cache for air-gapped use (`FASTEMBED_CACHE_PATH`).
- The index is a single SQLite file containing your document text; protect it as
  you would the source files.
- fidx loads the `sqlite-vec` SQLite extension (a pinned, bounded dependency);
  `fidx doctor` reports exactly which native components load.
- Releases are published via PyPI **Trusted Publishing (OIDC)** — no long-lived
  tokens — with build provenance attestations.
