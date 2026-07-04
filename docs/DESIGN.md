# fidx design

## Goal

A local-only semantic search CLI for markdown/text/chat/code that beats QMD
on retrieval recall and latency on the same corpora, using only freely
licensed open-source components, with no GPU or cloud dependency.

Success metrics (from the original requirement):

1. **Retrieval accuracy** — R@1, R@3, R@10 against known-item queries.
2. **Latency** — per-query median and p99.
3. **Storage** — index size on disk, balanced against speed and recall.

Target use cases:

1. Vague concept search — "a doc discussed the most recent indexing project".
2. Unique-name search — a person, a place, an identifier.
3. Scoped search — "only search the emails".

## Where the design comes from

### QMD study (v2.5.3, MIT)

QMD's pipeline: SQLite + FTS5 BM25 + sqlite-vec, GGUF embedding model
(embeddinggemma-300M Q8) via node-llama-cpp, LLM query expansion (fine-tuned
1.7B), LLM reranking (qwen3-reranker 0.6B), RRF fusion with position-aware
blending. Quality is good; the cost is latency — the LLM stages put a warm
hybrid `query` at ~10s and a cold CLI call at ~16s on CPU, and the recall
backbone (BM25 + vectors + RRF) does most of the work before the LLM stages.

What we kept: the storage shape (one SQLite file, FTS5 external-content +
sqlite-vec), structure-aware chunking with overlap, RRF fusion, collection
scoping, docids, JSON/agent output modes, a warm daemon.

What we dropped: every LLM call in the query path. No query expansion, no
reranker. One query embedding is the only model work per search.

What we changed: 768-dim ONNX embeddings (fastembed/onnxruntime) instead of
a 300M GGUF model — faster on CPU and dimensionally stronger; the original
requirement notes 384-dim degrades on ~100k-chunk corpora.

### Prior embedding-model research

Earlier CPU embedding-model benchmarking (a private precursor project)
measured e5-small (384d) at ~5ms/query vs Qwen3-0.6B at ~67ms, and
sentence-transformers cold load at ~18-35s — which motivated (a) the ONNX
runtime here instead, and (b) the warm daemon. Its eval harness (R@k + MRR,
p50/p95 latency, index size) shaped `bench/`.

## Architecture

```
src/fidx/
  config.py    profiles (model, dim, prefixes), db path resolution
  db.py        schema: documents + FTS5 + chunks + vectors (sqlite-vec)
  chunker.py   structural chunking, offset-based; title extraction
  embedder.py  FastEmbedder (ONNX) / HashEmbedder (test seam)
  indexer.py   scan -> upsert -> chunk -> batch embed; incremental by hash
  search.py    lexical (BM25), vector (KNN), hybrid (RRF)
  daemon.py    unix-socket warm server + client; JSON-line protocol
  cli.py       click CLI wiring
```

### Storage

One SQLite file. `documents.body` is the only copy of content; FTS5 is an
external-content table over it (BM25, porter unicode61) and chunks are
`(pos, length)` offsets.

Vectors go through a pluggable backend (`vector_store.py`):

- **sqlite-vec** (default): a `vec0` table in the same file (cosine, float32,
  brute-force KNN — exact, no ANN recall loss; fine well past 100k chunks).
- **duckdb**: a DuckDB VSS/HNSW sidecar (`<db>.vectors.duckdb`), a design
  validated in the prior research above — approximate KNN, faster at very
  large scale, extra storage; the sidecar is rebuildable from SQLite.

The embedding profile **and vector backend** are pinned in `meta` by the
**first embed run**, not by whichever command created the file; a mismatch in
either is a hard error, never a silent dimension/backend conflict.

### Query pipeline

```
query ──┬─ FTS5 MATCH (OR-of-quoted-tokens, phrases preserved) ─ top 50 ─┐
        │                                                                ├─ RRF(k=60) ─ top N
        └─ embed (768d, ONNX) ─ vec0 KNN top 200 chunks ─ best-per-doc ──┘
```

- The OR-joined token expression is recall-oriented; BM25 still ranks
  all-term documents first. User-quoted phrases pass through as FTS5 phrases.
- Vector search over-fetches 4× then dedups to best chunk per document.
- RRF: `score = Σ weight / (60 + rank + 1)`, equal weights. A document found
  by both sources outranks a single-source rank-1 — that is the hybrid bet.

### Latency budget

| path | cost |
|---|---|
| daemon hybrid query (~20k chunks) | ~5 ms p50 |
| cold CLI hybrid query | python + ONNX load, < 1 s |
| lexical-only query | no model load at all |

The daemon (`fidx serve`) holds the SQLite connection and ONNX model on a
unix socket; the CLI auto-detects it. This is the intended mode for agents.

## Invariants

- No LLM calls and no network calls in the query path, ever.
- `documents.body` is the single copy of content (FTS external-content,
  chunk offsets). Storage = corpus + FTS index + 3KB/chunk of vectors.
- Every chunk of an embedded document has exactly one `vectors` row.
- Search results are document-level; chunk granularity is an internal detail.
- The test suite must not download models (`HashEmbedder` is the seam).

## Anti-goals (for now)

- ANN indexes (HNSW/IVF) — unnecessary below ~1M chunks; revisit with data.
- Vector quantization (int8/binary) — a storage lever to pull if the
  benchmark shows storage is the losing metric.
- Rerankers — only if benchmark recall is behind QMD; a small ONNX
  cross-encoder on top-40 would cost ~100-300ms, still 30× under QMD.
- File watching; MCP server. Both are thin layers to add later.

## Validation

See `bench/README.md`: four corpora (docs / docs-small / chat / code),
500 generated known-item queries per corpus, identical queries replayed
against fidx and QMD (search / vsearch / query modes), reporting R@k,
latency p50/p99, index build time and on-disk size.
