# fidx benchmarks

Canonical results for fidx (distributed on PyPI as `fmdidx`) vs [QMD](https://github.com/tobi/qmd) and
[SymDex](https://github.com/husnainpk/SymDex), measured with the reproducible
harness in [`bench/`](../bench/README.md). Every number here can be
regenerated; see [Reproducing](#reproducing) for commands and cost warnings.

**House rule for reading (and quoting) these numbers:** result *quality* has
two axes that trade off against each other — **recall** (is the right document
in the top-k) and **purity** (is the top-k free of noise). Neither is
meaningful alone; tables here always carry both where quality is compared.
Latency is reported alongside as the third dimension.

## Conditions

- Hardware: one 32-core x86_64 Linux machine, 62 GiB RAM, CPU only (no GPU).
- fidx: warm daemon (its documented agentic mode), **one ONNX thread** for
  query embedding (a conservative laptop-like lower bound); profile
  `e5-768` / `sqlite-vec` unless stated.
- QMD **2.5.3**, `QMD_FORCE_CPU=1`, via its CLI / `mcp` server. Modes:
  `search` (BM25), `vsearch` (LLM query expansion + vector), `query` (full
  LLM hybrid), `mcp` (warm server — the number comparable to fidx's daemon).
- SymDex **v0.1.26**, semantic mode (MiniLM-384 default; a 768-d
  `all-mpnet-base-v2` A/B is included).
- Queries strictly sequential; latency measurement refuses to run when system
  load > 2.0. Full methodology, fairness notes (cache cleanup between QMD
  modes, isolated indexes, memory accounting): [`bench/README.md`](../bench/README.md).

## Corpora and queries

| corpus | source | size |
|---|---|---|
| `docs` | 20 Newsgroups (binary posts filtered out) | 18,821 documents |
| `docs-small` | 3-group subset of `docs` | 2,000 documents |
| `chat` | Cornell Movie-Dialogs rendered as **synthetic** WhatsApp-style chat exports (no real chats) | ~8,000 files / ~30k messages |
| `code` | 10 real repos, one per language, pinned tags | 92,294 files combined; also benchmarked per-repo (~10k files each) |

500 known-item queries per corpus (150 for docs-small), generated
reproducibly from randomly selected files in three types: **unique** (rarest
tokens), **phrase** (contiguous span), **vague** (shuffled mid-frequency
content words). Ground truth is the source file; the same queries are
replayed against every engine.

## Recall + latency

fidx hybrid (warm daemon) vs QMD modes. `p50` is per-query latency; fidx does
one embedding + FTS5 + vector KNN per query, **no LLM calls**.

| corpus | engine / mode | R@1 | R@3 | R@10 | p50 latency |
|---|---|---|---|---|---|
| docs-small | **fidx hybrid** | 0.740 | 0.927 | **1.000** | **20 ms** |
| docs-small | qmd search (FTS) | 0.820 | 0.907 | 0.920 | 78 ms |
| docs-small | qmd query (LLM hybrid) | 0.667 | 0.840 | 0.920 | 33 s |
| docs-small | qmd mcp (warm LLM) | 0.793 | 0.913 | 0.953 | 27 s |
| docs-small | qmd vsearch | 0.207 | 0.287 | 0.333 | 4.4 s |
| docs | **fidx hybrid** | 0.780 | **0.962** | **0.990** | **49 ms** |
| docs | qmd search | 0.784 | 0.876 | 0.896 | 87 ms |
| docs | qmd query | 0.636 | 0.810 | 0.916 | 36 s |
| docs | qmd mcp | 0.800 | 0.872 | 0.914 | 27 s |
| docs | qmd vsearch | 0.068 | 0.122 | 0.172 | 5.1 s |
| chat | fidx hybrid | 0.816 | 0.908 | 0.914 | **18 ms** |
| chat | qmd search | 0.892 | 0.904 | 0.916 | 81 ms |
| chat | qmd query | 0.866 | 0.900 | 0.914 | 18 s |
| chat | qmd mcp | 0.890 | 0.902 | 0.912 | 12 s |
| chat | qmd vsearch | 0.280 | 0.416 | 0.538 | 4.5 s |
| code (92.3k) | **fidx hybrid (e5)** | 0.710 | 0.862 | **0.900** | 452 ms |
| code (92.3k) | fidx hybrid (nomic) | 0.712 | 0.858 | 0.900 | 488 ms |
| code (92.3k) | qmd search | 0.686 | 0.760 | 0.784 | **121 ms** |
| code (92.3k) | qmd query | 0.588 | 0.698 | 0.786 | 33 s |
| code (92.3k) | qmd mcp | 0.686 | 0.724 | 0.786 | 29 s |
| code (92.3k) | qmd vsearch | 0.028 | 0.038 | 0.048 | 5.1 s |

Reading it:

- fidx matches or beats QMD on R@10 on every corpus, leads R@3 everywhere,
  at 18–49 ms vs 12–36 s for QMD's LLM modes (~300–1000×).
- **Where QMD wins:** `search` R@1 on docs-small/chat, and raw latency on the
  big code corpus (121 ms vs fidx's 452 ms — fidx's vector arm is a
  brute-force KNN over 92k vectors). Even there fidx stays sub-second and
  ~65× faster than QMD's LLM modes.
- **Code is the widest recall gap:** fidx 0.900 vs QMD's best 0.786
  (+11 pts); QMD's pure-vector mode collapses to 0.048 on code.
- Index cost at 92k files is fidx's real cost: ~6.6 h (e5) / ~13.5 h (nomic)
  of CPU embedding vs QMD's ~1.6 h. (QMD `search`/`vsearch`/`query` each had
  1 of 500 code queries fail to parse — a ~0.2 pt effect on those rows.)

## Purity (the second axis)

Purity = **noise-rate@10** (share of returned docs judged pure noise) and
**clean@10** (share of queries whose top-10 contains zero noise docs), scored
by an LLM judge over the union of all engines' top-10s (every engine scored
against the same judgments). Motivation: an engine that always fills 10 slots
"buys" recall with a noisy tail.

Raw engines first:

| corpus | engine / mode | R@10 | noise@10 | clean@10 |
|---|---|---|---|---|
| docs | fidx hybrid (no truncation) | 0.990 | 0.513 | 0.096 |
| docs | qmd query | 0.914 | 0.677 | 0.060 |
| docs | qmd search | 0.896 | 0.353 | **0.818** |
| chat | fidx hybrid (no truncation) | 0.914 | 0.354 | 0.258 |
| chat | qmd query | 0.912 | 0.472 | 0.186 |
| chat | qmd search | 0.916 | **0.086** | **0.964** |
| code | fidx hybrid e5 (no truncation) | 0.900 | 0.297 | 0.262 |
| code | qmd query | 0.782 | 0.713 | 0.056 |
| code | qmd search | 0.784 | 0.256 | **0.868** |

Untruncated fidx wins recall but pads its tail with related-but-irrelevant
docs; QMD `search` is the purest mode (short, abstaining lists) but trails on
recall. This motivated deterministic truncation.

### Truncation: hybrid-vs-hybrid on both axes

fidx hybrid + `knee` truncation (`--truncate knee`: parameter-free elbow cut,
~2 µs/query, no LLM) vs QMD `query` — the apple-to-apple hybrid comparison:

| corpus | engine | R@10 | noise@10 | clean@10 |
|---|---|---|---|---|
| docs-small | qmd query | 0.933 | 0.564 | 0.053 |
| docs-small | **fidx + knee** | 0.933 | **0.219** | **0.560** |
| docs | qmd query | 0.914 | 0.677 | 0.060 |
| docs | **fidx + knee** | **0.962** | **0.250** | **0.482** |
| chat | qmd query | 0.912 | 0.472 | 0.186 |
| chat | **fidx + knee** | 0.908 | **0.133** | **0.710** |
| code | qmd query | 0.782 | 0.713 | 0.056 |
| code | **fidx + knee** | **0.864** | **0.127** | **0.704** |

fidx+knee beats QMD's hybrid on every axis on docs and code, and dominates
purity at tied recall on docs-small/chat. The recall *cost* of knee on fidx
itself is 0–3.6 pts (e.g. code 0.900 → 0.864). The default remains
`--truncate off`; see [Threats to validity](#threats-to-validity) for why the
knee *recall* numbers should not be over-generalized, and `fidx calibrate`
for the corpus-adaptive alternative (derives a score floor from the indexed
corpus itself — self-retrieval positives + gibberish negatives — instead of
tuning to any benchmark).

## Paraphrase queries — semantic recall (the hard set)

An independent reviewer of the v0.1.0 benchmark made a correct observation:
the known-item queries are built from tokens taken from the target document,
so they reward lexical overlap and under-test semantic paraphrase. This
section measures the semantic regime directly.

**Query construction.** For every known-item query, an LLM wrote a paraphrase
query for the same target document — natural wording, 8–20 words, with the
document's distinctive terms (names, identifiers, rare words) deliberately
paraphrased away. A **separate** LLM pass then reviewed each query
(specific enough that the target is the best answer among thousands; no
copied distinctive terms); failed queries were regenerated once, and
still-invalid ones dropped. Final sets: docs-small 148, docs 491, chat 498,
code 485 (98.4% of the known-item counts). Query→document content-word
overlap averages **0.09–0.12** (the known-item sets are ~1.0 by
construction) — these are genuinely non-lexical. The sets are checked into
`bench/data/queries-*-paraphrase.jsonl`; the harness is
`bench/paraphrase_bench.py`.

**Results** (recall@10; same kept indexes as the main tables; fidx e5-768 /
sqlite-vec; fidx decomposed into its two arms):

| corpus | fidx hybrid | fidx lexical-only | fidx vector-only | qmd search (BM25) | qmd query (LLM) |
|---|---|---|---|---|---|
| docs-small (2k) | 0.419 | 0.149 | 0.541 | 0.000 | **0.635** |
| docs (18.8k) | 0.385 | 0.141 | **0.450** | 0.000 | pending† |
| chat (8k) | 0.317 | 0.078 | **0.373** | 0.000 | pending† |
| code (92.3k) | 0.039 | 0.016 | 0.041 | 0.000 | pending† |

† qmd's LLM mode costs ~33 s/query on CPU; it was run in full on docs-small
and is pending on the larger corpora (subsampled runs planned).

**What this says, honestly:**

1. **The reviewer was right about the known-item sets.** Every engine's
   near-perfect known-item recall is substantially lexical. On genuinely
   paraphrased queries, recall drops for everything.
2. **BM25 alone gets zero.** qmd `search` returned empty result lists for
   ~93% of paraphrase queries (verified genuine with known-item controls
   through the identical replay path) — with no shared terms, conjunctive
   FTS has nothing to match. fidx's lexical arm barely survives
   (0.08–0.15) only because its OR-token matching is looser.
3. **fidx's vector arm does real semantic work on prose** — 0.45–0.54 R@10
   on the doc corpora and 0.37 on chat, at millisecond latency. Source
   attribution confirms it: 95–98% of fidx hybrid's paraphrase hits involve
   the vector arm.
4. **LLM query expansion earns its latency in this regime.** On docs-small,
   qmd `query` leads (0.635 vs fidx's best 0.541) — at ~33 s vs 20 ms per
   query. This is the honest trade both tools embody: fidx gives you most
   of the semantic recall at reflex latency; an LLM in the query path buys
   the rest at ~1,500× the cost.
5. **Known weakness #1 — hybrid fusion drags on semantic queries.** fidx
   hybrid is WORSE than its own vector-only mode on every corpus here
   (e.g. docs-small 0.419 vs 0.541): RRF gives fusion credit to lexical
   matches that are noise in this regime. Query-adaptive arm weighting is
   now a roadmap item; until then, `--mode vector` is the better setting
   for purely conceptual queries.
6. **Known weakness #2 — semantic code search at 92k files does not work
   with a 768-d text embedder.** All modes collapse on code (≤0.04).
   fidx's strong known-item code recall (0.90) is overwhelmingly lexical
   (identifiers) — which is what code search mostly is, but callers wanting
   "find the file that implements X" semantics should know the limit.
   Code-tuned embeddings are the research direction.

## Per-repo code benchmark (10 languages, each indexed alone)

The combined 92.3k index mixes cross-language distractors, so each repo was
also benchmarked alone (~10k files each: node, grafana, home-assistant,
golang, aspnetcore, symfony, clickhouse, elasticsearch, rust, zephyr).
Summary across the ten repos:

- **Recall:** fidx R@10 ≥ QMD `query` on every repo where both completed
  (e.g. JS 0.97 vs 0.88, Go 0.75 vs 0.57, Rust 0.98 vs 0.84, C 0.99 vs 0.92)
  at 50–100 ms vs 31–79 s.
- **Purity:** `knee` cut noise@10 by 0.13–0.25 and lifted clean@10 to
  0.62–0.88 on all ten repos, for a 0–6 pt R@10 cost.
- **Index cost:** fidx indexed each repo 2–3× faster than QMD (LLM-free ONNX
  embedding vs LLM embedding on CPU).
- **Reliability:** fidx built all 10 indexes. QMD did not finish 3 of 10
  (symfony, clickhouse, elasticsearch): its LLM embed path stalled past the
  harness watchdog or its LLM session expired mid-embed. On two repos the
  harness declined to *measure fidx latency* because the machine was loaded
  by concurrent runs (a measurement-environment guard, not an engine failure;
  recall was recovered from the completed indexes).

## fidx vs SymDex (code corpora)

SymDex is the closest architectural neighbour — local-first, SQLite,
LLM-free indexing — but it indexes tree-sitter *symbols*, not full-text
chunks. Same ten per-repo corpora and queries:

| Metric (mean over 10 languages) | fidx | SymDex (semantic) |
|---|--:|--:|
| recall@10 | **0.90** | 0.32 |
| noise@10 / clean@10 | 0.27 / 0.28 raw, **0.10 / 0.77** with knee | 0.47 / 0.14 |
| query p50 (warm) | **50–100 ms** | 74–512 ms |
| index time / repo | 19–71 min | **3–12 min** |

Read this as **task fit, not defect**: SymDex's recall is capped by symbol
reachability (files without named symbols — tests, markdown, data — are
unreachable; per-language ceiling 0.59–1.00), and a larger 768-d embedder
(`all-mpnet-base-v2`) did **not** close the gap (mean R@10 0.28, worse on
7/10 languages) — the limit is architectural, not model size. SymDex targets
symbol/call-graph lookup ("find function X"), which these NL known-item
queries do not measure; it wins index speed 4–6× and matched fidx on
reliability (10/10 completed).

## docs-small deep dive (per-query-type behavior)

fidx e5/sqlite-vec vs QMD, docs-small, recall@1 by query type:

| type | fidx hybrid | qmd search | qmd query |
|---|---|---|---|
| unique (names/identifiers) | 0.90 | 0.92 | 0.82 |
| phrase (contiguous span) | 0.58 | 0.58 | 0.42 |
| vague (shuffled content words) | 0.72 | 0.92 | 0.74 |

- Phrase R@1 ≈ 0.58 *for everyone*: 20 Newsgroups contains near-duplicate
  texts (quoted reply chains, reposts), so R@1 against one specific file has
  a ceiling; R@3/R@10 absorb it (fidx phrase R@10 = 1.0).
- QMD `search` winning vague R@1 is a query-generation bias: generated
  "vague" queries only contain words that literally appear in the document,
  which favors conjunctive BM25. Human-vague queries (paraphrase, partial
  memory) would not all-terms-match.
- fidx cold CLI (no daemon): ~0.43–0.51 s/query. duckdb-HNSW backend at this
  scale is slower than sqlite-vec brute force and slightly lossy — its value
  starts around the ~30k-chunk crossover (see below).

## Scaling model (two measured scale points — planning numbers)

Fitted from docs-small (3,547 chunks) and docs (31,658 chunks), e5-768:

- **Chunks:** prose yields `N ≈ corpus_bytes / 1,130` (~450-token chunks,
  15% overlap). Code differs; re-fit there.
- **Disk (sqlite-vec):** `≈ 2 MB + 5.4 KB × N` (~4.7× raw corpus).
- **RAM (warm daemon):** sqlite-vec ~260 MB flat (model dominates;
  brute-force scan uses SQLite's bounded page cache). duckdb-HNSW
  `≈ 300 MB + 9.2 KB × N` (graph stays resident; ~9.5 GB at 1M chunks).
- **Latency (hybrid p50):** sqlite-vec `≈ 17 ms + 1.15 µs × N`
  (~130 ms @ 100k, ~1.2 s @ 1M). duckdb-HNSW effectively flat
  (~52 ms @ 100k, ~57 ms @ 1M). Crossover ≈ 30k chunks.

Caveats: two scale points cannot distinguish O(N) from O(N log N); constants
are specific to this machine, thread count, and profile. Treat as planning
numbers, not guarantees.

## Threats to validity

Read before quoting any number:

- **Known-item bias.** Queries have exactly one ground-truth file, and ~78%
  of answers sit at rank 0 of the untruncated list. `knee` keeps rank-0
  answers perfectly but only 0–67% of answers at rank ≥3, and multi-relevant
  retention is 0.885 (not the 0.962 single-answer figure) — so the knee
  *recall* wins are benchmark-easiness dependent and do not transfer to
  harder query mixes. The *purity* wins are robust (independent of answer
  depth). This is also why the shipped truncation default is `off` and
  `fidx calibrate` derives per-corpus policy instead of hardcoding a
  benchmark-tuned constant.
- **Corpus noise floor.** 20 Newsgroups near-duplicates depress R@1 for every
  engine equally; do not read absolute R@1 there as engine quality.
- **Query-generation bias — now measured, not just disclosed.** The three
  known-item query types are built from the target document's own tokens, so
  they reward lexical overlap (an independent reviewer flagged this after
  v0.1.0). The paraphrase section above quantifies it: on non-lexical
  queries, recall drops for every engine and the ranking changes. Read the
  known-item tables as the lexical-leaning regime and the paraphrase table
  as the semantic regime.
- **Single machine, single run.** All constants (latency, index time, RSS)
  are from one 32-core CPU box; per-repo index/latency numbers were measured
  under varying background load, with a load-gate refusing the worst cases.
- **Judge is an LLM.** Purity labels come from one LLM judge over pooled
  results (identical judgments across engines, so comparisons are fair;
  absolute noise rates carry judge bias). 100 deliberately unanswerable
  probe queries were used to sanity-check the metric.
- **QMD parse failures.** 1/500 code-corpus queries failed to parse per QMD
  mode (~0.2 pt); QMD's 3 per-repo DNFs are reported as DNF, not zero.

## Reproducing

```sh
# fastest path: download the prepared corpora + exact query sets from the
# v0.1.0 release assets (fidx-bench-*.tar.gz) and unpack into bench/data/
# — see fidx-bench-DATASETS.md there. Or rebuild from scratch:
uv run python bench/corpora.py
uv run python bench/gen_queries.py bench/data/docs -n 500

# fidx and qmd on one corpus
uv run python bench/run_bench.py run --engine fidx --corpus docs
bun install -g @tobilu/qmd@2.5.3
uv run python bench/run_bench.py run --engine qmd --corpus docs
uv run python bench/run_bench.py report
```

Cost warnings (CPU-only):

- QMD LLM modes run **hours** per 500-query corpus (~33 s/query); the full
  per-repo 10-language suite was ~34 h wall / ~294 h CPU across both engines.
- The nomic profile on the full docs corpus peaks ~21 GB RSS while indexing;
  don't run two such index jobs concurrently on a 64 GB box.
- Purity judging requires an LLM judge (see `bench/judge_purity.py`); recall
  and latency need none.
