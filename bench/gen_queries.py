#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Generate retrieval queries from randomly selected corpus files.

For each sampled file, one query of a rotating type (mirroring the three
target use cases):

  unique — the file's rarest tokens (names, identifiers): exact-name lookup
  phrase — a contiguous span of words from the file: near-exact recall
  vague  — shuffled mid-frequency content words, rare tokens excluded:
           conceptual "I remember a doc about…" search

Ground truth is the source file. The same queries are replayed against every
engine, so generation bias affects all engines equally.

Usage: python bench/gen_queries.py <corpus-dir> [-n 500] [-o queries.jsonl] [--seed 7]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
STOPWORDS = set(
    """the and for that with this from have are was were you your not but all can our out
    they them then than has had who whom what when where which while will would there
    their about into over under just like some more most other been being does did doing
    she her his him its it's don't can't won't isn't how why these those each very much
    many may might must shall could should a an is in on at to of as by or if we he it
    no yes one two three also any new now get got make made use used using""".split()
)
MAX_DOC_BYTES = 200_000

# QMD's indexer hard-excludes these directories and all dot-paths regardless
# of --mask (store.ts indexFiles). Queries must never expect a file QMD
# structurally cannot return, so both engines are scored on the indexable
# intersection. fidx still indexes the full corpus.
QMD_UNINDEXABLE = {"vendor", "node_modules", "dist", "build"}

# WhatsApp-style chat exports prefix every line with "dd/mm/YYYY, HH:MM -
# Speaker: ". A phrase query that includes this synthetic prefix hands exact
# lexical matchers a near-unique key no human would type — strip it before
# selecting phrase spans.
CHAT_PREFIX_RE = re.compile(r"^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} - [^:]*: ")


def qmd_indexable(rel_path: Path) -> bool:
    return not any(part in QMD_UNINDEXABLE or part.startswith(".")
                   for part in rel_path.parts)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 3]


def build_df(files: list[Path]) -> collections.Counter:
    df: collections.Counter = collections.Counter()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:MAX_DOC_BYTES]
        except OSError:
            continue
        df.update(set(content_words(text)))
    return df


def q_unique(words: list[str], df: collections.Counter, rng: random.Random) -> str | None:
    ranked = sorted(set(words), key=lambda w: (df[w], rng.random()))
    rare = [w for w in ranked if df[w] <= 3][:2]
    if not rare:
        return None
    anchor = [w for w in ranked if 3 < df[w] <= 50][:1]
    return " ".join(rare + anchor)


def q_phrase(text: str, rng: random.Random) -> str | None:
    lines = [CHAT_PREFIX_RE.sub("", ln.strip()) for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln.split()) >= 8]
    if not lines:
        return None
    words = rng.choice(lines).split()
    start = rng.randrange(0, max(1, len(words) - 8))
    return " ".join(words[start : start + rng.randint(6, 9)])


def q_vague(words: list[str], df: collections.Counter, rng: random.Random) -> str | None:
    mid = [w for w in dict.fromkeys(words) if 5 <= df[w] <= 500]
    if len(mid) < 4:
        return None
    picked = rng.sample(mid, min(6, len(mid)))
    rng.shuffle(picked)
    return " ".join(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("-n", "--num-queries", type=int, default=500)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    files = sorted(p for p in args.corpus.rglob("*") if p.is_file())
    if not files:
        sys.exit(f"no files under {args.corpus}")
    rng = random.Random(args.seed)
    print(f"computing document frequencies over {len(files)} files ...", file=sys.stderr)
    df = build_df(files)

    out_path = args.output or args.corpus.parent / f"queries-{args.corpus.name}.jsonl"
    kinds = ["unique", "phrase", "vague"]
    queries = []
    candidates = [p for p in files if qmd_indexable(p.relative_to(args.corpus))]
    rng.shuffle(candidates)
    for path in candidates:
        if len(queries) >= args.num_queries:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:MAX_DOC_BYTES]
        except OSError:
            continue
        words = content_words(text)
        if len(words) < 10:
            continue
        kind = kinds[len(queries) % 3]
        query = {
            "unique": lambda: q_unique(words, df, rng),
            "phrase": lambda: q_phrase(text, rng),
            "vague": lambda: q_vague(words, df, rng),
        }[kind]()
        if not query:
            continue
        queries.append(
            {
                "qid": f"{args.corpus.name}-{len(queries):04d}",
                "type": kind,
                "query": query,
                "expected": str(path.relative_to(args.corpus)),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    by_type = collections.Counter(q["type"] for q in queries)
    print(f"wrote {len(queries)} queries ({dict(by_type)}) -> {out_path}")


if __name__ == "__main__":
    main()
