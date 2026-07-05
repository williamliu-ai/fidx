#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Prepare the three benchmark corpora under bench/data/.

  docs       — 20 Newsgroups text documents
  docs-small — 3-group smoke subset derived from docs
  chat       — Cornell Movie-Dialogs rendered as WhatsApp-style chat exports
               (one file per conversation, real speaker names, synthetic timestamps)
  code       — one repo per top-10 programming language, each pinned to a tag
               and each >=10k files in-repo (the requirement's size rule); only
               that language's sources (+ .md) are copied into the corpus

The corpora are text-only: is_text_document() rejects binary payloads
(uuencoded/BinHex posts, NUL-containing files, minified bundles) at prepare
time — engines must never be benchmarked on binary content.

Each corpus is a plain directory of files, so any engine that indexes a
directory can be benchmarked on it. Re-running is idempotent.

Usage: python bench/corpora.py [docs|docs-small|chat|code ...] [--chat-conversations N]
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).parent / "data"

NEWS_URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
CORNELL_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
# One repo per top-10 language (GitHub Octoverse / TIOBE blend). Every repo
# has >=10k files at the pinned tag, satisfying the requirement's "repo with
# at least 10k files" rule, while each sub-corpus is kept near ~10k files
# after filtering. An optional 6th element lists top-level directories to
# exclude — vendored third-party code (node deps/, elasticsearch x-pack/) and
# test trees that dwarf the source they test (rust tests/ is ~22k tiny
# UI-test snippets; home-assistant tests/ is ~6k files).
CODE_REPOS = [
    # (dir, language, clone URL, tag, suffixes kept[, excluded top dirs])
    ("golang", "go", "https://github.com/golang/go", "go1.23.0", {".go", ".md"}),
    ("home-assistant", "python", "https://github.com/home-assistant/core", "2025.6.0",
     {".py", ".md"}, ("tests",)),
    ("node", "javascript", "https://github.com/nodejs/node", "v22.11.0",
     {".js", ".mjs", ".cjs", ".md"}, ("deps",)),
    ("grafana", "typescript", "https://github.com/grafana/grafana", "v11.2.2", {".ts", ".tsx", ".md"}),
    ("elasticsearch", "java", "https://github.com/elastic/elasticsearch", "v8.15.0",
     {".java", ".md"}, ("x-pack",)),
    ("zephyr", "c", "https://github.com/zephyrproject-rtos/zephyr", "v3.7.0", {".c", ".h", ".md"}),
    ("clickhouse", "cpp", "https://github.com/ClickHouse/ClickHouse", "v24.8.4.13-lts",
     {".cpp", ".cc", ".cxx", ".hpp", ".h", ".md"}),
    ("aspnetcore", "csharp", "https://github.com/dotnet/aspnetcore", "v9.0.0", {".cs", ".md"}),
    ("rust", "rust", "https://github.com/rust-lang/rust", "1.82.0", {".rs", ".md"}, ("tests",)),
    ("symfony", "php", "https://github.com/symfony/symfony", "v7.1.5", {".php", ".md"}),
]


# docs-small: alt.atheism + comp.graphics in full, plus this many files of
# comp.os.ms-windows.misc (sorted) — the historical smoke-corpus shape.
DOCS_SMALL_GROUPS = ("alt.atheism", "comp.graphics")
DOCS_SMALL_MISC_CAP = 228


def is_text_document(body: str) -> bool:
    """The corpora are text-only; binary payloads must never be benchmarked.

    Rejects true binaries (NUL bytes) and binary shipped as text — uuencoded
    or BinHex dumps in newsgroup posts, whose signature is a high share of
    long whitespace-free lines. Validated on 20 Newsgroups: catches all 25
    encoded-binary posts (the docs that OOMed embedding engines), keeps
    long legitimate text like FAQ lists."""
    if "\x00" in body:
        return False
    lines = [ln.strip() for ln in body.splitlines()]
    if not lines:
        return False
    dense = sum(1 for ln in lines if len(ln) > 40 and " " not in ln)
    return dense <= 0.25 * len(lines)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url} ...", file=sys.stderr)
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    return dest


def prepare_docs() -> None:
    out = DATA / "docs"
    if out.exists() and sum(1 for _ in out.rglob("*.txt")) > 10_000:
        print(f"docs corpus already prepared at {out}")
        return
    archive = _download(NEWS_URL, DATA / "raw" / "20news-bydate.tar.gz")
    out.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # members look like 20news-bydate-train/<group>/<msgid>
            parts = Path(member.name).parts
            if len(parts) != 3:
                continue
            _, group, msgid = parts
            body = tar.extractfile(member).read().decode("latin-1")
            if not is_text_document(body):
                skipped += 1
                continue
            target = out / group / f"{msgid}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            n += 1
    print(f"docs corpus: {n} documents ({skipped} binary-payload posts excluded) -> {out}")


def prepare_docs_small() -> None:
    """Derive the smoke corpus from the (already filtered) docs corpus."""
    src = DATA / "docs"
    out = DATA / "docs-small"
    if not src.exists():
        sys.exit("docs corpus missing — prepare docs first")
    if out.exists():
        shutil.rmtree(out)
    n = 0
    for group in DOCS_SMALL_GROUPS:
        files = sorted((src / group).glob("*.txt"))
        for f in files:
            (out / group).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out / group / f.name)
            n += 1
    misc = sorted((src / "comp.os.ms-windows.misc").glob("*.txt"))[:DOCS_SMALL_MISC_CAP]
    for f in misc:
        (out / "comp.os.ms-windows.misc").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out / "comp.os.ms-windows.misc" / f.name)
        n += 1
    print(f"docs-small corpus: {n} documents -> {out}")


def _cornell_tables(archive: Path) -> tuple[dict, dict, list]:
    sep = " +++$+++ "
    zf = zipfile.ZipFile(archive)
    base = "cornell movie-dialogs corpus/"

    def rows(name: str):
        with zf.open(base + name) as f:
            for line in io.TextIOWrapper(f, encoding="latin-1"):
                yield line.rstrip("\n").split(sep)

    titles = {r[0]: r[1] for r in rows("movie_titles_metadata.txt") if len(r) >= 2}
    lines = {r[0]: (r[3], r[4] if len(r) > 4 else "") for r in rows("movie_lines.txt") if len(r) >= 4}
    convs = [(r[2], r[3]) for r in rows("movie_conversations.txt") if len(r) >= 4]
    return titles, lines, convs


def prepare_chat(max_conversations: int) -> None:
    out = DATA / "chat"
    if out.exists() and sum(1 for _ in out.rglob("*.txt")) >= min(max_conversations, 1000):
        print(f"chat corpus already prepared at {out}")
        return
    archive = _download(CORNELL_URL, DATA / "raw" / "cornell_movie_dialogs_corpus.zip")
    titles, lines, convs = _cornell_tables(archive)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    n_msgs = 0
    written = 0
    for i, (movie_id, line_list) in enumerate(convs):
        if written >= max_conversations:
            break
        line_ids = [x.strip(" '") for x in line_list.strip("[]").split(",")]
        msgs = [lines[lid] for lid in line_ids if lid in lines]
        if len(msgs) < 2:
            continue
        ts = datetime(2020, 1, 1) + timedelta(
            days=rng.randint(0, 1500), hours=rng.randint(7, 23), minutes=rng.randint(0, 59)
        )
        title = titles.get(movie_id, movie_id)
        rendered = []
        for speaker, text in msgs:
            ts += timedelta(seconds=rng.randint(10, 300))
            stamp = ts.strftime("%d/%m/%Y, %H:%M")
            rendered.append(f"{stamp} - {speaker.title()}: {text.strip()}")
            n_msgs += 1
        fname = out / f"{movie_id}-conv{i:06d}.txt"
        header = f"WhatsApp chat export — group: {title.title()}\n"
        fname.write_text(header + "\n".join(rendered) + "\n", encoding="utf-8")
        written += 1
    print(f"chat corpus: {written} conversations, {n_msgs} messages -> {out}")


def prepare_code() -> None:
    out = DATA / "code"
    total, failed = 0, []
    for name, lang, url, tag, suffixes, *rest in CODE_REPOS:
        excluded_dirs = rest[0] if rest else ()
        src = DATA / "raw" / name
        dest = out / name
        if dest.exists() and any(p.is_file() for p in dest.rglob("*")):
            n = sum(1 for p in dest.rglob("*") if p.is_file())
            print(f"code/{name} ({lang}) already prepared: {n} files")
            total += n
            continue
        if not src.exists():
            print(f"cloning {url} @ {tag} (shallow) ...", file=sys.stderr)
            try:
                subprocess.run(["git", "clone", "--depth", "1", "--branch", tag,
                                "--single-branch", url, str(src)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"WARN: clone of {name} failed ({e}); skipping", file=sys.stderr)
                failed.append(name)
                continue
        n = 0
        for path in src.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in suffixes or path.stat().st_size > 512_000:
                continue
            if path.relative_to(src).parts[0] in excluded_dirs:
                continue
            # Suffix filtering is not enough: repos ship encoded blobs and
            # minified bundles under source suffixes (test fixtures etc.).
            if not is_text_document(path.read_text(encoding="utf-8", errors="replace")):
                continue
            rel = path.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(path, target)
            n += 1
        total += n
        print(f"code/{name} ({lang}): {n} files -> {dest}")
    # Per-repo corpora (code-<name>) are the same trees benchmarked alone —
    # materialize them as symlinks into the combined corpus so one command
    # yields both the combined index target and the per-repo targets that the
    # queries-code-<name>.jsonl sets expect.
    for name, *_ in CODE_REPOS:
        if name in failed:
            continue
        link = DATA / f"code-{name}"
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(Path("code") / name, target_is_directory=True)
            print(f"code-{name} -> code/{name} (per-repo corpus)")
        except OSError as e:  # e.g. Windows without symlink privilege
            print(f"WARN: could not link code-{name} ({e}); "
                  f"copy or junction bench/data/code/{name} manually", file=sys.stderr)
    print(f"code corpus: {total} files across {len(CODE_REPOS) - len(failed)} repos -> {out}")
    if failed:
        print(f"WARN: failed repos: {', '.join(failed)}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="*", default=["docs", "chat", "code"],
                        choices=["docs", "docs-small", "chat", "code"])
    parser.add_argument("--chat-conversations", type=int, default=8000,
                        help="conversations to render (~4 messages each)")
    args = parser.parse_args()
    for name in args.corpora or ["docs", "chat", "code"]:
        if name == "docs":
            prepare_docs()
        elif name == "docs-small":
            prepare_docs_small()
        elif name == "chat":
            prepare_chat(args.chat_conversations)
        elif name == "code":
            prepare_code()


if __name__ == "__main__":
    main()
