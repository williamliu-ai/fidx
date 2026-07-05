# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Structure-aware chunking for markdown/text.

Documents are split into overlapping chunks at the best structural break point
near the target size (headings > code-fence boundaries > horizontal rules >
blank lines > newlines), never inside a fenced code block. Chunks are stored
as (pos, length) offsets into the document body — text is never duplicated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sizes in characters (~4 chars/token): ~450-token chunks, 15% overlap.
TARGET_CHARS = 1800
OVERLAP_CHARS = 270
SEARCH_WINDOW = 600

_BREAK_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"^# .+$", re.M), 100),
    (re.compile(r"^## .+$", re.M), 90),
    (re.compile(r"^### .+$", re.M), 80),
    (re.compile(r"^#{4,6} .+$", re.M), 65),
    (re.compile(r"^(---|\*\*\*)\s*$", re.M), 60),
    (re.compile(r"\n\s*\n"), 20),
    (re.compile(r"\n"), 1),
]

_FENCE_RE = re.compile(r"^(```|~~~)", re.M)


@dataclass(frozen=True)
class Chunk:
    seq: int
    pos: int
    length: int


def _fence_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by fenced code blocks (inclusive of fences)."""
    ranges = []
    starts = [m.start() for m in _FENCE_RE.finditer(text)]
    for i in range(0, len(starts) - 1, 2):
        ranges.append((starts[i], starts[i + 1]))
    return ranges


def _in_fence(pos: int, fences: list[tuple[int, int]]) -> bool:
    return any(a < pos <= b for a, b in fences)


def _break_points(text: str, fences: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """All candidate break positions as (pos, score), best score per position."""
    best: dict[int, int] = {}
    for pattern, score in _BREAK_PATTERNS:
        for m in pattern.finditer(text):
            pos = m.start()
            if _in_fence(pos, fences):
                continue
            if best.get(pos, 0) < score:
                best[pos] = score
    return sorted(best.items())


def chunk_text(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS,
               window: int = SEARCH_WINDOW) -> list[Chunk]:
    if not text.strip():
        return []
    if len(text) <= target:
        return [Chunk(0, 0, len(text))]

    fences = _fence_ranges(text)
    breaks = _break_points(text, fences)
    chunks: list[Chunk] = []
    start = 0
    seq = 0
    while start < len(text):
        end = start + target
        if end >= len(text):
            end = len(text)
        else:
            # Pick the highest-scoring break in [end - window, end], with a
            # squared distance decay so nearby weak breaks can't always beat
            # distant strong ones (a heading at -500 chars beats a newline at 0).
            lo = max(start + overlap + 1, end - window)
            candidates = [(p, s) for p, s in breaks if lo <= p <= end]
            if candidates:
                def effective(item: tuple[int, int]) -> float:
                    p, s = item
                    dist = (end - p) / window
                    return s * (1 - dist * dist * 0.7)
                end = max(candidates, key=effective)[0]
                if end <= start:
                    end = start + target
            else:
                # No break in the window (e.g. inside a code fence): extend to
                # the next break after the target rather than cutting mid-fence.
                end = next((p for p, _ in breaks if p > end), len(text))
        chunks.append(Chunk(seq, start, end - start))
        seq += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        if _in_fence(start, fences):
            start = end  # don't let the overlap window begin mid-code-fence
    return chunks


_TITLE_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)


def extract_title(text: str, fallback: str) -> str:
    m = _TITLE_RE.search(text[:4000])
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return fallback
