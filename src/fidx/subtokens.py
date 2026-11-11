# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Pure identifier splitting and safe query expansion for derived lexical text."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _parts(token: str) -> list[str]:
    parts = []
    for word in token.split("_"):
        start = 0
        for i in range(1, len(word)):
            left, right = word[i - 1], word[i]
            following = word[i + 1] if i + 1 < len(word) else ""
            boundary = (
                (left.islower() and right.isupper())
                or (left.isupper() and right.isupper() and following.islower())
                or (left.isalpha() and right.isdigit())
                or (left.isdigit() and right.isalpha())
            )
            if boundary:
                parts.append(word[start:i].lower())
                start = i
        if word:
            parts.append(word[start:].lower())
    return parts


def subtoken_text(body: str) -> str:
    """Keep whole tokens, appending lowercase parts only for split identifiers."""
    stream = []
    for token in TOKEN_RE.findall(body):
        stream.append(token)
        parts = _parts(token)
        if len(parts) > 1:
            stream.extend(parts)
    return " ".join(stream)


def expand_query(query: str) -> str:
    """Replace split tokens with parts; retain user phrase delimiters."""
    def replace(match):
        token = match.group()
        parts = _parts(token)
        return " ".join(parts) if len(parts) > 1 else token
    return TOKEN_RE.sub(replace, query)


def subtoken_match_expr(query: str, conjunctive: bool = False) -> str | None:
    """Use exactly the main arm's safe quoting after expanding identifiers."""
    from .search import fts_match_expr
    return fts_match_expr(expand_query(query), conjunctive)
