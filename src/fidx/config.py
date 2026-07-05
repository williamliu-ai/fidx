# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Configuration: index location and embedding model profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GLOBS = ["**/*.md", "**/*.markdown", "**/*.txt"]


def default_db_path() -> Path:
    """Resolve the index database path.

    Precedence: FIDX_DB env var > XDG cache dir default.
    CLI --db flag overrides both (handled in cli.py).
    """
    env = os.environ.get("FIDX_DB")
    if env:
        return Path(env).expanduser()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "fidx" / "index.db"


@dataclass(frozen=True)
class EmbedProfile:
    """An embedding model profile. The fingerprint of (name, dim, prefixes)
    is stored in the index; changing profiles requires re-embedding.

    hf_source/model_file register a custom ONNX model with fastembed for
    models outside its built-in catalogue (mean pooling, normalized)."""

    name: str
    model: str
    dim: int
    query_prefix: str = ""
    doc_prefix: str = ""
    hf_source: str | None = None
    model_file: str | None = None


# CPU-friendly ONNX models served by fastembed. 768-dim is the default:
# 384-dim models degrade past ~100k chunks (see docs/DESIGN.md).
PROFILES: dict[str, EmbedProfile] = {
    # int8-quantized ONNX: ~4x faster than f32 on CPU (measured 15.6 vs 4.3
    # chunks/s on 32 cores) at near-identical retrieval quality.
    "nomic-768-q": EmbedProfile(
        name="nomic-768-q",
        model="nomic-ai/nomic-embed-text-v1.5-Q",
        dim=768,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
    ),
    "nomic-768": EmbedProfile(
        name="nomic-768",
        model="nomic-ai/nomic-embed-text-v1.5",
        dim=768,
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
    ),
    # e5-base-v2 (English): a family validated in earlier CPU embedding-model
    # benchmarking, via the Xenova ONNX conversion (int8-quantized).
    "e5-768": EmbedProfile(
        name="e5-768",
        model="intfloat/e5-base-v2",
        dim=768,
        query_prefix="query: ",
        doc_prefix="passage: ",
        hf_source="Xenova/e5-base-v2",
        model_file="onnx/model_quantized.onnx",
    ),
    "bge-768": EmbedProfile(
        name="bge-768",
        model="BAAI/bge-base-en-v1.5",
        dim=768,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # Small/fast profile for smoke tests and small corpora.
    "bge-384": EmbedProfile(
        name="bge-384",
        model="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}

DEFAULT_PROFILE = os.environ.get("FIDX_PROFILE", "nomic-768-q")


def get_profile(name: str | None = None) -> EmbedProfile:
    key = name or DEFAULT_PROFILE
    if key not in PROFILES:
        raise KeyError(f"unknown embedding profile {key!r}; known: {', '.join(PROFILES)}")
    return PROFILES[key]


def profile_fingerprint(p: EmbedProfile) -> str:
    import hashlib

    raw = f"{p.model}|{p.dim}|{p.query_prefix}|{p.doc_prefix}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]
