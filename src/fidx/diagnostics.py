# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Host capability diagnostics for `fidx doctor`.

Self-contained and import-safe: every probe guards its imports so the report runs
and explains failures even when native deps (sqlite-vec / onnxruntime) are
missing or built for the wrong architecture.
"""
from __future__ import annotations

import platform
import sqlite3
from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    remediation: str = ""
    hard: bool = False  # a hard failure means fidx cannot function on this host


def triple() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def capabilities() -> list[Check]:
    checks: list[Check] = []

    checks.append(Check(
        "python", True,
        f"{platform.python_version()} ({platform.python_implementation()}) on {triple()}",
    ))

    try:
        from . import __version__
        checks.append(Check("fidx", True, __version__))
    except Exception as e:  # pragma: no cover
        checks.append(Check("fidx", False, f"version unavailable: {e}"))

    # Which sqlite module is in use (stdlib, or pysqlite3 fallback).
    try:
        from . import db
        sqlite = db.sqlite_module()
    except Exception:  # pragma: no cover - db import is pure now
        sqlite = sqlite3
    modname = getattr(sqlite, "__name__", "sqlite3")
    checks.append(Check("sqlite", True, f"sqlite {sqlite.sqlite_version} via {modname}"))

    can_load = hasattr(sqlite.Connection, "enable_load_extension")
    checks.append(Check(
        "load_extension", can_load,
        "enable_load_extension available" if can_load
        else "this Python's sqlite3 was built without loadable-extension support",
        remediation="" if can_load else
        "Install fidx with `uv tool install` (bundles a capable Python) or use "
        "Homebrew Python; macOS system Python is compiled without extension loading.",
        hard=True,
    ))

    # FTS5 (hard).
    try:
        c = sqlite.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE _t USING fts5(x)")
        c.close()
        checks.append(Check("fts5", True, "FTS5 available", hard=True))
    except Exception as e:
        checks.append(Check(
            "fts5", False, f"FTS5 not available: {e}",
            remediation="Use a Python whose sqlite3 includes FTS5 (uv-managed or Homebrew).",
            hard=True,
        ))

    # sqlite-vec vec0 load (hard).
    if can_load:
        try:
            import sqlite_vec
            c = sqlite.connect(":memory:")
            c.enable_load_extension(True)
            sqlite_vec.load(c)
            (ver,) = c.execute("SELECT vec_version()").fetchone()
            c.close()
            checks.append(Check("sqlite_vec", True, f"vec0 loads (vec_version {ver})", hard=True))
        except Exception as e:
            checks.append(Check(
                "sqlite_vec", False, f"sqlite-vec failed to load: {e}",
                remediation="Ensure a sqlite-vec wheel exists for this platform/arch: "
                "`pip install --only-binary=:all: sqlite-vec`.",
                hard=True,
            ))
    else:
        checks.append(Check(
            "sqlite_vec", False, "skipped (extension loading unavailable)",
            remediation="Resolve load_extension first.", hard=True,
        ))

    # Cache presence is a broad hint; the profile-specific probe below checks
    # fastembed's own local-only resolver without loading an ONNX session.
    from .config import get_profile
    from .embedder import FastEmbedder

    cache = FastEmbedder.cache_path()
    present = cache.is_dir() and any(cache.iterdir())
    checks.append(Check(
        "model_cache", True,
        f"{cache} ({'files present' if present else 'empty — fetch or index downloads a model'})",
    ))
    from . import rerank_models
    try:
        rerank_models.probe_rerank_capability()
        rerank_capable = True
        rerank_detail = "FastEmbed cross-encoder available"
    except Exception as exc:
        rerank_capable = False
        rerank_detail = f"FastEmbed cross-encoder unavailable: {exc}"
    checks.append(Check(
        "rerank_capability", rerank_capable, rerank_detail,
        remediation="" if rerank_capable else "Install `fmdidx[rerank]`.",
    ))
    selected = rerank_models.DEFAULT_RERANK_MODEL
    available = (rerank_models.reranker_available_locally(selected)
                 if rerank_capable else False)
    checks.append(Check(
        "rerank_model", available,
        f"{selected} {'cached' if available else 'unavailable locally'}",
        remediation="" if available else
        f"Run `fidx models fetch rerank --model {selected}` while online.",
    ))
    try:
        profile = get_profile()
    except KeyError as exc:
        checks.append(Check(
            "embedding_model", False, str(exc),
            remediation="Choose a known embedding profile with FIDX_PROFILE.",
        ))
        return checks
    available = FastEmbedder(profile, local_files_only=True).available_locally()
    checks.append(Check(
        "embedding_model", available,
        f"{profile.name} ({profile.model}) {'cached' if available else 'unavailable locally'}",
        remediation="" if available else
        f"Run `fidx models fetch embed --profile {profile.name}` while online.",
    ))

    return checks


def hard_failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.hard and not c.ok]
