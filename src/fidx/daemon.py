# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Warm-search daemon over a Unix socket.

`fidx serve` keeps the SQLite connection and ONNX embedding model loaded so
repeated searches cost milliseconds instead of paying Python/model startup.
The CLI transparently uses the socket when it exists (see cli.py); the
protocol is one JSON request line in, one JSON response line out.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import socketserver
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import budget, config, db, freshness, rerank as rerankmod, rerank_models, search as searchmod, vector_store
from .db import get_meta

SEARCH_SCHEMA = "fidx.search.v2"
GUIDANCE_DEFAULT = False  # Internal seam; a public default flip requires M-G6.


def _validated_rerank_effective(state: object) -> dict:
    """Validate one exact, versioned rerank state from an untrusted socket."""
    if type(state) is not dict or type(state.get("enabled")) is not bool:
        raise ValueError("invalid rerank effective state")
    if state["enabled"]:
        if set(state) != {"enabled", "model"} or type(state["model"]) is not str:
            raise ValueError("invalid rerank effective state")
        try:
            canonical = rerank_models.canonical_model(state["model"])
        except ValueError as exc:
            raise ValueError("invalid rerank effective state") from exc
        if state["model"] != canonical:
            raise ValueError("rerank model must use canonical spelling")
    elif (set(state) != {"enabled", "advice_flag"}
          or state["advice_flag"] != "--no-rerank"):
        raise ValueError("invalid rerank effective state")
    return dict(state)


def _socket_rerank_request(req: dict) -> tuple[dict, dict | None]:
    """Private run_search markers are trusted only when derived at ingress."""
    if "_rerank_model" in req or "_rerank_advice_off" in req:
        raise ValueError("private rerank request fields are not accepted")
    if "rerank_protocol" not in req and "rerank_effective" not in req:
        return req, None
    if type(req.get("rerank_protocol")) is not int or req["rerank_protocol"] != 1:
        raise ValueError("unsupported rerank protocol")
    state = _validated_rerank_effective(req.get("rerank_effective"))
    internal = dict(req)
    internal.pop("rerank_protocol")
    internal.pop("rerank_effective")
    if state["enabled"]:
        internal["_rerank_model"] = state["model"]
    else:
        internal["_rerank_advice_off"] = True
    return internal, state


def socket_path_for(db_path: Path) -> Path:
    resolved = str(Path(db_path).expanduser().resolve(strict=False))
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        base = Path(runtime).expanduser() / "fidx"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        base = cache / "fidx" / "run"
    return base / f"{digest}.sock"


def _prepare_socket_path(db_path: Path) -> Path:
    sock_path = socket_path_for(db_path)
    sock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(sock_path.parent, 0o700)
    return sock_path


def _bind_private_unix_server(sock_path: Path, handler) -> socketserver.UnixStreamServer:
    if not hasattr(socketserver, "UnixStreamServer"):
        raise RuntimeError("Unix-socket daemon is unavailable on this platform")

    class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True
        request_queue_size = 128

    old_umask = os.umask(0o177)
    try:
        server = _ThreadedUnixServer(str(sock_path), handler)
    finally:
        os.umask(old_umask)
    os.chmod(sock_path, 0o600)
    return server


def _result_sources(result: searchmod.Result, mode: str) -> dict[str, float]:
    if result.sources:
        return {k: round(v, 5) for k, v in result.sources.items()}
    if mode in ("lexical", "vector"):
        return {mode: round(result.score, 5)}
    return {}


def serialize_results(results: list[searchmod.Result], mode: str) -> list[dict]:
    serialized = []
    for rank, r in enumerate(results, 1):
        row = {
            "rank": rank,
            "path": f"{r.collection}/{r.relpath}",
            "collection": r.collection,
            "relpath": r.relpath,
            "title": r.title,
            "docid": f"#{r.docid}",
            "score": round(r.score, 5),
            "snippet": r.snippet,
            "sources": _result_sources(r, mode),
        }
        if r.line_start is not None and r.line_end is not None:
            row["locations"] = [{"line_start": r.line_start, "line_end": r.line_end}]
        serialized.append(row)
    return serialized


def _unknown_collections(conn, collections: list[str]) -> list[str]:
    if not collections:
        return []
    known = set()
    for collection in collections:
        row = conn.execute(
            "SELECT name FROM collections WHERE name = ?",
            (collection,),
        ).fetchone()
        if row:
            known.add(row["name"])
    return sorted(c for c in collections if c not in known)


def _search_command(query: str, mode: str, collections: list[str], limit: int,
                    min_score, truncate, *, rerank_option: str | None = None,
                    fusion: str = "rrf") -> list[str]:
    cmd = ["fidx", "search", query, "--json"]
    if mode != "hybrid":
        cmd += ["--mode", mode]
    for collection in collections:
        cmd += ["-c", collection]
    cmd += ["-n", str(limit)]
    if min_score is not None:
        cmd += ["--min-score", str(min_score)]
    if truncate and truncate != "off":
        cmd += ["--truncate", str(truncate)]
    if fusion != "rrf":
        cmd += ["--fusion", fusion]
    if rerank_option is not None:
        cmd.append(rerank_option)
    return cmd


def _source_mix(results: list[searchmod.Result], mode: str) -> dict[str, int]:
    mix = {"both": 0, "lexical_only": 0, "vector_only": 0, "other": 0}
    for result in results:
        sources = set(result.sources)
        if not sources and mode in ("lexical", "vector"):
            sources = {mode}
        if {"lexical", "vector"} <= sources:
            mix["both"] += 1
        elif "lexical" in sources:
            mix["lexical_only"] += 1
        elif "vector" in sources:
            mix["vector_only"] += 1
        else:
            mix["other"] += 1
    return mix


def _confidence(results: list[searchmod.Result], mode: str) -> str:
    if not results:
        return "none"
    mix = _source_mix(results, mode)
    if mix["both"]:
        return "strong"
    if len(results) <= 2:
        return "narrow"
    return "mixed"


def _calibration(conn) -> dict:
    raw = get_meta(conn, "truncate_floor")
    if raw is None:
        return {"floor_available": False, "floor": None}
    try:
        return {"floor_available": True, "floor": round(float(raw), 5)}
    except ValueError:
        return {"floor_available": False, "floor": None}


def _score_profile(results: list[searchmod.Result]) -> dict:
    if not results:
        return {
            "count": 0,
            "top_score": None,
            "tail_score": None,
            "spread": None,
            "flat": True,
            "has_knee": False,
        }
    scores = [r.score for r in results]
    top = scores[0]
    tail = scores[-1]
    spread = top - tail
    flat = spread <= 1e-12
    has_knee = False
    if len(results) >= 4 and not flat:
        from . import truncate as truncatemod
        has_knee = len(truncatemod.truncate(results, "knee", "hybrid")) < len(results)
    return {
        "count": len(results),
        "top_score": round(top, 5),
        "tail_score": round(tail, 5),
        "spread": round(spread, 5),
        "flat": flat,
        "has_knee": has_knee,
    }


def _truncation_option(intent: str, truncate, lean: str, applicable: bool,
                       recommended: bool, reason: str, command: list[str] | None) -> dict:
    option = {
        "intent": intent,
        "truncate": truncate,
        "lean": lean,
        "applicable": applicable,
        "recommended": recommended,
        "reason": reason,
    }
    if applicable and command is not None:
        option["command"] = command
    return option


def _truncation_advice(query: str, mode: str, collections: list[str], limit: int,
                       min_score, truncate, pre_truncate: list[searchmod.Result],
                       final: list[searchmod.Result], diagnostics: dict,
                       *, rerank_option: str | None = None, fusion: str = "rrf") -> dict:
    search_command = lambda *args: _search_command(
        *args, rerank_option=rerank_option, fusion=fusion)
    effective_truncate = truncate or "off"
    filters = diagnostics["filters"]
    calibration = diagnostics["calibration"]
    profile = _score_profile(pre_truncate)
    floor_available = calibration["floor_available"]
    floor_applicable = floor_available and calibration.get("floor_applicable", True)
    unavailable_floor_reason = (
        "The stored fused-score calibration floor is inapplicable after reranking"
        if floor_available else "No calibration floor is stored"
    )

    recommendation = None
    primary_action = None
    lean = None
    reason = "Truncation is not useful until the search has candidate results."

    if final:
        if effective_truncate != "off":
            if effective_truncate == "calibrated" and not floor_applicable:
                if profile["has_knee"]:
                    recommendation = "knee"
                    primary_action = "clean_shortlist"
                    lean = "balanced"
                    reason = (
                        f"{unavailable_floor_reason}, so calibrated behaves like knee; "
                        "use knee as the explicit balanced truncation mode."
                    )
                else:
                    recommendation = "off"
                    primary_action = "disable_truncation"
                    lean = "recall"
                    reason = (
                        f"{unavailable_floor_reason} and the score profile has no "
                        "applicable knee; use off to preserve recall."
                    )
            elif effective_truncate == "knee" and not profile["has_knee"]:
                recommendation = "off"
                primary_action = "disable_truncation"
                lean = "recall"
                reason = (
                    "Knee truncation returned candidates but the score profile has no "
                    "applicable knee; use off for the same recall without implying a cut."
                )
            else:
                recommendation = effective_truncate if effective_truncate in ("knee", "calibrated") else "off"
                primary_action = "keep_current"
                lean = "purity" if effective_truncate == "calibrated" else "balanced"
                reason = "The current truncation setting returned candidates; inspect before loosening it."
        elif profile["count"] < 4:
            recommendation = "off"
            primary_action = "keep_current"
            lean = "recall"
            reason = (
                f"Only {profile['count']} candidate(s) returned; knee needs at least 4 "
                "scores, so keep truncation off to preserve recall."
            )
        elif profile["has_knee"]:
            recommendation = "knee"
            primary_action = "clean_shortlist"
            lean = "balanced"
            reason = (
                "The result list has enough scores for a knee cut; use it to trim "
                "the weak tail while keeping the confident head."
            )
        else:
            recommendation = "off"
            primary_action = "keep_current"
            lean = "recall"
            reason = (
                "The score curve does not expose a useful knee; keep truncation off "
                "and inspect or request more candidates."
            )
    elif filters["after_min_score"] == 0 and filters["raw_count"] > 0:
        reason = "The minimum-score filter removed all candidates; adjust that before truncation."
    elif filters["dropped_by_truncate"] > 0:
        if effective_truncate == "calibrated" and floor_applicable and profile["has_knee"]:
            recommendation = "knee"
            primary_action = "clean_shortlist"
            lean = "balanced"
            reason = (
                "Calibrated truncation removed every candidate; retry with knee to "
                "loosen the corpus floor while still cutting the score tail."
            )
        else:
            recommendation = "off"
            primary_action = "disable_truncation"
            lean = "recall"
            reason = "Truncation removed every candidate; disable it to recover recall."

    off_recommended = recommendation == "off"
    knee_applicable = profile["count"] >= 4 and profile["has_knee"]
    knee_recommended = recommendation == "knee"
    calibrated_recommended = recommendation == "calibrated" and floor_applicable
    if profile["count"] < 4:
        knee_reason = "Knee needs at least 4 result scores; it would be a no-op here."
    elif profile["flat"]:
        knee_reason = "The score curve is flat, so a knee cut would not separate a tail."
    elif knee_applicable:
        knee_reason = "Balanced option: cut the score-curve tail without corpus calibration."
    else:
        knee_reason = "No useful knee was detected in the score profile."
    if fusion != "rrf":
        calibrated_reason = "The RRF calibration floor is inapplicable to experimental weighted fusion."
    elif floor_applicable:
        calibrated_reason = (
            "Purity option: apply the stored corpus floor, then knee; this can abstain "
            "or drop borderline results."
        )
    elif floor_available:
        calibrated_reason = (
            "The stored fused-score calibration floor is inapplicable after reranking; "
            "calibrated would behave like knee, so it is not offered as a distinct action."
        )
    else:
        calibrated_reason = (
            "No stored truncate_floor is available; calibrated would behave like knee, "
            "so it is not offered as a distinct action."
        )
    return {
        "current": effective_truncate,
        "recommendation": recommendation,
        "primary_action": primary_action,
        "lean": lean,
        "reason": reason,
        "score_profile": profile,
        "options": [
            _truncation_option(
                "keep_current" if effective_truncate == "off" else "disable_truncation",
                "off",
                "recall",
                True,
                off_recommended,
                "Recall option: keep every ranked candidate and inspect the tail manually.",
                search_command(query, mode, collections, limit, min_score, "off"),
            ),
            _truncation_option(
                "clean_shortlist",
                "knee",
                "balanced",
                knee_applicable,
                knee_recommended,
                knee_reason,
                search_command(query, mode, collections, limit, min_score, "knee"),
            ),
            _truncation_option(
                "use_calibrated_abstention",
                "calibrated",
                "purity",
                floor_applicable,
                calibrated_recommended,
                calibrated_reason,
                search_command(query, mode, collections, limit, min_score, "calibrated"),
            ),
        ],
    }


def _next_actions(query: str, mode: str, collections: list[str], limit: int, min_score,
                  truncate, results: list[searchmod.Result], diagnostics: dict,
                  truncation_advice: dict, *, rerank_option: str | None = None,
                  fusion: str = "rrf") -> list[dict]:
    search_command = lambda *args: _search_command(
        *args, rerank_option=rerank_option, fusion=fusion)
    actions: list[dict] = []
    effective_truncate = truncate or "off"

    def add(intent: str, reason: str, command: list[str]) -> None:
        action = {"intent": intent, "reason": reason, "command": command}
        if action not in actions:
            actions.append(action)

    if results:
        best = results[0]
        add("inspect_best_match",
            "Open the highest-ranked candidate before deciding whether to refine the query.",
            ["fidx", "get", "--head", f"#{best.docid}"])
        options = {o["intent"]: o for o in truncation_advice["options"]}
        for intent in ("clean_shortlist", "use_calibrated_abstention"):
            option = options[intent]
            if option["truncate"] == effective_truncate:
                continue
            if option["applicable"] and (
                option["recommended"] or truncation_advice["recommendation"] == "knee"
            ):
                add(intent, option["reason"], option["command"])
        if effective_truncate != "off" and (
            diagnostics["filters"]["dropped_by_truncate"] > 0
            or truncation_advice["primary_action"] == "disable_truncation"
        ):
            if effective_truncate == "calibrated" and options["clean_shortlist"]["applicable"]:
                option = options["clean_shortlist"]
                add("clean_shortlist", option["reason"], option["command"])
            add("disable_truncation",
                "Truncation hid candidates; rerun without a tail cut if the target is missing.",
                search_command(query, mode, collections, limit, min_score, "off"))
        if mode != "lexical" and fusion == "rrf":
            add("try_exact_terms",
                "If the desired item has exact names, paths, errors, or identifiers, try lexical mode.",
                search_command(query, "lexical", collections, limit, None, "off"))
        if mode != "vector" and fusion == "rrf":
            add("try_conceptual_terms",
                "If the desired item uses different wording, try vector mode.",
                search_command(query, "vector", collections, limit, None, "off"))
        if len(results) <= 2 and diagnostics["filters"]["after_truncate"] < limit:
            add("broaden_query",
                "Only a narrow set returned; use fewer or broader terms if the target is missing.",
                search_command("<broader query>", mode, collections, limit, None, "off"))
        if diagnostics["filters"]["after_truncate"] >= limit:
            add("increase_limit",
                "The result list reached the requested limit; ask for more candidates.",
                search_command(query, mode, collections, limit * 2, min_score, effective_truncate))
        return actions

    if diagnostics["index_empty"]:
        actions += [
            {"intent": "add_collection",
             "reason": "The index has no active documents.",
             "command": ["fidx", "collection", "add", "<path>", "--name", "<collection>"]},
            {"intent": "index",
             "reason": "After adding a collection, build the index before searching.",
             "command": ["fidx", "index"]},
        ]
        return actions

    if diagnostics["unknown_collections"]:
        add("retry_without_scope",
            "One or more requested collections are not registered.",
            search_command(query, mode, [], limit, min_score, effective_truncate))
    elif collections:
        add("retry_without_scope",
            "The collection scope may be too narrow.",
            search_command(query, mode, [], limit, min_score, effective_truncate))

    filters = diagnostics["filters"]
    if min_score is not None and filters["raw_count"] > filters["after_min_score"]:
        add("remove_min_score",
            "The minimum score filter removed available candidates.",
            search_command(query, mode, collections, limit, None, effective_truncate))
    if effective_truncate != "off" and filters["after_min_score"] > filters["after_truncate"]:
        options = {o["intent"]: o for o in truncation_advice["options"]}
        action = truncation_advice["primary_action"]
        if action == "clean_shortlist" and options["clean_shortlist"]["applicable"]:
            option = options["clean_shortlist"]
            add("clean_shortlist", option["reason"], option["command"])
        else:
            add("disable_truncation",
                "Truncation removed all candidates; rerun without a tail cut.",
                search_command(query, mode, collections, limit, min_score, "off"))
    if mode != "hybrid":
        add("try_hybrid",
            "Hybrid mode combines lexical and vector evidence and is the recall-first default.",
            search_command(query, "hybrid", collections, limit, None, "off"))
    if mode != "lexical" and fusion == "rrf":
        add("try_exact_terms",
            "Use lexical mode for exact remembered words, names, paths, errors, or identifiers.",
            search_command(query, "lexical", collections, limit, None, "off"))
    if mode != "vector" and fusion == "rrf":
        add("try_conceptual_terms",
            "Use vector mode for synonym-heavy or conceptual wording.",
            search_command(query, "vector", collections, limit, None, "off"))
    add("broaden_query",
        "Use fewer or broader terms if the query may be too specific.",
        search_command("<broader query>", mode, collections, limit, None, "off"))
    return actions


def _agent_envelope(conn, req: dict, mode: str, collections: list[str], limit: int,
                    min_score, truncate, raw: list[searchmod.Result],
                    after_min: list[searchmod.Result],
                    final: list[searchmod.Result], *,
                    rerank_succeeded: bool = False) -> dict:
    unknown = _unknown_collections(conn, collections)
    filters = {
        "raw_count": len(raw),
        "after_min_score": len(after_min),
        "after_truncate": len(final),
        "dropped_by_min_score": len(raw) - len(after_min),
        "dropped_by_truncate": len(after_min) - len(final),
    }
    active_docs = conn.execute(
        "SELECT count(*) AS n FROM documents WHERE active = 1").fetchone()["n"]
    index_empty = active_docs == 0
    diagnostics = {
        "index_empty": index_empty,
        "unknown_collections": unknown,
        "filters": filters,
        "calibration": _calibration(conn),
    }
    fusion = req.get("fusion", "rrf")
    if rerank_succeeded or fusion != "rrf":
        diagnostics["calibration"]["floor_applicable"] = False
    results = serialize_results(final, mode)
    status = "ok" if final else ("empty_index" if index_empty else "no_results")
    top_score = round(final[0].score, 5) if final else None
    request = {
        "mode": mode,
        "collections": collections,
        "limit": limit,
        "min_score": min_score,
        "truncate": truncate or "off",
    }
    if fusion != "rrf":
        request["fusion"] = fusion
    if "_rerank_model" in req:
        request["rerank"] = {"model": req["_rerank_model"]}
        rerank_option = f"--rerank={req['_rerank_model']}"
    elif req.get("_rerank_advice_off"):
        rerank_option = "--no-rerank"
    else:
        rerank_option = None
    truncation_advice = _truncation_advice(
        req["query"], mode, collections, limit, min_score, truncate, after_min,
        final, diagnostics, rerank_option=rerank_option, fusion=fusion)
    summary = {
        "result_count": len(final),
        "confidence": _confidence(final, mode),
        "limit_reached": len(final) >= limit,
        "top_score": top_score,
        "source_mix": _source_mix(final, mode),
        "truncation_advice": truncation_advice,
    }
    index = freshness.index_summary(conn)
    if index is not None:
        summary["index"] = index
    return {
        "schema": SEARCH_SCHEMA,
        "query": req["query"],
        "status": status,
        "request": request,
        "summary": summary,
        "results": results,
        "diagnostics": diagnostics,
        "next_actions": _next_actions(req["query"], mode, collections, limit,
                                      min_score, truncate, final, diagnostics,
                                      truncation_advice,
                                      rerank_option=rerank_option, fusion=fusion),
    }


def run_search(conn, store, embedder, req: dict,
               *, trace: searchmod.SearchTrace | None = None,
               reranker: rerankmod.Reranker | None = None,
               rerank_n: int = 20) -> dict:
    with db.read_snapshot(conn):
        return _run_search(conn, store, embedder, req, trace=trace,
                           reranker=reranker, rerank_n=rerank_n)


def _run_search(conn, store, embedder, req: dict,
                *, trace: searchmod.SearchTrace | None = None,
                reranker: rerankmod.Reranker | None = None,
                rerank_n: int = 20) -> dict:
    query = req["query"]
    mode = req.get("mode", "hybrid")
    fusion = searchmod.parse_fusion(req.get("fusion", "rrf"))
    if fusion != "rrf" and mode != "hybrid":
        raise ValueError("--fusion weighted requires --mode hybrid")
    if fusion != "rrf" and req.get("truncate") == "calibrated":
        raise ValueError("--fusion weighted conflicts with --truncate calibrated")
    if fusion != "rrf":
        req = {**req, "fusion": fusion}
    if ("guidance" in req and type(req["guidance"]) is not bool) or (
            "explain" in req and type(req["explain"]) is not bool):
        raise ValueError("guidance and explain must be booleans")
    guidance_on = req.get("guidance", GUIDANCE_DEFAULT)
    explain = req.get("explain", False)
    if (guidance_on or explain) and trace is None:
        trace = searchmod.SearchTrace()
    if trace is not None:
        trace.mode = mode
        trace.lexical = ()
        trace.vector = ()
        trace.fused_pool = ()
        trace.count_relation = "gte"
    collections = list(req.get("collections") or [])
    search_collections = collections or None
    limit = int(req.get("limit", 10))
    if reranker is not None and (
            type(rerank_n) is not int or not 1 <= rerank_n <= 100):
        raise ValueError("rerank_n must be an integer from 1 to 100")

    def retrieve(window: int) -> list[searchmod.Result]:
        # A valid empty index has no vectors to query. Skip model loading
        # entirely, including on machines with a cold embedding cache.
        if mode in ("lexical", "vector", "hybrid") and not conn.execute(
                "SELECT 1 FROM documents WHERE active = 1 LIMIT 1").fetchone():
            return []
        if mode == "lexical":
            found = searchmod.search_lexical(conn, query, search_collections, window)
            if trace is not None:
                trace.lexical = tuple(found)
                trace.fused_pool = tuple(found)
            return found
        if mode == "vector":
            found = searchmod.search_vector(
                conn, store, embedder, query, search_collections, window)
            if trace is not None:
                trace.vector = tuple(found)
                trace.fused_pool = tuple(found)
            return found
        if mode == "hybrid":
            fusion_options = {"fusion": fusion} if fusion != "rrf" else {}
            if trace is None:
                return searchmod.search_hybrid(
                    conn, store, embedder, query, search_collections, window, **fusion_options)
            return searchmod.search_hybrid(
                conn, store, embedder, query, search_collections, window, trace=trace, **fusion_options)
        raise ValueError(f"unknown mode {mode!r}")

    raw = retrieve(max(limit, rerank_n) if reranker is not None else limit)
    rerank_succeeded = False
    rerank_failed = False
    if reranker is not None and raw:
        outcome = rerankmod.rerank_results(conn, query, raw, mode, reranker)
        if outcome.failed:
            rerank_failed = True
            # The wider lexical/vector retrieval window can change candidates
            # and scores; retry at the original limit for truthful parity.
            raw = retrieve(limit)
        else:
            raw = list(outcome.results[:limit])
            rerank_succeeded = True
    min_score = req.get("min_score")
    results = raw
    if min_score is not None:
        results = [r for r in results if r.score >= float(min_score)]
    after_min = results
    truncate = req.get("truncate")
    if truncate == "calibrated":
        # corpus-calibrated abstention floor (fidx calibrate --store) + knee tail
        from . import truncate as truncatemod
        floor = get_meta(conn, "truncate_floor")
        if floor is not None and not rerank_succeeded:
            results = truncatemod.truncate(results, f"abs:{float(floor)}", mode)
        results = truncatemod.truncate(results, "knee", mode)
    elif truncate:
        from . import truncate as truncatemod
        results = truncatemod.truncate(results, truncate, mode)
    envelope = _agent_envelope(conn, req, mode, collections, limit, min_score, truncate,
                               raw, after_min, results,
                               rerank_succeeded=rerank_succeeded)
    if "guidance" in req or guidance_on:
        envelope["request"]["guidance"] = guidance_on
    if guidance_on or explain:
        from . import guidance, guidance_diagnostics
        coverage = guidance.term_coverage(conn, query, collections=collections)
        if guidance_on:
            projection = guidance_diagnostics.assemble(
                envelope, trace, coverage,
                guidance_diagnostics.ExecutionEvidence(True, None), explain=explain)
            envelope["summary"].update(projection["summary"])
            envelope["diagnostics"].update(projection["diagnostics"])
        else:
            signals = guidance.compute_qpp(trace, term_coverage_ratio=coverage.ratio)
            envelope["diagnostics"]["qpp"] = guidance_diagnostics.qpp_projection(signals)
    if rerank_failed:
        envelope["diagnostics"]["warnings"] = [
            {"component": "rerank", "code": "runtime_failure", "fallback": "fused"}]
    elif (rerank_succeeded and truncate == "calibrated" and
          envelope["diagnostics"]["calibration"]["floor_available"]):
        envelope["diagnostics"]["warnings"] = [
            {"component": "rerank", "code": "fused_calibration_ignored", "fallback": "knee"}]
    if (get_meta(conn, "subtokens") == "1" and not db.subtokens_ready(conn)):
        envelope["diagnostics"].setdefault("warnings", []).append({
            "component": "subtokens", "code": "derived_index_stale", "fallback": "main_fts",
            "message": "Identifier subtokens are stale or missing; run fidx index --subtokens to rebuild.",
        })
    requested = req.get("budget_tokens")
    return budget.apply_budget(envelope, requested) if requested is not None else envelope


def serve(db_path: Path, conn, store, embedder) -> None:
    if not hasattr(socketserver, "UnixStreamServer"):
        raise RuntimeError("Unix-socket daemon is unavailable on this platform")
    instance_id = get_meta(conn, "instance_id")
    daemon_fingerprint = config.profile_fingerprint(embedder.profile)
    embed_lock = threading.Lock()
    duckdb_lock = threading.Lock()
    rerank_lock = threading.Lock()
    rerank_slot: dict[str, object | None] = {"model": None, "scorer": None}

    class LockedEmbedder:
        def embed_queries(self, texts):
            with embed_lock:
                return embedder.embed_queries(texts)

    locked_embedder = LockedEmbedder()

    def search_request(req: dict, rerank_state: dict | None) -> dict:
        request_conn = db.connect(db_path, create=False, read_only=True)
        try:
            # open_store only selects the backend; DuckDB opens its sidecar
            # lazily in run_search. Recheck per request because first indexing
            # may pin DuckDB after this daemon started with NullStore.
            request_store = vector_store.open_store(
                request_conn, db_path, read_only=True)
            with duckdb_lock if request_store.name == "duckdb" else nullcontext():
                try:
                    if req.get("mode", "hybrid") in ("vector", "hybrid"):
                        pinned_fingerprint = get_meta(request_conn, "embed_fingerprint")
                        if (pinned_fingerprint is not None
                                and pinned_fingerprint != daemon_fingerprint):
                            pinned_profile = get_meta(request_conn, "embed_profile")
                            raise RuntimeError(
                                f"index embedding profile {pinned_profile!r} differs "
                                f"from daemon profile {embedder.profile.name!r}; "
                                "restart fidx serve"
                            )
                    if rerank_state is not None and rerank_state["enabled"]:
                        # Scoring is a lazy iterator. Keep the lock through the
                        # whole request so a model swap cannot overlap its use.
                        with rerank_lock:
                            rerank_models.probe_rerank_capability()
                            active = request_conn.execute(
                                "SELECT 1 FROM documents WHERE active = 1 LIMIT 1"
                            ).fetchone() is not None
                            scorer = None
                            if active:
                                model = rerank_state["model"]
                                if rerank_slot["model"] != model:
                                    rerank_slot["model"] = None
                                    rerank_slot["scorer"] = None
                                    scorer, _ = rerank_models.resolve_reranker_model(
                                        model, local_files_only=True)
                                    rerank_slot["model"] = model
                                    rerank_slot["scorer"] = scorer
                                scorer = rerank_slot["scorer"]
                            return run_search(request_conn, request_store, locked_embedder,
                                              req, reranker=scorer)
                    return run_search(request_conn, request_store, locked_embedder, req)
                finally:
                    if hasattr(request_store, "close"):
                        request_store.close()
        finally:
            request_conn.close()

    sock_path = _prepare_socket_path(db_path)
    if sock_path.exists() or sock_path.is_symlink():
        if not sock_path.is_socket():
            raise RuntimeError(f"refusing to remove non-socket at {sock_path}")
        sock_path.unlink()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            if not line:
                return
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("request must be an object")
                req, rerank_state = _socket_rerank_request(req)
                protocol = req.get("guidance_protocol")
                if protocol is not None and (type(protocol) is not int or protocol != 1):
                    raise ValueError("unsupported guidance protocol")
                if req.get("cmd") == "ping":
                    resp = {"ok": True, "pong": True}
                else:
                    resp = {"ok": True, "results": search_request(req, rerank_state)}
                    fusion = searchmod.parse_fusion(req.get("fusion", "rrf"))
                    if fusion != "rrf":
                        resp["fusion_effective"] = fusion
                    if protocol == 1:
                        resp["guidance_protocol"] = 1
                    if rerank_state is not None:
                        resp["rerank_protocol"] = 1
                        resp["rerank_effective"] = rerank_state
            except Exception as exc:  # protocol boundary: report, don't crash server
                resp = {"ok": False, "error": str(exc)}
            resp["instance_id"] = instance_id
            self.wfile.write((json.dumps(resp) + "\n").encode())

    # Warm the model before accepting connections.
    embedder.embed_queries(["warmup"])
    server = _bind_private_unix_server(sock_path, Handler)
    print(f"fidx daemon listening on {sock_path} (pid {os.getpid()})", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sock_path.unlink(missing_ok=True)


def client_request(db_path: Path, req: dict, timeout: float = 30.0) -> dict | None:
    """Send a request to a running daemon; None if no usable daemon."""
    if not hasattr(socketserver, "UnixStreamServer"):
        return None
    sock_path = socket_path_for(db_path)
    if not sock_path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(sock_path))
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                data = s.recv(65536)
                if not data:
                    break
                buf += data
        return json.loads(buf) if buf else None
    except (OSError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class ClientOutcome:
    """Internal routing evidence for a future guidance-aware CLI caller."""

    path: Literal["daemon", "cold"]
    response: dict | None = None
    fallback_cause: str | None = None
    daemon_error: str | None = None
    stale_instance: bool = False


def client_request_outcome(db_path: Path, req: dict, timeout: float = 30.0, *,
                           expected_instance_id: str | None = None,
                           require_guidance_protocol: bool = False,
                           rerank_effective: dict | None = None) -> ClientOutcome:
    """Classify a daemon attempt without changing the legacy client contract.

    A protocol request deliberately fails closed against a daemon that cannot
    acknowledge it. The server only gains that acknowledgement when guidance
    fields are implemented; this internal seam does not advertise them.
    """
    if (req.get("cmd") == "search" and
            (not isinstance(expected_instance_id, str) or not expected_instance_id)):
        raise ValueError("search outcome requires expected_instance_id")
    if not hasattr(socketserver, "UnixStreamServer"):
        return ClientOutcome("cold")
    sock_path = socket_path_for(db_path)
    if not sock_path.exists():
        return ClientOutcome("cold")
    wire_req = dict(req)
    if require_guidance_protocol:
        wire_req["guidance_protocol"] = 1
    if rerank_effective is not None:
        state = _validated_rerank_effective(rerank_effective)
        wire_req.pop("_rerank_model", None)
        wire_req.pop("_rerank_advice_off", None)
        wire_req["rerank_protocol"] = 1
        wire_req["rerank_effective"] = state
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(sock_path))
            s.sendall((json.dumps(wire_req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                data = s.recv(65536)
                if not data:
                    break
                buf += data
    except ConnectionRefusedError:
        return ClientOutcome("cold", fallback_cause="connect_refused")
    except TimeoutError:
        return ClientOutcome("cold", fallback_cause="timeout")
    except OSError:
        return ClientOutcome("cold", fallback_cause="transport_error")
    if not buf.endswith(b"\n"):
        return ClientOutcome("cold", fallback_cause="malformed_response")
    try:
        response = json.loads(buf)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ClientOutcome("cold", fallback_cause="malformed_response")
    if not isinstance(response, dict):
        return ClientOutcome("cold", fallback_cause="malformed_response")
    if response.get("ok") is False:
        error = response.get("error")
        if isinstance(error, str) and error:
            return ClientOutcome("cold", fallback_cause="daemon_error", daemon_error=error)
        return ClientOutcome("cold", fallback_cause="malformed_response")
    if response.get("ok") is not True:
        return ClientOutcome("cold", fallback_cause="malformed_response")
    if (expected_instance_id is not None and
            response.get("instance_id") != expected_instance_id):
        return ClientOutcome("cold", stale_instance=True)
    if req.get("cmd") == "search":
        candidate = response.get("results")
        if not isinstance(candidate, dict) or candidate.get("schema") != SEARCH_SCHEMA:
            return ClientOutcome("cold", fallback_cause="malformed_response")
    if (require_guidance_protocol and
            (type(response.get("guidance_protocol")) is not int or
             response["guidance_protocol"] != 1)):
        return ClientOutcome("cold", fallback_cause="malformed_response")
    if rerank_effective is not None:
        if (type(response.get("rerank_protocol")) is not int or
                response["rerank_protocol"] != 1):
            return ClientOutcome("cold", fallback_cause="malformed_response")
        try:
            returned = _validated_rerank_effective(response.get("rerank_effective"))
        except ValueError:
            return ClientOutcome("cold", fallback_cause="malformed_response")
        if returned != state:
            return ClientOutcome("cold", fallback_cause="malformed_response")
    fusion = searchmod.parse_fusion(req.get("fusion", "rrf"))
    if fusion != "rrf" and response.get("fusion_effective") != fusion:
        return ClientOutcome("cold", fallback_cause="malformed_response")
    return ClientOutcome("daemon", response=response)
