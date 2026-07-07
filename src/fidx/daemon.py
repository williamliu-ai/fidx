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

import json
import os
import socket
import socketserver
import sys
from pathlib import Path

from . import search as searchmod
from .db import get_meta

SEARCH_SCHEMA = "fidx.search.v2"


def socket_path_for(db_path: Path) -> Path:
    return db_path.with_suffix(".sock")


def _result_sources(result: searchmod.Result, mode: str) -> dict[str, float]:
    if result.sources:
        return {k: round(v, 5) for k, v in result.sources.items()}
    if mode in ("lexical", "vector"):
        return {mode: round(result.score, 5)}
    return {}


def serialize_results(results: list[searchmod.Result], mode: str) -> list[dict]:
    return [
        {
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
        for rank, r in enumerate(results, 1)
    ]


def _known_collections(conn) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT name FROM collections ORDER BY name")]


def _search_command(query: str, mode: str, collections: list[str], limit: int,
                    min_score, truncate) -> list[str]:
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
                       final: list[searchmod.Result], diagnostics: dict) -> dict:
    effective_truncate = truncate or "off"
    filters = diagnostics["filters"]
    calibration = diagnostics["calibration"]
    profile = _score_profile(pre_truncate)
    floor_available = calibration["floor_available"]

    recommendation = None
    primary_action = None
    lean = None
    reason = "Truncation is not useful until the search has candidate results."

    if final:
        if effective_truncate != "off":
            if effective_truncate == "calibrated" and not floor_available:
                if profile["has_knee"]:
                    recommendation = "knee"
                    primary_action = "clean_shortlist"
                    lean = "balanced"
                    reason = (
                        "No calibration floor is stored, so calibrated behaves like knee; "
                        "use knee as the explicit balanced truncation mode."
                    )
                else:
                    recommendation = "off"
                    primary_action = "disable_truncation"
                    lean = "recall"
                    reason = (
                        "No calibration floor is stored and the score profile has no "
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
        if effective_truncate == "calibrated" and floor_available and profile["has_knee"]:
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
    calibrated_recommended = recommendation == "calibrated" and floor_available
    if profile["count"] < 4:
        knee_reason = "Knee needs at least 4 result scores; it would be a no-op here."
    elif profile["flat"]:
        knee_reason = "The score curve is flat, so a knee cut would not separate a tail."
    elif knee_applicable:
        knee_reason = "Balanced option: cut the score-curve tail without corpus calibration."
    else:
        knee_reason = "No useful knee was detected in the score profile."
    if floor_available:
        calibrated_reason = (
            "Purity option: apply the stored corpus floor, then knee; this can abstain "
            "or drop borderline results."
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
                _search_command(query, mode, collections, limit, min_score, "off"),
            ),
            _truncation_option(
                "clean_shortlist",
                "knee",
                "balanced",
                knee_applicable,
                knee_recommended,
                knee_reason,
                _search_command(query, mode, collections, limit, min_score, "knee"),
            ),
            _truncation_option(
                "use_calibrated_abstention",
                "calibrated",
                "purity",
                floor_available,
                calibrated_recommended,
                calibrated_reason,
                _search_command(query, mode, collections, limit, min_score, "calibrated"),
            ),
        ],
    }


def _next_actions(query: str, mode: str, collections: list[str], limit: int, min_score,
                  truncate, results: list[searchmod.Result], diagnostics: dict,
                  truncation_advice: dict) -> list[dict]:
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
                _search_command(query, mode, collections, limit, min_score, "off"))
        if mode != "lexical":
            add("try_exact_terms",
                "If the desired item has exact names, paths, errors, or identifiers, try lexical mode.",
                _search_command(query, "lexical", collections, limit, None, "off"))
        if mode != "vector":
            add("try_conceptual_terms",
                "If the desired item uses different wording, try vector mode.",
                _search_command(query, "vector", collections, limit, None, "off"))
        if len(results) <= 2 and diagnostics["filters"]["after_truncate"] < limit:
            add("broaden_query",
                "Only a narrow set returned; use fewer or broader terms if the target is missing.",
                _search_command("<broader query>", mode, collections, limit, None, "off"))
        if diagnostics["filters"]["after_truncate"] >= limit:
            add("increase_limit",
                "The result list reached the requested limit; ask for more candidates.",
                _search_command(query, mode, collections, limit * 2, min_score, effective_truncate))
        return actions

    if diagnostics["active_docs"] == 0:
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
            _search_command(query, mode, [], limit, min_score, effective_truncate))
    elif collections:
        add("retry_without_scope",
            "The collection scope may be too narrow.",
            _search_command(query, mode, [], limit, min_score, effective_truncate))

    filters = diagnostics["filters"]
    if min_score is not None and filters["raw_count"] > filters["after_min_score"]:
        add("remove_min_score",
            "The minimum score filter removed available candidates.",
            _search_command(query, mode, collections, limit, None, effective_truncate))
    if effective_truncate != "off" and filters["after_min_score"] > filters["after_truncate"]:
        options = {o["intent"]: o for o in truncation_advice["options"]}
        action = truncation_advice["primary_action"]
        if action == "clean_shortlist" and options["clean_shortlist"]["applicable"]:
            option = options["clean_shortlist"]
            add("clean_shortlist", option["reason"], option["command"])
        else:
            add("disable_truncation",
                "Truncation removed all candidates; rerun without a tail cut.",
                _search_command(query, mode, collections, limit, min_score, "off"))
    if mode != "hybrid":
        add("try_hybrid",
            "Hybrid mode combines lexical and vector evidence and is the recall-first default.",
            _search_command(query, "hybrid", collections, limit, None, "off"))
    if mode != "lexical":
        add("try_exact_terms",
            "Use lexical mode for exact remembered words, names, paths, errors, or identifiers.",
            _search_command(query, "lexical", collections, limit, None, "off"))
    if mode != "vector":
        add("try_conceptual_terms",
            "Use vector mode for synonym-heavy or conceptual wording.",
            _search_command(query, "vector", collections, limit, None, "off"))
    add("broaden_query",
        "Use fewer or broader terms if the query may be too specific.",
        _search_command("<broader query>", mode, collections, limit, None, "off"))
    return actions


def _agent_envelope(conn, req: dict, mode: str, collections: list[str], limit: int,
                    min_score, truncate, raw: list[searchmod.Result],
                    after_min: list[searchmod.Result],
                    final: list[searchmod.Result]) -> dict:
    known = _known_collections(conn)
    known_set = set(known)
    unknown = sorted(c for c in collections if c not in known_set)
    filters = {
        "raw_count": len(raw),
        "after_min_score": len(after_min),
        "after_truncate": len(final),
        "dropped_by_min_score": len(raw) - len(after_min),
        "dropped_by_truncate": len(after_min) - len(final),
    }
    active_docs = conn.execute(
        "SELECT count(*) AS n FROM documents WHERE active = 1").fetchone()["n"]
    diagnostics = {
        "active_docs": active_docs,
        "known_collections": known,
        "unknown_collections": unknown,
        "filters": filters,
        "calibration": _calibration(conn),
    }
    results = serialize_results(final, mode)
    status = "ok" if final else ("empty_index" if active_docs == 0 else "no_results")
    top_score = round(final[0].score, 5) if final else None
    request = {
        "mode": mode,
        "collections": collections,
        "limit": limit,
        "min_score": min_score,
        "truncate": truncate or "off",
    }
    truncation_advice = _truncation_advice(
        req["query"], mode, collections, limit, min_score, truncate, after_min,
        final, diagnostics)
    return {
        "schema": SEARCH_SCHEMA,
        "query": req["query"],
        "status": status,
        "request": request,
        "summary": {
            "result_count": len(final),
            "confidence": _confidence(final, mode),
            "limit_reached": len(final) >= limit,
            "top_score": top_score,
            "source_mix": _source_mix(final, mode),
            "truncation_advice": truncation_advice,
        },
        "results": results,
        "diagnostics": diagnostics,
        "next_actions": _next_actions(req["query"], mode, collections, limit,
                                      min_score, truncate, final, diagnostics,
                                      truncation_advice),
    }


def run_search(conn, store, embedder, req: dict) -> dict:
    query = req["query"]
    mode = req.get("mode", "hybrid")
    collections = list(req.get("collections") or [])
    search_collections = collections or None
    limit = int(req.get("limit", 10))
    if mode == "lexical":
        raw = searchmod.search_lexical(conn, query, search_collections, limit)
    elif mode == "vector":
        raw = searchmod.search_vector(conn, store, embedder, query, search_collections, limit)
    elif mode == "hybrid":
        raw = searchmod.search_hybrid(conn, store, embedder, query, search_collections, limit)
    else:
        raise ValueError(f"unknown mode {mode!r}")
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
        if floor is not None:
            results = truncatemod.truncate(results, f"abs:{float(floor)}", mode)
        results = truncatemod.truncate(results, "knee", mode)
    elif truncate:
        from . import truncate as truncatemod
        results = truncatemod.truncate(results, truncate, mode)
    return _agent_envelope(conn, req, mode, collections, limit, min_score, truncate,
                           raw, after_min, results)


def serve(db_path: Path, conn, store, embedder) -> None:
    sock_path = socket_path_for(db_path)
    if sock_path.exists():
        sock_path.unlink()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            if not line:
                return
            try:
                req = json.loads(line)
                if req.get("cmd") == "ping":
                    resp = {"ok": True, "pong": True}
                else:
                    resp = {"ok": True, "results": run_search(conn, store, embedder, req)}
            except Exception as exc:  # protocol boundary: report, don't crash server
                resp = {"ok": False, "error": str(exc)}
            self.wfile.write((json.dumps(resp) + "\n").encode())

    # Warm the model before accepting connections.
    embedder.embed_queries(["warmup"])
    server = socketserver.UnixStreamServer(str(sock_path), Handler)
    print(f"fidx daemon listening on {sock_path} (pid {os.getpid()})", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sock_path.unlink(missing_ok=True)


def client_request(db_path: Path, req: dict, timeout: float = 30.0) -> dict | None:
    """Send a request to a running daemon; None if no usable daemon."""
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
