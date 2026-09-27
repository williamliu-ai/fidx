# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Deterministic presentation budget for the versioned search envelope.

The estimate is intentionally simple, not a model tokenizer: canonical compact
UTF-8 JSON bytes divided by four, rounded up. A budget can trim presentation
fields and trailing results, but never changes retrieval order or scores.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any


_MINIMUM = "minimum"
_RETAINED = "retained"
_STATUS = "status-preserved"
_LADDER = "ladder"

_TOP_FIELDS = {
    "schema": _MINIMUM, "query": _MINIMUM, "status": _MINIMUM,
    "request": _RETAINED, "summary": _MINIMUM, "results": _MINIMUM,
    "next_actions": _STATUS, "diagnostics": _LADDER,
    "facets": _LADDER, "query_suggestions": _LADDER,
}
_SUMMARY_FIELDS = {
    "result_count": _MINIMUM, "budget": _MINIMUM,
    "confidence": _RETAINED, "limit_reached": _RETAINED,
    "top_score": _RETAINED, "source_mix": _RETAINED,
    "truncation_advice": _LADDER, "confidence_reasons": _LADDER,
    "count_relation": _LADDER, "index": _LADDER,
}
_RESULT_FIELDS = {
    "rank": _MINIMUM, "path": _MINIMUM, "collection": _MINIMUM,
    "relpath": _MINIMUM, "docid": _MINIMUM, "score": _MINIMUM,
    "title": _LADDER, "snippet": _LADDER, "sources": _LADDER,
    "locations": _LADDER,
}
_DIAGNOSTIC_FIELDS = {
    "index_empty": _LADDER, "unknown_collections": _LADDER,
    "filters": _LADDER, "calibration": _LADDER,
    "execution": _LADDER, "term_coverage": _LADDER,
    "no_results": _LADDER, "qpp": _LADDER,
    "warnings": _LADDER,
}
_NO_RESULTS_FIELDS = {"causes": _STATUS}
_GUIDANCE_FIELDS = ("facets", "query_suggestions", "term_coverage",
                    "confidence_reasons", "count_relation", "causes")


def canonical_compact_json(payload: Any) -> str:
    """The stable serialization used by budgeted CLI output and accounting."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def estimate_tokens(payload: Any) -> int:
    """Estimate tokens from sorted-key, no-whitespace UTF-8 JSON."""
    return math.ceil(len(canonical_compact_json(payload).encode("utf-8")) / 4)


def unclassified_fields(payload: dict) -> list[str]:
    """List emitted envelope fields with no budget class.

    Nested fields of an atomic object (request, sources, advice, etc.) inherit
    that object's class. Diagnostics are enumerated because only one nested
    cause field is preserved on abstention.
    """
    missing = [key for key in payload if key not in _TOP_FIELDS]
    summary = payload.get("summary")
    if isinstance(summary, dict):
        missing.extend(f"summary.{key}" for key in summary if key not in _SUMMARY_FIELDS)
    for index, row in enumerate(payload.get("results", [])):
        if isinstance(row, dict):
            missing.extend(f"results[{index}].{key}" for key in row
                           if key not in _RESULT_FIELDS)
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        missing.extend(f"diagnostics.{key}" for key in diagnostics
                       if key not in _DIAGNOSTIC_FIELDS)
        no_results = diagnostics.get("no_results")
        if isinstance(no_results, dict):
            missing.extend(f"diagnostics.no_results.{key}" for key in no_results
                           if key not in _NO_RESULTS_FIELDS)
    return missing


def _update_estimate(payload: dict) -> int:
    budget = payload["summary"]["budget"]
    for _ in range(10):
        value = estimate_tokens(payload)
        if budget["estimated_used"] == value:
            return value
        budget["estimated_used"] = value
    raise RuntimeError("budget estimate failed to converge")


def apply_budget(envelope: dict, requested: int) -> dict:
    """Trim an envelope in the specified ladder order, retaining its floor."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError("budget_tokens must be a non-negative integer")
    unknown = unclassified_fields(envelope)
    if unknown:
        raise ValueError(f"unclassified search envelope fields: {', '.join(unknown)}")

    payload = copy.deepcopy(envelope)
    summary = payload["summary"]
    diagnostics = payload.get("diagnostics", {})
    guidance_active = any((
        "facets" in payload, "query_suggestions" in payload,
        "term_coverage" in diagnostics,
        "confidence_reasons" in summary, "count_relation" in summary,
        "causes" in diagnostics.get("no_results", {}),
    ))
    dropped: dict[str, int | bool] = {
        "snippets": 0, "titles": 0, "advice": False,
        "next_actions": False, "diagnostics": False,
        "sources": 0, "locations": 0, "results": 0,
    }
    if guidance_active:
        dropped.update({"facets": False, "suggestions": 0,
                        "term_coverage": False,
                        "confidence_reasons": False, "count_relation": False})
    if "index" in summary:
        dropped["index"] = False
    summary["budget"] = {
        "requested": requested, "estimated_used": 0,
        "over_minimum": False, "dropped": dropped,
    }

    def fits() -> bool:
        return _update_estimate(payload) <= requested

    if fits():
        return payload

    if "facets" in payload:
        del payload["facets"]
        dropped["facets"] = True
        if fits():
            return payload
    suggestions = payload.get("query_suggestions")
    if isinstance(suggestions, list) and len(suggestions) > 1:
        dropped["suggestions"] += len(suggestions) - 1
        del suggestions[1:]
        if fits():
            return payload
    if "term_coverage" in diagnostics:
        del diagnostics["term_coverage"]
        dropped["term_coverage"] = True
        if fits():
            return payload
    if "query_suggestions" in payload:
        dropped["suggestions"] += len(payload["query_suggestions"])
        del payload["query_suggestions"]
        if fits():
            return payload

    for field, counter in (("snippet", "snippets"), ("title", "titles")):
        for result in reversed(payload["results"]):
            if field in result:
                del result[field]
                dropped[counter] += 1
                if fits():
                    return payload
    if "truncation_advice" in summary:
        del summary["truncation_advice"]
        dropped["advice"] = True
        if fits():
            return payload
    if payload["status"] == "ok" and "next_actions" in payload:
        del payload["next_actions"]
        dropped["next_actions"] = True
        if fits():
            return payload

    if "diagnostics" in payload:
        before = payload["diagnostics"]
        causes = before.get("no_results", {}).get("causes") if payload["status"] != "ok" else None
        kept = {"no_results": {"causes": causes}} if causes is not None else {}
        if before != kept:
            if kept:
                payload["diagnostics"] = kept
            else:
                del payload["diagnostics"]
            dropped["diagnostics"] = True
            if fits():
                return payload
    for field in ("confidence_reasons", "count_relation", "index"):
        if field in summary:
            del summary[field]
            dropped[field] = True
            if fits():
                return payload

    for field, counter in (("sources", "sources"), ("locations", "locations")):
        for result in reversed(payload["results"]):
            if field in result:
                del result[field]
                dropped[counter] += 1
                if fits():
                    return payload
    while len(payload["results"]) > 1:
        payload["results"].pop()
        dropped["results"] += 1
        if fits():
            return payload

    # The applicable floor itself exceeds the request. It is returned intact
    # with an explicit signal, even when the caller asked for zero tokens.
    if not fits():
        summary["budget"]["over_minimum"] = True
        _update_estimate(payload)
    return payload
