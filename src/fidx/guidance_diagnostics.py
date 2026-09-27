# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Internal, deterministic guidance diagnostics with no envelope activation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from . import guidance
from .guidance import QPPSignals, TermCoverage
from .search import SearchTrace, fts_match_expr


@dataclass(frozen=True)
class ExecutionEvidence:
    """Explicit state; current successful searches always use (True, None)."""

    complete: bool
    partial_reason: str | None
    semantic_unavailable: bool = False

    def __post_init__(self) -> None:
        if (type(self.complete) is not bool or type(self.semantic_unavailable) is not bool or
                (self.complete and self.partial_reason is not None) or
                (not self.complete and
                 (not isinstance(self.partial_reason, str) or not self.partial_reason))):
            raise ValueError("execution state is inconsistent")


def execution_projection(execution: ExecutionEvidence) -> dict:
    """Only the completion fields; G2c adds daemon/cold route provenance."""
    return {"complete": execution.complete, "partial_reason": execution.partial_reason}


def qpp_projection(signals: QPPSignals) -> dict:
    """Raw signal data for G2c's explicit explain-only path."""
    payload = asdict(signals)
    values = (*payload["lexical"].values(), *payload["vector"].values(),
              *payload["fused"].values(), payload["channel_agreement"],
              payload["term_coverage"])
    if any(not math.isfinite(value) for value in values):
        raise ValueError("non-finite QPP signal")
    return payload


def no_result_causes(envelope: dict, coverage: TermCoverage,
                     execution: ExecutionEvidence) -> tuple[str, ...]:
    """Rank only causes directly supported by this search's evidence.

    The current engine never supplies partial or semantic-unavailable success
    state. Those two states exist for explicit future/injected fixtures only.
    """
    status = envelope["status"]
    if status == "ok":
        return ()
    if status not in ("no_results", "empty_index"):
        raise ValueError("no-result causes require a search result envelope")
    request, diag, summary = envelope["request"], envelope["diagnostics"], envelope["summary"]
    filters = diag["filters"]
    raw_count = filters["raw_count"]
    collections = request["collections"]
    unknown = diag["unknown_collections"]
    index_empty = diag["index_empty"] is True
    valid_query = fts_match_expr(envelope["query"]) is not None
    positive_limit = type(request["limit"]) is int and request["limit"] > 0
    unknown_only_scope = bool(collections) and set(collections) <= set(unknown)
    causes: list[str] = []
    if index_empty:
        causes.append("empty_index")
    if not valid_query or not positive_limit:
        causes.append("query_invalid")
    if unknown:
        causes.append("unknown_collection")
    if (raw_count > 0 and filters["after_truncate"] == 0 and
            (filters["dropped_by_min_score"] > 0 or filters["dropped_by_truncate"] > 0)):
        causes.append("filters_dropped_all")
    if not execution.complete:
        causes.append("execution_partial")
    if execution.semantic_unavailable:
        causes.append("semantic_unavailable")
    if isinstance(summary.get("index"), dict) and summary["index"].get("stale_hint") is True:
        causes.append("index_incomplete")
    if (not index_empty and valid_query and positive_limit and not unknown_only_scope and
            raw_count == 0 and not coverage.matched and coverage.unmatched):
        causes.append("vocabulary_mismatch")
    if (not causes and raw_count == 0 and execution.complete and
            request.get("mode", "hybrid") != "vector"):
        causes.append("likely_absent")
    return tuple(causes)


def confidence_reasons(trace: SearchTrace, coverage: TermCoverage,
                       envelope: dict) -> tuple[str, ...]:
    """Uncalibrated structural flags over the pre-filter candidate head."""
    if envelope["status"] != "ok":
        return ()
    if trace.mode != envelope["request"]["mode"]:
        raise ValueError("trace mode differs from search request")
    head = trace.fused_pool[:10]
    scores = [result.score for result in head]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("non-finite candidate score")
    filters = envelope["diagnostics"]["filters"]
    reasons: list[str] = []
    if len(scores) >= 2 and max(scores) - min(scores) <= 1e-12:
        reasons.append("flat_scores")
    if trace.mode == "hybrid" and trace.lexical and trace.vector:
        lexical_ids = {result.doc_id for result in trace.lexical}
        vector_ids = {result.doc_id for result in trace.vector}
        if not lexical_ids & vector_ids:
            reasons.append("low_channel_agreement")
    if coverage.unmatched:
        reasons.append("low_term_coverage")
    if len(head) >= 3:
        directories: dict[tuple[str, str], int] = {}
        for result in head:
            directory = result.relpath.split("/", 1)[0] if "/" in result.relpath else "."
            key = (result.collection, directory)
            directories[key] = directories.get(key, 0) + 1
        if len(directories) >= 2 and max(directories.values()) * 2 <= len(head):
            reasons.append("diffuse_directories")
    if (trace.mode in ("hybrid", "lexical") and
            not any(result.tier in ("phrase", "and") for result in trace.lexical)):
        reasons.append("weak_lexical_anchor")
    if (filters["raw_count"] >= 2 and filters["dropped_by_truncate"] > 0 and
            filters["after_truncate"] < 2):
        reasons.append("truncated_before_stable_head")
    return tuple(reasons)


def assemble(envelope: dict, trace: SearchTrace, coverage: TermCoverage,
             execution: ExecutionEvidence, *, explain: bool = False) -> dict:
    """Build dark guidance fields; G2c alone may attach them to an envelope."""
    if (trace.count_relation != "gte" or
            trace.mode != envelope["request"]["mode"] or
            envelope["summary"]["result_count"] != len(envelope["results"])):
        raise ValueError("candidate relation or exact returned count changed")
    result = {
        "summary": {"confidence_reasons": list(confidence_reasons(trace, coverage, envelope)),
                    "count_relation": "gte"},
        "diagnostics": {
            "term_coverage": {"matched": list(coverage.matched),
                              "unmatched": list(coverage.unmatched),
                              "rare": list(coverage.rare)},
            "execution": execution_projection(execution),
        },
    }
    if envelope["status"] != "ok":
        result["diagnostics"]["no_results"] = {
            "causes": list(no_result_causes(envelope, coverage, execution))}
    if explain:
        result["diagnostics"]["qpp"] = qpp_projection(guidance.compute_qpp(
            trace, term_coverage_ratio=coverage.ratio))
    return result
