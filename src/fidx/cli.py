# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""fidx command-line interface."""

from __future__ import annotations

import json
import os
import string
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import click

from . import __version__, budget, config, daemon, db, errors, freshness, indexer, vector_store
from .embedder import FastEmbedder
from .locking import index_lock


def open_index(db_path: Path, profile_name: str | None = None, *, create: bool = True):
    """Open an index, optionally creating it, returning (conn, profile).

    Once embeddings exist, the index pins its profile in meta; a conflicting
    --profile is an error rather than a silent dim mismatch.
    """
    conn = db.connect(db_path) if create else db.connect(db_path, create=False)
    try:
        if create:
            db.init_schema(conn)
        else:
            db.validate_index(conn)
        stored = db.get_meta(conn, "embed_profile")
        if stored is not None and profile_name and profile_name != stored:
            raise click.ClickException(
                f"index {db_path} is pinned to profile {stored!r}; "
                f"re-index into a fresh --db to use {profile_name!r}"
            )
        profile = config.get_profile(stored or profile_name)
        return conn, profile
    except Exception:
        conn.close()
        raise


def _json_requested(args: list[str]) -> bool:
    """A literal --json after -- is a positional value, not an output option."""
    return "--json" in args[:args.index("--") if "--" in args else len(args)]


def _command_name(args: list[str]) -> str | None:
    """Find the root command for doctor report precedence."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--db":
            i += 2
        elif arg.startswith("--"):
            i += 1
        else:
            return arg
    return None


class ErrorBoundaryGroup(click.Group):
    """Render execution failures after Click has parsed arguments and options."""

    def main(self, args=None, prog_name=None, standalone_mode=True, **extra):
        argv = list(sys.argv[1:] if args is None else args)
        json_mode = _json_requested(argv)
        doctor_command = _command_name(argv) == "doctor"
        try:
            if json_mode and not doctor_command:
                # A command may print progress before failing. Keep stdout a
                # single parseable document in JSON error mode.
                with tempfile.SpooledTemporaryFile(
                    mode="w+t", encoding="utf-8", max_size=1024 * 1024
                ) as output:
                    with redirect_stdout(output):
                        result = super().main(args=argv, prog_name=prog_name,
                                              standalone_mode=False, **extra)
                    output.seek(0)
                    shutil.copyfileobj(output, sys.stdout)
                return result
            return super().main(args=argv, prog_name=prog_name,
                                standalone_mode=False, **extra)
        except click.UsageError as exc:
            exc.show()
            raise SystemExit(errors.ExitCode.USAGE) from None
        except Exception as exc:
            if os.environ.get("FIDX_DEBUG") == "1":
                raise
            payload, code = errors.envelope(exc)
            if json_mode:
                if doctor_command:
                    # Doctor's diagnostic report is its result even on failure.
                    click.echo(json.dumps({
                        "schema": "fidx.doctor.v1", "ok": False,
                        "checks": [{
                            "name": "doctor", "ok": False,
                            "detail": payload["message"],
                            "remediation": "Run with FIDX_DEBUG=1 for a traceback.",
                            "hard": True,
                        }],
                    }))
                else:
                    click.echo(json.dumps(payload, ensure_ascii=False))
            else:
                click.echo(f"Error: {payload['message']}", err=True)
            raise SystemExit(code) from None


@click.group(cls=ErrorBoundaryGroup)
@click.version_option(__version__)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="Index database path (default: $FIDX_DB or ~/.cache/fidx/index.db).")
@click.option("--json", "error_json", is_flag=True,
              help="Emit command errors as JSON; use command-level --json for result output.")
@click.pass_context
def main(ctx: click.Context, db_path: Path | None, error_json: bool) -> None:
    """fidx — fast local semantic search for markdown, text, chat and code.

    \b
    Agent quick path:
      fidx collection add <path> --name <scope>
      fidx index
      fidx serve
      fidx search "<query>" --json -n 10
      fidx get --head "#docid"

    \b
    fidx is a local AI search engine: hybrid BM25 + vector search in one
    SQLite-backed index, CPU-only, offline, and no LLM calls in the query path.
    For agent/tool selection notes, see docs/AGENT_GUIDE.md in the repository.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = (db_path or config.default_db_path()).expanduser()
    ctx.obj["error_json"] = error_json


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")
def doctor(as_json: bool) -> None:
    """Diagnose host capabilities (sqlite extensions, FTS5, sqlite-vec, model cache).

    Safe to run anywhere: never imports native deps eagerly, so it diagnoses (not
    crashes) when sqlite-vec/onnxruntime are missing or built for another arch.
    Exits non-zero if a hard capability fails.
    """
    from . import diagnostics
    as_json = as_json or click.get_current_context().find_root().obj["error_json"]
    checks = diagnostics.capabilities()
    hard = diagnostics.hard_failures(checks)
    if as_json:
        click.echo(json.dumps({
            "schema": "fidx.doctor.v1", "ok": not hard,
            "checks": [c.__dict__ for c in checks],
        }, indent=2))
    else:
        for c in checks:
            click.echo(f"[{'OK  ' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
            if not c.ok and c.remediation:
                click.echo(f"        -> {c.remediation}")
    if hard:
        if not as_json:
            click.echo(f"\n{len(hard)} hard check(s) failed — fidx cannot run on this host.",
                       err=True)
        raise SystemExit(errors.ExitCode.ERROR)
    if not as_json:
        click.echo("\nAll hard checks passed — fidx is ready.")


@main.group()
def models() -> None:
    """Manage locally cached models."""


@models.command("fetch")
@click.argument("target", type=click.Choice(["embed", "rerank"]), default="embed")
@click.option("--profile", default=None, help="Embedding profile to fetch.")
def models_fetch(target: str, profile: str | None) -> None:
    """Fetch an embedding model for offline search."""
    if target == "rerank":
        raise click.ClickException("rerank model fetch is available when reranking is enabled")
    try:
        prof = config.get_profile(profile)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        cache = FastEmbedder(prof).fetch()
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not fetch embedding model {prof.model!r}: {exc}") from exc
    click.echo(f"fetched embedding model {prof.model} [{prof.name}] into {cache}")


@main.group()
def collection() -> None:
    """Manage indexed collections (named directory trees)."""


@collection.command("add")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", required=True, help="Collection name (used for scoping searches).")
@click.option("--glob", "globs", multiple=True, help="Glob pattern(s) relative to root; repeatable. "
              f"Default: {' '.join(config.DEFAULT_GLOBS)}")
@click.option("--exclude-glob", "exclude_globs", multiple=True,
              help="Glob pattern(s) to skip after includes are matched; repeatable.")
@click.pass_context
def collection_add(ctx: click.Context, root: Path, name: str, globs: tuple[str, ...],
                   exclude_globs: tuple[str, ...]) -> None:
    """Register ROOT as a collection. Run `fidx index` afterwards."""
    conn, _ = open_index(ctx.obj["db_path"])
    indexer.add_collection(conn, name, root, list(globs) or None, list(exclude_globs) or None)
    click.echo(f"collection {name!r} -> {root}")


@collection.command("list")
@click.pass_context
def collection_list(ctx: click.Context) -> None:
    conn, _ = open_index(ctx.obj["db_path"], create=False)
    for row in conn.execute("SELECT name, root, globs FROM collections ORDER BY name"):
        count = conn.execute("SELECT count(*) AS n FROM documents WHERE collection = ?",
                             (row["name"],)).fetchone()["n"]
        click.echo(f"{row['name']}\t{row['root']}\t{count} docs\t{json.loads(row['globs'])}")


@collection.command("remove")
@click.argument("name")
@click.pass_context
def collection_remove(ctx: click.Context, name: str) -> None:
    conn, _ = open_index(ctx.obj["db_path"], create=False)
    store = vector_store.open_store(conn, ctx.obj["db_path"])
    indexer.remove_collection(conn, store, name)
    click.echo(f"removed collection {name!r}")


@main.command()
@click.option("-c", "--collection", "only", default=None, help="Only index this collection.")
@click.option("--profile", default=None, help=f"Embedding profile for a new index "
              f"(default {config.DEFAULT_PROFILE}; known: {', '.join(config.PROFILES)}).")
@click.option("--threads", type=int, default=None, help="ONNX intra-op threads.")
@click.option("--parallel", type=int, default=None,
              help="Data-parallel embedding workers (0 = all cores).")
@click.option("--backend", type=click.Choice(vector_store.BACKENDS), default=None,
              help="Vector backend for a new index (default sqlite-vec; "
                   "duckdb = HNSW sidecar, needs the [duckdb] extra).")
@click.option("--calibrate/--no-calibrate", "do_calibrate", default=True, show_default=True,
              help="Maintain the truncation floor (used by search --truncate calibrated). "
                   "Recalibrates only on first build or after enough drift; small "
                   "incremental updates skip it.")
@click.option("--recalibrate-after", type=float, default=0.1, show_default=True,
              help="Recalibrate once changed docs since last calibration reach this "
                   "fraction of the corpus (min 100 docs).")
@click.option("--json", "as_json", is_flag=True, help="Versioned machine-readable index result.")
@click.pass_context
def index(ctx: click.Context, only: str | None, profile: str | None, threads: int | None,
          parallel: int | None, backend: str | None, do_calibrate: bool,
          recalibrate_after: float, as_json: bool) -> None:
    """Scan collections and (re)build chunks + embeddings incrementally."""
    db_path = ctx.obj["db_path"]
    with index_lock(db_path):
        _index_unlocked(ctx, only, profile, threads, parallel, backend,
                        do_calibrate, recalibrate_after, as_json)


def _index_unlocked(ctx: click.Context, only: str | None, profile: str | None,
                    threads: int | None, parallel: int | None, backend: str | None,
                    do_calibrate: bool, recalibrate_after: float, as_json: bool) -> None:
    db_path = ctx.obj["db_path"]
    conn, prof = open_index(db_path, profile)
    db.ensure_instance_id(conn)
    stored_backend = db.get_meta(conn, "vector_backend")
    store = vector_store.make_store(backend or stored_backend or "sqlite-vec", conn, db_path)
    names = [only] if only else [r["name"] for r in conn.execute("SELECT name FROM collections")]
    if not names:
        raise click.ClickException("no collections registered; use `fidx collection add` first")
    stats = indexer.IndexStats()
    t0 = time.perf_counter()
    try:
        db.ensure_vectors(conn, store, prof.dim, config.profile_fingerprint(prof),
                          prof.model, prof.name)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))
    for name in names:
        indexer.sync_collection(conn, store, name, stats)
    indexer.embed_pending(conn, store, FastEmbedder(prof, threads=threads, parallel=parallel),
                          stats, progress=True)
    elapsed = time.perf_counter() - t0
    if not as_json:
        click.echo(
            f"indexed in {elapsed:.1f}s: +{stats.added} added, ~{stats.updated} updated, "
            f"-{stats.removed} removed, {stats.unchanged} unchanged, "
            f"{stats.embedded_chunks} chunks embedded [{prof.name}/{store.name}]"
        )
    for err in stats.errors:
        click.echo(f"warning: {err}", err=True)

    calibration_state = "skipped"
    if do_calibrate:
        # Calibration estimates a corpus-wide score-distribution floor, so a few
        # changed docs barely move it. Accumulate changes in meta and recalibrate
        # only on first build or once drift crosses the threshold — incremental
        # updates then cost ~nothing beyond a meta counter write.
        from . import calibrate as calibratemod
        n_now = conn.execute("SELECT count(*) FROM documents WHERE active = 1").fetchone()[0]
        changed = stats.added + stats.updated + stats.removed
        try:
            prior = int(db.get_meta(conn, "calib_changes") or 0)
        except ValueError:
            prior = 0  # corrupt counter -> treat as 0 (will recalibrate sooner)
        pending = prior + changed
        have_floor = db.get_meta(conn, "truncate_floor") is not None
        threshold = calibratemod.recalibration_threshold(n_now, recalibrate_after)
        if not calibratemod.should_recalibrate(have_floor, pending, n_now, recalibrate_after):
            db.set_meta(conn, "calib_changes", str(pending))
            if not as_json:
                click.echo(f"calibration skipped (+{changed} changed, {pending}/{threshold} "
                           f"since last; floor={db.get_meta(conn, 'truncate_floor')})")
        else:
            cal = calibratemod.calibrate(conn, store, FastEmbedder(prof, threads=threads), seed=0)
            if cal.get("n_pos"):
                calibration_state = "applied"
                db.set_meta(conn, "truncate_floor", str(cal["floor"]))
                db.set_meta(conn, "calib_changes", "0")
                if not as_json:
                    click.echo(
                        f"calibrated truncate_floor={cal['floor']} "
                        f"(rejects {cal['neg_rejected_at_floor']:.0%} noise, "
                        f"keeps {cal['pos_retained_at_floor']:.0%} answers; "
                        f"search --truncate calibrated)")
            else:
                calibration_state = "deferred"
                # too few/no samples (e.g. emptied corpus): keep accumulating the
                # change count so it isn't lost, and retry next index.
                db.set_meta(conn, "calib_changes", str(pending))
                if not as_json:
                    click.echo("calibration deferred (insufficient samples)")
    db.set_meta(conn, "last_index_completed_at", freshness.completed_at())
    if as_json:
        click.echo(json.dumps({
            "schema": "fidx.index.v1", "added": stats.added,
            "updated": stats.updated, "removed": stats.removed,
            "embedded_chunks": stats.embedded_chunks, "seconds": elapsed,
            "calibration": calibration_state, "warnings": stats.errors,
        }, ensure_ascii=False))


def _guidance_response_valid(candidate: dict, req: dict, *, guidance_on: bool,
                             guidance_explicit: bool, explain: bool) -> bool:
    """Reject a capability acknowledgement that did not honor the request."""
    request = candidate.get("request")
    summary = candidate.get("summary")
    diagnostics = candidate.get("diagnostics")
    if not all(isinstance(value, dict) for value in (request, summary, diagnostics)):
        return False
    if (not isinstance(candidate.get("results"), list) or
            candidate.get("status") not in ("ok", "no_results", "empty_index") or
            type(summary.get("result_count")) is not int or
            summary["result_count"] != len(candidate["results"]) or
            candidate.get("query") != req["query"] or
            any(request.get(key) != req[key] for key in (
                "mode", "collections", "limit", "min_score")) or
            request.get("truncate") != (req["truncate"] or "off")):
        return False
    if guidance_explicit or guidance_on:
        if type(request.get("guidance")) is not bool or request["guidance"] is not guidance_on:
            return False
    elif "guidance" in request:
        return False
    if guidance_on:
        execution = diagnostics.get("execution")
        if (not isinstance(summary.get("confidence_reasons"), list) or
                summary.get("count_relation") != "gte" or
                not isinstance(diagnostics.get("term_coverage"), dict) or
                not isinstance(execution, dict) or
                execution.get("complete") is not True or
                execution.get("partial_reason", object()) is not None):
            return False
        if candidate.get("status") != "ok":
            no_results = diagnostics.get("no_results")
            if (not isinstance(no_results, dict) or
                    not isinstance(no_results.get("causes"), list)):
                return False
    else:
        no_results = diagnostics.get("no_results")
        if ("confidence_reasons" in summary or "count_relation" in summary or
                "term_coverage" in diagnostics or "execution" in diagnostics or
                (isinstance(no_results, dict) and "causes" in no_results) or
                (no_results is not None and not isinstance(no_results, dict))):
            return False
    return isinstance(diagnostics.get("qpp"), dict) if explain else "qpp" not in diagnostics


@main.command()
@click.argument("query")
@click.option("-c", "--collection", "collections", multiple=True,
              help="Restrict to collection(s); repeatable.")
@click.option("-n", "--limit", default=10, show_default=True)
@click.option("--mode", type=click.Choice(["hybrid", "lexical", "vector"]), default="hybrid",
              show_default=True, help="hybrid = BM25 + vector with RRF fusion.")
@click.option("--json", "as_json", is_flag=True,
              help="Machine-readable agent envelope with results, diagnostics and next actions.")
@click.option("--files", "files_only", is_flag=True, help="Print matching paths only.")
@click.option("--min-score", type=float, default=None)
@click.option("--truncate", default=None,
              help="Deterministic tail truncation, e.g. ratio:0.5, gap:0.5, knee, "
                   "mad:3, source:0.4,0.2. See fidx/truncate.py.")
@click.option("--budget-tokens", type=click.IntRange(min=0), default=None,
              help="Estimate a compact JSON response budget; requires --json.")
@click.option("--guidance/--no-guidance", default=None,
              help="Opt in to search guidance; disabled by default.")
@click.option("--explain", is_flag=True, help="Include search quality signals in JSON diagnostics.")
@click.option("--no-daemon", is_flag=True, help="Skip the warm daemon even if running.")
@click.pass_context
def search(ctx: click.Context, query: str, collections: tuple[str, ...], limit: int,
           mode: str, as_json: bool, files_only: bool, min_score: float | None,
           truncate: str | None, budget_tokens: int | None, guidance: bool | None,
           explain: bool, no_daemon: bool) -> None:
    """Search the index. Default mode is hybrid (best recall).

    \b
    Agent use:
      --json returns the fidx.search.v2 envelope with ranked results,
      diagnostics, summary.truncation_advice and runnable next_actions.

    \b
      Use --mode lexical for exact names, paths, errors and identifiers.
      Use --mode vector for synonym-heavy or conceptual wording.
      Use -c/--collection to scope to memory, docs, code or another source.
      Use --truncate only when the JSON truncation_advice recommends it.
      Use --budget-tokens with --json to trim presentation fields in rank order.
    """
    if budget_tokens is not None and not (as_json or ctx.obj["error_json"]):
        raise click.UsageError("--budget-tokens requires --json")
    db_path: Path = ctx.obj["db_path"]
    conn, prof = open_index(db_path, create=False)
    instance_id = db.get_meta(conn, "instance_id")
    req = {"cmd": "search", "query": query, "mode": mode,
           "collections": list(collections), "limit": limit, "min_score": min_score,
           "truncate": truncate}
    guidance_on = daemon.GUIDANCE_DEFAULT if guidance is None else guidance
    protocol_required = guidance is not None or guidance_on or explain
    if guidance is not None or guidance_on:
        req["guidance"] = guidance_on
    if explain:
        req["explain"] = True

    payload: dict | None = None
    stale_daemon = False
    fallback_cause: str | None = None
    daemon_error: str | None = None
    execution_path = "cold"
    if not no_daemon:
        if protocol_required:
            outcome = daemon.client_request_outcome(
                db_path, req, expected_instance_id=instance_id,
                require_guidance_protocol=True)
            stale_daemon = outcome.stale_instance
            fallback_cause = outcome.fallback_cause
            daemon_error = outcome.daemon_error
            if outcome.path == "daemon":
                candidate = outcome.response["results"]
                if _guidance_response_valid(
                    candidate, req, guidance_on=guidance_on,
                    guidance_explicit=guidance is not None, explain=explain,
                ):
                    payload = candidate
                    execution_path = "daemon"
                else:
                    fallback_cause = "malformed_response"
        else:
            resp = daemon.client_request(db_path, req)
            if resp is not None and resp.get("ok"):
                if resp.get("instance_id") != instance_id:
                    stale_daemon = True
                else:
                    candidate = resp.get("results")
                    if isinstance(candidate, dict) and candidate.get("schema") == daemon.SEARCH_SCHEMA:
                        payload = candidate
    if payload is None:
        store = vector_store.open_store(conn, db_path)
        has_documents = conn.execute(
            "SELECT 1 FROM documents WHERE active = 1 LIMIT 1").fetchone() is not None
        embedder = (FastEmbedder(prof, local_files_only=True)
                    if has_documents and mode in ("vector", "hybrid") else None)
        try:
            payload = daemon.run_search(conn, store, embedder, req)
        except ValueError as exc:
            raise click.ClickException(str(exc))
    if stale_daemon:
        payload["diagnostics"]["execution"] = {
            "daemon": "stale_instance",
            "message": "Restart fidx serve to use the current index database.",
        }
    if guidance_on:
        execution = payload["diagnostics"]["execution"]
        execution.update({"complete": True, "partial_reason": None,
                          "path": execution_path})
        if fallback_cause is not None:
            execution["fallback_cause"] = fallback_cause
        if daemon_error is not None:
            execution["error"] = daemon_error
    if budget_tokens is not None:
        payload = budget.apply_budget(payload, budget_tokens)

    if as_json or ctx.obj["error_json"]:
        click.echo(budget.canonical_compact_json(payload) if budget_tokens is not None
                   else json.dumps(payload, ensure_ascii=False))
    elif files_only:
        for r in payload["results"]:
            click.echo(r["path"])
    else:
        if not payload["results"]:
            click.echo("no results")
        for r in payload["results"]:
            click.echo(f"{r['score']:.4f}  {r['docid']}  {r['path']}  — {r['title']}")
            if r["snippet"]:
                click.echo(f"        {r['snippet'][:160]}")


_GET_JSON_DEFAULT_MAX_BYTES = 65_536
_GET_TRUNCATION_MARKER = "\n… [truncated]"


def _bounded_get_body(body: str, max_bytes: int) -> tuple[str, bool]:
    """Cap source bytes at a UTF-8 character boundary before JSON escaping."""
    if max_bytes == 0:
        return body, False
    encoded = body.encode("utf-8")
    if len(encoded) <= max_bytes:
        return body, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + _GET_TRUNCATION_MARKER, True


@main.command()
@click.argument("target")
@click.option("--full/--head", default=True, help="--head prints the first 40 lines.")
@click.option("--json", "as_json", is_flag=True,
              help="Versioned document JSON with a 64 KiB raw UTF-8 body cap.")
@click.option("--max-bytes", type=click.IntRange(min=0), default=None,
              help="JSON body cap in raw UTF-8 bytes (0 = unlimited).")
@click.pass_context
def get(ctx: click.Context, target: str, full: bool, as_json: bool,
        max_bytes: int | None) -> None:
    """Print a document by path (collection/relpath) or docid (#abc123)."""
    if max_bytes is not None and not as_json:
        raise click.UsageError("--max-bytes requires get --json")
    conn, _ = open_index(ctx.obj["db_path"], create=False)
    if target.startswith("#"):
        prefix = target[1:]
        if not prefix or any(ch not in string.hexdigits for ch in prefix):
            raise click.ClickException(f"document {target!r} not found")
        matches = conn.execute(
            "SELECT body, collection, relpath, hash FROM documents WHERE hash LIKE ? "
            "ORDER BY collection, relpath LIMIT 11", (prefix + "%",)
        ).fetchall()
        if len(matches) > 1:
            raise errors.AmbiguousReference(
                f"document reference {target!r} matches multiple documents",
                candidates=[
                    {"docid": "#" + match["hash"][:6],
                     "collection": match["collection"], "relpath": match["relpath"]}
                    for match in matches[:10]
                ],
            )
        row = matches[0] if matches else None
    else:
        coll, _, rel = target.partition("/")
        row = conn.execute("SELECT body, collection, relpath, hash FROM documents "
                           "WHERE collection = ? AND relpath = ?", (coll, rel)).fetchone()
    if row is None:
        raise click.ClickException(f"document {target!r} not found")
    body = row["body"]
    if not full:
        body = "\n".join(body.splitlines()[:40])
    if as_json:
        bounded, truncated = _bounded_get_body(
            body, _GET_JSON_DEFAULT_MAX_BYTES if max_bytes is None else max_bytes)
        click.echo(json.dumps({
            "schema": "fidx.get.v1",
            "docid": "#" + row["hash"][:6],
            "collection": row["collection"],
            "relpath": row["relpath"],
            "truncated": truncated,
            "body": bounded,
        }, ensure_ascii=False))
    else:
        click.echo(body)


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Versioned machine-readable status.")
@click.pass_context
def status(ctx: click.Context, as_json: bool) -> None:
    """Index health: collections, documents, chunks, vectors, size."""
    db_path: Path = ctx.obj["db_path"]
    conn, prof = open_index(db_path, create=False)
    docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
    chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    store = vector_store.open_store(conn, db_path)
    vecs = store.count()
    if as_json:
        collections = []
        for row in conn.execute("SELECT name, root FROM collections ORDER BY name"):
            count = conn.execute(
                "SELECT count(*) AS n FROM documents WHERE collection = ?", (row["name"],)
            ).fetchone()["n"]
            collections.append({"name": row["name"], "root": row["root"],
                                "documents": count})
        click.echo(json.dumps({
            "schema": "fidx.status.v1", "db_path": str(db_path),
            "profile": prof.name, "backend": store.name,
            "collections": collections, "documents": docs, "chunks": chunks,
            "embedded_chunks": vecs,
        }, ensure_ascii=False))
        return
    size_mb = db_path.stat().st_size / 1e6 if db_path.exists() else 0
    sidecar = vector_store.sidecar_path(db_path)
    if store.name == "duckdb" and sidecar.exists():
        size_mb += sidecar.stat().st_size / 1e6
    click.echo(f"db: {db_path} ({size_mb:.1f} MB)")
    click.echo(f"profile: {prof.name} ({prof.model}, {prof.dim}d)  backend: {store.name}")
    click.echo(f"documents: {docs}  chunks: {chunks}  vectors: {vecs}")
    response = daemon.client_request(db_path, {"cmd": "ping"})
    daemon_state = ("running" if response and response.get("ok") and
                    response.get("instance_id") == db.get_meta(conn, "instance_id")
                    else "not running")
    click.echo(f"daemon: {daemon_state}")
    for row in conn.execute("SELECT collection, count(*) AS n FROM documents GROUP BY collection"):
        click.echo(f"  {row['collection']}: {row['n']} docs")


@main.command()
@click.option("--threads", type=int, default=None,
              help="ONNX intra-op threads for query embedding (also $FIDX_THREADS).")
@click.pass_context
def serve(ctx: click.Context, threads: int | None) -> None:
    """Run the warm-search daemon (foreground). Searches then take ~ms."""
    conn, prof = open_index(ctx.obj["db_path"], create=False)
    store = vector_store.open_store(conn, ctx.obj["db_path"])
    daemon.serve(ctx.obj["db_path"], conn, store,
                 FastEmbedder(prof, threads=threads, local_files_only=True))


@main.command()
@click.option("--sample", type=int, default=200, show_default=True,
              help="Documents to sample for self-retrieval calibration.")
@click.option("--target", type=float, default=0.85, show_default=True,
              help="Fraction of corpus self-answers to keep (reference floor).")
@click.option("--reject", type=float, default=0.9, show_default=True,
              help="Fraction of gibberish noise queries the floor should reject.")
@click.option("--dropout", type=int, default=2, show_default=True,
              help="Pseudo-query term dropout (higher = harder/safer floor).")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--store", is_flag=True,
              help="Persist the floor to the index (used by --truncate calibrated).")
@click.pass_context
def calibrate(ctx: click.Context, sample: int, target: float, reject: float,
              dropout: int, seed: int, store: bool) -> None:
    """Calibrate the truncation floor from the indexed corpus (no benchmark).

    Probes the corpus with self-retrieval pseudo-queries (positives: the source
    doc is the known answer) and gibberish queries (negatives: the noise
    ceiling). The recommended floor rejects ~`reject` of noise; `search
    --truncate calibrated` applies it + the knee cut. With --store it is saved.
    """
    from . import calibrate as calibratemod
    db_path: Path = ctx.obj["db_path"]
    conn, prof = open_index(db_path, create=False)
    vstore = vector_store.open_store(conn, db_path)
    res = calibratemod.calibrate(
        conn, vstore, FastEmbedder(prof, local_files_only=True), sample=sample,
        target=target, reject=reject, dropout=dropout, seed=seed)
    if not res.get("n_pos"):
        raise click.ClickException(res.get("note", "calibration produced no samples"))
    click.echo(f"positives n={res['n_pos']} (p5={res['pos_p5']} median={res['pos_median']})  "
               f"negatives n={res['n_neg']} (p50={res['neg_p50']} p90={res['neg_p90']} max={res['neg_max']})")
    click.echo(f"separation (pos_p25 - neg_p90) = {res['separation']}  "
               f"({'clean gap' if res['separation'] > 0 else 'OVERLAP — limited abstention'})")
    click.echo(f"recommended floor = {res['floor']}  "
               f"(rejects {res['neg_rejected_at_floor']:.0%} noise, keeps {res['pos_retained_at_floor']:.0%} answers; "
               f"{res['pos_collides_with_noise']:.0%} of answers collide with the noise ceiling)")
    click.echo(f"  alt: floor_retention={res['floor_retention']} (keep {target:.0%} answers)  "
               f"floor_reject={res['floor_reject']} (reject {reject:.0%} noise)")
    if store:
        db.set_meta(conn, "truncate_floor", str(res["floor"]))
        db.set_meta(conn, "calib_changes", "0")  # floor is now fresh
        click.echo(f"stored truncate_floor={res['floor']} (use: search --truncate calibrated)")


if __name__ == "__main__":
    main()
