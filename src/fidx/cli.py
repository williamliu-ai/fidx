"""fidx command-line interface."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from . import __version__, config, daemon, db, indexer, vector_store
from .embedder import FastEmbedder


def open_index(db_path: Path, profile_name: str | None = None):
    """Open (or create) an index, returning (conn, profile).

    Once embeddings exist, the index pins its profile in meta; a conflicting
    --profile is an error rather than a silent dim mismatch.
    """
    conn = db.connect(db_path)
    db.init_schema(conn)
    stored = db.get_meta(conn, "embed_profile")
    if stored is not None and profile_name and profile_name != stored:
        raise click.ClickException(
            f"index {db_path} is pinned to profile {stored!r}; "
            f"re-index into a fresh --db to use {profile_name!r}"
        )
    profile = config.get_profile(stored or profile_name)
    return conn, profile


@click.group()
@click.version_option(__version__)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="Index database path (default: $FIDX_DB or ~/.cache/fidx/index.db).")
@click.pass_context
def main(ctx: click.Context, db_path: Path | None) -> None:
    """fidx — fast local semantic search for markdown, text, chat and code."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = (db_path or config.default_db_path()).expanduser()


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output.")
def doctor(as_json: bool) -> None:
    """Diagnose host capabilities (sqlite extensions, FTS5, sqlite-vec, model cache).

    Safe to run anywhere: never imports native deps eagerly, so it diagnoses (not
    crashes) when sqlite-vec/onnxruntime are missing or built for another arch.
    Exits non-zero if a hard capability fails.
    """
    from . import diagnostics
    checks = diagnostics.capabilities()
    if as_json:
        click.echo(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        for c in checks:
            click.echo(f"[{'OK  ' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
            if not c.ok and c.remediation:
                click.echo(f"        -> {c.remediation}")
    hard = diagnostics.hard_failures(checks)
    if hard:
        if not as_json:
            click.echo(f"\n{len(hard)} hard check(s) failed — fidx cannot run on this host.",
                       err=True)
        raise SystemExit(1)
    if not as_json:
        click.echo("\nAll hard checks passed — fidx is ready.")


@main.group()
def collection() -> None:
    """Manage indexed collections (named directory trees)."""


@collection.command("add")
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--name", required=True, help="Collection name (used for scoping searches).")
@click.option("--glob", "globs", multiple=True, help="Glob pattern(s) relative to root; repeatable. "
              f"Default: {' '.join(config.DEFAULT_GLOBS)}")
@click.pass_context
def collection_add(ctx: click.Context, root: Path, name: str, globs: tuple[str, ...]) -> None:
    """Register ROOT as a collection. Run `fidx index` afterwards."""
    conn, _ = open_index(ctx.obj["db_path"])
    indexer.add_collection(conn, name, root, list(globs) or None)
    click.echo(f"collection {name!r} -> {root}")


@collection.command("list")
@click.pass_context
def collection_list(ctx: click.Context) -> None:
    conn, _ = open_index(ctx.obj["db_path"])
    for row in conn.execute("SELECT name, root, globs FROM collections ORDER BY name"):
        count = conn.execute("SELECT count(*) AS n FROM documents WHERE collection = ?",
                             (row["name"],)).fetchone()["n"]
        click.echo(f"{row['name']}\t{row['root']}\t{count} docs\t{json.loads(row['globs'])}")


@collection.command("remove")
@click.argument("name")
@click.pass_context
def collection_remove(ctx: click.Context, name: str) -> None:
    conn, _ = open_index(ctx.obj["db_path"])
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
@click.pass_context
def index(ctx: click.Context, only: str | None, profile: str | None, threads: int | None,
          parallel: int | None, backend: str | None, do_calibrate: bool,
          recalibrate_after: float) -> None:
    """Scan collections and (re)build chunks + embeddings incrementally."""
    db_path = ctx.obj["db_path"]
    conn, prof = open_index(db_path, profile)
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
    click.echo(
        f"indexed in {elapsed:.1f}s: +{stats.added} added, ~{stats.updated} updated, "
        f"-{stats.removed} removed, {stats.unchanged} unchanged, "
        f"{stats.embedded_chunks} chunks embedded [{prof.name}/{store.name}]"
    )
    for err in stats.errors:
        click.echo(f"warning: {err}", err=True)

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
            click.echo(f"calibration skipped (+{changed} changed, {pending}/{threshold} "
                       f"since last; floor={db.get_meta(conn, 'truncate_floor')})")
        else:
            cal = calibratemod.calibrate(conn, store, FastEmbedder(prof, threads=threads), seed=0)
            if cal.get("n_pos"):
                db.set_meta(conn, "truncate_floor", str(cal["floor"]))
                db.set_meta(conn, "calib_changes", "0")
                click.echo(
                    f"calibrated truncate_floor={cal['floor']} "
                    f"(rejects {cal['neg_rejected_at_floor']:.0%} noise, "
                    f"keeps {cal['pos_retained_at_floor']:.0%} answers; "
                    f"search --truncate calibrated)")
            else:
                # too few/no samples (e.g. emptied corpus): keep accumulating the
                # change count so it isn't lost, and retry next index.
                db.set_meta(conn, "calib_changes", str(pending))
                click.echo("calibration deferred (insufficient samples)")


@main.command()
@click.argument("query")
@click.option("-c", "--collection", "collections", multiple=True,
              help="Restrict to collection(s); repeatable.")
@click.option("-n", "--limit", default=10, show_default=True)
@click.option("--mode", type=click.Choice(["hybrid", "lexical", "vector"]), default="hybrid",
              show_default=True, help="hybrid = BM25 + vector with RRF fusion.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output for agents.")
@click.option("--files", "files_only", is_flag=True, help="Print matching paths only.")
@click.option("--min-score", type=float, default=None)
@click.option("--truncate", default=None,
              help="Deterministic tail truncation, e.g. ratio:0.5, gap:0.5, knee, "
                   "mad:3, source:0.4,0.2. See fidx/truncate.py.")
@click.option("--no-daemon", is_flag=True, help="Skip the warm daemon even if running.")
@click.pass_context
def search(ctx: click.Context, query: str, collections: tuple[str, ...], limit: int,
           mode: str, as_json: bool, files_only: bool, min_score: float | None,
           truncate: str | None, no_daemon: bool) -> None:
    """Search the index. Default mode is hybrid (best recall)."""
    db_path: Path = ctx.obj["db_path"]
    req = {"cmd": "search", "query": query, "mode": mode,
           "collections": list(collections), "limit": limit, "min_score": min_score,
           "truncate": truncate}

    results: list[dict] | None = None
    if not no_daemon:
        resp = daemon.client_request(db_path, req)
        if resp is not None and resp.get("ok"):
            results = resp["results"]
    if results is None:
        conn, prof = open_index(db_path)
        store = vector_store.open_store(conn, db_path)
        embedder = FastEmbedder(prof) if mode in ("vector", "hybrid") else None
        try:
            raw = daemon.run_search(conn, store, embedder, req)
        except ValueError as exc:
            raise click.ClickException(str(exc))
        results = raw

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False))
    elif files_only:
        for r in results:
            click.echo(r["path"])
    else:
        if not results:
            click.echo("no results")
        for r in results:
            click.echo(f"{r['score']:.4f}  {r['docid']}  {r['path']}  — {r['title']}")
            if r["snippet"]:
                click.echo(f"        {r['snippet'][:160]}")


@main.command()
@click.argument("target")
@click.option("--full/--head", default=True, help="--head prints the first 40 lines.")
@click.pass_context
def get(ctx: click.Context, target: str, full: bool) -> None:
    """Print a document by path (collection/relpath) or docid (#abc123)."""
    conn, _ = open_index(ctx.obj["db_path"])
    if target.startswith("#"):
        row = conn.execute("SELECT body, collection, relpath FROM documents WHERE hash LIKE ?",
                           (target[1:] + "%",)).fetchone()
    else:
        coll, _, rel = target.partition("/")
        row = conn.execute("SELECT body, collection, relpath FROM documents "
                           "WHERE collection = ? AND relpath = ?", (coll, rel)).fetchone()
    if row is None:
        raise click.ClickException(f"document {target!r} not found")
    body = row["body"]
    if not full:
        body = "\n".join(body.splitlines()[:40])
    click.echo(body)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Index health: collections, documents, chunks, vectors, size."""
    db_path: Path = ctx.obj["db_path"]
    conn, prof = open_index(db_path)
    docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
    chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    store = vector_store.open_store(conn, db_path)
    vecs = store.count()
    size_mb = db_path.stat().st_size / 1e6 if db_path.exists() else 0
    sidecar = vector_store.sidecar_path(db_path)
    if store.name == "duckdb" and sidecar.exists():
        size_mb += sidecar.stat().st_size / 1e6
    click.echo(f"db: {db_path} ({size_mb:.1f} MB)")
    click.echo(f"profile: {prof.name} ({prof.model}, {prof.dim}d)  backend: {store.name}")
    click.echo(f"documents: {docs}  chunks: {chunks}  vectors: {vecs}")
    daemon_state = "running" if daemon.client_request(db_path, {"cmd": "ping"}) else "not running"
    click.echo(f"daemon: {daemon_state}")
    for row in conn.execute("SELECT collection, count(*) AS n FROM documents GROUP BY collection"):
        click.echo(f"  {row['collection']}: {row['n']} docs")


@main.command()
@click.option("--threads", type=int, default=None,
              help="ONNX intra-op threads for query embedding (also $FIDX_THREADS).")
@click.pass_context
def serve(ctx: click.Context, threads: int | None) -> None:
    """Run the warm-search daemon (foreground). Searches then take ~ms."""
    conn, prof = open_index(ctx.obj["db_path"])
    store = vector_store.open_store(conn, ctx.obj["db_path"])
    daemon.serve(ctx.obj["db_path"], conn, store, FastEmbedder(prof, threads=threads))


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
    conn, prof = open_index(db_path)
    vstore = vector_store.open_store(conn, db_path)
    res = calibratemod.calibrate(conn, vstore, FastEmbedder(prof), sample=sample,
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
