# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

from concurrent.futures import Future
from dataclasses import asdict
import threading

import pytest

from fidx import indexer, search
from fidx.embedder import FastEmbedder


@pytest.fixture()
def populated(conn, store, embedder, tmp_path):
    root = tmp_path / 'corpus'
    root.mkdir()
    for i in range(12):
        body = ('before ' * 30 + 'alpha beta ' + 'after ' * 30 if i % 3 == 0
                else 'alpha distant beta' if i % 3 == 1 else 'alphaBeta gamma')
        (root / f'{i:02d}.txt').write_text(body)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, 'notes', root)
    indexer.sync_collection(conn, store, 'notes', stats)
    indexer.embed_pending(conn, store, embedder, stats)
    return conn


@pytest.mark.parametrize('mode', ['lexical', 'hybrid'])
@pytest.mark.parametrize('query', ['alpha beta', 'alpha missing', 'alphaBeta', 'beta gamma'])
def test_snippets_match_eager_fts_and_only_fetch_returned_rows(
        populated, store, embedder, monkeypatch, mode, query):
    conn = populated
    original = search._fts_query
    original_tier = search._lexical_tier

    def eager(conn, expr, collections, limit, tier):
        clause, params = search._collection_clause(collections)
        rows = conn.execute(f'''
            SELECT d.id, d.collection, d.relpath, d.title, d.hash,
                   bm25(documents_fts, 4.0, 1.0, 2.0) AS rank,
                   snippet(documents_fts, 1, '', '', '…', {search.SNIPPET_TOKENS}) AS snip
            FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ? AND d.active = 1{clause}
            ORDER BY rank, d.collection, d.relpath LIMIT ?
        ''', [expr, *params, limit]).fetchall()
        return [search.Result(r['id'], r['collection'], r['relpath'], r['title'],
                              r['hash'][:6], -r['rank'] / (1 - r['rank']),
                              snippet=r['snip'], tier=tier) for r in rows]

    def eager_tier(conn, expr, sub_expr, collections, limit, tier):
        main = eager(conn, expr, collections, limit, tier)
        if sub_expr is None:
            return main
        candidates = {r.doc_id: r for r in main}
        for sub in search._sub_fts_query(conn, sub_expr, collections, limit, tier):
            if sub.doc_id in candidates:
                result = candidates[sub.doc_id]
                result.score = max(result.score, sub.score)
            else:
                row = conn.execute(
                    f"SELECT snippet(documents_fts, 1, '', '', '…', {search.SNIPPET_TOKENS}) "
                    "FROM documents_fts WHERE rowid = ? AND documents_fts MATCH ?",
                    (sub.doc_id, sub_expr)).fetchone()
                sub.snippet = row[0] if row else ""
                candidates[sub.doc_id] = sub
        return sorted(candidates.values(), key=lambda r: (-r.score, r.collection, r.relpath))

    def run():
        if mode == 'lexical':
            return search.search_lexical(conn, query, limit=2)
        return search.search_hybrid(conn, store, embedder, query, limit=2)

    monkeypatch.setattr(search, '_fts_query', eager)
    monkeypatch.setattr(search, '_lexical_tier', eager_tier)
    expected = run()
    monkeypatch.setattr(search, '_fts_query', original)
    monkeypatch.setattr(search, '_lexical_tier', original_tier)
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        actual = run()
    finally:
        conn.set_trace_callback(None)
    assert [asdict(r) for r in actual] == [asdict(r) for r in expected]
    snippets = [s for s in statements if 'snippet(' in s.lower()]
    assert snippets
    returned = {r.doc_id for r in actual}
    for sql in snippets:
        # Every snippet statement must target a single returned document.
        assert any(f'WHERE rowid = {doc_id} AND' in sql for doc_id in returned)
        assert 'ORDER BY' not in sql


class InlinePool:
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def test_hybrid_overlaps_embedding_and_reuses_worker(populated, store, embedder, monkeypatch):
    pool = search._QUERY_POOL
    workers = []
    started = threading.Event()
    original_lex = search.search_lexical
    caller = threading.get_ident()

    class RecordingEmbedder:
        def embed_queries(self, texts):
            workers.append(threading.get_ident())
            started.set()
            return embedder.embed_queries(texts)

    def checked_lex(*args, **kwargs):
        assert threading.get_ident() == caller
        assert started.wait(2), 'embedding did not start before lexical search'
        return original_lex(*args, **kwargs)

    monkeypatch.setattr(search, 'search_lexical', checked_lex)
    concurrent = [search.search_hybrid(populated, store, RecordingEmbedder(), 'alpha beta')
                  for _ in range(2)]
    assert workers[0] == workers[1] != caller
    assert search._QUERY_POOL is pool
    monkeypatch.setattr(search, '_QUERY_POOL', InlinePool())
    serial = search.search_hybrid(populated, store, RecordingEmbedder(), 'alpha beta')
    assert concurrent[0] == concurrent[1] == serial


@pytest.mark.parametrize(('cpu', 'env', 'explicit', 'expected'), [
    (32, None, None, 8), (4, None, None, 4), (None, None, None, 1),
    (32, '3', None, 3), (32, '3', 2, 2), (32, None, 2, 2),
])
def test_query_thread_resolution(monkeypatch, embedder, cpu, env, explicit, expected):
    monkeypatch.setattr('os.cpu_count', lambda: cpu)
    monkeypatch.delenv('FIDX_THREADS', raising=False)
    if env is not None:
        monkeypatch.setenv('FIDX_THREADS', env)
    model = FastEmbedder(embedder.profile, threads=explicit)
    assert model._query_threads == expected
    assert model._threads == (explicit if explicit is not None else int(env) if env else None)


def test_query_session_options_leave_indexing_unchanged(monkeypatch, embedder):
    import fastembed

    monkeypatch.delenv('FIDX_THREADS', raising=False)
    monkeypatch.setattr('os.cpu_count', lambda: 32)
    calls = []
    monkeypatch.setattr(fastembed.TextEmbedding, '__init__', lambda self, **kw: calls.append(kw))
    model = FastEmbedder(embedder.profile)
    model._build_model()
    model._build_model(query=True)
    assert calls[0]['threads'] is None
    assert 'inter_op_num_threads' not in calls[0]
    assert calls[1]['threads'] == 8
    assert calls[1]['enable_cpu_mem_arena'] is False


def test_default_query_backend_sets_actual_onnx_inter_op_threads(monkeypatch, embedder):
    import fastembed
    import onnxruntime
    from fastembed.text.onnx_embedding import OnnxTextEmbedding

    monkeypatch.delenv('FIDX_THREADS', raising=False)
    registries = []

    def capture(self, **kwargs):
        registries.append(self.EMBEDDINGS_REGISTRY)

    monkeypatch.setattr(fastembed.TextEmbedding, '__init__', capture)
    model = FastEmbedder(embedder.profile)
    model._build_model(query=True)
    backend = next(cls for cls in registries[0] if issubclass(cls, OnnxTextEmbedding))
    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 8
    backend.add_extra_session_options(options, {'enable_cpu_mem_arena': False})
    assert options.inter_op_num_threads == 1
    assert options.enable_cpu_mem_arena is False
