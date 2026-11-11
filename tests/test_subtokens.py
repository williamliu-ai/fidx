# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from fidx import cli, config, daemon, db, indexer, search, vector_store
from conftest import TEST_PROFILE
import test_rerank_daemon
from test_rerank_daemon import _live_server
from test_weighted_fusion import test_base_json_fingerprint, indexed  # noqa: F401


@pytest.mark.parametrize('token,parts', [
    ('camelCase', 'camel case'), ('HTTPClient2', 'http client 2'),
    ('parseHTTPResponse', 'parse http response'), ('utf8', 'utf 8'),
    ('v2beta', 'v 2 beta'), ('snake_case', 'snake case'),
    ('ALLCAPS', None), ('x', None), ('größeWert', 'größe wert'),
    ("don't", None), ("user'sName", "user's name"), ('', None),
])
def test_pure(token, parts):
    from fidx.subtokens import subtoken_text, expand_query
    assert subtoken_text(token) == (f'{token} {parts}' if parts else token)
    assert expand_query(token) == (parts or token)


def test_phrase_expansion():
    from fidx.subtokens import expand_query, subtoken_match_expr
    assert expand_query('"getUserName utf8" other') == '"get user name utf 8" other'
    assert subtoken_match_expr('"getUserName utf8" other') == '"get user name utf 8" OR "other"'
    assert subtoken_match_expr('getUserName', conjunctive=True) == '"get" "user" "name"'
    assert subtoken_match_expr('getUserName') == '"get" OR "user" OR "name"'


@pytest.fixture
def identifiers(conn, tmp_path, monkeypatch, embedder):
    root = tmp_path / 'code'
    root.mkdir()
    for name, text in [('snake', 'get_user_name'), ('pascal', 'GetUserName'),
                       ('camel', 'getUserName'), ('other', 'unrelated')]:
        (root / f'{name}.txt').write_text(text)
    indexer.add_collection(conn, 'code', root, ['*.txt'])
    indexer.sync_collection(conn, vector_store.NullStore(), 'code', indexer.IndexStats())
    monkeypatch.setattr(config, 'get_profile', lambda *a, **k: TEST_PROFILE)
    monkeypatch.setattr(cli, 'FastEmbedder', lambda *a, **k: embedder)
    return root


def command(conn, *args):
    path = conn.execute('PRAGMA database_list').fetchone()['file']
    out = CliRunner().invoke(cli.main, ['--db', path, *args])
    assert out.exit_code == 0, (out.output, out.exception)
    return out


def enable(conn):
    command(conn, 'index', '--subtokens', '--no-calibrate')


def paths(conn, query='getUserName', **kwargs):
    return {r.relpath for r in search.search_lexical(conn, query, **kwargs)}


def test_retrieval_and_disable(conn, identifiers):
    before = paths(conn)
    assert before == {'camel.txt', 'pascal.txt'}
    assert not conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'documents_sub%'").fetchall()
    enable(conn)
    assert paths(conn) == {'snake.txt', 'camel.txt', 'pascal.txt'}
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='documents_sub_fts'").fetchone()[0]
    assert 'content=' not in sql and 'porter unicode61' in sql
    assert db.get_meta(conn, 'subtokens') == '1'
    assert db.get_meta(conn, 'subtokens_dirty') is None
    command(conn, 'index', '--no-calibrate')  # absent keeps enabled
    assert paths(conn) == {'snake.txt', 'camel.txt', 'pascal.txt'}
    command(conn, 'index', '--no-subtokens', '--no-calibrate')
    assert paths(conn) == before
    assert not conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'documents_sub%'").fetchall()
    assert not conn.execute("SELECT key FROM meta WHERE key LIKE 'subtokens%'").fetchall()
    enable(conn)
    assert len(paths(conn)) == 3


def test_backfill_all_stored_documents(conn, identifiers, tmp_path):
    root = tmp_path / 'second'
    root.mkdir()
    (root / 'extra.txt').write_text('HTTPClient2')
    indexer.add_collection(conn, 'second', root, ['*.txt'])
    indexer.sync_collection(conn, vector_store.NullStore(), 'second', indexer.IndexStats())
    # The unselected collection's file differs; backfill must use stored body.
    (root / 'extra.txt').write_text('replacement')
    command(conn, 'index', '-c', 'code', '--subtokens', '--no-calibrate')
    assert paths(conn, 'http client 2', collections=['second']) == {'extra.txt'}
    assert paths(conn, collections=['absent']) == set()
    conn.execute("UPDATE documents SET active=0 WHERE relpath='snake.txt'")
    conn.commit()
    assert 'snake.txt' not in paths(conn)
    assert conn.execute('SELECT count(*) FROM documents_sub_fts').fetchone()[0] == 5


def test_update_delete(conn, identifiers):
    enable(conn)
    store = vector_store.NullStore()
    fresh = identifiers / 'new.txt'
    fresh.write_text('uniqueIdentifier42')
    indexer.sync_collection(conn, store, 'code', indexer.IndexStats())
    assert paths(conn, 'unique identifier 42') == {'new.txt'}
    fresh.write_text('replacement text with different length')
    indexer.sync_collection(conn, store, 'code', indexer.IndexStats())
    assert not paths(conn, 'unique identifier 42')
    fresh.write_text('deleteIdentifier99')
    indexer.sync_collection(conn, store, 'code', indexer.IndexStats())
    assert paths(conn, 'delete identifier 99') == {'new.txt'}
    fresh.unlink()
    indexer.sync_collection(conn, store, 'code', indexer.IndexStats())
    assert not paths(conn, 'delete identifier 99')
    assert conn.execute('SELECT count(*) FROM documents_sub_fts').fetchone()[0] == 4
    assert not conn.execute('SELECT rowid FROM documents_sub_fts EXCEPT SELECT id FROM documents').fetchall()
    assert db.get_meta(conn, 'subtokens_dirty') is None


def test_collection_remove(conn, identifiers):
    enable(conn)
    indexer.remove_collection(conn, vector_store.NullStore(), 'code')
    assert not paths(conn)
    assert conn.execute('SELECT count(*) FROM documents_sub_fts').fetchone()[0] == 0
    assert db.get_meta(conn, 'subtokens_dirty') is None


@pytest.mark.parametrize('damage', ['update', 'insert', 'delete', 'missing'])
def test_dirty_guard_and_rebuild(conn, identifiers, embedder, damage):
    enable(conn)
    if damage == 'missing':
        conn.execute('DROP TABLE documents_sub_fts')
    elif damage == 'update':
        conn.execute("UPDATE documents SET body='different' WHERE relpath='snake.txt'")
    elif damage == 'insert':
        conn.execute("INSERT INTO documents(collection,relpath,body,hash) VALUES ('code','raw.txt','rawWriter2','raw')")
    else:
        conn.execute("DELETE FROM documents WHERE relpath='snake.txt'")
    conn.commit()
    if damage != 'missing':
        assert db.get_meta(conn, 'subtokens_dirty') == '1'
    req = {'query': 'getUserName', 'mode': 'lexical'}
    guarded = daemon.run_search(conn, vector_store.NullStore(), embedder, req)
    warning = guarded['diagnostics']['warnings'][0]
    assert warning['component'] == 'subtokens'
    assert 'fidx index --subtokens' in warning['message']
    conn.execute("DELETE FROM meta WHERE key='subtokens'")
    conn.commit()
    off = daemon.run_search(conn, vector_store.NullStore(), embedder, req)
    assert guarded['results'] == off['results']
    conn.execute("INSERT INTO meta VALUES ('subtokens', '1')")
    conn.commit()
    command(conn, 'index', '--no-calibrate')
    # An ordinary sync cannot certify old-writer damage or a missing table.
    assert not db.subtokens_ready(conn)
    enable(conn)
    assert db.subtokens_ready(conn)
    assert db.get_meta(conn, 'subtokens_dirty') is None
    assert 'warnings' not in daemon.run_search(conn, vector_store.NullStore(), embedder, req)['diagnostics']


def test_scores_tiers_and_snippets(conn, identifiers):
    enable(conn)
    query = 'getUserName'
    hits = search.search_lexical(conn, query)
    for hit in hits:
        main = conn.execute('SELECT -bm25(documents_fts,4.0,1.0,2.0) FROM documents_fts WHERE documents_fts MATCH ? AND rowid=?',
                            (search.fts_match_expr(query, True), hit.doc_id)).fetchone()
        sub = conn.execute('SELECT -bm25(documents_sub_fts) FROM documents_sub_fts WHERE documents_sub_fts MATCH ? AND rowid=?',
                           ('"get" "user" "name"', hit.doc_id)).fetchone()
        raw = max(main[0] if main else 0, .5 * sub[0])
        assert hit.score == pytest.approx(raw / (1 + raw), rel=1e-12)
        assert hit.tier == 'and'
        if hit.relpath == 'snake.txt':
            assert hit.snippet == 'get_user_name'  # main body MATCH expanded terms
        else:
            assert hit.snippet in ('GetUserName', 'getUserName')
    tiers = search.search_lexical(conn, 'get user name')
    assert tiers[0].tier == 'phrase'
    assert len({r.doc_id for r in tiers}) == len(tiers)
    assert len(search.search_lexical(conn, query, limit=1)) == 1


@pytest.mark.parametrize('query', ['"', '*', ':', '^', 'NEAR(', '-', 'body_sub:x',
    '"getUserName', 'getUserName" OR unrelated', '"getUserName"*', 'getUserName -unrelated'])
def test_hostile(conn, identifiers, query):
    from fidx.subtokens import subtoken_match_expr
    enable(conn)
    for conjunctive in (True, False):
        # Expression must exactly equal the original safe builder applied to
        # plain, expanded text; all metacharacters remain data.
        from fidx.subtokens import expand_query
        assert subtoken_match_expr(query, conjunctive) == search.fts_match_expr(expand_query(query), conjunctive)
    search.search_lexical(conn, query)
    assert paths(conn, '"getUserName"*') == {'snake.txt', 'pascal.txt', 'camel.txt'}
    assert 'other.txt' not in paths(conn, '"getUserName"*')


@pytest.mark.skipif(os.name != 'posix', reason='daemon uses Unix sockets')
def test_warm_activation(conn, identifiers, tmp_path, monkeypatch):
    path = Path(conn.execute('PRAGMA database_list').fetchone()['file'])
    monkeypatch.setattr(test_rerank_daemon, 'get_profile', lambda: TEST_PROFILE)
    req = {'cmd': 'search', 'query': 'getUserName', 'mode': 'lexical'}
    with _live_server(path, tmp_path, monkeypatch):
        before = daemon.client_request(path, req)
        assert before['ok'] and len(before['results']['results']) == 2
        enable(conn)
        after = daemon.client_request(path, req)
        cold = json.loads(command(conn, 'search', 'getUserName', '--mode', 'lexical', '--json', '--no-daemon').output)
        assert after['ok'] and len(after['results']['results']) == 3
        assert after['results'] == cold
        command(conn, 'index', '--no-subtokens', '--no-calibrate')
        assert len(daemon.client_request(path, req)['results']['results']) == 2


def test_status_and_help(conn, identifiers):
    assert json.loads(command(conn, 'status', '--json').output)['subtokens'] is False
    enable(conn)
    assert json.loads(command(conn, 'status', '--json').output)['subtokens'] is True
    help_text = command(conn, 'index', '--help').output
    assert '--subtokens / --no-subtokens' in help_text
    assert 'experimental' in help_text.lower()


def test_old_writer_during_scan_cannot_be_certified_clean(conn, identifiers, monkeypatch):
    import sqlite3
    enable(conn)
    path = Path(conn.execute('PRAGMA database_list').fetchone()['file'])
    other = db.connect(path, create=False)
    other.execute('PRAGMA busy_timeout=0')
    original = indexer._scan_files
    blocked = []

    def scan(*args, **kwargs):
        try:
            other.execute("UPDATE documents SET body='replacement' WHERE relpath='snake.txt'")
            other.commit()
        except sqlite3.OperationalError as exc:
            assert 'locked' in str(exc)
            other.rollback()
            blocked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(indexer, '_scan_files', scan)
    try:
        indexer.sync_collection(conn, vector_store.NullStore(), 'code', indexer.IndexStats())
        assert blocked == [True]
        other.execute("UPDATE documents SET body='replacement' WHERE relpath='snake.txt'")
        other.commit()
        assert db.get_meta(conn, 'subtokens_dirty') == '1'
        assert paths(conn) == {'camel.txt', 'pascal.txt'}
    finally:
        other.close()


@pytest.mark.parametrize('envelope', [False, True])
def test_concurrent_disable_uses_read_snapshot(conn, identifiers, monkeypatch, embedder, envelope):
    enable(conn)
    path = Path(conn.execute('PRAGMA database_list').fetchone()['file'])
    other = db.connect(path, create=False)
    original = search._fts_query
    disabled = []

    def main_query(*args, **kwargs):
        results = original(*args, **kwargs)
        if not disabled:
            db.configure_subtokens(other, False)
            disabled.append(True)
        return results

    monkeypatch.setattr(search, '_fts_query', main_query)
    try:
        if envelope:
            payload = daemon.run_search(conn, vector_store.NullStore(), embedder,
                                        {'query': 'getUserName', 'mode': 'lexical'})
            assert len(payload['results']) == 3
            assert 'warnings' not in payload['diagnostics']
        else:
            assert len(paths(conn)) == 3
        assert not conn.in_transaction
        assert paths(conn) == {'camel.txt', 'pascal.txt'}
    finally:
        other.close()


def test_failed_sub_sync_rolls_back_documents_and_marker(conn, identifiers, monkeypatch):
    enable(conn)
    (identifiers / 'new.txt').write_text('newIdentifier')

    def fail(body):
        raise RuntimeError('derived write failed')

    monkeypatch.setattr(indexer, 'subtoken_text', fail)
    with pytest.raises(RuntimeError, match='derived write failed'):
        indexer.sync_collection(conn, vector_store.NullStore(), 'code', indexer.IndexStats())
    assert not conn.in_transaction
    assert conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 4
    assert conn.execute('SELECT count(*) FROM documents_sub_fts').fetchone()[0] == 4
    assert db.get_meta(conn, 'subtokens_dirty') is None


def test_sub_only_tier_precedence_and_empty_snippet(conn, identifiers):
    for name, body in [('phrase', 'alphaBeta zeta'), ('and', 'zeta alphaBeta'),
                       ('or', 'alphaBeta')]:
        (identifiers / f'{name}.txt').write_text(body)
    enable(conn)
    hits = search.search_lexical(conn, 'alpha beta zeta')
    assert [(r.relpath, r.tier) for r in hits] == [
        ('phrase.txt', 'phrase'), ('and.txt', 'and'), ('or.txt', 'or')]
    # Main MATCH cannot split the camel identifier, so phrase/AND snippets
    # are unavailable; exposing generated text would misrepresent the body.
    assert hits[0].snippet == hits[1].snippet == ''
    assert [r.relpath for r in search.search_lexical(conn, '"alphaBeta zeta"')] == ['phrase.txt']
    assert [r.relpath for r in search.search_lexical(conn, 'alpha beta zeta', limit=1)] == ['phrase.txt']
