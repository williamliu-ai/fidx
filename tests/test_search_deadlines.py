# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Search deadlines interrupt SQL and fail closed at the CLI boundary."""
import json
import sqlite3
import time

import pytest
from click.testing import CliRunner

from fidx import daemon, db, errors, vector_store
from fidx.cli import main


def test_cold_sql_deadline(conn, monkeypatch):
    def slow(*args, **kwargs):
        conn.create_function('pause', 1, lambda n: (time.sleep(.002), n)[1])
        conn.execute('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000) SELECT sum(pause(x)) FROM n').fetchone()
    monkeypatch.setattr(daemon, '_run_search', slow)
    started = time.monotonic()
    with pytest.raises(Exception) as caught:
        daemon.run_search(conn, vector_store.NullStore(), None, {'query': 'slow', 'timeout': .03})
    assert type(caught.value).__name__ == 'SearchTimeout'
    assert time.monotonic() - started < .3
    assert conn.execute('SELECT 1').fetchone()[0] == 1


@pytest.mark.parametrize('code', ['timeout', 'busy', 'error'])
@pytest.mark.parametrize('flags', [[], ['--guidance']])
def test_cli_does_not_repeat_daemon_failure(tmp_path, monkeypatch, code, flags):
    path = tmp_path / 'index.db'
    conn = db.connect(path); db.init_schema(conn); conn.close()
    response = {'ok': False, 'error_code': code, 'error': 'daemon ' + code}
    monkeypatch.setattr(daemon, 'client_request', lambda *a, **k: response)
    monkeypatch.setattr(daemon, 'client_request_outcome', lambda *a, **k: daemon.ClientOutcome('cold', fallback_cause='daemon_error', daemon_error='daemon '+code, response=response))
    calls = []
    monkeypatch.setattr(daemon, 'run_search', lambda *a, **k: calls.append(True))
    result = CliRunner().invoke(main, ['--db', str(path), 'search', 'alpha', '--mode', 'lexical', '--json', *flags])
    assert not calls
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload['schema'] == 'fidx.error.v1'
    assert code in payload['message']


def test_timeout_option_overrides_environment(tmp_path, monkeypatch):
    path = tmp_path / 'index.db'
    conn = db.connect(path); db.init_schema(conn); conn.close()
    monkeypatch.setenv('FIDX_SEARCH_TIMEOUT', '.02')
    seen = []
    def search(*args, **kwargs):
        seen.append(args[3]['timeout'])
        raise errors.FidxError('observed')
    monkeypatch.setattr(daemon, 'run_search', search)
    result = CliRunner().invoke(main, ['--db', str(path), 'search', 'alpha', '--no-daemon', '--mode', 'lexical', '--timeout', '.05'])
    assert seen == [.05], result.output


def test_embedding_deadline_checked_before_and_after(conn, monkeypatch):
    from fidx import search
    calls = []
    class SlowEmbedder:
        def embed_queries(self, texts):
            calls.append(texts)
            time.sleep(.04)
            return [[0.0]]
    monkeypatch.setattr(daemon, '_run_search', lambda *a, **k: search._embed_query(SlowEmbedder(), 'alpha'))
    with pytest.raises(Exception) as caught:
        daemon.run_search(conn, vector_store.NullStore(), None, {'query': 'alpha', 'timeout': .01})
    assert type(caught.value).__name__ == 'SearchTimeout'
    assert len(calls) == 1
    # Already-expired admission must never start an embedding.
    with pytest.raises(errors.SearchTimeout):
        daemon.run_search(conn, vector_store.NullStore(), None, {'query': 'alpha', '_deadline': time.monotonic() - 1})
    assert len(calls) == 1
