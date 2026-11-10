# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

"""Real stdio checks, also runnable with only the installed MCP extra."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import uuid

QUERY = "café 東京"
BODY = "café 東京\n"
LARGE = BODY * 12000
HOOK = r'''
import json, os, socket, sys
from pathlib import Path
intentional = True
log = Path(os.environ['FIDX_TEST_LOG'])
def deny(original):
    def guarded(self, address):
        frame = sys._getframe(1)
        while frame:
            if frame.f_code is getattr(socket.socketpair, '__code__', None):
                return original(self, address)
            frame = frame.f_back
        if self.family in (socket.AF_INET, socket.AF_INET6):
            if not intentional:
                with log.open('a', encoding='utf-8') as out:
                    out.write(json.dumps({'unexpected_connect': True}) + '\n')
            raise PermissionError('fidx-test-network-denied')
        return original(self, address)
    return guarded
socket.socket.connect = deny(socket.socket.connect)
socket.socket.connect_ex = deny(socket.socket.connect_ex)
for address, method in ((address, method) for address in ('127.0.0.1', '192.0.2.1')
                        for method in ('connect', 'connect_ex')):
    with socket.socket() as probe:
        try:
            getattr(probe, method)((address, 9))
        except PermissionError as exc:
            assert str(exc) == 'fidx-test-network-denied'
        else:
            raise AssertionError('network guard absent')
left, right = socket.socketpair()
with left, right:
    left.send(b'x')
    assert right.recv(1) == b'x'
intentional = False
from fidx.config import default_db_path, DEFAULT_PROFILE
from fidx.embedder import FastEmbedder
from fidx.daemon import socket_path_for
args = sys.orig_argv
role = 'client' if '--client' in args else 'unknown'
if '-m' in args and args[args.index('-m') + 1] == 'fidx.cli':
    role = args[args.index('-m') + 4]
identity = dict(nonce=os.environ['FIDX_TEST_NONCE'], pid=os.getpid(),
                ppid=os.getppid(), role=role, db=str(default_db_path().resolve()),
                cache=str(FastEmbedder.cache_path().resolve()), profile=DEFAULT_PROFILE,
                socket=str(socket_path_for(default_db_path()).resolve()), denied=True)
if sys.platform == 'win32':
    import ctypes as c
    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = c.c_void_p
    handle = c.c_void_p(kernel.OpenProcess(0x1400, False, os.getppid()))
    try:
        info, image, size = (c.c_void_p * 6)(), c.create_unicode_buffer(32768), c.c_ulong(32768)
        assert c.WinDLL('ntdll').NtQueryInformationProcess(handle, 0, c.byref(info), c.sizeof(info), None) == 0
        assert kernel.QueryFullProcessImageNameW(handle, 0, image, c.byref(size)), c.get_last_error()
        identity.update(launcher_ppid=info[5], launcher_exe=str(Path(image.value).resolve()))
    finally:
        kernel.CloseHandle(handle)
if default_db_path().stem == 'empty' or role != 'search':
    def forbidden(*args, **kwargs):
        raise AssertionError('model-independent operation constructed embedder')
    FastEmbedder.__init__ = forbidden
if default_db_path().stem == 'fake':
    import fastembed
    import numpy as np
    class FakeTextEmbedding:
        def __init__(self, *, local_files_only, **kwargs):
            assert local_files_only
            self.model = type('Model', (), {'tokenizer': None})()
        def embed(self, texts, **kwargs):
            return iter(np.ones(384, dtype=np.float32) for _ in texts)
    fastembed.TextEmbedding = FakeTextEmbedding
with log.open('a', encoding='utf-8') as out:
    out.write(json.dumps(identity) + '\n')
'''


def run_probe(root, installed):
    root = Path(root).resolve()
    hook = root / 'hook'
    hook.mkdir()
    (hook / 'sitecustomize.py').write_text(HOOK, encoding='utf-8')
    env = {key: os.environ[key] for key in
           ('PATH', 'SYSTEMROOT', 'WINDIR', 'PATHEXT', 'TEMP', 'TMP') if key in os.environ}
    for key in ('HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME',
                'XDG_CACHE_HOME', 'XDG_RUNTIME_DIR', 'FASTEMBED_CACHE_PATH', 'HF_HOME'):
        directory = root / key.lower()
        directory.mkdir()
        env[key] = str(directory)
    home = Path(env['HOME'])
    env.update(USERPROFILE=str(home), APPDATA=str(home / 'AppData' / 'Roaming'),
               LOCALAPPDATA=str(home / 'AppData' / 'Local'), HOMEDRIVE=home.drive,
               HOMEPATH=str(home)[len(home.drive):], FIDX_PROFILE='bge-384',
               FIDX_THREADS='1', FIDX_RERANK='off', HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_XET='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               FIDX_TEST_NONCE=uuid.uuid4().hex, PYTHONPATH=str(hook))
    if not installed:
        env['PYTHONPATH'] += os.pathsep + str(Path(__file__).resolve().parents[1] / 'src')
    for case in ('empty', 'populated', 'fake', 'missing'):
        log = root / (case + '.jsonl')
        env.update(FIDX_DB=str(root / (case + '.db')), FIDX_TEST_LOG=str(log))
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--client',
                                    case, str(int(installed))], cwd=root, env=env,
                                   capture_output=True, timeout=120)
        assert completed.returncode == 0, completed.stderr.decode('utf-8', errors='strict')
        rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
        clients = [row for row in rows if row.get('role') == 'client']
        servers = [row for row in rows if row.get('role') == 'mcp']
        children = [row for row in rows if row.get('role') == 'search']
        assert len(clients) == len(servers) == 1
        assert children if case != 'missing' else not children
        digest = hashlib.sha256(env['FIDX_DB'].encode()).hexdigest()[:16]
        for row in rows:
            assert row.get('nonce') == env['FIDX_TEST_NONCE'] and row['denied']
            assert row['cache'] == env['FASTEMBED_CACHE_PATH'] and row['db'] == env['FIDX_DB']
            assert row['socket'] == str(Path(env['XDG_RUNTIME_DIR']) / 'fidx' / (digest + '.sock'))
            assert row['profile'] == 'bge-384'
            parent = os.getpid() if row['role'] == 'client' else (
                clients[0]['pid'] if row['role'] == 'mcp' else servers[0]['pid'])
            assert row['ppid'] == parent or (sys.platform == 'win32' and
                row['launcher_ppid'] == parent and row['launcher_exe'] == str(Path(sys.executable).resolve()))
            assert not Path(row['socket']).exists()
        for key in ('FASTEMBED_CACHE_PATH', 'HF_HOME'):
            assert not any(p.is_file() for p in Path(env[key]).rglob('*'))
    print('real pipes: empty Unicode search, Unicode get/cap/head, collections, cold missing; fake-only success')


async def client(case, installed):
    import fidx
    from fidx import db, mcp_server
    from importlib.metadata import version, requires
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    assert (1, 30) <= tuple(map(int, version('mcp').split('.')[:2])) < (2, 0)
    assert any('mcp<2,>=1.30' in item.replace(' ', '') for item in requires('fmdidx'))
    if installed:
        for module in (fidx, mcp_server):
            assert any(Path(module.__file__).resolve().is_relative_to(Path(sysconfig.get_path(key)).resolve())
                       for key in ('purelib', 'platlib'))
    path = Path(os.environ['FIDX_DB'])
    if case != 'missing':
        conn = db.connect(path)
        db.init_schema(conn)
        db.set_meta(conn, 'embed_profile', 'bge-384')
        db.ensure_instance_id(conn)
        if case != 'empty':
            for name in ('zeta', 'notes'):
                conn.execute("INSERT INTO collections(name, root, globs) VALUES (?, ?, '[]')", (name, str(path.parent)))
            for relpath, body, digest, active in (('é.md', BODY, 'abcdef', 1),
                                                 ('large.md', LARGE, 'fedcba', 0)):
                conn.execute('INSERT INTO documents(collection,relpath,body,hash,active) VALUES (?,?,?,?,?)',
                             ('notes', relpath, body, digest, active))
        conn.commit()
        conn.close()
    params = StdioServerParameters(command=sys.executable, args=['-m', 'fidx.cli', '--db', str(path), 'mcp'],
                                   env=dict(os.environ), cwd=str(path.parent),
                                   encoding='utf-8', encoding_error_handler='strict')
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert {tool.name for tool in (await session.list_tools()).tools} == {
                'fidx_search', 'fidx_get', 'fidx_collections'}
            async def call(name, arguments, schema, status=None, error=False):
                result = await session.call_tool(name, arguments)
                payload = result.structuredContent
                assert payload == json.loads(result.content[0].text)
                assert payload['schema'] == schema and bool(result.isError) == error, payload
                if status:
                    assert payload['status'] == status, payload
                return payload
            collections = await call('fidx_collections', {},
                                     'fidx.error.v1' if case == 'missing' else 'fidx.collections.v1',
                                     'index_missing' if case == 'missing' else None, case == 'missing')
            if case == 'missing':
                return
            assert [(row['name'], row['doc_count']) for row in collections['collections']] == (
                [] if case == 'empty' else [('notes', 2), ('zeta', 0)])
            if case == 'empty':
                for guidance in (None, False, True):
                    arguments = {'query': QUERY, 'budget_tokens': 4000}
                    if guidance is not None:
                        arguments['guidance'] = guidance
                    payload = await call('fidx_search', arguments, 'fidx.search.v2', 'empty_index')
                    assert payload['query'].encode('utf-8') == QUERY.encode('utf-8')
                    assert payload['request'].get('guidance') is guidance
                    assert ('guidance' in payload['request']) == (guidance is not None)
                    assert 'rerank' not in payload['request']
                    assert payload['summary']['budget']['requested'] == 4000
                return
            for docid, head, expected in (('notes/é.md', False, BODY),
                                         ('#fedcba', False, LARGE.encode()[:65536].decode('utf-8', errors='ignore') + '\n… [truncated]'),
                                         ('#fedcba', True, '\n'.join(LARGE.splitlines()[:40]))):
                payload = await call('fidx_get', {'docid': docid, 'head': head}, 'fidx.get.v1')
                assert payload['body'].encode('utf-8') == expected.encode('utf-8')
                assert payload['truncated'] == (docid == '#fedcba' and not head)
            payload = await call('fidx_search', {'query': QUERY, 'guidance': True},
                                 'fidx.search.v2' if case == 'fake' else 'fidx.error.v1',
                                 'ok' if case == 'fake' else 'model_missing', case != 'fake')
            if case == 'fake':
                assert payload['results'][0]['docid'] == '#abcdef'
                assert payload['diagnostics']['execution']['path'] == 'cold'
            else:
                assert any(action['intent'] == 'fetch_model' and
                           action['command'] == ['fidx', 'models', 'fetch', 'embed', '--profile', 'bge-384']
                           for action in payload['next_actions']), payload


def test_guard_socketpair_and_denied_connects(tmp_path):
    result = subprocess.run([sys.executable, '-c', HOOK.split('from fidx.config')[0]],
                            env={**os.environ, 'FIDX_TEST_LOG': str(tmp_path / 'guard.jsonl')},
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode('utf-8')


def test_real_pipe_offline(tmp_path):
    import pytest
    pytest.importorskip('mcp')
    run_probe(tmp_path, installed=False)


def test_guard_rejects_altered_child_hook(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip('mcp')
    altered = HOOK.replace("json.dumps(identity)",
                           "json.dumps({**identity, 'nonce': 'altered'} if role == 'search' else identity)")
    monkeypatch.setitem(globals(), 'HOOK', altered)
    with pytest.raises(AssertionError):
        run_probe(tmp_path, installed=False)


if __name__ == '__main__':
    if '--client' in sys.argv:
        import anyio
        anyio.run(client, sys.argv[2], bool(int(sys.argv[3])))
    else:
        assert sys.argv[1:] == ['--installed']
        with tempfile.TemporaryDirectory(prefix='fidx-mcp-') as directory:
            run_probe(Path(directory), installed=True)
