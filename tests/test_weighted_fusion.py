# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import hashlib
import json
import os
from dataclasses import asdict

import pytest
from click.testing import CliRunner

from fidx import cli, config, daemon, db, embedder as embeddermod, freshness, indexer, search
from conftest import TEST_PROFILE
from test_daemon_outcome import _request as socket_response
import test_rerank_daemon
from test_rerank_daemon import _live_server


def result(id, score, path=None, **kwargs):
    return search.Result(id, "notes", path or f"{id}.md", "Title", f"{id:06}", score, **kwargs)


@pytest.mark.parametrize("spec", ["bad", "weighted:", "weighted:1", "weighted:1,2,3",
    "weighted:-1,1", "weighted:0,0", "weighted:nan,1", "weighted:inf,1", "weighted:x,1"])
def test_parse_rejects(spec):
    with pytest.raises(ValueError):
        search.parse_fusion(spec)


@pytest.mark.parametrize("spec,expected", [
    ("rrf", "rrf"), ("weighted:2,2", "weighted:0.5,0.5"),
    ("weighted:0,1", "weighted:0,1"), ("weighted:1,0", "weighted:1,0"),
    ("weighted:1e308,1e308", "weighted:0.5,0.5"),
    ("weighted:-0,1", "weighted:0,1"),
    ("weighted:1,2", "weighted:0.3333333333333333,0.6666666666666667"),
])
def test_parse_normalizes(spec, expected):
    # Canonical spelling uses Python's shortest round-trip float repr, with
    # integer endpoints written as 0/1; parsing canonical weights is idempotent.
    assert search.parse_fusion(spec) == expected
    assert search.parse_fusion(expected) == expected


def test_minmax_missing_and_metadata():
    lex = [result(1, 8, snippet="lex"), result(2, 4), result(3, 0)]
    vec = [result(2, 0.5, snippet="vec", matched_chunk_id=7, line_start=2, line_end=4),
           result(4, -0.5)]
    lists = [("lexical", 2, lex), ("vector", 1, vec)]
    fused = search.weighted_fuse(lists, (1, 1), 10)
    scores = {r.doc_id: r.score for r in fused}
    assert scores == {1: 0.5, 2: 0.75, 3: 0, 4: 0}
    original = {r.doc_id: asdict(r) for r in search.rrf_fuse(lists, 10)}
    for r in fused:
        fields = asdict(r)
        fields.pop("score")
        expected = original[r.doc_id]
        expected.pop("score")
        assert fields == expected
    assert lex[0].score == 8


@pytest.mark.parametrize("scores", [[0.4], [0.4, 0.4]])
def test_zero_range(scores):
    lists = [("lexical", 1, [result(i, s) for i, s in enumerate(scores)]),
             ("vector", 1, [result(99, 0.7)])]
    fused = search.weighted_fuse(lists, (3, 1), 10)
    assert {r.doc_id: r.score for r in fused} == {
        **{i: 0.75 for i in range(len(scores))}, 99: 0.25}


def test_ties_use_rrf_before_key_then_key():
    lists = [("lexical", 1, [result(1, 7, "z.md"), result(2, 7, "a.md")]),
             ("vector", 1, [result(3, 7, "b.md"), result(4, 7, "c.md")])]
    # All weighted scores equal; rank-1 z beats rank-2 a despite path order.
    assert [r.relpath for r in search.weighted_fuse(lists, (1, 1), 10)] == [
        "b.md", "z.md", "a.md", "c.md"]


@pytest.fixture
def indexed(conn, store, embedder, corpus_dir, monkeypatch):
    monkeypatch.setattr(embeddermod, "hash", lambda token: int.from_bytes(
        hashlib.sha256(token.encode()).digest()[:8], "big"), raising=False)
    stats = indexer.IndexStats()
    indexer.add_collection(conn, "notes", corpus_dir)
    indexer.sync_collection(conn, store, "notes", stats)
    indexer.embed_pending(conn, store, embedder, stats)
    db.set_meta(conn, "truncate_floor", "0.12")
    conn.commit()
    monkeypatch.setattr(cli, "FastEmbedder", lambda *a, **k: embedder)
    monkeypatch.setattr(config, "get_profile", lambda *a, **k: TEST_PROFILE)
    monkeypatch.setattr(freshness, "index_summary", lambda conn: None)
    return conn


def run_cli(conn, *options):
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    return CliRunner().invoke(cli.main, ["--db", path, "search", "carbonara", "--json", *options])


@pytest.mark.parametrize("fusion,arm", [("weighted:1,0", "lexical"), ("weighted:0,1", "vector")])
def test_real_index_endpoint_and_trace(indexed, store, embedder, fusion, arm):
    trace = search.SearchTrace()
    fused = search.search_hybrid(indexed, store, embedder, "carbonara", limit=10,
                                 fusion=fusion, trace=trace)
    source = getattr(trace, arm)
    assert [r.doc_id for r in fused if r.doc_id in {s.doc_id for s in source}] == [
        s.doc_id for s in source]
    assert fused == list(trace.fused_pool)
    assert fused[0].relpath == "recipes.md"


@pytest.mark.parametrize("options", [["--mode", "lexical"], ["--mode", "vector"],
                                      ["--truncate", "calibrated"]])
def test_cli_conflicts(indexed, options):
    out = run_cli(indexed, "--fusion", "weighted:1,1", *options)
    assert out.exit_code == 2
    assert "--fusion" in out.output
    assert options[0] in out.output


@pytest.mark.parametrize("spec", ["bad", "weighted:", "weighted:1", "weighted:1,2,3",
    "weighted:-1,1", "weighted:0,0", "weighted:nan,1", "weighted:inf,1"])
def test_cli_bad_parameter(indexed, spec):
    out = run_cli(indexed, "--fusion", spec)
    assert out.exit_code == 2
    assert "Invalid value for --fusion" in out.output


@pytest.mark.parametrize("mode", ["hybrid", "lexical", "vector"])
def test_default_equals_explicit_rrf(indexed, mode):
    ordinary = run_cli(indexed, "--no-daemon", "--mode", mode)
    explicit = run_cli(indexed, "--no-daemon", "--mode", mode, "--fusion", "rrf")
    assert ordinary.exit_code == explicit.exit_code == 0
    assert ordinary.output == explicit.output
    assert "fusion" not in json.loads(ordinary.output)["request"]


def test_advice_and_numeric_truncation(indexed, store, embedder):
    req = {"query": "carbonara", "fusion": "weighted:2,2", "guidance": True}
    trace = search.SearchTrace()
    out = daemon.run_search(indexed, store, embedder, req, trace=trace)
    assert out["request"]["fusion"] == "weighted:0.5,0.5"
    assert out["diagnostics"]["calibration"]["floor_applicable"] is False
    commands = [a["command"] for a in out["next_actions"] if "command" in a]
    options = out["summary"]["truncation_advice"]["options"]
    commands += [o["command"] for o in options if "command" in o]
    for cmd in commands:
        if cmd[:2] == ["fidx", "search"]:
            assert cmd[cmd.index("--fusion") + 1] == "weighted:0.5,0.5"
            assert "calibrated" not in cmd
            assert "--mode" not in cmd
    calibrated = next(o for o in options if o["truncate"] == "calibrated")
    assert not calibrated["applicable"] and not calibrated["recommended"]
    assert "weighted" in calibrated["reason"]
    filtered = daemon.run_search(indexed, store, embedder, {**req, "truncate": "abs:0.8"})
    assert all(r["score"] >= 0.8 for r in filtered["results"])
    assert len(filtered["results"]) < len(out["results"])


@pytest.mark.parametrize("extra", [{"mode": "lexical"}, {"mode": "vector"},
                                   {"truncate": "calibrated"}])
def test_daemon_rejects_conflicts(indexed, store, embedder, extra):
    with pytest.raises(ValueError, match="fusion"):
        daemon.run_search(indexed, store, embedder,
                          {"query": "carbonara", "fusion": "weighted:1,1", **extra})


@pytest.mark.skipif(os.name != "posix", reason="daemon uses Unix sockets")
def test_real_daemon_parity(indexed, tmp_path, monkeypatch):
    path = daemon.Path(indexed.execute("PRAGMA database_list").fetchone()["file"])
    instance = db.get_meta(indexed, "instance_id")
    monkeypatch.setattr(test_rerank_daemon, "get_profile", lambda: TEST_PROFILE)
    with _live_server(path, tmp_path, monkeypatch):
        for fusion in (None, "rrf", "weighted:0.5,0.5"):
            req = {"cmd": "search", "query": "carbonara"}
            if fusion:
                req["fusion"] = fusion
            response = daemon.client_request_outcome(path, req, expected_instance_id=instance)
            assert response.path == "daemon", response
            warm = response.response
            if fusion and fusion != "rrf":
                assert warm["fusion_effective"] == fusion
            else:
                assert "fusion_effective" not in warm
            options = ["--fusion", fusion] if fusion else []
            cold = run_cli(indexed, "--no-daemon", *options)
            assert cold.exit_code == 0, cold.output
            assert warm["results"] == json.loads(cold.output)
            via_cli = run_cli(indexed, *options)
            assert via_cli.output == cold.output


@pytest.mark.parametrize("ack", [None, "rrf", "weighted:1,0"])
def test_old_daemon_falls_back(indexed, tmp_path, monkeypatch, ack):
    cold = run_cli(indexed, "--no-daemon", "--fusion", "weighted:1,0")
    assert cold.exit_code == 0, cold.output
    payload = json.loads(cold.output)
    # Poison scores so an accepted old answer cannot accidentally pass parity.
    remote = json.loads(cold.output)
    remote["results"][0]["score"] = -99
    response = {"ok": True, "instance_id": db.get_meta(indexed, "instance_id"), "results": remote}
    if ack is not None:
        response["fusion_effective"] = ack
    socket_response(monkeypatch, tmp_path, response)
    actual = run_cli(indexed, "--fusion", "weighted:1,0")
    assert actual.exit_code == 0, actual.output
    if ack == "weighted:1,0":
        assert json.loads(actual.output)["results"] == remote["results"]
    else:
        assert json.loads(actual.output)["results"] == payload["results"]


def test_legacy_index_stays_cold(indexed, monkeypatch):
    indexed.execute("DELETE FROM meta WHERE key = 'instance_id'")
    indexed.commit()
    monkeypatch.setattr(daemon, "client_request_outcome", lambda *a, **k: pytest.fail("must stay cold"))
    assert run_cli(indexed, "--fusion", "weighted:1,0").exit_code == 0


@pytest.mark.parametrize("spec", ["weighted:0.5211480915483105,0.47885190845168946",
                                   "weighted:3,11", "weighted:1e-200,1e200"])
def test_canonical_is_stable(spec):
    canonical = search.parse_fusion(spec)
    assert search.parse_fusion(canonical) == canonical


@pytest.mark.parametrize("extra", [{"min_score": 2}, {"truncate": "abs:2"},
                                   {"collections": ["absent"]}])
def test_no_result_followups_preserve_fusion(indexed, store, embedder, extra):
    payload = daemon.run_search(indexed, store, embedder, {
        "query": "carbonara", "fusion": "weighted:1,0", **extra})
    assert payload["results"] == []
    commands = [a["command"] for a in payload["next_actions"]
                if a["command"][:2] == ["fidx", "search"]]
    assert commands
    assert all("--mode" not in c and "--fusion" in c and "calibrated" not in c for c in commands)


@pytest.mark.parametrize("mode", ["hybrid", "lexical", "vector"])
def test_base_json_fingerprint(indexed, mode):
    ordinary = run_cli(indexed, "--no-daemon", "--mode", mode)
    assert ordinary.exit_code == 0, ordinary.output
    # Captured from the public base before implementing fusion; fixture omits
    # only the volatile index freshness summary, for portable byte comparison.
    expected = {
        "hybrid": "242ebc0471f3c28a4f60cbb5a5eda9493c313eba847e6debb003322e0adb8623",  # pragma: allowlist secret
        "lexical": "39dd6dd53e161d436f871f5272282193e6c03afa14dfe8ead250039333d2615c",  # pragma: allowlist secret
        "vector": "984a8370c1c4c4f66cd4cc51b1ba279df037eb4b139ae5d6d6a6099a9f2f191e",  # pragma: allowlist secret
    }
    assert hashlib.sha256(ordinary.output.encode()).hexdigest() == expected[mode]


@pytest.mark.parametrize("lex,vec,expected", [
    ([result(1, 1), result(2, 0)], [result(3, 1)], [1, 3, 2]),
    ([result(1, .1, tier="phrase"), result(2, .8, tier="and")],
     [result(3, 1)], [2, 1, 3]),
    ([result(1, 1), result(2, 1)], [result(2, 1)], [2, 1]),
])
def test_endpoint_obeys_score_and_rrf_rules(lex, vec, expected):
    # A present minimum ties an absent source at zero; lexical tiers and equal
    # source scores also need not retain lexical list order under score fusion.
    lists = [("lexical", 2, lex), ("vector", 1, vec)]
    assert [r.doc_id for r in search.weighted_fuse(lists, (1, 0), 10)] == expected


@pytest.mark.parametrize("flags", [["--explain"], ["--guidance", "--budget-tokens", "0"]])
def test_weighted_explain_and_budget(indexed, flags):
    outcome = run_cli(indexed, "--no-daemon", "--fusion", "weighted:2,2", *flags)
    assert outcome.exit_code == 0, outcome.output
    payload = json.loads(outcome.output)
    assert payload["request"]["fusion"] == "weighted:0.5,0.5"
    if "--explain" in flags:
        assert "qpp" in payload["diagnostics"]
    else:
        assert payload["summary"]["budget"]["requested"] == 0


@pytest.mark.parametrize("wrong_fusion", [None, "rrf", "weighted:0,1"])
def test_matching_ack_with_wrong_envelope_falls_back(indexed, tmp_path, monkeypatch, wrong_fusion):
    cold = run_cli(indexed, "--no-daemon", "--fusion", "weighted:1,0")
    remote = json.loads(cold.output)
    if wrong_fusion is None:
        remote["request"].pop("fusion")
    else:
        remote["request"]["fusion"] = wrong_fusion
    remote["results"][0]["score"] = -99
    socket_response(monkeypatch, tmp_path, {
        "ok": True, "instance_id": db.get_meta(indexed, "instance_id"),
        "fusion_effective": "weighted:1,0", "results": remote})
    actual = run_cli(indexed, "--fusion", "weighted:1,0")
    assert actual.exit_code == 0, actual.output
    assert json.loads(actual.output)["results"] == json.loads(cold.output)["results"]


def test_weighted_rerank_preserves_fused_scores_and_advice(indexed, store, embedder):
    class Scorer:
        def rerank(self, query, documents):
            return iter([1.0] * len(documents))

    model = "Xenova/ms-marco-MiniLM-L-6-v2"
    req = {"query": "carbonara", "fusion": "weighted:1,0", "_rerank_model": model}
    fused = daemon.run_search(indexed, store, embedder, req)
    out = daemon.run_search(indexed, store, embedder, req, reranker=Scorer())
    assert out["request"]["fusion"] == "weighted:1,0"
    expected = {r["docid"]: r["score"] for r in fused["results"]}
    assert all(r["sources"]["fused"] == expected[r["docid"]] for r in out["results"])
    options = out["summary"]["truncation_advice"]["options"]
    commands = [o["command"] for o in options if "command" in o]
    assert commands
    assert all("--fusion" in c and "--rerank=" + model in c and "calibrated" not in c
               for c in commands)
