import pytest

from fidx import truncate
from fidx.search import Result


def _result(doc_id, score, sources=None):
    return Result(
        doc_id=doc_id,
        collection="notes",
        relpath=f"{doc_id}.md",
        title=f"Title {doc_id}",
        docid=str(doc_id),
        score=score,
        sources=dict(sources) if sources is not None else {},
    )


def _results(scores, sources=None):
    source_rows = sources if sources is not None else [None] * len(scores)
    return [_result(i + 1, score, source_rows[i]) for i, score in enumerate(scores)]


def _assert_identity_subsequence(original, kept):
    next_index = 0
    indices = []
    for item in kept:
        for index in range(next_index, len(original)):
            if original[index] is item:
                indices.append(index)
                next_index = index + 1
                break
        else:
            pytest.fail("kept item was not an identity-preserving subsequence member")
    assert indices == sorted(indices)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, ("off", [])),
        ("off", ("off", [])),
        ("abs:0.1", ("abs", [0.1])),
        ("ratio:0.5", ("ratio", [0.5])),
        ("gap:0.5", ("gap", [0.5])),
        ("knee", ("knee", [])),
        ("mad:3", ("mad", [3.0])),
        ("source:0.3,0.2", ("source", [0.3, 0.2])),
        ("source:0.3", ("source", [0.3])),
    ],
)
def test_parse_spec_valid_forms(spec, expected):
    assert truncate.parse_spec(spec) == expected


def test_parse_spec_unknown_method_raises_value_error():
    with pytest.raises(ValueError, match="unknown truncate method"):
        truncate.parse_spec("unknown:0.5")


@pytest.mark.parametrize("spec", ["abs", "abs:", "ratio", "ratio:", "source", "source:"])
def test_parse_spec_missing_required_params_raise_value_error(spec):
    with pytest.raises(ValueError):
        truncate.parse_spec(spec)


def test_parse_spec_non_numeric_param_raises_value_error():
    with pytest.raises(ValueError):
        truncate.parse_spec("ratio:abc")


def test_abs_truncates_by_absolute_score_floor():
    results = _results([0.3, 0.1, 0.099, 0.0])
    assert truncate.truncate(results, "abs:0.1") == results[:2]


def test_ratio_truncates_by_alpha_times_top_score_scale_free():
    results = _results([10.0, 6.0, 4.9, 1.0])
    scaled = _results([1.0, 0.6, 0.49, 0.1])

    assert truncate.truncate(results, "ratio:0.5") == results[:2]
    assert truncate.truncate(scaled, "ratio:0.5") == scaled[:2]


def test_gap_cuts_before_first_big_proportional_drop():
    results = _results([1.0, 0.8, 0.39, 0.38])
    assert truncate.truncate(results, "gap:0.5") == results[:2]


def test_knee_cuts_before_noise_tail_on_cliff_curve():
    results = _results([1.0, 0.9, 0.1, 0.09])
    assert truncate.truncate(results, "knee") == results[:2]


def test_knee_keeps_short_or_flat_lists():
    short = _results([1.0, 0.1, 0.01])
    flat = _results([0.4, 0.4, 0.4, 0.4])

    assert truncate.truncate(short, "knee") == short
    assert truncate.truncate(flat, "knee") == flat


def test_mad_keeps_robust_outlier_above_tail():
    results = _results([1.0, 0.2, 0.19, 0.18, 0.17])
    assert truncate.truncate(results, "mad:3") == results[:1]


def test_mad_keeps_short_or_degenerate_spreads():
    short = _results([1.0, 0.1])
    equal = _results([0.2, 0.2, 0.2, 0.2])

    assert truncate.truncate(short, "mad:3") == short
    assert truncate.truncate(equal, "mad:3") == equal


def test_mad_can_return_empty_when_no_score_clears_cut():
    results = _results([0.3, 0.2, 0.1])
    assert truncate.truncate(results, "mad:3") == []


def test_source_hybrid_keeps_present_vector_or_lexical_floor_matches_only():
    results = _results(
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [
            {"vector": 0.31},
            {"lexical": 0.21},
            {"vector": 0.29, "lexical": 0.19},
            {"vector": 0.1},
            {},
        ],
    )

    assert truncate.truncate(results, "source:0.3,0.2", mode="hybrid") == results[:2]


def test_source_hybrid_one_param_raises_value_error():
    results = _results([0.9], [{"vector": 0.9}])
    with pytest.raises(ValueError, match="requires vmin,lmin"):
        truncate.truncate(results, "source:0.3", mode="hybrid")


def test_source_vector_mode_uses_first_param_as_score_floor():
    results = _results([0.31, 0.3, 0.29])
    assert truncate.truncate(results, "source:0.3,0.9", mode="vector") == results[:2]


def test_source_lexical_mode_uses_second_param_when_present():
    results = _results([0.25, 0.19])
    assert truncate.truncate(results, "source:0.4,0.2", mode="lexical") == results[:1]


def test_source_lexical_mode_uses_first_param_when_only_one_present():
    results = _results([0.31, 0.29])
    assert truncate.truncate(results, "source:0.3", mode="lexical") == results[:1]


def test_truncate_preserves_order_identity_and_is_deterministic():
    results = _results(
        [1.0, 0.75, 0.3, 0.12, 0.04],
        [
            {"vector": 0.9, "lexical": 0.1},
            {"vector": 0.2, "lexical": 0.8},
            {"vector": 0.1},
            {"lexical": 0.05},
            {},
        ],
    )
    cases = [
        (None, "hybrid"),
        ("off", "hybrid"),
        ("abs:0.2", "hybrid"),
        ("ratio:0.5", "hybrid"),
        ("gap:0.5", "hybrid"),
        ("knee", "hybrid"),
        ("mad:3", "hybrid"),
        ("source:0.5,0.5", "hybrid"),
        ("source:0.5", "vector"),
        ("source:0.5,0.2", "lexical"),
    ]

    for spec, mode in cases:
        first = truncate.truncate(results, spec, mode=mode)
        second = truncate.truncate(results, spec, mode=mode)

        _assert_identity_subsequence(results, first)
        assert [id(item) for item in second] == [id(item) for item in first]


def test_uniform_weak_hybrid_source_floor_abstains():
    results = _results(
        [0.2, 0.19, 0.18],
        [
            {"vector": 0.1, "lexical": 0.1},
            {"vector": 0.09, "lexical": 0.08},
            {"vector": 0.05},
        ],
    )

    assert truncate.truncate(results, "source:0.9,0.9", mode="hybrid") == []


def test_mad_all_zero_scores_keeps_all():
    results = _results([0.0, 0.0, 0.0, 0.0])
    assert truncate.truncate(results, "mad:3") == results


def test_source_hybrid_singleton_below_floor_is_droppable():
    results = _results([0.1], [{"vector": 0.1}])
    assert truncate.truncate(results, "source:0.4,0.2", mode="hybrid") == []


def test_source_hybrid_missing_lexical_key_does_not_satisfy_floor():
    results = _results([0.1, 0.05], [{"vector": 0.1}, {"vector": 0.05}])
    assert truncate.truncate(results, "source:0.4,0.2", mode="hybrid") == []
