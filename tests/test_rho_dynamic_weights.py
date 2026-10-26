from __future__ import annotations

import copy
import json
import math
import random

import pytest

from new_interest.training.rho_dynamic_weights import (
    bounded_midrank_weights, build_classwise_weights, canonical_sha256,
    correlations, midranks, source_identity,
)


def sources(classes=("A", "B"), n=4):
    return [{"source_sample_id": f"{target}-{i}", "user_id": f"user-{target}-{i}",
             "target_subcategory": target, "messages": [
                 {"role": "user", "content": f"History {target} {i}"},
                 {"role": "assistant", "content": f"<interest_{class_index:03d}>"}]}
            for class_index, target in enumerate(classes) for i in range(n)]


def scores(rows, values, evaluation="reference-evaluation", key="full_vocab_first_sid_nll"):
    return [{**{field: row[field] for field in ("source_sample_id", "user_id", "target_subcategory")},
             "input_sha256": source_identity(row), key: value, "evaluation_identity": evaluation}
            for row, value in zip(rows, values)]


def build(rows, mode, reference=None, current=None, **kwargs):
    return build_classwise_weights(rows, mode=mode, reference_rows=reference, current_rows=current,
                                    expected_classes=None, expected_rows_per_class=None, **kwargs)


def test_exact_tie_midranks_no_arbitrary_source_order_tiebreak():
    assert midranks([0., 0., 1., 2.]) == [.5, .5, 2., 3.]
    result = bounded_midrank_weights([0., 0., 1., 2.])
    assert result == pytest.approx([2 / 3, 2 / 3, 7 / 6, 1.5])
    assert result[0] == result[1]
    assert math.fsum(result) == pytest.approx(4., abs=1e-14)
    # Different real priorities are not collapsed by a tolerance-based tie rule.
    assert midranks([1., math.nextafter(1., math.inf)]) == [0., 1.]


@pytest.mark.parametrize("priorities", [[0.], [-100.], [0.] * 200, [3.5] * 7])
def test_singleton_and_all_equal_priorities_have_exact_unit_weights(priorities):
    assert bounded_midrank_weights(priorities) == [1.] * len(priorities)


def test_static_and_dynamic_priorities_protect_learnable_hard_examples():
    rows = sources(classes=("A",))
    reference = scores(rows, [.2, .8, 3.1, 2.])
    current = scores(rows, [.3, 3., 3., 4.], evaluation="student-snapshot")
    static = build(rows, "static", reference, current)
    dynamic = build(rows, "dynamic", reference, current)
    assert [row["priority"] for row in static["rows"]] == [-.2, -.8, -3.1, -2.]
    assert [row["sample_weight"] for row in static["rows"]] == pytest.approx([1.5, 7 / 6, .5, 5 / 6])
    assert [row["priority"] for row in dynamic["rows"]] == pytest.approx([.1, 2.2, -.1, 2.])
    assert [row["sample_weight"] for row in dynamic["rows"]] == pytest.approx([5 / 6, 1.5, .5, 7 / 6])
    # Both middle examples have current loss 3; the cross-user-learnable one is upweighted.
    assert dynamic["rows"][1]["sample_weight"] > dynamic["rows"][2]["sample_weight"]


def test_dynamic_weights_change_when_student_learns_but_reference_stays_fixed():
    rows = sources(classes=("A",))
    reference = scores(rows, [.5, 1., 2., 3.])
    before = build(rows, "dynamic", reference, scores(rows, [.5, 4., 3., 3.5]))
    after = build(rows, "dynamic", reference, scores(rows, [.5, 1., 3., 3.5]))
    assert before["rows"][1]["sample_weight"] == 1.5
    assert after["rows"][1]["sample_weight"] < 1.
    assert before["diagnostics"]["reference_score_sha256"] == after["diagnostics"]["reference_score_sha256"]
    assert before["diagnostics"]["current_score_sha256"] != after["diagnostics"]["current_score_sha256"]


def test_class_mass_preserved_even_when_difficulty_scales_differ():
    rows = sources()
    result = build(rows, "static", scores(rows, [1., 2., 3., 4., 100., 200., 300., 400.]))
    for target in ("A", "B"):
        class_rows = [row for row in result["rows"] if row["target_subcategory"] == target]
        assert math.fsum(row["sample_weight"] for row in class_rows) == pytest.approx(4.)
        assert [row["sample_weight"] for row in class_rows] == pytest.approx([1.5, 7 / 6, 5 / 6, .5])
        audit = result["diagnostics"]["per_class"][target]
        assert audit["correlations"]["reference_nll_vs_weight"]["spearman"] == pytest.approx(-1.)
    audit = result["diagnostics"]["global"]
    weights = [row["sample_weight"] for row in result["rows"]]
    assert audit["effective_sample_size"] == pytest.approx(64 / math.fsum(w * w for w in weights))
    assert 0 < audit["effective_sample_fraction"] <= 1


def test_uniform_and_zero_dynamic_difference_are_exactly_uniform():
    rows = sources()
    reference = scores(rows, [float(i) for i in range(8)])
    for result in (build(rows, "uniform", reference, reference), build(rows, "dynamic", reference, reference)):
        assert [row["sample_weight"] for row in result["rows"]] == [1.] * 8
        assert result["diagnostics"]["global"]["effective_sample_size"] == 8
        assert result["diagnostics"]["global"]["correlations"]["priority_vs_weight"] == {"pearson": None, "spearman": None}


def test_join_preserves_source_order_is_immutable_and_tracks_score_identity():
    rows = sources()
    reference = scores(rows, [float(i % 4) for i in range(8)])
    original_rows, original_reference = copy.deepcopy(rows), copy.deepcopy(reference)
    result = build(rows, "static", list(reversed(reference)))
    assert rows == original_rows and reference == original_reference
    assert [row["source_sample_id"] for row in result["rows"]] == [row["source_sample_id"] for row in rows]
    assert result["diagnostics"]["reference_evaluation_identities"] == ["reference-evaluation"]
    assert result["diagnostics"]["weight_records_sha256"] == canonical_sha256(result["rows"])
    permuted = build(list(reversed(rows)), "static", reference)
    assert {r["source_sample_id"]: r["sample_weight"] for r in result["rows"]} == {
        r["source_sample_id"]: r["sample_weight"] for r in permuted["rows"]}
    json.dumps(result, allow_nan=False)


def test_source_hash_matches_reference_scorer_canonical_fields():
    row = sources(classes=("中文类别",), n=1)[0]
    row["prompt"] = "an unrelated duplicated field does not change the scoring identity"
    expected = canonical_sha256({key: row[key] for key in ("source_sample_id", "user_id", "target_subcategory", "messages")})
    assert source_identity(row) == expected
    hash_only = {key: row[key] for key in ("source_sample_id", "user_id", "target_subcategory")}
    hash_only["input_sha256"] = expected
    result = build([hash_only], "uniform")
    assert result["diagnostics"]["source_identities_from_provided_hash"] == 1
    with pytest.raises(ValueError, match="differ from supplied"):
        source_identity({**row, "input_sha256": "0" * 64})


@pytest.mark.parametrize("corruption", ["missing", "extra", "duplicate_id", "duplicate_user", "user", "target", "input"])
def test_exact_score_coverage_and_identity_are_required(corruption):
    rows = sources()
    reference = scores(rows, [1.] * 8)
    if corruption == "missing":
        reference.pop()
    elif corruption == "extra":
        reference.append({**reference[0], "source_sample_id": "extra", "user_id": "extra-user"})
    elif corruption == "duplicate_id":
        reference.append(copy.deepcopy(reference[0]))
    elif corruption == "duplicate_user":
        reference[1]["user_id"] = reference[0]["user_id"]
    elif corruption == "user":
        reference[1]["user_id"] = "wrong-user"
    elif corruption == "target":
        reference[1]["target_subcategory"] = "wrong-target"
    elif corruption == "input":
        reference[1]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        build(rows, "static", reference)
    # Uniform must not silently accept bad metadata supplied for diagnostics.
    with pytest.raises(ValueError):
        build(rows, "uniform", reference)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1.0", None])
def test_nonfinite_or_nonscalar_metadata_rejected(value):
    rows = sources(classes=("A",), n=1)
    with pytest.raises((ValueError, TypeError)):
        build(rows, "dynamic", scores(rows, [0.]), scores(rows, [value]))
    with pytest.raises((ValueError, TypeError)):
        bounded_midrank_weights([value])


def test_torch_loss_tensor_is_rejected_without_silently_detaching():
    torch = pytest.importorskip("torch")
    value = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(TypeError, match="scalar metadata"):
        bounded_midrank_weights([value])
    assert value.requires_grad and value.grad is None


def test_production_class_counts_and_required_inputs_are_strict():
    rows = sources()
    with pytest.raises(ValueError, match="Expected 108 classes"):
        build_classwise_weights(rows, mode="uniform")
    with pytest.raises(ValueError, match="exactly 200"):
        build_classwise_weights(rows, mode="uniform", expected_classes=2)
    for mode in ("static", "dynamic"):
        with pytest.raises(ValueError, match="requires reference_rows"):
            build(rows, mode)
    with pytest.raises(ValueError, match="requires current_rows"):
        build(rows, "dynamic", scores(rows, [1.] * 8))
    with pytest.raises(ValueError, match="Duplicate source"):
        build([rows[0], rows[0]], "uniform")


def test_correlations_are_tie_aware_and_large_finite_values_do_not_overflow():
    assert correlations([1., 1., 2.], [3., 3., 4.])["spearman"] == pytest.approx(1.)
    assert correlations([1e300, 2e300, 3e300], [3e300, 2e300, 1e300])["pearson"] == pytest.approx(-1.)
    assert correlations([1., 1.], [2., 3.]) == {"pearson": None, "spearman": None}


def test_tied_random_classes_remain_bounded_and_mass_preserving():
    generator = random.Random(42)
    for n in (2, 3, 7, 108, 200, 501):
        priorities = [float(generator.randrange(7)) for _ in range(n)]
        weights = bounded_midrank_weights(priorities)
        assert min(weights) >= .5 - 1e-12 and max(weights) <= 1.5 + 1e-12
        assert math.fsum(weights) == pytest.approx(n, rel=0, abs=1e-11)
        tied = {}
        for priority, weight in zip(priorities, weights):
            assert tied.setdefault(priority, weight) == weight


def test_current_loss_field_is_explicitly_configurable():
    rows = sources(classes=("A",), n=1)
    result = build(rows, "dynamic", scores(rows, [2.]), scores(rows, [3.], key="current_nll"), current_loss_key="current_nll")
    assert result["rows"][0]["priority"] == 1.
    assert result["rows"][0]["sample_weight"] == 1.
