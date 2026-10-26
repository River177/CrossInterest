import pytest

from new_interest.training.short_think_grpo_reward import advantages, group_statistics, score


def reward(predicted="a", valid=True, length=100):
    return score(valid=valid, predicted=predicted, target="a", history=["d"],
                 parent_by_token={"a": "P", "b": "P", "c": "Q", "d": "P"},
                 thought_tokens=length)


@pytest.mark.parametrize("predicted,valid,expected", [("a", True, 1), ("b", True, .2),
    ("c", True, 0), ("d", True, -.1), (None, False, -.2)])
def test_outcomes(predicted, valid, expected):
    assert reward(predicted, valid)["H"] == expected


def test_correctness_dominates_length_and_wrong_groups_get_no_length_signal():
    assert reward(length=10000)["HL"] == .9
    assert reward(length=240)["HL"] == .95
    assert reward(length=160)["HL"] == 1
    for sid in ["b", "c", "d"]:
        assert reward(sid, length=10000)["HL"] == reward(sid, length=1)["HL"]
    assert advantages([0, 0, 0]) == [0, 0, 0]
    assert max(abs(x) for x in advantages([1, .999])) < .01


def test_data_errors_are_not_model_failures():
    with pytest.raises(ValueError):
        score(valid=False, predicted=None, target="a", history=["a"], parent_by_token={"a":"P"}, thought_tokens=None)
    with pytest.raises(ValueError):
        reward("unknown")
    with pytest.raises(ValueError):
        reward(length=0)


def test_groups_cannot_drop_duplicate_or_missing_rollouts():
    base={"source_sample_id":"x", "generation_index":0, "valid_format":True,
          "budget_exhausted":False, "generated_tokens":120, "think_tokens":100, "reward":reward()}
    with pytest.raises(ValueError):
        group_statistics([base, base], generations=2)
    r=group_statistics([base, {**base,"generation_index":1,"reward":reward("c")}], generations=2)
    assert r["variable_group_rates"]["E"] == 1
    assert r["exact_rate"] == .5
