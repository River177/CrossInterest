import copy

import pytest

from new_interest.data.build_history_length_ablation import audit_pair, truncate_record


def record():
    history = [f"{i}. [Topic {i % 3}] Headline {i}" for i in range(1, 51)]
    prompt = "Frozen instruction\n\n" + "\n".join(history) + "\n\nFrozen output instruction"
    return {"sample_id": "a", "prompt": prompt, "history_length": 50,
            "target_subcategory": "New topic", "response": "<interest_001>",
            "history_subcategories": [f"Topic {i % 3}" for i in range(1, 51)],
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": "<interest_001>"}]}


@pytest.mark.parametrize("limit", [5, 10, 30, 50])
def test_recent_history_preserves_order_repetitions_labels_and_messages(limit):
    source = record()
    saved = copy.deepcopy(source)
    target = truncate_record(source, limit)
    assert source == saved
    assert target["history_length"] == limit
    assert f"1. [Topic {(51-limit) % 3}] Headline {51-limit}\n" in target["prompt"]
    assert target["history_subcategories"] == source["history_subcategories"][-limit:]
    assert target["messages"][1:] == source["messages"][1:]
    assert target["messages"][0]["content"] == target["prompt"]
    assert target["response"] == source["response"]
    assert audit_pair([source], [target], limit)["rows"] == 1
    if limit == 50:
        assert source == target


def test_short_history_is_unchanged_and_truncation_is_nested():
    source = record()
    h5 = truncate_record(source, 5)
    assert truncate_record(h5, 10) == h5
    assert truncate_record(truncate_record(source, 30), 5) == h5


def test_changed_labels_or_corrupt_numbering_fail_closed():
    source = record()
    target = truncate_record(source, 5)
    target["response"] = "<interest_002>"
    with pytest.raises(ValueError):
        audit_pair([source], [target], 5)
    source["prompt"] = source["prompt"].replace("50. [", "49. [")
    with pytest.raises(ValueError):
        truncate_record(source, 5)
