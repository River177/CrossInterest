import copy

import pytest

from new_interest.data.build_cir_dataset import render_history
from new_interest.data.build_cir_frequency_dataset import check_view, frequency_matches, paired_views


def fixture(categories=None):
    history = [{"subcategory": c, "title": f"headline {i}", "news_id": f"N{i}"}
               for i, c in enumerate(categories or ["a", "b", "b", "b", "b", "c", "c", "d", "e", "f"])]
    source = {"sample_id": "s", "user_id": "u", "impression_id": "i", "timestamp": "t", "history": history}
    training = {"user_id": "u", "impression_id": "i", "timestamp": "t", "split": "train",
                "task_type": "behavior_interest", "prompt": "Header\n\n" + render_history(history[-50:]) + "\n\nNI"}
    mapping = {c: f"<interest_{i:03d}>" for i, c in enumerate("abcdef")}
    return source, training, mapping


def test_frequency_boundary_is_one_versus_at_least_four():
    assert [n for n in range(6) if frequency_matches(n, "singleton")] == [1]
    assert [n for n in range(6) if frequency_matches(n, "repeated")] == [4, 5]
    with pytest.raises(ValueError):
        frequency_matches(1, "unknown")


def test_pair_deletes_whole_class_and_independent_reader_detects_corruption():
    source, training, mapping = fixture()
    pair = paired_views(source, training, mapping)
    assert pair["repeated"][0]["target_subcategory"] == "b"
    assert pair["repeated"][1]["deleted_positions_zero_based"] == [1, 2, 3, 4]
    assert len(pair["singleton"][1]["deleted_positions_zero_based"]) == 1
    for arm, (row, manifest) in pair.items():
        check_view(source, row, manifest, mapping, arm)
        altered = copy.deepcopy(row)
        altered["prompt"] = altered["prompt"].replace("headline", "CORRUPTED", 1)
        with pytest.raises(ValueError):
            check_view(source, altered, manifest, mapping, arm)


def test_frequency_count_after_h50_and_no_backfill():
    source, training, mapping = fixture(["b"] * 4 + ["c"] * 45 + ["a"])
    source["history"] = [{"subcategory": "a", "title": "older a", "news_id": "old"}] + source["history"]
    pair = paired_views(source, training, mapping)
    assert pair["singleton"][0]["target_subcategory"] == "a"
    for arm, (row, manifest) in pair.items():
        check_view(source, row, manifest, mapping, arm)
        assert "older a" not in row["prompt"]


def test_missing_pool_is_not_filled_from_other_frequency():
    source, training, mapping = fixture(["a", "b", "c", "d", "e", "f"])
    pair = paired_views(source, training, mapping)
    assert pair["singleton"][0] is not None
    assert pair["repeated"][0] is None


def test_repeated_target_must_leave_five_behaviors():
    source, training, mapping = fixture(["b", "b", "b", "b", "a", "c", "d", "e"])
    assert paired_views(source, training, mapping)["repeated"][0] is None


def test_pair_selection_ignores_future_labels_and_source_order():
    source, training, mapping = fixture()
    first = paired_views(source, training, mapping)
    training.update(target_subcategory="different future", response="different SID", messages=[])
    source["sample_id"] = "different-future-index"
    second = paired_views(source, training, dict(reversed(list(mapping.items()))))
    for arm in first:
        assert first[arm][0]["prompt"] == second[arm][0]["prompt"]
        assert first[arm][0]["response"] == second[arm][0]["response"]
