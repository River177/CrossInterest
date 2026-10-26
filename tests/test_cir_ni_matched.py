import copy

import pytest

from new_interest.data.build_cir_ni_matched import select_pair, training_row


def history(labels):
    return [{"subcategory": c, "title": "Article " + str(i), "news_id": str(i)} for i, c in enumerate(labels)]


def test_true_prefix_and_whole_category_erasure_share_label_and_length():
    h = history(list("ABADBCAE CF".replace(" ", "")))
    pair, reason = select_pair(h, h, "C", "u")
    assert reason == "kept" and pair["ni_positions"] == [0, 1, 2, 3, 4]
    assert pair["cir_positions"] == sorted(pair["cir_positions"])
    assert len(pair["cir_positions"]) == 5
    assert all(h[i]["subcategory"] != "C" for i in pair["cir_positions"])
    row = {"user_id": "u", "impression_id": "1", "target_subcategory": "C", "response": "<interest_002>", "source_sample_id": "future-label-7"}
    ni = training_row(row, h, pair["ni_positions"], "ni_prefix")
    cir = training_row(row, h, pair["cir_positions"], "cir_length_matched")
    assert (ni["sample_id"], ni["response"], ni["history_length"]) == (cir["sample_id"], cir["response"], cir["history_length"])
    assert ni["training_task"] == "new_interest"
    assert cir["training_task"] == "cross_interest_reconstruction"


def test_first_occurrence_before_h50_is_not_falsely_novel():
    full = history(["C"] + ["A"] * 10 + ["B"] * 40 + ["C"])
    assert select_pair(full, full[-50:], "C", "u") == (None, "target_seen_before_H50")


def test_short_prefix_and_source_mismatch_rejected():
    h = history(list("ABCAAAAA"))
    assert select_pair(h, h, "C", "u")[1] == "fewer_than_five_pre_target_events_in_H50"
    with pytest.raises(ValueError, match="reproduce H50"):
        select_pair(h, h[::-1], "C", "u")


def test_selection_and_prompt_do_not_read_original_future_ni_label():
    h = history(list("ABADBCAE CF".replace(" ", "")))
    original = copy.deepcopy(h)
    pair, _ = select_pair(h, h, "C", "u")
    assert select_pair(h, h, "C", "u")[0] == pair and h == original
    row = {"user_id": "u", "impression_id": "1", "target_subcategory": "C", "response": "<interest_002>", "source_sample_id": "future-label-7"}
    one = training_row(row, h, pair["cir_positions"], "cir_length_matched")
    two = training_row({**row, "source_sample_id": "future-label-98"}, h, pair["cir_positions"], "cir_length_matched")
    assert one["sample_id"] == two["sample_id"] and one["prompt"] == two["prompt"]
