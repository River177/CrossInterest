import copy

from new_interest.data.build_cir_dataset import (
    PROMPT, history_from_record, history_key, make_view, render_history, select_category,
)


def fixture():
    history = [{"subcategory": c, "title": f"headline {i}", "news_id": f"N{i}"}
               for i, c in enumerate(["a", "b", "a", "b", "c", "b", "c", "b", "c"])]
    source = {"sample_id": "s1", "user_id": "u1", "impression_id": "i1", "timestamp": "t1", "history": history}
    training = {"user_id": "u1", "impression_id": "i1", "timestamp": "t1", "split": "train",
                "task_type": "behavior_interest", "history_length": len(history),
                "prompt": "Header\n\n" + render_history(history) + "\n\nNI instruction",
                "target_subcategory": "future"}
    return source, training, {"a": "<interest_000>", "b": "<interest_001>", "c": "<interest_002>"}


def test_whole_category_removed_order_and_minimum_retained():
    source, training, mapping = fixture()
    row, manifest = make_view(source, training, mapping)
    target = row["target_subcategory"]
    remaining = [x for x in source["history"] if x["subcategory"] != target]
    assert row["history_length"] == len(remaining) >= 5
    assert history_from_record(row) == render_history(remaining)
    assert row["response"] == mapping[target]
    assert manifest["retained_news_ids"] == [x["news_id"] for x in remaining]
    assert row["messages"] == [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}]


def test_future_label_never_affects_selection_or_prompt():
    source, training, mapping = fixture()
    first = make_view(source, training, mapping)
    changed = copy.deepcopy(training)
    changed.update(target_subcategory="different future", response="different target", messages=[])
    assert make_view(source, changed, mapping) == first
    changed_source = {**source, "sample_id": "same-impression-another-future-label"}
    changed_row, _ = make_view(changed_source, changed, mapping)
    assert changed_row["target_subcategory"] == first[0]["target_subcategory"]
    assert changed_row["prompt"] == first[0]["prompt"]


def test_history_window_is_before_deletion_no_backfill():
    source, training, mapping = fixture()
    last50 = [source["history"][i % 9] for i in range(50)]
    source["history"] = [{"subcategory": "hidden", "title": "hidden", "news_id": "old"}] + last50
    training.update(history_length=50, prompt="Header\n\n" + render_history(last50) + "\n\nInstruction")
    row, manifest = make_view(source, training, mapping)
    assert "hidden" not in row["prompt"]
    assert row["history_length"] == 50 - len(manifest["deleted_positions_zero_based"])


def test_empty_or_short_remaining_histories_not_candidates():
    h = [{"subcategory": "a"}] * 5 + [{"subcategory": "b"}]
    assert select_category(h, {"a", "b"}, "s", 42) == ("b", ["b"])
    assert select_category(h[:5], {"a", "b"}, "s", 42) == (None, [])
    assert select_category(h, {"a"}, "s", 42) == (None, [])


def test_duplicate_key_ignores_task_instruction_but_preserves_order():
    source, training, _ = fixture()
    other = {**training, "prompt": training["prompt"].replace("NI instruction", "OI instruction"), "target_subcategory": "a"}
    assert history_key(history_from_record(training)) == history_key(history_from_record(other))
    assert history_key(render_history(source["history"])) != history_key(render_history(source["history"][::-1]))


def test_news_click_prompt_and_no_target_instructions():
    assert "news" in PROMPT and "next click" in PROMPT
    assert "purchase" not in PROMPT and "product" not in PROMPT


def test_structured_heldout_matches_rendered_history():
    source, training, _ = fixture()
    row = {"history_titles": [x["title"] for x in source["history"]],
           "history_subcategories": [x["subcategory"] for x in source["history"]]}
    assert history_from_record(row) == history_from_record(training)
    assert history_from_record({"prompt": training["prompt"]}) == history_from_record(training)
