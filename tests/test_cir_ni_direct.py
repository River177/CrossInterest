from copy import deepcopy
from pathlib import Path

import pytest

from new_interest.data.build_cir_dataset import make_view, render_history
from new_interest.data.build_cir_ni_direct import build, make_pair, unique_index
from new_interest.data.build_cir_ni_prompt_dataset import template_parts


def fixture():
    mapping = {c: f"<interest_{i:03d}>" for i,c in enumerate("ABCD")}
    history = [{"news_id": f"N{i}", "subcategory": c, "title": f"Headline {i}"}
               for i,c in enumerate("AB"*6)]
    source = {"sample_id": "source1", "user_id": "u1", "impression_id": "i1",
              "timestamp": "2019-11-11", "history": history}
    prompt = "Browse history.\n\n"+render_history(history)+"\n\nPredict a new interest."
    ni = {"sample_id": "ni:source1", "source_sample_id": "source1", "split": "train",
          "task_type": "behavior_interest", "user_id": "u1", "impression_id": "i1",
          "timestamp": source["timestamp"], "history_length": 12, "prompt": prompt,
          "response": mapping["C"], "target_subcategory": "C"}
    ni["messages"] = [{"role": "user", "content": prompt},
                      {"role": "assistant", "content": ni["response"]}]
    cir, _ = make_view(source, ni, mapping)
    return source, ni, cir, mapping


def test_pair_preserves_original_and_erases_entire_category():
    source, ni, cir, mapping = fixture()
    saved = deepcopy((source, ni, cir))
    c, n, pair = make_pair(cir, ni, source, mapping)
    assert (source, ni, cir) == saved
    assert n == {**ni, "training_task": "new_interest"}
    assert c["response"] == cir["response"]
    assert c["history_length"] == 6 and n["history_length"] == 12
    assert template_parts(c) == template_parts(n)
    assert c["training_task"] == n["training_task"] == "new_interest"
    assert pair["erased_target_clicks_h50"] == 6
    assert len(pair["retained_news_ids"]) == 6


def test_CIR_selection_does_not_depend_on_future_NI_target():
    source, ni, cir, mapping = fixture()
    first = make_pair(cir, ni, source, mapping)[0]
    ni["target_subcategory"], ni["response"] = "D", mapping["D"]
    ni["messages"][1]["content"] = ni["response"]
    assert make_pair(cir, ni, source, mapping)[0] == first


@pytest.mark.parametrize("mutation,error", [
    (lambda c: c.update(user_id="other"), "Paired identity"),
    (lambda c: c.update(source_sample_id="other"), "Source pairing"),
    (lambda c: c.update(response="<interest_099>"), "Frozen CIR differs"),
])
def test_rejects_corrupted_CIR_source(mutation, error):
    source, ni, cir, mapping = fixture()
    mutation(cir)
    with pytest.raises(ValueError, match=error):
        make_pair(cir, ni, source, mapping)


def test_original_NI_novelty_checks_full_history_not_only_H50():
    source, ni, cir, mapping = fixture()
    source["history"] = [{"news_id": "old", "subcategory": "C", "title": "Old"}] + source["history"]
    with pytest.raises(ValueError, match="full history"):
        make_pair(cir, ni, source, mapping)


def test_duplicate_source_identity_rejected():
    with pytest.raises(ValueError, match="Duplicate identity"):
        unique_index([{"id": "same"}, {"id": "same"}], "id")


def test_existing_output_is_never_replaced(tmp_path):
    (tmp_path/"keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        build(Path("missing"), Path("missing"), tmp_path)
    assert (tmp_path/"keep").read_text() == "unchanged"
