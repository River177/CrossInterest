from copy import deepcopy

import pytest

from new_interest.data.build_cir_ni_prompt_dataset import replace_prompt, template_parts


def row():
    prompt = 'Removed category.\n\n1. [A] Title one\n2. [C] Title two\n\nReconstruct.'
    return {"prompt": prompt, "response": "<interest_001>", "history_length": 2,
            "target_subcategory": "B", "sample_id": "cir:user1", "training_task": "cross_interest_reconstruction",
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": "<interest_001>"}]}


def test_only_instruction_changes_with_messages_synchronized():
    source = row()
    saved = deepcopy(source)
    result = replace_prompt(source, ("NI input.", "Predict unseen interest."))
    assert source == saved
    assert {k for k in source if source[k] != result[k]} == {"prompt", "messages"}
    assert result["prompt"] == 'NI input.\n\n1. [A] Title one\n2. [C] Title two\n\nPredict unseen interest.'
    assert result["messages"][0]["content"] == result["prompt"]
    assert result["messages"][1] == source["messages"][1]
    assert template_parts(result) == ("NI input.", "Predict unseen interest.")


def test_rejects_target_present_in_history():
    source = row()
    source["target_subcategory"] = "A"
    with pytest.raises(ValueError, match="still appears"):
        replace_prompt(source, ("NI input.", "Predict."))


def test_rejects_unsynchronized_training_messages():
    source = row()
    source["messages"][0]["content"] = "stale training text"
    with pytest.raises(ValueError, match="prompt/messages"):
        replace_prompt(source, ("NI input.", "Predict."))
