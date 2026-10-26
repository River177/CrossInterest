from __future__ import annotations

import pytest

from new_interest.evaluation.evaluate_candidate_qwen import (
    _generated_token_metadata,
    _normalize_shard_output,
    _pilot_records,
    apply_candidate_chat_template,
    parse_generated_output,
)


class _UnsupportedThinkingTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        del messages
        self.calls.append(kwargs)
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "rendered"


class _BrokenTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        raise TypeError("unrelated tokenizer failure")


def test_llama_chat_template_retries_without_enable_thinking() -> None:
    tokenizer = _UnsupportedThinkingTokenizer()

    rendered = apply_candidate_chat_template(
        tokenizer,
        [{"role": "user", "content": "prompt"}],
        add_generation_prompt=True,
        enable_thinking=False,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    assert rendered == "rendered"
    assert tokenizer.calls == [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        {
            "tokenize": False,
            "add_generation_prompt": True,
        },
    ]


def test_qwen_chat_template_does_not_hide_type_error() -> None:
    with pytest.raises(TypeError, match="enable_thinking"):
        apply_candidate_chat_template(
            _UnsupportedThinkingTokenizer(),
            [{"role": "user", "content": "prompt"}],
            add_generation_prompt=True,
            enable_thinking=True,
            model_name="Qwen/Qwen3-8B",
        )


def test_llama_chat_template_does_not_hide_unrelated_type_error() -> None:
    with pytest.raises(TypeError, match="unrelated tokenizer failure"):
        apply_candidate_chat_template(
            _BrokenTokenizer(),
            [{"role": "user", "content": "prompt"}],
            add_generation_prompt=True,
            enable_thinking=False,
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )


def test_closed_think_block_is_separated_from_final_output() -> None:
    parsed = parse_generated_output(
        "<think>\ncompare the candidates\n</think>\n"
        "New topic: Sports"
    )

    assert parsed == {
        "raw_output": (
            "<think>\ncompare the candidates\n</think>\n"
            "New topic: Sports"
        ),
        "output": "New topic: Sports",
        "final_output": "New topic: Sports",
        "think_content": "compare the candidates",
        "has_think_block": True,
        "closed_think_block": True,
        "missing_final_answer": False,
    }


def test_unclosed_think_block_marks_missing_final_answer() -> None:
    parsed = parse_generated_output("<think>still reasoning")

    assert parsed["output"] == ""
    assert parsed["final_output"] == ""
    assert parsed["think_content"] == "still reasoning"
    assert parsed["has_think_block"] is True
    assert parsed["closed_think_block"] is False
    assert parsed["missing_final_answer"] is True


def test_old_output_only_shard_remains_readable() -> None:
    parsed = _normalize_shard_output({"output": "New topic: Sports"})

    assert parsed["raw_output"] == "New topic: Sports"
    assert parsed["output"] == "New topic: Sports"
    assert parsed["final_output"] == "New topic: Sports"
    assert parsed["generated_token_count"] is None
    assert parsed["reached_max_new_tokens"] is None


def test_raw_only_shard_remains_readable() -> None:
    parsed = _normalize_shard_output(
        {"raw_output": "<think>reasoning</think>New topic: Sports"}
    )

    assert parsed["raw_output"].startswith("<think>")
    assert parsed["output"] == "New topic: Sports"
    assert parsed["final_output"] == "New topic: Sports"


def test_generated_token_metadata_distinguishes_eos_from_limit() -> None:
    assert _generated_token_metadata(
        [11, 12, 2, 0],
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=4,
    ) == (3, False)
    assert _generated_token_metadata(
        [11, 12, 13, 14],
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=4,
    ) == (4, True)


def test_category_covering_pilot_is_deterministic() -> None:
    records = [
        {
            "selection_id": f"{category}-{index}",
            "answer_category": category,
        }
        for category in ("A", "B", "C")
        for index in range(4)
    ]

    first = _pilot_records(
        records,
        5,
        17,
        cover_categories=True,
    )
    second = _pilot_records(
        list(reversed(records)),
        5,
        17,
        cover_categories=True,
    )

    assert {row["answer_category"] for row in first} == {"A", "B", "C"}
    assert {row["selection_id"] for row in first} == {
        row["selection_id"] for row in second
    }
