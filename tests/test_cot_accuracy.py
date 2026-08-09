from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from new_interest.data.build_grpo_dataset import append_cot_instruction
from new_interest.evaluation.aggregate_accuracy_results import (
    aggregate_accuracy_predictions,
)
from new_interest.evaluation.evaluate_cot_accuracy import (
    evaluate_accuracy_outputs,
    generate_greedy_outputs,
)


def _record(
    impression_id: str,
    acceptable: list[str],
    history: list[str] | None = None,
) -> dict[str, object]:
    return {
        "impression_id": impression_id,
        "prompt": f"prompt {impression_id}",
        "acceptable_subcategories": acceptable,
        "history_subcategories": history or [],
    }


def test_multi_target_prediction_is_set_correct() -> None:
    metrics, rows = evaluate_accuracy_outputs(
        [_record("1", ["Music", "Golf"])],
        ["<think>Golf is novel.</think>\nNew topic: Golf"],
        {"Music", "Golf", "Film"},
        mode="cot",
    )

    assert metrics["set_accuracy_at_1"] == 1.0
    assert metrics["strict_set_accuracy_at_1"] == 1.0
    assert rows[0]["set_hit_at_1"] is True
    assert rows[0]["acceptable_target_count"] == 2


def test_malformed_but_correct_cot_only_counts_as_loose_accuracy() -> None:
    metrics, rows = evaluate_accuracy_outputs(
        [_record("1", ["Music"])],
        ["<think>Unclosed reason\nNew topic: Music"],
        {"Music", "Golf"},
        mode="cot",
        response_tokens=[12],
        reached_max_new_tokens=[True],
    )

    assert metrics["set_accuracy_at_1"] == 1.0
    assert metrics["strict_set_accuracy_at_1"] == 0.0
    assert metrics["strict_format_rate"] == 0.0
    assert metrics["truncation_rate"] == 1.0
    assert rows[0]["parsed_label"] == "Music"


def test_metrics_group_by_target_count_and_summarize_predictions() -> None:
    records = [
        _record("1", ["Music", "Golf"]),
        _record("2", ["Film"], ["Film"]),
        _record("3", ["Music"]),
    ]
    metrics, _ = evaluate_accuracy_outputs(
        records,
        [
            "New topic: Golf",
            "New topic: Film",
            "invalid",
        ],
        {"Music", "Golf", "Film"},
        mode="no_cot",
        response_tokens=[2, 4, 6],
    )

    assert metrics["set_accuracy_at_1"] == 2 / 3
    assert metrics["valid_label_rate"] == 2 / 3
    assert metrics["novelty_compliance_rate"] == 1 / 3
    assert metrics["response_tokens"] == {
        "mean": 4,
        "median": 4,
        "p95": 5.8,
        "max": 6,
    }
    assert metrics["by_acceptable_target_count"]["2"][
        "set_accuracy_at_1"
    ] == 1.0
    assert metrics["prediction_counts"] == {"Film": 1, "Golf": 1}
    assert metrics["prediction_entropy_bits"] == 1.0


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        self.calls.append((content, kwargs["enable_thinking"]))
        return content

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[1, 1] for _ in texts]),
            "attention_mask": torch.tensor([[1, 1] for _ in texts]),
        }

    def decode(self, token_ids, **kwargs):
        del token_ids, kwargs
        return "New topic: Music"


class _Model:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(
            temperature=1.0,
            top_p=1.0,
            top_k=20,
        )

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        suffix = torch.tensor([[7, 2] for _ in input_ids])
        return torch.cat([input_ids, suffix], dim=1)


def test_direct_mode_uses_original_prompt_without_thinking() -> None:
    tokenizer = _Tokenizer()
    generations = generate_greedy_outputs(
        _Model(),
        tokenizer,
        [_record("1", ["Music"])],
        mode="no_cot",
        batch_size=1,
        max_new_tokens=8,
    )

    assert tokenizer.calls == [("prompt 1", False)]
    assert generations[0]["raw_output"] == "New topic: Music"
    assert generations[0]["response_tokens"] == 1
    assert generations[0]["reached_max_new_tokens"] is False


def test_cot_mode_appends_instruction_and_enables_thinking() -> None:
    tokenizer = _Tokenizer()
    generate_greedy_outputs(
        _Model(),
        tokenizer,
        [_record("1", ["Music"])],
        mode="cot",
        batch_size=1,
        max_new_tokens=8,
    )

    rendered_prompt, enable_thinking = tokenizer.calls[0]
    assert rendered_prompt == append_cot_instruction("prompt 1")
    assert enable_thinking is True


def test_aggregate_accuracy_requires_exact_dev_coverage(
    tmp_path,
) -> None:
    records = [
        _record("1", ["Music"]),
        _record("2", ["Golf"]),
    ]
    eval_file = tmp_path / "eval.parquet"
    pq.write_table(pa.Table.from_pylist(records), eval_file)
    _, rows = evaluate_accuracy_outputs(
        records,
        ["New topic: Music", "New topic: Film"],
        {"Music", "Golf", "Film"},
        mode="no_cot",
        record_indices=[0, 1],
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.Table.from_pylist([rows[0]]), first)
    pq.write_table(pa.Table.from_pylist([rows[1]]), second)

    metrics = aggregate_accuracy_predictions(
        [second, first],
        eval_file,
        tmp_path / "combined",
    )
    assert metrics["samples"] == 2
    assert metrics["set_accuracy_at_1"] == 0.5
    combined = pq.read_table(
        tmp_path / "combined/predictions.parquet"
    ).to_pylist()
    assert [row["record_index"] for row in combined] == [0, 1]

    with pytest.raises(ValueError, match="exactly once"):
        aggregate_accuracy_predictions(
            [first],
            eval_file,
            tmp_path / "invalid",
        )
