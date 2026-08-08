from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from new_interest.evaluation.aggregate_cot_label_results import (
    aggregate_prediction_shards,
)
from new_interest.evaluation.evaluate_cot_label_ranking import (
    _write_rows_atomically,
    build_reasoning_prefix,
    compute_all_metrics,
    rank_labels,
    repair_reasoning,
    score_label_continuations,
)


class CharacterTokenizer:
    def __init__(self) -> None:
        alphabet = (
            "\n <>/:abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.tokens = {
            character: index + 2
            for index, character in enumerate(alphabet)
        }
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {
            "input_ids": [
                self.tokens[character] for character in text
            ]
        }


@pytest.mark.parametrize(
    ("output", "reasoning", "repair_type"),
    [
        (
            "<think>Brief reason.</think>\nNew topic: Music",
            "Brief reason.",
            "none",
        ),
        (
            "<think>Unclosed reason.\nNew topic: Music",
            "Unclosed reason.",
            "close_unclosed_think_before_answer",
        ),
        (
            "<think>Reason.</think>\nExtra\nNew topic: Music",
            "Reason.",
            "rebuild_after_closed_think",
        ),
        (
            "Explanation\nNew topic: Music",
            "Explanation",
            "wrap_text_before_answer",
        ),
        (
            "New topic: Music",
            "",
            "empty_reasoning",
        ),
    ],
)
def test_repair_reasoning(
    output: str,
    reasoning: str,
    repair_type: str,
) -> None:
    repaired = repair_reasoning(output)
    assert repaired.reasoning == reasoning
    assert repaired.repair_type == repair_type


def test_build_reasoning_prefix_has_no_answer_space() -> None:
    prefix = build_reasoning_prefix("PROMPT", "Reason")
    assert prefix == (
        "PROMPT<think>\nReason\n</think>\nNew topic:"
    )


def test_rank_labels_is_deterministic_for_ties() -> None:
    ranked = rank_labels(["Z", "A", "B"], [0.5, 0.5, 0.7])
    assert [row["label"] for row in ranked] == ["B", "A", "Z"]


def _brute_force_scores(
    model: GPT2LMHeadModel,
    tokenizer: CharacterTokenizer,
    prefix: str,
    labels: list[str],
) -> list[float]:
    scores = []
    for label in labels:
        prefix_ids = tokenizer(
            prefix,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = tokenizer(
            f"{prefix} {label}",
            add_special_tokens=False,
        )["input_ids"]
        continuation = full_ids[len(prefix_ids) :]
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for offset, token_id in enumerate(continuation):
            total += float(
                log_probs[
                    0,
                    len(prefix_ids) + offset - 1,
                    token_id,
                ].item()
            )
        total += float(
            log_probs[
                0,
                len(full_ids) - 1,
                tokenizer.eos_token_id,
            ].item()
        )
        scores.append(total / (len(continuation) + 1))
    return scores


def test_shared_prefix_scores_match_brute_force() -> None:
    torch.manual_seed(7)
    tokenizer = CharacterTokenizer()
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=max(tokenizer.tokens.values()) + 1,
            n_positions=64,
            n_ctx=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    model.eval()
    labels = ["A", "BC", "Z"]
    prefix = "Prompt\nNew topic:"
    cached = score_label_continuations(
        model,
        tokenizer,
        prefix,
        labels,
        candidate_batch_size=2,
    )
    brute = _brute_force_scores(model, tokenizer, prefix, labels)
    assert cached == pytest.approx(brute, abs=1e-6)


def _record() -> dict[str, object]:
    return {
        "impression_id": "1",
        "prompt": "prompt",
        "acceptable_subcategories": ["A"],
        "acceptable_responses": ["New topic: A"],
        "history_subcategories": [],
    }


def _row(*, strict_format: bool) -> dict[str, object]:
    return {
        "record_index": 0,
        "impression_id": "1",
        "prompt": "prompt",
        "raw_output": "output",
        "response_tokens": 2,
        "reached_max_new_tokens": False,
        "strict_format": strict_format,
        "strict_reasoning": "reason" if strict_format else None,
        "repaired_reasoning": "reason",
        "repair_type": "none" if strict_format else "empty_reasoning",
        "greedy_answer": "New topic: A",
        "greedy_label": "A",
        "greedy_valid_label": True,
        "greedy_set_hit": True,
        "greedy_strict_hit": strict_format,
        "novelty_compliant": True,
        "acceptable_subcategories": ["A"],
        "history_subcategories": [],
        "ranked_labels": ["A", "B", "C"],
        "ranked_scores": [0.9, 0.8, 0.7],
    }


def test_malformed_rows_are_zero_only_in_strict_metrics() -> None:
    metrics, predictions = compute_all_metrics(
        [_record()],
        [_row(strict_format=False)],
        cutoffs=(1, 3),
    )
    assert metrics["strict_ranking"]["recall_at_1"] == 0.0
    assert metrics["strict_ranking"]["ndcg_at_3"] == 0.0
    assert metrics["repaired_ranking"]["recall_at_1"] == 1.0
    assert metrics["repaired_ranking"]["ndcg_at_3"] == 1.0
    assert predictions[0]["strict_recall_at_1"] == 0.0
    assert predictions[0]["repaired_recall_at_1"] == 1.0


def test_prediction_shard_is_atomic_and_readable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shards" / "000000000-000000001.parquet"
    rows = [_row(strict_format=True)]
    _write_rows_atomically(path, rows)
    assert path.is_file()
    loaded = __import__("pyarrow.parquet").parquet.read_table(
        path
    ).to_pylist()
    assert loaded[0]["impression_id"] == "1"
    assert not list(path.parent.glob(f".{path.name}.*"))


@pytest.mark.parametrize(
    (
        "source_job",
        "adapter_relative",
        "expected_condition",
        "expected_task",
    ),
    [
        (
            "qwen3-interest-22k",
            "sft/final_adapter",
            "sft22k",
            "interest_only",
        ),
        (
            "grpo-1k-qwen3-behavior-22k",
            "grpo/final_adapter",
            "grpo1k",
            "behavior_only",
        ),
    ],
)
def test_map_script_resolves_adapter_and_task(
    tmp_path: Path,
    source_job: str,
    adapter_relative: str,
    expected_condition: str,
    expected_task: str,
) -> None:
    code_dir = tmp_path / "code"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    adapter = input_dir / adapter_relative
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    for task_type in ("interest_only", "behavior_only"):
        eval_path = (
            code_dir
            / "data/training/sft_experiments/eval"
            / task_type
            / "dev_10k_grouped.parquet"
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_bytes(b"eval")
    labels_path = (
        code_dir
        / "data/training/generative_new_interest_balanced_200_h50"
        / "retained_interests.parquet"
    )
    labels_path.parent.mkdir(parents=True)
    labels_path.write_bytes(b"labels")
    arguments_path = tmp_path / "arguments.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ARGUMENTS_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = (
        Path(__file__).parents[1]
        / "scripts/amlt/evaluation/run_qwen_cot_label_eval_map.sh"
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "ARGUMENTS_PATH": str(arguments_path),
        "AMLT_MAP_INPUT_JOB_NAME": source_job,
        "AMLT_MAP_INPUT_DIR": str(input_dir),
        "AMLT_CODE_DIR": str(code_dir),
        "AMLT_OUTPUT_DIR": str(output_dir),
    }
    subprocess.run(
        ["bash", str(script), "smoke"],
        check=True,
        env=environment,
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--condition") + 1] == expected_condition
    assert arguments[arguments.index("--task-type") + 1] == expected_task
    assert str(adapter) in arguments
    assert arguments[arguments.index("--max-examples") + 1] == "128"


def test_map_script_builds_disjoint_dev10k_shard(
    tmp_path: Path,
) -> None:
    code_dir = tmp_path / "code"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    adapter = input_dir / "sft/final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    eval_path = (
        code_dir
        / "data/training/sft_experiments/eval/interest_only"
        / "dev_10k_grouped.parquet"
    )
    eval_path.parent.mkdir(parents=True)
    eval_path.write_bytes(b"eval")
    labels_path = (
        code_dir
        / "data/training/generative_new_interest_balanced_200_h50"
        / "retained_interests.parquet"
    )
    labels_path.parent.mkdir(parents=True)
    labels_path.write_bytes(b"labels")
    arguments_path = tmp_path / "arguments.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ARGUMENTS_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = (
        Path(__file__).parents[1]
        / "scripts/amlt/evaluation/run_qwen_cot_label_eval_map.sh"
    )
    subprocess.run(
        ["bash", str(script), "full", "2", "4"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "ARGUMENTS_PATH": str(arguments_path),
            "AMLT_MAP_INPUT_JOB_NAME": "qwen3-interest-22k",
            "AMLT_MAP_INPUT_DIR": str(input_dir),
            "AMLT_CODE_DIR": str(code_dir),
            "AMLT_OUTPUT_DIR": str(output_dir),
        },
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--start-index") + 1] == "5000"
    assert arguments[arguments.index("--max-examples") + 1] == "2500"
    assert (
        arguments[arguments.index("--run-name") + 1]
        == "qwen3-sft22k-interest-dev10k-s2of4-cot-label-ranking"
    )


def test_aggregate_prediction_shards_requires_exact_coverage(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = [_record(), {**_record(), "impression_id": "2"}]
    eval_file = tmp_path / "eval.parquet"
    pq.write_table(pa.Table.from_pylist(records), eval_file)
    first = _row(strict_format=True)
    second = {**_row(strict_format=False), "record_index": 1, "impression_id": "2"}
    shard_0 = tmp_path / "s0.parquet"
    shard_1 = tmp_path / "s1.parquet"
    pq.write_table(pa.Table.from_pylist([first]), shard_0)
    pq.write_table(pa.Table.from_pylist([second]), shard_1)

    metrics = aggregate_prediction_shards(
        prediction_files=[shard_1, shard_0],
        eval_file=eval_file,
        output_dir=tmp_path / "combined",
        condition="sft22k",
        task_type="interest_only",
        cutoffs=(1, 3),
    )
    assert metrics["samples"] == 2
    assert metrics["strict_ranking"]["recall_at_1"] == 0.5
    combined = pq.read_table(
        tmp_path / "combined/predictions.parquet"
    ).to_pylist()
    assert [row["record_index"] for row in combined] == [0, 1]

    with pytest.raises(ValueError, match="exactly once"):
        aggregate_prediction_shards(
            prediction_files=[shard_0],
            eval_file=eval_file,
            output_dir=tmp_path / "invalid",
            condition="sft22k",
            task_type="interest_only",
            cutoffs=(1, 3),
        )
