from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.generate_preference_state_teacher import (
    TRAPI_DEPLOYMENT,
    format_completion,
    generate,
    history_section,
    history_subcategories,
    select_teacher_rows,
    student_prompt,
    transition_prompt,
    validate_profile,
    validate_transition,
)


def _prompt(category: str = "Sports") -> str:
    return (
        "The headlines and corresponding topic categories of the news items I "
        "recently viewed, in chronological order, are listed below. Each item "
        'is formatted as "[Topic category] Headline":\n\n'
        f"1. [{category}] Team wins title\n"
        "2. [Business] Markets rise\n\n"
        "Based on both the headlines and their topic categories, predict a new "
        "topic category. Respond with only <interest_NNN>."
    )


def test_reasoning_contract_hides_the_final_category_until_answer() -> None:
    prompt = _prompt()
    assert history_subcategories(prompt) == ["Sports", "Business"]
    assert "Based on both" not in history_section(prompt)
    transformed = student_prompt(prompt)
    assert "<think>" in transformed and "<answer>" in transformed

    profile = validate_profile(
        {
            "history_summary": "The history repeatedly mixes sports and markets.",
            "preference_state": "The user follows competition and commercial stakes.",
        }
    )
    request = transition_prompt(
        **profile,
        target_subcategory="Sports Betting",
        target_category="Sports",
    )
    assert "Sports Betting" in request
    expansion = validate_transition(
        {"potential_expansion": "A related unseen direction combines competitive outcomes with financial stakes."},
        target_subcategory="Sports Betting",
    )
    completion = format_completion(
        **profile,
        potential_expansion=expansion,
        target_token="<interest_007>",
    )
    assert "Sports Betting" not in completion
    assert completion.endswith("<answer><interest_007></answer>")
    with pytest.raises(ValueError, match="reveals"):
        validate_transition(
            {"potential_expansion": "Sports Betting is plausible."},
            target_subcategory="Sports Betting",
        )
    with pytest.raises(ValueError, match="32 words"):
        validate_profile(
            {
                "history_summary": " ".join(["word"] * 33),
                "preference_state": "Concise preference.",
            }
        )


def test_balanced_selection_interleaves_one_row_per_class(tmp_path: Path) -> None:
    rows = []
    mapping = []
    vocabulary = []
    for class_index, target in enumerate(("Alpha", "Zulu")):
        mapping.append(
            {
                "target_subcategory": target,
                "interest_token": f"<interest_{class_index:03d}>",
            }
        )
        vocabulary.append({"category": "Parent", "subcategory": target})
        for offset in range(10_800):
            rows.append(
                {
                    "sample_id": f"{target}:{offset}:behavior_interest",
                    "source_sample_id": f"{target}:{offset}",
                    "impression_id": str(offset),
                    "user_id": f"u-{target}-{offset}",
                    "history_length": 2,
                    "prompt": _prompt(),
                    "target_subcategory": target,
                }
            )
    train = tmp_path / "train.parquet"
    mapping_file = tmp_path / "mapping.parquet"
    vocabulary_file = tmp_path / "vocabulary.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train)
    pq.write_table(pa.Table.from_pylist(mapping), mapping_file)
    pq.write_table(pa.Table.from_pylist(vocabulary), vocabulary_file)

    # Patch the module constants to keep this unit fixture small in class count.
    import new_interest.data.generate_preference_state_teacher as module

    original_classes = module.CLASSES
    module.CLASSES = 2
    try:
        selected = select_teacher_rows(
            train_file=train,
            mapping_file=mapping_file,
            vocabulary_file=vocabulary_file,
            per_class=2,
        )
    finally:
        module.CLASSES = original_classes
    assert [row["within_class_rank"] for row in selected] == [0, 0, 1, 1]
    assert [row["target_subcategory"] for row in selected[:2]] == ["Alpha", "Zulu"]


class _Completions:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        if "Private fine-grained target" in prompt:
            content = json.dumps(
                {
                    "potential_expansion": (
                        "A related unseen direction combines the observed themes "
                        "with a different practical motivation."
                    )
                }
            )
        else:
            content = json.dumps(
                {
                    "history_summary": "The history mixes sports and business coverage.",
                    "preference_state": "The user follows competition and commercial context.",
                }
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )


def test_generation_uses_two_gpt55_calls_and_writes_audited_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import new_interest.data.generate_preference_state_teacher as module

    selection = [
        {
            "selection_rank": 0,
            "within_class_rank": 0,
            "selection_priority": "0" * 64,
            "sample_id": "s:behavior_interest",
            "source_sample_id": "s",
            "impression_id": "i",
            "user_id": "u",
            "history_length": 2,
            "history_subcategories": ["Sports", "Business"],
            "original_prompt": _prompt(),
            "student_prompt": student_prompt(_prompt()),
            "target_subcategory": "Alpha",
            "target_category": "Parent",
            "target_token": "<interest_000>",
        }
    ]
    monkeypatch.setattr(module, "select_teacher_rows", lambda **_: selection)
    for name in ("train.parquet", "mapping.parquet", "vocabulary.parquet"):
        (tmp_path / name).write_bytes(name.encode())
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    output = tmp_path / "out"
    metadata = generate(
        {
            "train_file": str(tmp_path / "train.parquet"),
            "mapping_file": str(tmp_path / "mapping.parquet"),
            "vocabulary_file": str(tmp_path / "vocabulary.parquet"),
            "output_dir": str(output),
            "deployment": TRAPI_DEPLOYMENT,
            "reasoning_effort": "none",
            "per_class": 10,
            "max_samples": 1,
            "concurrency": 1,
            "checkpoint_size": 1,
        },
        client=client,
        sleep_fn=lambda _: None,
    )
    assert len(completions.calls) == 2
    assert all(call["reasoning_effort"] == "none" for call in completions.calls)
    assert all("temperature" not in call for call in completions.calls)
    records = pq.read_table(output / "teacher_records.parquet").to_pylist()
    assert records[0]["status"] == "success"
    assert records[0]["completion"].endswith(
        "<answer><interest_000></answer>"
    )
    assert metadata["status_counts"] == {"success": 1}
    assert metadata["generated_this_run"] == 1
    assert metadata["resumed_successes"] == 0
    assert metadata["artifact_sha256"]["teacher_records.parquet"]

    second = generate(
        {
            "train_file": str(tmp_path / "train.parquet"),
            "mapping_file": str(tmp_path / "mapping.parquet"),
            "vocabulary_file": str(tmp_path / "vocabulary.parquet"),
            "output_dir": str(output),
            "deployment": TRAPI_DEPLOYMENT,
            "reasoning_effort": "none",
            "per_class": 10,
            "max_samples": 1,
            "concurrency": 1,
            "checkpoint_size": 1,
        },
        client=client,
        sleep_fn=lambda _: None,
    )
    assert len(completions.calls) == 2
    assert second["generated_this_run"] == 0
    assert second["resumed_successes"] == 1
