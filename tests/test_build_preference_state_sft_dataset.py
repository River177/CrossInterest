from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_preference_state_sft_dataset import (
    build_reasoning_sft_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_reasoning_sft_dataset_joins_exact_teacher_rows(
    tmp_path: Path,
) -> None:
    source_rows = []
    teacher_rows = []
    selection_rank = 0
    for target_index, target in enumerate(("Alpha", "Zulu")):
        token = f"<interest_{target_index:03d}>"
        for offset in range(2):
            source_id = f"{target}:{offset}"
            prompt = f"History {source_id}\n\nBased on both: direct"
            source_rows.append(
                {
                    "sample_id": f"{source_id}:behavior_interest",
                    "source_sample_id": source_id,
                    "prompt": prompt,
                    "response": token,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": token},
                    ],
                    "target_subcategory": target,
                    "history_length": 1,
                }
            )
            student = f"Student {source_id}"
            completion = (
                "<think>\nHistory summary: observed history.\n"
                "Preference state: broad preference.\n"
                "Potential expansion: an unseen adjacent direction.\n"
                f"</think>\n<answer>{token}</answer>"
            )
            teacher_rows.append(
                {
                    "selection_rank": selection_rank,
                    "source_sample_id": source_id,
                    "status": "success",
                    "target_subcategory": target,
                    "target_token": token,
                    "original_prompt": prompt,
                    "student_prompt": student,
                    "completion": completion,
                }
            )
            selection_rank += 1
    train_file = tmp_path / "source.parquet"
    teacher_file = tmp_path / "teacher.parquet"
    teacher_metadata = tmp_path / "teacher.json"
    pq.write_table(pa.Table.from_pylist(source_rows), train_file)
    pq.write_table(pa.Table.from_pylist(teacher_rows), teacher_file)
    teacher_metadata.write_text(
        json.dumps(
            {
                "requested_rows": 4,
                "status_counts": {"success": 4},
                "source": {"train_sha256": _sha256(train_file)},
                "artifact_sha256": {
                    "teacher_records.parquet": _sha256(teacher_file)
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    build_reasoning_sft_dataset(
        train_file=train_file,
        teacher_records=teacher_file,
        teacher_metadata=teacher_metadata,
        output_dir=output,
        expected_rows=4,
        expected_classes=2,
        expected_per_class=2,
    )
    rows = pq.read_table(output / "train.parquet").to_pylist()
    assert len(rows) == 4
    assert rows[0]["messages"][0]["content"] == "Student Alpha:0"
    assert rows[0]["messages"][1]["content"].endswith(
        "<answer><interest_000></answer>"
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["target_disclosure"] == "final-answer-only"
    assert metadata["artifact_sha256"]["train.parquet"] == _sha256(
        output / "train.parquet"
    )
