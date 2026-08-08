from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.diagnose_candidate_bias import diagnose


def test_diagnose_catches_below_random_position_bias(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    candidates = [f"Topic {index}" for index in range(10)]
    data = []
    predictions = []
    for index in range(100):
        correct_position = index % 10
        data.append(
            {
                "selection_id": f"s{index}",
                "candidates": candidates,
                "answer_category": candidates[correct_position],
                "correct_position": correct_position,
            }
        )
        selected = candidates[0]
        predictions.append(
            {
                "selection_id": f"s{index}",
                "normalized_category": selected,
                "candidate_valid": True,
                "category_correct": selected
                == candidates[correct_position],
            }
        )
    pq.write_table(pa.Table.from_pylist(data), data_path)
    pq.write_table(
        pa.Table.from_pylist(predictions),
        prediction_path,
    )

    result = diagnose(data_path, prediction_path)

    assert result["selection_accuracy"] == 0.1
    assert result["chosen_position_rates"]["1"] == 1.0
    assert result["correct_position_counts"]["1"] == 10
    assert result["correct_position_counts"]["10"] == 10
