from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def diagnose(
    data_file: Path,
    predictions_file: Path,
) -> dict[str, object]:
    data = pq.read_table(
        data_file,
        columns=[
            "selection_id",
            "candidates",
            "answer_category",
            "correct_position",
        ],
    ).to_pylist()
    predictions = pq.read_table(
        predictions_file,
        columns=[
            "selection_id",
            "normalized_category",
            "candidate_valid",
            "category_correct",
        ],
    ).to_pylist()
    if len(data) != len(predictions):
        raise ValueError("Data and prediction row counts differ")

    chosen_positions: Counter[int] = Counter()
    correct_support: Counter[int] = Counter()
    correct_by_position: Counter[int] = Counter()
    predicted_categories: Counter[str] = Counter()
    target_categories: Counter[str] = Counter()
    valid = 0
    correct = 0

    for row, prediction in zip(data, predictions):
        if row["selection_id"] != prediction["selection_id"]:
            raise ValueError("Selection IDs differ")
        correct_position = int(row["correct_position"])
        correct_support[correct_position] += 1
        correct_by_position[correct_position] += int(
            prediction["category_correct"]
        )
        target_categories[str(row["answer_category"])] += 1
        correct += int(prediction["category_correct"])
        if prediction["candidate_valid"]:
            valid += 1
            category = str(prediction["normalized_category"])
            predicted_categories[category] += 1
            chosen_positions[
                list(row["candidates"]).index(category)
            ] += 1

    samples = len(data)
    accuracy = correct / samples
    random_probability = 1.0 / 10.0
    random_standard_error = math.sqrt(
        random_probability * (1 - random_probability) / samples
    )
    return {
        "samples": samples,
        "selection_accuracy": accuracy,
        "random_accuracy": random_probability,
        "z_vs_random": (
            accuracy - random_probability
        )
        / random_standard_error,
        "candidate_valid_rate": valid / samples,
        "correct_position_counts": {
            str(position + 1): correct_support[position]
            for position in range(10)
        },
        "chosen_position_rates": {
            str(position + 1): chosen_positions[position] / samples
            for position in range(10)
        },
        "accuracy_by_correct_position": {
            str(position + 1): (
                correct_by_position[position]
                / correct_support[position]
            )
            for position in range(10)
        },
        "top_predicted_categories": predicted_categories.most_common(20),
        "top_target_categories": target_categories.most_common(20),
        "unique_predicted_categories": len(predicted_categories),
        "unique_target_categories": len(target_categories),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose K10 candidate selection biases."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = diagnose(args.data_file, args.predictions_file)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
