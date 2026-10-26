from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_frontier_configs_change_only_weight_and_output_identity() -> None:
    for size in (5400, 21600):
        frontier = yaml.safe_load(
            (
                ROOT
                / "configs/sft/experiments/"
                f"static_sud_semantic_id_qwen35_behavior_interest_frontier_{size}_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        vanilla = yaml.safe_load(
            (
                ROOT
                / "configs/sft/experiments/"
                f"static_sud_semantic_id_qwen35_behavior_interest_vanilla_{size}_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        assert frontier["semantic_id_loss_scope"] == "full_vocabulary"
        assert frontier["sample_weight_file"].endswith(
            f"frontier_qwen35_behavior_interest_{size}_v1/sample_weights.parquet"
        )
        ignored = {
            "semantic_id_loss_scope",
            "sample_weight_file",
            "output_dir",
            "run_name",
        }
        assert {
            key: value for key, value in frontier.items() if key not in ignored
        } == {
            key: value for key, value in vanilla.items() if key not in ignored
        }
