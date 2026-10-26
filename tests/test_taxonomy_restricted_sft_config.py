from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
def test_taxonomy_softmax_changes_only_the_objective_and_output_identity() -> None:
    for size in (5400, 21600):
        taxonomy = yaml.safe_load(
            (
                ROOT
                / "configs/sft/experiments/"
                "static_sud_semantic_id_qwen35_behavior_interest_"
                f"taxonomy_softmax_{size}_v1.yaml"
            ).read_text(encoding="utf-8")
        )
        vanilla = yaml.safe_load(
            (
                ROOT
                / "configs/sft/experiments/"
                "static_sud_semantic_id_qwen35_behavior_interest_"
                f"vanilla_{size}_v1.yaml"
            ).read_text(encoding="utf-8")
        )

        assert taxonomy["model_name"] == "Qwen/Qwen3.5-9B"
        assert taxonomy["semantic_id_loss_scope"] == "taxonomy"
        assert "sample_weight_file" not in taxonomy
        assert taxonomy["enable_thinking"] is False

        ignored = {"semantic_id_loss_scope", "output_dir", "run_name"}
        assert {
            key: value for key, value in taxonomy.items() if key not in ignored
        } == {
            key: value for key, value in vanilla.items() if key not in ignored
        }
