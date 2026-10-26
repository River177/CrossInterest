from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_softmax_smoke_is_fail_closed_and_four_gpu() -> None:
    config_path = (
        ROOT
        / "configs/amlt/archive/2026-10-26/static_sud_taxonomy_softmax_qwen35_smoke_0901_v1.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["code"]["local_dir"].endswith(
        "amlt_inputs/static_sud_taxonomy_softmax_qwen35_smoke_0901_v1"
    )
    assert len(config["jobs"]) == 1
    job = config["jobs"][0]
    assert job["name"] == "qwen35-bi-taxonomy-softmax-smoke"
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert job["command"][0] == "set -euo pipefail"
    assert "WANDB_API_KEY" not in str(config)

    script = (
        ROOT
        / "scripts/archive/amlt/2026-10-26/sft/run_static_sud_taxonomy_softmax_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "--batch-size 1" in script
    assert "--nproc_per_node=4" in script
    assert "SEMANTIC_ID_INDEPENDENT_SMOKE_GATE_OK" in script
    assert "semantic_id_loss_scope" in script
    assert 'drift["changed_input_rows"] != 0' in script
    assert "Taxonomy loss must train output Semantic-ID rows only" in script
    assert 'drift["changed_input_rows"] != 32' in script
    assert "Frontier single smoke must change its 32 input rows" in script
    assert "verify_semantic_id_token_adapter" in script
    assert "--max-examples 64" in script


def test_frontier_smoke_uses_the_same_independent_gate() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / 'configs/amlt/archive/2026-10-26/static_sud_frontier_reweighting_qwen35_smoke_0901_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    job = config["jobs"][0]
    assert job["name"] == "qwen35-bi-frontier-reweighting-smoke"
    assert job["sku"] == "80G4-A100"
    assert job["submit_args"]["env"]["SEMANTIC_ID_SMOKE_METHOD"] == (
        "frontier"
    )
    assert "run_static_sud_taxonomy_softmax_smoke.sh" in "\n".join(
        job["command"]
    )
