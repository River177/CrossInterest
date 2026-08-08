from __future__ import annotations

from new_interest.run_serial_experiments import (
    EXPERIMENTS,
    _beam_script,
    _best_experiment,
    _train_script,
)


def test_best_experiment_uses_documented_tiebreaks() -> None:
    state = {
        "experiments": {
            experiment.key: {
                "metrics_10k": {
                    "ndcg_at_20": 0.2,
                    "recall_at_20": (
                        0.4 if experiment.size == "5k" else 0.3
                    ),
                    "ndcg_at_10": 0.1,
                }
            }
            for experiment in EXPERIMENTS
        }
    }
    assert _best_experiment(state, "interest_only").size == "5k"
    assert _best_experiment(state, "behavior_only").size == "5k"


def test_best_experiment_prefers_smaller_exact_tie() -> None:
    state = {
        "experiments": {
            experiment.key: {
                "metrics_10k": {
                    "ndcg_at_20": 0.2,
                    "recall_at_20": 0.3,
                    "ndcg_at_10": 0.1,
                }
            }
            for experiment in EXPERIMENTS
        }
    }
    assert _best_experiment(state, "interest_only").size == "1k"


def test_remote_scripts_include_required_settings() -> None:
    experiment = next(
        value
        for value in EXPERIMENTS
        if value.key == "behavior_only-5k"
    )
    train_script = _train_script(experiment, "artifacts/example")
    assert "behavior_only/5k.parquet" in train_script
    assert "--wandb-mode offline" in train_script
    assert "WANDB_DIR=artifacts/example/wandb" in train_script

    beam_script = _beam_script(
        experiment,
        "artifacts/example",
        "artifacts/eval",
        "10k",
    )
    assert '"num_beams": 32' in beam_script
    assert '"top_k": 20' in beam_script
    assert '"checkpoint_size": 64' in beam_script
    assert "dev_10k_grouped.parquet" in beam_script
    assert "json.loads" in beam_script
