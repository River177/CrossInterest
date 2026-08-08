from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Experiment:
    task_type: str
    size: str
    reuse_remote_adapter: str | None = None

    @property
    def key(self) -> str:
        return f"{self.task_type}-{self.size}"

    @property
    def short_task(self) -> str:
        return "interest" if self.task_type == "interest_only" else "behavior"

    @property
    def run_name(self) -> str:
        return f"qwen3-8b-{self.short_task}-{self.size}-seed42"


EXPERIMENTS = (
    Experiment(
        "interest_only",
        "1k",
        "artifacts/sft/pilot-qwen3-8b-interest-1k",
    ),
    Experiment("behavior_only", "1k"),
    Experiment("interest_only", "5k"),
    Experiment("behavior_only", "5k"),
    Experiment("interest_only", "10k"),
    Experiment("behavior_only", "10k"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now()}] $ {' '.join(command)}\n")
        log.flush()
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: "
            f"{' '.join(command)}; see {log_path}"
        )


def _remote_shell(
    host: str,
    remote_root: str,
    script: str,
    log_path: Path,
) -> None:
    command = [
        "ssh",
        host,
        (
            "set -euo pipefail; "
            "source ~/venvs/cross-interest-sft/bin/activate; "
            f"cd {remote_root}; "
            "export PYTHONPATH=$PWD/src; "
            "export TOKENIZERS_PARALLELISM=false; "
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
            f"{script}"
        ),
    ]
    _run(command, log_path)


def _remote_exists(
    host: str,
    remote_root: str,
    relative_path: str,
) -> bool:
    return subprocess.run(
        [
            "ssh",
            host,
            f"test -e {remote_root}/{relative_path}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _sync_remote_directory(
    host: str,
    remote_path: str,
    local_path: Path,
    log_path: Path,
) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            f"{host}:{remote_path.rstrip('/')}/",
            f"{local_path}/",
        ],
        log_path,
    )


def _load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "created_at": _now(),
        "updated_at": _now(),
        "experiments": {
            experiment.key: {
                **asdict(experiment),
                "status": "pending",
                "history": [],
            }
            for experiment in EXPERIMENTS
        },
    }


def _update_status(
    state_path: Path,
    state: dict[str, Any],
    experiment: Experiment,
    status: str,
    **extra: Any,
) -> None:
    entry = state["experiments"][experiment.key]
    entry["status"] = status
    entry.update(extra)
    entry["history"].append({"status": status, "at": _now()})
    state["updated_at"] = _now()
    _atomic_json(state_path, state)


def _best_experiment(
    state: dict[str, Any],
    task_type: str,
) -> Experiment:
    candidates = [
        experiment
        for experiment in EXPERIMENTS
        if experiment.task_type == task_type
    ]

    def ranking_key(experiment: Experiment) -> tuple[float, float, float, int]:
        metrics = state["experiments"][experiment.key].get("metrics_10k")
        if not isinstance(metrics, dict):
            raise ValueError(
                f"{experiment.key}: missing Dev 10K metrics"
            )
        size_value = int(experiment.size.removesuffix("k"))
        return (
            float(metrics["ndcg_at_20"]),
            float(metrics["recall_at_20"]),
            float(metrics["ndcg_at_10"]),
            -size_value,
        )

    return max(candidates, key=ranking_key)


def _train_script(
    experiment: Experiment,
    remote_output: str,
) -> str:
    train_file = (
        "data/training/sft_experiments/train/"
        f"{experiment.task_type}/{experiment.size}.parquet"
    )
    return (
        f"mkdir -p {shlex.quote(remote_output)}/wandb; "
        f"export WANDB_DIR={shlex.quote(remote_output)}/wandb; "
        "python -m new_interest.train_sft "
        "--config configs/sft/qwen3_8b_lora.yaml "
        f"--train-file {train_file} "
        f"--output-dir {remote_output} "
        f"--run-name {experiment.run_name} "
        "--wandb-mode offline"
    )


def _beam_script(
    experiment: Experiment,
    remote_adapter: str,
    remote_output: str,
    eval_scope: str,
    max_examples: int | None = None,
) -> str:
    eval_file = (
        "data/training/sft_experiments/eval/"
        f"{experiment.task_type}/dev_{eval_scope}_grouped.parquet"
    )
    config = {
        "model_name": "Qwen/Qwen3-8B",
        "adapter_path": f"{remote_adapter}/final_adapter",
        "eval_file": eval_file,
        "retained_interests_file": (
            "data/training/"
            "generative_new_interest_balanced_200_h50/"
            "retained_interests.parquet"
        ),
        "output_dir": remote_output,
        "run_name": f"{experiment.run_name}-beam-{eval_scope}",
        "wandb_project": "cross-interest-generative-sft",
        "wandb_mode": "offline",
        "enable_thinking": False,
        "batch_size": 2,
        "checkpoint_size": 64,
        "num_beams": 32,
        "top_k": 20,
        "max_new_tokens": 32,
        "cutoffs": [1, 5, 10, 20],
    }
    if max_examples is not None:
        config["max_examples"] = max_examples
    serialized = json.dumps(config)
    return (
        f"mkdir -p {shlex.quote(remote_output)}/wandb; "
        f"export WANDB_DIR={shlex.quote(remote_output)}/wandb; "
        "python -c "
        + shlex.quote(
            "import json; "
            "from new_interest.evaluate_beam import evaluate_beam; "
            "config=json.loads("
            f"{serialized!r}"
            "); "
            "print(json.dumps(evaluate_beam(config), indent=2))"
        )
    )


def run_serial(
    *,
    host: str,
    remote_root: str,
    local_root: Path,
    state_path: Path,
) -> None:
    state = _load_state(state_path)
    logs_dir = local_root / "logs"

    for experiment in EXPERIMENTS:
        entry = state["experiments"][experiment.key]
        log_path = logs_dir / f"{experiment.key}.log"
        local_experiment = (
            local_root / experiment.task_type / experiment.size
        )
        remote_output = (
            f"artifacts/sft/qwen3/{experiment.task_type}/"
            f"{experiment.size}"
        )
        if experiment.reuse_remote_adapter is not None:
            remote_output = experiment.reuse_remote_adapter

        try:
            adapter_ready = _remote_exists(
                host,
                remote_root,
                f"{remote_output}/final_adapter/adapter_model.safetensors",
            )
            if not adapter_ready:
                _update_status(
                    state_path,
                    state,
                    experiment,
                    "training",
                )
                _remote_shell(
                    host,
                    remote_root,
                    _train_script(experiment, remote_output),
                    log_path,
                )
                adapter_ready = _remote_exists(
                    host,
                    remote_root,
                    (
                        f"{remote_output}/final_adapter/"
                        "adapter_model.safetensors"
                    ),
                )
                if not adapter_ready:
                    raise RuntimeError(
                        f"{experiment.key}: final adapter missing"
                    )
            _update_status(
                state_path,
                state,
                experiment,
                "trained",
                remote_adapter=remote_output,
            )

            remote_eval = (
                f"artifacts/beam/qwen3/{experiment.task_type}/"
                f"{experiment.size}/dev10k"
            )
            metrics_ready = _remote_exists(
                host,
                remote_root,
                f"{remote_eval}/metrics.json",
            )
            if not metrics_ready:
                _update_status(
                    state_path,
                    state,
                    experiment,
                    "evaluating_10k",
                )
                _remote_shell(
                    host,
                    remote_root,
                    _beam_script(
                        experiment,
                        remote_output,
                        remote_eval,
                        "10k",
                    ),
                    log_path,
                )

            _update_status(
                state_path,
                state,
                experiment,
                "persisting",
            )
            _sync_remote_directory(
                host,
                f"{remote_root}/{remote_output}/final_adapter",
                local_experiment / "final_adapter",
                log_path,
            )
            for filename in (
                "run_summary.json",
                "resolved_config.json",
                "token_audit.json",
            ):
                if _remote_exists(
                    host,
                    remote_root,
                    f"{remote_output}/{filename}",
                ):
                    _run(
                        [
                            "rsync",
                            "-az",
                            (
                                f"{host}:{remote_root}/{remote_output}/"
                                f"{filename}"
                            ),
                            str(local_experiment / filename),
                        ],
                        log_path,
                    )
            _sync_remote_directory(
                host,
                f"{remote_root}/{remote_eval}",
                local_experiment / "eval_10k",
                log_path,
            )
            if _remote_exists(
                host,
                remote_root,
                f"{remote_output}/wandb",
            ):
                _sync_remote_directory(
                    host,
                    f"{remote_root}/{remote_output}/wandb",
                    local_experiment / "wandb_train",
                    log_path,
                )
            _remote_shell(
                host,
                remote_root,
                (
                    f"rm -rf {remote_output}/checkpoints; "
                    "true"
                ),
                log_path,
            )
            metrics = json.loads(
                (local_experiment / "eval_10k/metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            train_summary_path = local_experiment / "run_summary.json"
            train_summary = (
                json.loads(train_summary_path.read_text(encoding="utf-8"))
                if train_summary_path.is_file()
                else None
            )
            _update_status(
                state_path,
                state,
                experiment,
                "evaluated_10k",
                metrics_10k=metrics,
                train_summary=train_summary,
                local_path=str(local_experiment.resolve()),
            )
        except Exception as error:
            _update_status(
                state_path,
                state,
                experiment,
                "failed",
                error=str(error),
                log_path=str(log_path.resolve()),
            )
            raise

    best_models: dict[str, str] = {}
    for task_type in ("interest_only", "behavior_only"):
        experiment = _best_experiment(state, task_type)
        best_models[task_type] = experiment.key
        entry = state["experiments"][experiment.key]
        remote_adapter = str(entry["remote_adapter"])
        remote_eval = (
            f"artifacts/beam/qwen3/{experiment.task_type}/"
            f"{experiment.size}/devfull"
        )
        local_experiment = (
            local_root / experiment.task_type / experiment.size
        )
        log_path = logs_dir / f"{experiment.key}.log"
        try:
            metrics_ready = _remote_exists(
                host,
                remote_root,
                f"{remote_eval}/metrics.json",
            )
            if not metrics_ready:
                _update_status(
                    state_path,
                    state,
                    experiment,
                    "evaluating_full",
                )
                _remote_shell(
                    host,
                    remote_root,
                    _beam_script(
                        experiment,
                        remote_adapter,
                        remote_eval,
                        "full",
                    ),
                    log_path,
                )
            _sync_remote_directory(
                host,
                f"{remote_root}/{remote_eval}",
                local_experiment / "eval_full",
                log_path,
            )
            full_metrics = json.loads(
                (local_experiment / "eval_full/metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            _update_status(
                state_path,
                state,
                experiment,
                "completed",
                metrics_full=full_metrics,
            )
        except Exception as error:
            _update_status(
                state_path,
                state,
                experiment,
                "failed",
                error=str(error),
                log_path=str(log_path.resolve()),
            )
            raise

    state["best_models"] = best_models
    state["completed_at"] = _now()
    state["updated_at"] = _now()
    _atomic_json(state_path, state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable Qwen3 experiments serially on GCR."
    )
    parser.add_argument("--host", default="gcr-a100")
    parser.add_argument(
        "--remote-root",
        default="~/CrossInterest",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("artifacts/gcr/qwen3"),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("artifacts/gcr/qwen3/serial_state.json"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_serial(
        host=args.host,
        remote_root=args.remote_root,
        local_root=args.local_root,
        state_path=args.state_path,
    )


if __name__ == "__main__":
    main()
