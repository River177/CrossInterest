from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from new_interest.training.train_sft import (
    _build_parser,
    _restore_configured_save_steps,
    _gradient_checkpointing_kwargs,
    _resolve_adapter_path,
    _unwrap_model_for_reload_reference,
    audit_trainable_parameters,
    configure_row_only_adapter_saves,
)


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class FakeConfig:
    model_type = "qwen3_5_text"
    architectures = ["Qwen3_5ForCausalLM"]


class FakeModel:
    config = FakeConfig()

    def named_parameters(self):
        return iter(
            [
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    FakeParameter(100, requires_grad=False),
                ),
                (
                    "model.layers.0.self_attn.q_proj.lora_A.weight",
                    FakeParameter(10, requires_grad=True),
                ),
                (
                    "model.layers.0.mlp.up_proj.lora_A.weight",
                    FakeParameter(20, requires_grad=True),
                ),
            ]
        )


def test_reload_reference_removes_accelerate_fp32_wrapper() -> None:
    class WrappedModel:
        def __init__(self) -> None:
            self._original_forward = object()

    class FakeAccelerator:
        def __init__(self) -> None:
            self.calls = []

        def unwrap_model(
            self,
            model,
            *,
            keep_fp32_wrapper: bool,
            keep_torch_compile: bool,
        ):
            self.calls.append(
                (model, keep_fp32_wrapper, keep_torch_compile)
            )
            delattr(model, "_original_forward")
            return model

    class FakeTrainer:
        def __init__(self, model) -> None:
            self.model_wrapped = model
            self.accelerator = FakeAccelerator()

    model = WrappedModel()
    trainer = FakeTrainer(model)
    result = _unwrap_model_for_reload_reference(trainer)
    assert result is model
    assert trainer.accelerator.calls == [(model, False, True)]
    assert not hasattr(result, "_original_forward")


def test_resolve_adapter_path_allows_new_lora() -> None:
    assert _resolve_adapter_path({}) is None
    assert _resolve_adapter_path({"adapter_path": None}) is None


def test_resolve_adapter_path_requires_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Input adapter not found"):
        _resolve_adapter_path({"adapter_path": tmp_path})

    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert _resolve_adapter_path({"adapter_path": str(tmp_path)}) == tmp_path


def test_trainable_parameter_audit_records_model_and_groups() -> None:
    audit = audit_trainable_parameters(FakeModel())
    assert audit["base_model_class"] == "FakeModel"
    assert audit["model_type"] == "qwen3_5_text"
    assert audit["total_parameters"] == 130
    assert audit["trainable_parameters"] == 30
    assert audit["trainable_parameter_groups"] == {
        "q_proj": 10,
        "up_proj": 20,
    }
    assert audit["forbidden_trainable_parameters"] == []


def test_trainable_parameter_audit_detects_forbidden_modules() -> None:
    class VisionModel(FakeModel):
        def named_parameters(self):
            yield (
                "model.visual.merger.q_proj.lora_A.weight",
                FakeParameter(10, requires_grad=True),
            )

    audit = audit_trainable_parameters(
        VisionModel(),
        forbidden_substrings=("vision", "visual"),
    )
    assert audit["forbidden_trainable_parameters"] == [
        "model.visual.merger.q_proj.lora_A.weight"
    ]


def test_semantic_adapter_saves_force_row_only_state() -> None:
    class SaveModel:
        def __init__(self) -> None:
            self.calls = []

        def save_pretrained(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "saved"

    model = SaveModel()
    configure_row_only_adapter_saves(model)
    assert model.save_pretrained("output", safe_serialization=True) == "saved"
    assert model.calls == [
        (
            ("output",),
            {
                "safe_serialization": True,
                "save_embedding_layers": False,
            },
        )
    ]
    configure_row_only_adapter_saves(model)
    with pytest.raises(ValueError, match="must not save full embedding"):
        model.save_pretrained("output", save_embedding_layers=True)


@pytest.mark.parametrize("task", ("behavior", "interest"))
def test_qwen35_22k_config_preserves_comparison_contract(
    task: str,
) -> None:
    path = Path(
        f"configs/sft/experiments/qwen35_{task}_22k.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["model_name"] == "Qwen/Qwen3.5-9B"
    assert config["train_file"].endswith(
        f"{task}_only/22k.parquet"
    )
    assert config["enable_thinking"] is False
    assert config["epochs"] == 3
    assert config["learning_rate"] == 2e-5
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["target_modules"] == "all-linear"
    assert set(
        config["forbidden_trainable_module_substrings"]
    ) == {"vision", "visual"}


def test_sft_cli_accepts_distributed_batch_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--per-device-train-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--save-steps",
            "25",
            "--resume-from-checkpoint",
            "/tmp/checkpoint-25",
            "--checkpoint-dir",
            "/scratch/checkpoints",
            "--persistent-checkpoint-dir",
            "/mnt/output/checkpoints",
        ]
    )
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.save_steps == 25
    assert args.resume_from_checkpoint == "/tmp/checkpoint-25"
    assert args.checkpoint_dir == Path("/scratch/checkpoints")
    assert args.persistent_checkpoint_dir == Path(
        "/mnt/output/checkpoints"
    )


def test_resume_restores_configured_checkpoint_cadence() -> None:
    class Args:
        save_steps = 100

    class State:
        save_steps = 675

    state = State()
    _restore_configured_save_steps(Args(), state)
    assert state.save_steps == 100


@pytest.mark.parametrize(
    "model,view,condition,learning_rate",
    (
        (model, view, condition, learning_rate)
        for model, learning_rate in (("qwen35", 0.00002), ("llama31", 0.0002))
        for view in ("behavior", "interest")
        for condition in ("vanilla", "weighted")
    ),
)
def test_crossfit_configs_enable_periodic_checkpointing(
    model: str,
    view: str,
    condition: str,
    learning_rate: float,
) -> None:
    path = Path(
        "configs/sft/experiments/"
        f"static_sud_crossfit_{model}_{view}_{condition}.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 100
    assert config["save_total_limit"] == 3
    assert config["persistent_checkpoint_limit"] == 3
    assert config["gradient_accumulation_steps"] == 8
    assert config["learning_rate"] == learning_rate
    if model == "llama31":
        assert config["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        if condition == "weighted":
            assert "reweighting_llama31_v1" in config["sample_weight_file"]


def test_crossfit_wrapper_persists_and_auto_resumes_checkpoints() -> None:
    script = Path(
        'scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_crossfit_reweighting.sh'
    ).read_text(encoding="utf-8")
    assert '4|8) ;;' in script
    assert "GRADIENT_ACCUMULATION_STEPS=$((32 / NUM_GPUS))" in script
    assert '--checkpoint-dir "$LOCAL_CHECKPOINT_ROOT"' in script
    assert (
        '--persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT"'
        in script
    )
    assert "--resume-from-checkpoint auto" in script
    assert '--save-steps "$CHECKPOINT_SAVE_STEPS"' in script
    assert "CROSSFIT_CHECKPOINT_RECOVERY_SMOKE_OK" in script
    assert 'MODEL_FAMILY="${2:-qwen35}"' in script
    assert "LLAMA31_SHARED_CACHE_READY" in script
    assert 'local_files_only=True' in script
    assert '"$MODEL_FAMILY" == "llama31"' in script
    assert '"${EVAL_NAMES[*]}" != "real"' in script
    assert 'MODEL_WEIGHT_FILE="$WEIGHT_ROOT/sample_weights/${VIEW}.parquet"' in script


def test_llama_crossfit_amlt_configs_are_two_phase_and_real_only() -> None:
    smoke = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/static_sud_crossfit_llama31_smoke_0822_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    behavior = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sud_crossfit_llama31_oof_behavior_0822_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    interest = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sud_crossfit_llama31_oof_interest_0822_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    final = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sud_crossfit_llama31_final_real_0822_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    assert len(smoke["jobs"]) == 2
    assert len(behavior["jobs"]) == 5
    assert len(interest["jobs"]) == 5
    assert len(final["jobs"]) == 4
    for job in smoke["jobs"] + behavior["jobs"] + interest["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["submit_args"]["env"]["STATIC_SUD_CROSSFIT_NUM_GPUS"] == "4"
        assert "HF_TOKEN" not in job["submit_args"]["env"]
    for job in final["jobs"]:
        assert job["sku"] == "80G8-A100"
        env = job["submit_args"]["env"]
        assert env["STATIC_SUD_CROSSFIT_NUM_GPUS"] == "8"
        assert env["CROSSFIT_EVAL_NAMES"] == "real"
        assert "HF_TOKEN" not in env
        command = "\n".join(job["command"])
        assert "final llama31" in command


def test_crossfit_g8_configs_preserve_global_batch_contract() -> None:
    smoke = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sud_crossfit_checkpoint_g8_smoke_0820_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    recovery = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sud_crossfit_oof_interest_fold2_recovery_g8_0820_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    final = yaml.safe_load(
        Path(
            "configs/amlt/archive/2026-10-26/static_sud_crossfit_final_0818_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    jobs = smoke["jobs"] + recovery["jobs"] + final["jobs"]
    assert len(jobs) == 6
    for job in jobs:
        assert job["sku"] == "80G8-A100"
        env = job["submit_args"]["env"]
        assert env["STATIC_SUD_CROSSFIT_NUM_GPUS"] == "8"
        assert env["CROSSFIT_CHECKPOINT_SAVE_STEPS"] == "100"


def test_behavior_eval_recovery_is_eval_only_and_pins_source_adapters() -> None:
    config = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sud_crossfit_behavior_eval_recovery_0821_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    assert [job["name"] for job in config["jobs"]] == [
        "behavior-vanilla-eval-recovery",
        "behavior-weighted-eval-recovery",
    ]
    expected_sources = {
        "behavior-vanilla-eval-recovery": (
            "7212788530.09261-77e8fede-8ba2"
        ),
        "behavior-weighted-eval-recovery": (
            "7212788530.09227-18411bb2-ac34"
        ),
    }
    for job in config["jobs"]:
        assert job["sku"] == "80G8-A100"
        assert job["sla_tier"] == "Standard"
        env = job["submit_args"]["env"]
        result_id = expected_sources[job["name"]]
        assert env["STATIC_SUD_CROSSFIT_NUM_GPUS"] == "8"
        assert env["CROSSFIT_SOURCE_RESULT_ID"] == result_id
        assert env["CROSSFIT_SOURCE_ROOT"].endswith(result_id)
        command = "\n".join(job["command"])
        assert "run_static_user_disjoint_crossfit_behavior_eval_recovery.sh" in command
        assert "run_static_user_disjoint_crossfit_reweighting.sh final" not in command

    script = Path(
        'scripts/archive/amlt/2026-10-26/evaluation/run_static_user_disjoint_crossfit_behavior_eval_recovery.sh'
    ).read_text(encoding="utf-8")
    assert "new_interest.training.train_sft" not in script
    assert "checkpoint-2025" in script
    assert "validate_persistent_checkpoint" in script
    assert "--expected-records 31392" in script
    assert "CROSSFIT_BEHAVIOR_EVAL_RECOVERY_GATE_OK" in script


def test_static_sud_wrapper_parameterizes_distributed_gpu_count() -> None:
    script = Path(
        "scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_balanced_sft.sh"
    ).read_text(encoding="utf-8")
    assert 'NUM_GPUS="${STATIC_SUD_NUM_GPUS:-8}"' in script
    assert 'DISTRIBUTED_GRAD_ACCUM=$((32 / NUM_GPUS))' in script
    assert '--nproc_per_node="$NUM_GPUS"' in script
    assert '--gradient-accumulation-steps "$DISTRIBUTED_GRAD_ACCUM"' in script
    assert '$(seq 0 $((NUM_GPUS - 1)))' in script
    assert '--num-shards "$NUM_GPUS"' in script


@pytest.mark.parametrize(
    "model,learning_rate",
    (
        ("qwen35", 2e-5),
        ("llama31", 2e-4),
    ),
)
def test_behavior_interest_sft_configs_preserve_protocol(
    model: str,
    learning_rate: float,
) -> None:
    path = Path(
        "configs/sft/experiments/"
        f"static_sud_{model}_behavior_interest_21600.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["train_file"].endswith(
        "behavior_interest_v1/train/behavior_interest/21600.parquet"
    )
    assert config["enable_thinking"] is False
    assert config["epochs"] == 3
    assert config["max_seq_length"] == 2048
    assert config["learning_rate"] == learning_rate
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["lora_dropout"] == 0.05
    assert config["target_modules"] == "all-linear"
    assert set(config["forbidden_trainable_module_substrings"]) == {
        "vision",
        "visual",
    }


def test_behavior_interest_amlt_job_is_four_gpu_and_serial() -> None:
    config = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sft_behavior_interest_serial_g4_0818_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    assert len(config["jobs"]) == 1
    job = config["jobs"][0]
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert job["submit_args"]["env"]["STATIC_SUD_NUM_GPUS"] == "4"
    assert (
        job["submit_args"]["env"]["BEHAVIOR_INTEREST_SAVE_STEPS"]
        == "100"
    )
    command = "\n".join(job["command"])
    assert command.startswith("set -euo pipefail")
    assert "run_static_user_disjoint_behavior_interest_serial.sh" in command

    script = Path(
        'scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_behavior_interest_serial.sh'
    ).read_text(encoding="utf-8")
    qwen_start = script.index('echo "SERIAL_PHASE_START qwen35"')
    qwen_complete = script.index('echo "SERIAL_PHASE_COMPLETE qwen35"')
    llama_start = script.index('echo "SERIAL_PHASE_START llama31"')
    assert qwen_start < qwen_complete < llama_start
    dependency_install = script.index(
        "install_qwen35_worker_deps.sh"
    )
    parquet_audit = script.index(
        'python - "$DATASET_DIR" "$TRAIN_FILE"'
    )
    assert dependency_install < parquet_audit
    assert '--nproc_per_node="$NUM_GPUS"' in script
    assert "--gradient-accumulation-steps 8" in script
    assert (
        'CHECKPOINT_SAVE_STEPS="${BEHAVIOR_INTEREST_SAVE_STEPS:-100}"'
        in script
    )
    assert '--save-steps "$CHECKPOINT_SAVE_STEPS"' in script
    assert '"expected_optimizer_steps": 2025' in script
    assert 'if [[ -z "${AMLT_MAP_INPUT_DIR:-}" ]]' in script
    assert 'cp -R "$source_root/." "$SERIAL_ROOT/"' in script
    assert 'echo "RECOVERY_BOOTSTRAP_OK $latest_checkpoint"' in script
    assert "optimizer.pt" in script
    assert "scheduler.pt" in script


def test_behavior_interest_llama_parallel_job_trains_then_evaluates() -> None:
    config = yaml.safe_load(
        Path(
            'configs/amlt/archive/2026-10-26/static_sft_behavior_interest_llama_parallel_g4_0819_v1.yaml'
        ).read_text(encoding="utf-8")
    )
    assert len(config["jobs"]) == 1
    job = config["jobs"][0]
    assert job["sku"] == "80G4-A100"
    assert job["sla_tier"] == "Standard"
    assert job["submit_args"]["env"]["STATIC_SUD_NUM_GPUS"] == "4"
    assert (
        job["submit_args"]["env"]["BEHAVIOR_INTEREST_SAVE_STEPS"]
        == "100"
    )
    command = "\n".join(job["command"])
    assert command.startswith("set -euo pipefail")
    assert "install_training_worker_deps.sh" in command
    assert (
        "run_static_user_disjoint_behavior_interest_llama_parallel.sh"
        in command
    )

    beam = json.loads(
        Path(
            "configs/beam/"
            "static_sud_llama31_behavior_interest_full.json"
        ).read_text(encoding="utf-8")
    )
    assert beam["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert beam["eval_file"].endswith(
        "behavior_interest/dev_full_grouped.parquet"
    )
    assert beam["num_beams"] == 32
    assert beam["top_k"] == 20

    script = Path(
        'scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_behavior_interest_llama_parallel.sh'
    ).read_text(encoding="utf-8")
    torchrun = script.index("python -m torch.distributed.run")
    evaluation = script.index("new_interest.evaluation.evaluate_beam")
    assert torchrun < evaluation
    assert '--nproc_per_node="$NUM_GPUS"' in script
    assert "--gradient-accumulation-steps 8" in script
    assert '--save-steps "$CHECKPOINT_SAVE_STEPS"' in script
    assert '"expected_optimizer_steps": 2025' in script
    assert "--expected-records 31392" in script
    assert 'metrics["supported_interest_count"]' in script
    assert "BEHAVIOR_INTEREST_LLAMA_PARALLEL_COMPLETE" in script


@pytest.mark.parametrize(
    "path",
    (
        "configs/amlt/archive/2026-10-26/static_sft_llama_ddp_smoke_0817_v3_g4.yaml",
        "configs/amlt/archive/2026-10-26/static_sft_qwen_beam_recovery_0817_v3_g4.yaml",
        "configs/amlt/archive/2026-10-26/static_sft_llama_full_recovery_0817_v3_g4.yaml",
    ),
)
def test_static_sud_g4_amlt_configs_bind_four_a100s(path: str) -> None:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for job in config["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        assert job["submit_args"]["env"]["STATIC_SUD_NUM_GPUS"] == "4"


@pytest.mark.parametrize(
    "path,mode",
    (
        (
            "configs/amlt/archive/2026-10-26/static_semantic_id_token_smoke_0818_v1.yaml",
            "smoke",
        ),
        (
            "configs/amlt/archive/2026-10-26/static_semantic_id_token_full_0818_v1.yaml",
            "full",
        ),
    ),
)
def test_semantic_id_amlt_configs_bind_protocol(
    path: str,
    mode: str,
) -> None:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert config["target"] == {
        "service": "sing",
        "name": "Feeds",
        "workspace_name": (
            "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
        ),
    }
    assert len(config["jobs"]) == 4
    names = {job["name"] for job in config["jobs"]}
    assert names == {
        f"qwen35-behavior-semantic-id-{mode}",
        f"qwen35-interest-semantic-id-{mode}",
        f"llama31-behavior-semantic-id-{mode}",
        f"llama31-interest-semantic-id-{mode}",
    }
    for job in config["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        env = job["submit_args"]["env"]
        assert env["SEMANTIC_ID_NUM_GPUS"] == "4"
        if job["name"].startswith("qwen35-"):
            assert env["HF_HOME"] == "/mnt/default/huggingface-cache"
        else:
            assert env["HF_HOME"] == (
                "/scratch/semantic-id-llama31-hf-cache"
            )
        command = "\n".join(job["command"])
        assert command.startswith("set -euo pipefail")
        assert (
            "run_static_user_disjoint_semantic_id_token_sft.sh"
            in command
        )
        assert command.endswith(mode)


@pytest.mark.parametrize(
    "path,expected_jobs,mode",
    (
        (
            "configs/amlt/archive/2026-10-26/static_semantic_id_token_recovery_smoke_0819_v1.yaml",
            4,
            "recovery-smoke",
        ),
        (
            "configs/amlt/archive/2026-10-26/static_semantic_id_token_recovery_full_0819_v1.yaml",
            4,
            "full",
        ),
    ),
)
def test_semantic_id_recovery_configs_are_persistent_and_resumable(
    path: str,
    expected_jobs: int,
    mode: str,
) -> None:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert len(config["jobs"]) == expected_jobs
    families = {job["name"].split("-", 1)[0] for job in config["jobs"]}
    assert families == {"qwen35", "llama31"}
    for job in config["jobs"]:
        assert job["sku"] == "80G4-A100"
        assert job["sla_tier"] == "Standard"
        env = job["submit_args"]["env"]
        assert env["HF_HOME"] == "/mnt/default/huggingface-cache"
        assert env["PIP_CACHE_DIR"].startswith("/mnt/default/")
        assert "HF_TOKEN" not in env
        assert env["SEMANTIC_ID_NUM_GPUS"] == "4"
        command = "\n".join(job["command"])
        assert command.startswith("set -euo pipefail")
        assert command.endswith(mode)


def test_semantic_id_wrapper_runs_all_smoke_and_full_gates() -> None:
    script = Path(
        'scripts/archive/amlt/2026-10-26/sft/run_static_user_disjoint_semantic_id_token_sft.sh'
    ).read_text(encoding="utf-8")
    assert 'NUM_GPUS="${SEMANTIC_ID_NUM_GPUS:-4}"' in script
    assert (
        'HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"'
        in script
    )
    assert '--nproc_per_node="$NUM_GPUS"' in script
    assert "audit_semantic_id_token_gradients" in script
    assert "--max-samples 32" in script
    assert "verify_semantic_id_token_adapter" in script
    assert "evaluate_id_token_ranking" in script
    assert "merge_id_token_shards" in script
    assert "--expected-records 31392" in script
    assert 'metrics["history_prediction_filtering"] is not False' in script
    assert "snapshot_download(name, local_files_only=True)" in script
    assert "SEMANTIC_ID_RECOVERY_CHECKPOINT_READY" in script
    assert 'FULL_SAVE_STEPS="${SEMANTIC_ID_FULL_SAVE_STEPS:-25}"' in script
    assert '--save-steps "$FULL_SAVE_STEPS"' in script
    assert 'RESUME_ARGS=(--resume-from-checkpoint "$RESUME_CHECKPOINT")' in script


def test_semantic_id_llama_requirements_pin_peft_018() -> None:
    requirements = Path(
        "requirements-semantic-id-token-llama-worker.txt"
    ).read_text(encoding="utf-8")
    assert "transformers==4.53.3" in requirements
    assert "peft==0.18.0" in requirements
    installer = Path(
        "scripts/amlt/evaluation/"
        "install_semantic_id_token_llama_worker_deps.sh"
    ).read_text(encoding="utf-8")
    assert "requirements-semantic-id-token-llama-worker.txt" in installer
    assert '"peft": "0.18.0"' in installer


def test_gradient_checkpointing_defaults_to_reentrant() -> None:
    assert _gradient_checkpointing_kwargs({}) == {
        "use_reentrant": True
    }


@pytest.mark.parametrize("task", ("behavior", "interest"))
def test_static_llama_disables_reentrant_checkpointing(
    task: str,
) -> None:
    path = Path(
        "configs/sft/experiments/"
        f"static_sud_llama31_{task}_21600.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _gradient_checkpointing_kwargs(config) == {
        "use_reentrant": False
    }


def test_static_qwen35_behavior_interest_disables_reentrant_checkpointing(
) -> None:
    path = Path(
        "configs/sft/experiments/"
        "static_sud_qwen35_behavior_interest_21600.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _gradient_checkpointing_kwargs(config) == {
        "use_reentrant": False
    }


@pytest.mark.parametrize("task", ("behavior", "interest"))
def test_semantic_id_qwen35_disables_reentrant_checkpointing(
    task: str,
) -> None:
    path = Path(
        "configs/sft/experiments/"
        f"static_sud_semantic_id_qwen35_{task}_21600.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _gradient_checkpointing_kwargs(config) == {
        "use_reentrant": False
    }


def test_qwen35_distributed_training_prefetches_model_once() -> None:
    script = Path(
        "scripts/archive/amlt/2026-10-26/sft/run_qwen35_22k.sh"
    ).read_text(encoding="utf-8")
    prefetch = script.index("snapshot_download")
    offline = script.index("HF_HUB_OFFLINE=1")
    torchrun = script.index("torch.distributed.run")
    assert prefetch < offline < torchrun


@pytest.mark.parametrize("task", ("behavior", "interest"))
def test_qwen3_cot_v2_config_uses_native_thinking_contract(
    task: str,
) -> None:
    path = Path(
        f"configs/sft/experiments/qwen3_{task}_cot_v2_1k.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["model_name"] == "Qwen/Qwen3-8B"
    assert config["train_file"].endswith(
        f"cot_sft_v2_experiments/{task}_only/1k.parquet"
    )
    assert config["enable_thinking"] is True
    assert config["epochs"] == 2
    assert config["learning_rate"] == 2e-5
    assert (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
        == 32
    )
