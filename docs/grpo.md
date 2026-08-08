# GRPO

GRPO uses 1K or 5K veRL-formatted CoT datasets derived from the SFT subsets. It
does not use either SFT worker requirements file. AMLT jobs install CUDA
PyTorch, then `requirements-verl.lock`, then source
`scripts/amlt/grpo/prepare_writable_verl.sh`.

## Build Data

```bash
PYTHONPATH=src python -m new_interest.data.build_grpo_dataset
```

The builder writes `data/training/grpo_experiments/<task>/{1k,5k}.parquet`.
Rows contain veRL chat prompts, ground truth reward data, and source identity.
The prompt requires one short `<think>...</think>` sentence followed by exactly
`New topic: <category name>`.

## Runtime

`scripts/amlt/grpo/prepare_writable_verl.sh` requires `PROJECT_DIR`. It extracts
`vendor/verl-safegrpo-0.7.0.dev.zip` to `VERL_RUNTIME_DIR` (default
`$HOME/verl-runtime`), patches one FSDP attention-loading call, prepends the
runtime to `PYTHONPATH`, and imports veRL. Archive provenance and checksum are in
[vendor/README.md](../vendor/README.md).

`scripts/amlt/grpo/run_verl_grpo_new_interest.sh` invokes `verl.trainer.main_ppo` with the
custom reward in `src/new_interest/training/grpo_reward.py`. It requires model, adapter,
Train file, output, and experiment environment variables; use the AMLT wrappers
rather than invoking it with guessed defaults.

## AMLT Flow

1. Restore SFT adapter inputs and GRPO data.
2. Run `configs/amlt/qwen_grpo_preflight_map.yaml` to validate reward behavior.
3. Run `configs/amlt/qwen_grpo_smoke_map.yaml` for a two-step smoke job.
4. Run `configs/amlt/qwen_grpo_1k_map.yaml` for the mapped 1K jobs.

`scripts/amlt/grpo/run_qwen_grpo_map_job.sh` binds AMLT paths, selects the task, copies the
latest LoRA adapter to the durable job output, and records its global step.
AMLT injects `AMLT_CODE_DIR`, `AMLT_OUTPUT_DIR`, `AMLT_MAP_INPUT_DIR`, and mapped
job identity. Do not treat the wrapper as a portable local launcher.

Checkpoint storage defaults to `/scratch` and is ephemeral; preserve the final
adapter and run metadata in the job output. Record restoration details through
[manifests/README.md](../manifests/README.md).
