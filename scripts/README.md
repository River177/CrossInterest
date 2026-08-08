# Scripts

Shell scripts are grouped by lifecycle and execution environment. Run them from
the repository root; most AMLT wrappers require injected environment variables
and are not general local CLIs.

## Active AMLT

| Path | Purpose |
|---|---|
| `amlt/evaluation/install_training_worker_deps.sh` | Install and verify pinned Qwen3 SFT/evaluation dependencies |
| `amlt/evaluation/run_qwen_cot_label_eval_map.sh` | Run mapped CoT label-ranking evaluation |
| `amlt/grpo/prepare_writable_verl.sh` | Extract and patch the vendored veRL runtime |
| `amlt/grpo/run_verl_grpo_new_interest.sh` | Invoke veRL GRPO from environment variables |
| `amlt/grpo/run_qwen_grpo_map_job.sh` | Bind AMLT SFT inputs and GRPO outputs |

Active job definitions are directly under `configs/amlt/`. YAML and its shell
wrapper form one interface: job-name parsing plus `AMLT_CODE_DIR`,
`AMLT_OUTPUT_DIR`, and `AMLT_MAP_INPUT_DIR` are part of it.

## Archived AMLT

Completed SFT/beam and candidate wrappers live under `archive/amlt/`; matching
job definitions live under `configs/amlt/archive/`. They remain executable for
reproduction but are not the default starting point for new experiments.

## Local Utility

`sync_wandb_offline.sh` synchronizes local offline W&B runs.

Environment boundaries are documented in [training](../docs/training.md) and
[GRPO](../docs/grpo.md). Vendored runtime details are in
[vendor/README.md](../vendor/README.md).
