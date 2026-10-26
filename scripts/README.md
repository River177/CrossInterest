# Scripts

Shell scripts are grouped by lifecycle and execution environment. Run them from
the repository root; most AMLT wrappers require injected environment variables
and are not general local CLIs.

## Shared and Mainline AMLT Workers

| Path | Purpose |
|---|---|
| `amlt/evaluation/install_training_worker_deps.sh` | Install and verify pinned Qwen3 SFT/evaluation dependencies |
| `amlt/sft/run_cir_schedule.sh` | CIR scheduling and subsequent NI training |
| `amlt/sft/run_history_token.sh` | Text125/Token125 NI training |
| `amlt/sft/run_token_joint_alignment.sh` | NI plus bidirectional name/ID alignment |
| `amlt/grpo/prepare_writable_verl.sh` | Extract and patch the vendored veRL runtime |
| `amlt/grpo/run_verl_grpo_new_interest.sh` | Invoke veRL GRPO from environment variables |
| `amlt/grpo/run_qwen_grpo_map_job.sh` | Bind AMLT SFT inputs and GRPO outputs |

Mainline launch references and required templates are under `configs/amlt/`. YAML and its shell
wrapper form one interface: job-name parsing plus `AMLT_CODE_DIR`,
`AMLT_OUTPUT_DIR`, and `AMLT_MAP_INPUT_DIR` are part of it.

## Archived AMLT

Historical wrappers live under `archive/amlt/`; matching job definitions live
under `configs/amlt/archive/`. The 2026-10-26 cleanup moved 45 workers to
`archive/amlt/2026-10-26/`. Repository-root launchers use those new paths;
frozen-bundle launchers keep their internal worker paths. Preparation scripts
and shared runtime helpers remain in `amlt/` when still referenced. Archived
launchers require their original external inputs and worker environment.

## Local Utility

`sync_wandb_offline.sh` synchronizes local offline W&B runs.

`check_repository.py` checks Git candidates for large outputs and credential
patterns, and verifies archived file hashes and AMLT code roots.

Environment boundaries are documented in [training](../docs/training.md) and
[GRPO](../docs/grpo.md). Vendored runtime details are in
[vendor/README.md](../vendor/README.md).
