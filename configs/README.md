# Configurations

All paths are repository-relative. CLI path, run-name, and sample-limit options
normally override config values, which is preferable for smoke runs.

| Directory | Consumer | Contents |
|---|---|---|
| `sft/` | `python -m new_interest.training.train_sft` | Qwen3/Llama LoRA recipes and fixed 22K experiment variants |
| `beam/` | `python -m new_interest.evaluation.evaluate_beam` | Model, task, adapter, grouped Dev, beam, and output settings |
| `candidate/` | Candidate evaluators | Base/SFT, sampling, likelihood, permutation, frequency-matched, and prompt variants |
| `amlt/` | `amlt` | Mainline launch references and required preparation templates; historical definitions are under `amlt/archive/` |

The 2026-10-26 cleanup retained 25 launch references/templates and moved 165
historical definitions to `amlt/archive/2026-10-26/`. Their `code.local_dir`
values still resolve to the same repository or frozen-bundle roots. Placement
in `amlt/` does not mean a job is running. See the
[path mapping](../manifests/repository_archive_20261026.json) for older report
and manifest references, which preserve original submitted paths.

Canonical configs may point at full ignored datasets, online W&B, Hugging Face
models, and A100 targets. For local validation, use `--max-samples` or
`--max-examples`, a fresh output directory, and `--wandb-mode disabled` where
supported. Do not submit an AMLT config merely to syntax-check local code.

AMLT YAML and its referenced script are one interface. Job-name parsing and
injected `AMLT_CODE_DIR`, `AMLT_OUTPUT_DIR`, and `AMLT_MAP_INPUT_DIR` variables
are part of that interface. See [scripts/README.md](../scripts/README.md),
[training](../docs/training.md), [evaluation](../docs/evaluation.md), and
[GRPO](../docs/grpo.md).
