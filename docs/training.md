# Training

SFT trains completion-only LoRA adapters for `behavior_only` or `interest_only`
prompts. Canonical Train data is under
`data/training/sft_experiments/train/<task>/<size>.parquet`, where sizes 1K,
5K, 10K, and 22K are nested class-balanced subsets.

## Environment

The editable package only installs local data dependencies. GPU training and
Qwen3 evaluation use the exact pins in `requirements-training-worker.txt`.
Qwen3.5 uses `requirements-qwen35-worker.txt`, whose Transformers 5.x stack is
incompatible with the Qwen3 worker environment; keep them separate.

```bash
python -m pip install -r requirements-training-worker.txt
python -m pip install -e .
```

On AMLT workers, `scripts/amlt/evaluation/install_training_worker_deps.sh` installs and verifies
the Qwen3 pins.

## Local Smoke Run

Canonical configs target full datasets, online W&B, and GPU models. Override
paths and limits rather than editing those configs:

```bash
python -m new_interest.training.train_sft \
  --config configs/sft/qwen3_8b_lora.yaml \
  --train-file data/training/sft_experiments/train/interest_only/1k.parquet \
  --output-dir artifacts/smoke/sft \
  --max-samples 8 \
  --wandb-mode disabled
```

Training audits every sequence before model construction and aborts if any row
exceeds `max_seq_length`; inputs are not silently truncated. A run writes
checkpoints, `final_adapter/`, `token_audit.json`, resolved configuration, and a
summary under its output directory.

## Configurations And Jobs

- `configs/sft/qwen3_8b_lora.yaml` and `llama31_8b_lora.yaml` are base recipes.
- `configs/sft/experiments/` fixes the four canonical 22K model/task conditions.
- `configs/amlt/archive/sft-beam/qwen3_8b_scaling.yaml` records Qwen3 1K/5K/10K SFT plus beam.
- `configs/amlt/archive/sft-beam/qwen_llama_22k.yaml` records Qwen3/Llama 22K SFT plus beam.
- `scripts/archive/amlt/sft-beam/run_qwen_sft_job.sh` and `run_sft_22k_job.sh` are archived AMLT worker entry
  points and depend on injected AMLT environment variables.

See [configs/README.md](../configs/README.md), [scripts/README.md](../scripts/README.md),
and the [experiment index](experiments/README.md).
