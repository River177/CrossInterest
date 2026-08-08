# Evaluation

The repository has two distinct evaluation families. Use a fresh output
directory whenever data or config identity changes; evaluators keep resumable
shards and may reject mismatched identities or otherwise encounter stale work.

## Constrained Beam Ranking

`new_interest.evaluation.evaluate_beam` ranks the retained interest vocabulary by
constrained generation. It consumes grouped Dev impressions with one prompt and
all acceptable targets for that impression.

```bash
python -m new_interest.evaluation.evaluate_beam \
  --config configs/beam/qwen3_interest_1k.json \
  --adapter-path artifacts/example/sft/final_adapter \
  --output-dir artifacts/smoke/beam \
  --max-examples 8 \
  --wandb-mode disabled
```

Outputs include `beam_predictions.parquet`, `metrics.json`,
`resolved_config.json`, and resumable `shards/`. Metrics are Recall, NDCG, and
Hit at configured cutoffs; definitions and multi-target behavior are documented
in [metrics/recall_ndcg_hit.md](metrics/recall_ndcg_hit.md).

## K=10 Candidate Selection

Candidate evaluation consumes expanded per-target rows from
`data/training/candidate_selection_k10`, not grouped beam files. Local Qwen
models use `new_interest.evaluation.evaluate_candidate_qwen`; hosted models use
`new_interest.evaluation.evaluate_candidate_trapi` and require their own runtime packages
and TRAPI authentication.

```bash
python -m new_interest.evaluation.evaluate_candidate_qwen \
  --config configs/candidate/qwen_base_interest.json \
  --output-dir artifacts/smoke/candidate \
  --max-examples 8 \
  --wandb-mode disabled
```

Candidate outputs include predictions, metrics, resolved config, and shards.
The candidate data rules and examples are in
[experiments/2026-08-08/candidate_selection_data_examples.md](experiments/2026-08-08/candidate_selection_data_examples.md).

## Configuration Choice

- `configs/beam/`: model/task/subset constrained-ranking runs.
- `configs/candidate/`: base, SFT, sampling, likelihood, permutation, and prompt
  variants for candidate selection.
- `configs/amlt/`: remote orchestration; the AMLT YAML and referenced worker
  script form one interface.

CLI path, run-name, and sample-limit arguments override config values. Preserve
`metrics.json`, predictions, resolved config, and source hashes when reporting a
result. See [artifacts.md](artifacts.md) for ignored output handling.
