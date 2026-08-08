# CrossInterest

CrossInterest studies prediction of a user's next novel news interest from
reclassified MINDlarge histories. The repository covers deterministic dataset
construction, LoRA SFT, constrained-beam and candidate-selection evaluation,
and veRL GRPO experiments.

## Quick Start

Python 3.10+ is required. The editable install supports local data builders and
tests; GPU experiments use separate pinned environments.

```bash
python -m pip install -e .
pytest -q
PYTHONPATH=src python -m new_interest.data.build_dataset \
  --input-dir data/raw/MINDlarge_reclassified \
  --output-dir data/processed/new_interest
```

`data/` and `artifacts/` are ignored. Before running builders or experiments,
restore the required inputs as described in [manifests/README.md](manifests/README.md).
Builders do not replace existing outputs unless `--overwrite` is explicit.

## Project Map

| Path | Purpose |
|---|---|
| [docs/](docs/README.md) | Data, training, evaluation, GRPO, artifact, metric, and experiment documentation |
| `src/new_interest/` | Dataset builders, trainers, evaluators, rewards, and reporting modules |
| [configs/](configs/README.md) | SFT, beam, candidate, and AMLT experiment definitions |
| [scripts/](scripts/README.md) | AMLT worker entry points and environment helpers |
| [manifests/](manifests/README.md) | Templates for recording restoration of ignored data and artifacts |
| [slides/](slides/README.md) | Presentation source, assets, and generated outputs |
| [vendor/](vendor/README.md) | Vendored veRL runtime archive used by GRPO |
| `tests/` | Local CPU test suite |

## Main Workflows

1. Build datasets: [docs/data/pipeline.md](docs/data/pipeline.md)
2. Train LoRA adapters: [docs/training.md](docs/training.md)
3. Evaluate models: [docs/evaluation.md](docs/evaluation.md)
4. Run GRPO: [docs/grpo.md](docs/grpo.md)
5. Locate or restore outputs: [docs/artifacts.md](docs/artifacts.md)

Experiment results are indexed in
[docs/experiments/README.md](docs/experiments/README.md).
