# CrossInterest Agent Notes

## Environment

- This is one Python 3.10+ `src/` package. `pyproject.toml` installs only the local data-building dependency (`pyarrow`); it is not a complete experiment environment.
- Use `python -m pip install -e .` for local dataset work. GPU training/Qwen3 evaluation uses the exact pins in `requirements-training-worker.txt`; Qwen3.5 uses the separate, incompatible `requirements-qwen35-worker.txt` (Transformers 5.x rather than 4.x).
- GRPO does not use either worker file. Its AMLT jobs install CUDA PyTorch, then `requirements-verl.lock`, then source `scripts/amlt/grpo/prepare_writable_verl.sh`; that script unpacks and patches `vendor/verl-safegrpo-0.7.0.dev.zip` and requires `PROJECT_DIR`.

## Package Layout

- Active modules are grouped by responsibility: builders in `new_interest.data`, SFT/rewards in `new_interest.training`, metrics/evaluators in `new_interest.evaluation`, and orchestration/reporting in `new_interest.experiments`.
- The former package-root modules such as `new_interest.build_dataset` and `new_interest.evaluate_beam` were removed. Use the migration table in `docs/migration-2026-08.md`; do not add forwarding wrappers unless an external consumer requires them.
- `new_interest.cli` is reserved for a future unified CLI. Keep substantive `main()` functions with their implementation rather than creating shallow pass-through modules.
- Put shared code in `new_interest.infrastructure` only when multiple workflows benefit from one real seam; workflow-specific helpers stay with their owning module.

## Verify Changes

- Full unit suite: `pytest -q`. Pytest already adds `src` through `pyproject.toml`.
- Focused test: `pytest -q tests/test_build_dataset.py` or select a collected node ID with `pytest -q path/to/test.py::test_name`.
- Tests are CPU/local; Torch-dependent cases skip when Torch is unavailable. There is no configured lint, formatter, or static typecheck command.

## Data Pipeline

- The dependency chain is `data/raw/MINDlarge_reclassified` -> `data/processed/new_interest` -> `data/training/generative_new_interest_balanced_200_h50` -> `data/training/sft_experiments`. Candidate-selection and GRPO data both branch from `sft_experiments`.
- Run package modules from the repository root, for example `python -m new_interest.data.build_dataset` after editable install, or use `PYTHONPATH=src python -m new_interest.data.build_dataset` without installation.
- Dataset builders are deterministic and write through temporary staging directories. They refuse to replace existing outputs unless `--overwrite` is explicit; do not add it casually because `data/` contains large, reproducibility-tracked artifacts.
- The SFT subsets are nested class-balanced prefixes (`1k`, `5k`, `10k`, `22k`), not independent samples. Beam evaluation consumes grouped Dev impressions, while candidate evaluation consumes expanded per-target rows; do not interchange those Parquet files.
- Builder metadata records source/output SHA-256 values and rule versions. Repository inputs must be serialized with `infrastructure.paths.portable_path`, not checkout-specific absolute paths. When changing schemas, prompts, sampling, or selection rules, update metadata/statistics and tests rather than only the Parquet writer.

## Experiment Boundaries

- `configs/sft/` drives LoRA training, `configs/beam/` constrained ranking, `configs/candidate/` K=10 evaluation, and active remote jobs live directly in `configs/amlt/`; completed definitions are under `configs/amlt/archive/`. Matching active wrappers live in `scripts/amlt/`, while completed wrappers are under `scripts/archive/amlt/`.
- Moving an archived AMLT YAML changes `$CONFIG_DIR`; keep `code.local_dir` resolving to the repository root. Treat each YAML, its wrapper, injected AMLT variables, and job-name parsing as one interface.
- CLI path/run arguments override config values; prefer overrides for smoke runs instead of editing canonical configs.
- Canonical configs commonly target full datasets, online W&B, Hugging Face models, and A100 jobs. For local smoke checks pass `--max-samples` or `--max-examples` and `--wandb-mode disabled`; do not run a config unchanged merely to validate code.
- SFT writes `checkpoints/`, `final_adapter/`, `token_audit.json`, and resolved config/summary files. Training aborts if any sample exceeds `max_seq_length`; fix or regenerate the input rather than silently truncating it.
- Beam and candidate evaluators checkpoint resumable Parquet shards inside their output directory. Reusing an output directory with different data/config can trigger identity checks or mix stale work, so use a fresh output directory when experiment identity changes.
- TRAPI candidate evaluation authenticates with `AzureCliCredential` followed by managed identity for scope `api://trapi/.default`; `azure-identity` and `openai` are runtime requirements but are not in the local project dependencies.

## Outputs And Reports

- `artifacts/` contains experiment outputs, including offline W&B runs; `docs/experiments/YYYY-MM-DD/` contains the human reports and `docs/experiments/README.md` is their index.
- `data/`, `artifacts/`, generated slide outputs, and `legacy/toys_and_games/raw_meta_Toys_and_Games/` are intentionally ignored. Record known restoration details in `manifests/`; never commit credentials, signed URLs, or large local data.
- `legacy/toys_and_games/` is unrelated historical material. Active code, configs, and documentation must not depend on it.
- Do not infer current behavior from reports when executable configs/scripts disagree. Preserve resolved configs, metrics, predictions, and source hashes when producing a new report.
