# Repository Reorganization Migration

The August 2026 reorganization grouped Python modules by responsibility and
separated active AMLT jobs from completed experiment definitions. Behavior and
data formats did not change.

## Python Modules

Use these package prefixes:

| Previous prefix | Current prefix |
|---|---|
| `new_interest.build_*` | `new_interest.data.build_*` |
| `new_interest.train_sft` | `new_interest.training.train_sft` |
| `new_interest.sft_data` | `new_interest.training.sft_data` |
| `new_interest.grpo_reward` | `new_interest.training.grpo_reward` |
| `new_interest.evaluate_*` | `new_interest.evaluation.evaluate_*` |
| Metrics, diagnostics, preflight, and aggregation at package root | `new_interest.evaluation.*` |
| Serial runners and reports at package root | `new_interest.experiments.*` |

The five installed dataset-builder commands keep their command names; only
their internal entry-point targets changed.

## Configs And Scripts

- Active AMLT definitions remain in `configs/amlt/`.
- Completed definitions moved to `configs/amlt/archive/{sft-beam,candidate}/`.
- Active wrappers moved to `scripts/amlt/{evaluation,grpo}/`.
- Completed wrappers moved to `scripts/archive/amlt/`.

Update custom automation to the current paths rather than importing removed
package-root modules.

## Legacy Material

The unrelated Toys & Games category analysis moved from `docs/archive/` to
`legacy/toys_and_games/docs/`. Its local raw data moved beside it and remains
ignored by Git.
