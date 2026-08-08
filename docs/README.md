# Documentation

Use this page as the documentation entry point. Paths and commands are relative
to the repository root unless stated otherwise.

## Workflows

- [Data pipeline](data/pipeline.md): dataset definitions, schemas, stages, and reproducibility.
- [Training](training.md): SFT data, environments, commands, and outputs.
- [Evaluation](evaluation.md): constrained beam, candidate selection, and metrics.
- [GRPO](grpo.md): veRL dataset, runtime preparation, preflight, and jobs.
- [Artifacts](artifacts.md): ignored outputs, provenance, and restoration records.

## Reference

- [Experiment reports](experiments/README.md)
- [Generative data examples](generative_data_examples.md)
- [Recall, NDCG, and Hit](metrics/recall_ndcg_hit.md)
- [Recall, NDCG, and Hit (Chinese)](metrics/recall_ndcg_hit_zh.md)
- [Configuration guide](../configs/README.md)
- [Script guide](../scripts/README.md)
- [Restoration manifests](../manifests/README.md)
- [August 2026 repository migration](migration-2026-08.md)

`docs/archive/` contains historical CrossInterest material and is not
authoritative for the current executable pipeline. Unrelated project material
is isolated under [`legacy/`](../legacy/README.md).
