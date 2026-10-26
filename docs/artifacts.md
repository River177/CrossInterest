# Data And Artifacts

`data/` and `artifacts/` are intentionally ignored by Git. Generated slide
outputs under `slides/output/` are also ignored. A clone therefore contains the
builders, configs, reports, and manifest templates, but not necessarily the
inputs or outputs needed to reproduce a reported run.

Local staging/readback trees `amlt_inputs/`, `outputs/`, `tmp/`, and
`.codex_tmp/` are also ignored. They remain on disk after the October 2026
cleanup; ignored frozen runtime copies must not replace versioned source
provenance. See the [current repository guide](repository.md).

## Conventions

- `data/raw/`: externally prepared source datasets.
- `data/processed/`: deterministic intermediate datasets.
- `data/training/`: generative, SFT, candidate, and GRPO datasets.
- `artifacts/`: local or downloaded training/evaluation outputs and offline W&B
  runs.
- `docs/experiments/`: durable human reports; these are tracked.

## Restore

Some experiment-specific manifests record real remote result prefixes and
hashes, including Text125/Token125 and joint alignment. Other inputs (including
raw MIND in this checkout) still need an authorized source. Consult the
specific experiment's restoration record, then:

1. Copy or download the required tree to the exact repository-relative path.
2. Verify checksums and sizes supplied by that source.
3. Record what was restored using the templates in
   [manifests/](../manifests/README.md).
4. Compare dataset `metadata.json`, `statistics.json`, source hashes, and
   resolved experiment configs with the report that consumes them.

Do not commit credentials, signed URLs, tokens, private endpoints, datasets,
model weights, or large predictions in a manifest. A manifest documents a
restoration event; it is not itself a downloader.

## Preserve A Result

Keep the resolved config, metrics, predictions, source hashes, model/adapter
identity, and logs needed to interpret the run. Evaluation output directories
also contain resumable shards; do not reuse them after changing data or config.
Add or update a report in [experiments/](experiments/README.md) and record where
the ignored files can be restored without inventing a location.
