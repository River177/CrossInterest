# Data Pipeline

The pipeline converts reclassified MINDlarge behaviors into novel-interest
examples, generative prompts, nested SFT subsets, and evaluation/GRPO branches.
All builders are deterministic, write through staging directories, and require
`--overwrite` before replacing existing outputs.

## Novel-Interest Definition

A currently clicked interest is novel for one behavior when it does not occur
in that behavior's clicked history:

```text
target_interest in current_clicked_interests
target_interest not in history_clicked_interests
```

An interest is the combination of a top-level and second-level category, for
example `Crime, Law & Justice / Courts & Legal Cases`. Construction:

- uses clicked (`-1`) current news and does not use unclicked (`-0`) news;
- does not require the interest to have appeared in earlier exposures;
- emits one row per distinct novel interest in an impression;
- keeps all current clicked news associated with that target interest;
- does not build negatives or balance classes at this first stage;
- builds Train and Dev only because MIND Test has no click labels.

Novelty is limited to the history supplied in a MIND behavior. Earlier omitted
clicks and prior unclicked exposures are unknowable, and the same user may emit
the same target in multiple behaviors.

## Dependency Chain

```text
data/raw/MINDlarge_reclassified
  -> data/processed/new_interest
  -> data/training/generative_new_interest_balanced_200_h50
  -> data/training/sft_experiments
       -> data/training/candidate_selection_k10
       -> data/training/grpo_experiments
```

The complete, unbalanced generative dataset can also be written to
`data/training/generative_new_interest/`. See
[manifests/README.md](../../manifests/README.md) before restoring ignored data.

## 1. Base Dataset

Expected raw layout: `data/raw/MINDlarge_reclassified/{train,dev}/` with the
reclassified `news.tsv` and labeled `behaviors.tsv` files.

```bash
PYTHONPATH=src python -m new_interest.build_dataset \
  --input-dir data/raw/MINDlarge_reclassified \
  --output-dir data/processed/new_interest
```

Outputs:

```text
data/processed/new_interest/
├── train.parquet
├── dev.parquet
├── interest_vocabulary.parquet
├── statistics.json
└── metadata.json
```

Observed dataset statistics recorded for the current corpus:

| Split | Behaviors | Behaviors with novel interest | Rows | Target interests |
|---|---:|---:|---:|---:|
| Train | 2,186,664 | 1,219,553 | 1,489,348 | 124 |
| Dev | 365,199 | 199,644 | 243,118 | 119 |

The full vocabulary has 125 category/subcategory combinations.

| Field | Meaning |
|---|---|
| `sample_id` | Stable split/impression/interest identifier |
| `split` | `train` or `dev` |
| `impression_id`, `user_id`, `timestamp` | Source behavior identity |
| `history` | Ordered clicked news with ID, category, subcategory, and title |
| `target_interest` | Novel category and subcategory |
| `target_news` | All currently clicked news for the target interest |

Read large files by batch:

```python
import pyarrow.parquet as pq

parquet = pq.ParquetFile("data/processed/new_interest/train.parquet")
for batch in parquet.iter_batches(batch_size=8192):
    rows = batch.to_pylist()
```

## 2. Generative Data

The model receives no candidates and must answer
`New topic: <subcategory>`. `behavior_only` uses ordered news titles;
`interest_only` uses the corresponding ordered subcategories, including
repetitions. The full builder preserves complete histories:

```bash
PYTHONPATH=src python -m new_interest.build_generative_dataset
```

Each task contains `train.parquet` and `dev.parquet`, plus shared
`statistics.json` and `metadata.json`. Rows include `sample_id`,
`source_sample_id`, source identity fields, `task_type`, `history_length`,
`prompt`, `response`, `messages`, and `target_subcategory`. They intentionally
contain no top-level target category. Concrete prompts are in
[generative_data_examples.md](../generative_data_examples.md).

## 3. Balanced Recent-50 Data

Canonical training uses one deterministic Train target per impression, 200
examples for each class with at least 200 source examples, and the most recent
50 history items:

```bash
PYTHONPATH=src python -m new_interest.build_generative_dataset \
  --output-dir data/training/generative_new_interest_balanced_200_h50 \
  --one-target-per-train-impression \
  --history-limit 50 \
  --train-samples-per-class 200 \
  --train-min-class-count 200 \
  --filter-dev-to-train-classes \
  --sampling-seed 20260806
```

The result has 110 retained classes and 22,000 Train rows per task. Dev is not
balanced or reduced to one target; it is filtered to retained Train classes,
leaving 242,839 rows and 109 observed classes. Selection is independent of file
iteration order:

```text
target_priority  = SHA256("20260806:target:<source_sample_id>")
balance_priority = SHA256("20260806:balance:<source_sample_id>")
```

`selected_train_samples.parquet` records selection priorities and
`retained_interests.parquet` records source, selected, and Dev class counts.
Both tasks use `history[-50:]`; `history_length` is the number entering the
prompt. The source row remains recoverable through `source_sample_id`.

## 4. SFT Subsets

```bash
PYTHONPATH=src python -m new_interest.build_sft_subsets
```

This creates class-balanced, nested Train prefixes at 1K, 5K, 10K, and 22K in
`data/training/sft_experiments/train/`. It also creates grouped Dev files for
beam evaluation. Nested means every smaller subset is a prefix of the larger
selection, not an independent sample. Do not substitute expanded per-target
candidate files for grouped beam Dev files.

## 5. Evaluation And GRPO Branches

```bash
PYTHONPATH=src python -m new_interest.build_candidate_selection
PYTHONPATH=src python -m new_interest.build_grpo_dataset
```

Candidate selection creates paired `behavior_only` and `interest_only` K=10
rows from SFT evaluation data. GRPO creates 1K and 5K veRL-formatted CoT Train
sets from SFT subsets. Continue with [evaluation.md](../evaluation.md) or
[grpo.md](../grpo.md).

## Reproducibility

- Builder metadata records rule versions, source/output hashes, and row counts.
- Statistics record class frequencies and history/target distributions.
- Selection manifests retain stable sample IDs and deterministic priorities.
- Existing outputs are refused unless `--overwrite` is supplied; inspect the
  destination before using it.
- Run `pytest -q tests/test_build_dataset.py` for the focused local test suite.
