# Data Pipeline

The pipeline converts reclassified MINDlarge behaviors into a user-level
future-interest benchmark, impression-level diagnostic datasets, generative
prompts, nested SFT subsets, and evaluation/GRPO branches. All builders are
deterministic, write through staging directories, and require `--overwrite`
before replacing existing outputs.

## Novel-Interest Definition

The preferred benchmark for future novel-interest retrieval uses the fixed
history supplied by MIND as the user profile:

```text
profile(user) = row-provided fixed MIND history

future_novel_targets(user)
  = all clicked topics in the labeled Train+Dev future window
    - fixed history topics
```

Its unit is one user profile mapped to a multi-label target set. It preserves
ordered and repeated click events, while the target set naturally contains
each taxonomy label once. Impression candidates are not model inputs; they are
used only to separate exposed-but-unclicked weak negatives from never-exposed
unknown labels.

The separate dynamic dataset remains useful for impression-level diagnosis. A
currently clicked interest is dynamic-novel when it does not occur in the
user's initial MIND history or any strictly earlier positive click:

```text
dynamic_history(user, time)
  = initial MIND history + all earlier positive clicks

novel_targets
  = current clicked interests - dynamic history interests
```

Train events are replayed first. Their final per-user state carries into Dev,
and earlier Dev clicks update later Dev examples. Events sharing one timestamp
are all labeled before any click from that timestamp updates history.

The legacy static target-row dataset also defines novelty against fixed
history, but splits one profile's future outcomes into separate
impression-target rows. It lacks complete exposure sets and zero-target users,
so it is not the source for the new user-level benchmark.

The full diagnosis of this behavior, the legacy processing path, and why
earlier impression clicks were omitted is documented in
[static_history_and_dynamic_replay.md](static_history_and_dynamic_replay.md).

An interest is the combination of a top-level and second-level category, for
example `Crime, Law & Justice / Courts & Legal Cases`.

Dynamic construction:

- uses clicked (`-1`) current news and does not use unclicked (`-0`) news;
- emits one row per impression, including impressions with no novel target;
- records all clicked, novel, and repeated interests separately;
- appends every positive click after labeling the current timestamp group;
- preserves the full chronological click history before each impression;
- does not balance classes at this first stage;
- builds Train and Dev only because MIND Test has no click labels.

The initial MIND history predates the labeled window and has no per-click
timestamps. Prior unclicked exposures remain unknowable.

## Dependency Chain

```text
data/raw/MINDlarge_reclassified
  -> data/processed/future_novel_interest_users_v1
       -> artifacts/future-novel-interest/baselines-v1
       -> docs/experiments/2026-08-12/future_novel_interest_baselines_report.md

static user-disjoint single-target branch:
data/raw/MINDlarge_reclassified
  -> data/processed/new_interest_static_user_disjoint_v1

diagnostic dynamic branch:
data/raw/MINDlarge_reclassified
  -> data/processed/new_interest_dynamic_v1
  -> data/training/generative_new_interest_dynamic_balanced_200_h50

legacy static branch:
data/raw/MINDlarge_reclassified
  -> data/processed/new_interest
  -> data/training/generative_new_interest_balanced_200_h50
  -> data/training/sft_experiments
       -> data/training/candidate_selection_k10
       -> data/training/grpo_experiments
       -> artifacts/sft-training-data-analysis/v1
            -> docs/experiments/2026-08-13/
               sft_training_data_similarity_analysis.md
```

The user-level benchmark is independent of the existing SFT/GRPO branches.
The dynamic generative balanced branch is available. SFT subset,
candidate-selection, and GRPO datasets still consume the legacy static branch
until they are explicitly regenerated from the dynamic generative output.

The complete, unbalanced generative dataset can also be written to
`data/training/generative_new_interest/`. See
[manifests/README.md](../../manifests/README.md) before restoring ignored data.

## Static User-Disjoint Single-Target Dataset

This static-history branch preserves the original MIND Train/Dev boundary while
removing user leakage and conflicting Train labels:

```bash
PYTHONPATH=src python -m \
  new_interest.data.build_static_user_disjoint_dataset
```

Default output:

```text
data/processed/new_interest_static_user_disjoint_v1/
├── train.parquet
├── dev.parquet
├── interest_vocabulary.parquet
├── train_selection_manifest.parquet
├── statistics.json
└── metadata.json
```

Train enumerates every static-history novel `impression × interest` target for
each user and selects the minimum SHA-256 priority using seed `20260806`. The
manifest records each user's candidate, distinct-interest, and eligible-
impression counts together with the selected sample and priority. Dev removes
every user ID present anywhere in raw Train, including users without an
eligible Train target, and retains all target rows for the remaining users.

Both splits use the complete ordered History field carried by the current raw
behavior row. Earlier Train or Dev impression clicks are never appended, and
the base dataset performs no H50 truncation. Selection and exclusion counts are
recorded under `statistics.json -> train` and `statistics.json -> dev`.

For user-level semantic diagnostics, recover every distinct eligible Train
target per user and deterministically reserve 10% of Train users for model
selection:

```bash
PYTHONPATH=src python -m \
  new_interest.data.build_static_user_semantic_benchmark
```

This writes
`data/processed/new_interest_static_user_disjoint_semantic_users_v1/` with
`fit.parquet`, `validation.parquet`, and user-aggregated `dev.parquet`. It is a
derived benchmark and does not change the canonical single-target corpus. Its
Dev exposure field contains observed positives only, so it is valid for the
full-taxonomy experiment and must not be used as an exposure-restricted track.

The frozen title-semantic actual-vs-matched-shuffle experiment is configured in
`configs/future_interest/static_user_disjoint_semantic_v1.json` and runs with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m \
  new_interest.experiments.semantic_history_signal
```

Observed output for the current reclassified MINDlarge snapshot:

| Statistic | Count |
|---|---:|
| Train candidate target rows before selection | 1,489,348 |
| Eligible / selected Train users | 587,187 |
| Train users with multiple candidate rows | 356,625 |
| Train users with multiple distinct interests | 348,441 |
| Train users with multiple eligible impressions | 316,549 |
| Multi-row users whose candidates repeat one interest only | 8,184 |
| Candidate rows discarded by one-user selection | 902,161 |
| Dev target rows after raw-Train user exclusion | 37,817 |
| Dev users after raw-Train user exclusion | 27,125 |

The Train output therefore has exactly one row per selected user. Dev contains
31,451 impressions because one unseen user or impression may retain multiple
correct target rows.

## 1. User-Level Fixed-Profile Benchmark

```bash
PYTHONPATH=src python -m \
  new_interest.data.build_future_interest_benchmark \
  configs/future_interest/benchmark_v1.json
```

Default output:

```text
data/processed/future_novel_interest_users_v1/
├── train.parquet
├── validation.parquet
├── test.parquet
├── interest_vocabulary.parquet
├── split_manifest.parquet
├── statistics.json
└── metadata.json
```

The split unit is the SHA-256 hash of the complete ordered history news-ID
sequence. This keeps both users and exact input profiles disjoint across
Train/Validation/Test.

Observed output:

| Split | Users | Users with novel targets |
|---|---:|---:|
| Train | 589,588 | 504,980 |
| Validation | 73,588 | 63,107 |
| Test | 73,164 | 62,687 |

The complete benchmark contains 736,340 users, 726,561 exact profile groups,
and all 125 taxonomy labels. See the
[dual-Track baseline report](../experiments/2026-08-12/future_novel_interest_baselines_report.md)
for Exposure-restricted and Full-taxonomy definitions and results.

## 2. Dynamic Natural-Distribution Dataset

```bash
PYTHONPATH=src python -m new_interest.data.build_dynamic_dataset
```

Default output:

```text
data/processed/new_interest_dynamic_v1/
├── train.parquet
├── dev.parquet
├── interest_vocabulary.parquet
├── statistics.json
└── metadata.json
```

Each Train/Dev row is one impression:

| Field | Meaning |
|---|---|
| `sample_id`, `split`, `impression_id`, `user_id`, `timestamp` | Source event identity |
| `initial_history_length` | Size of the fixed MIND seed history |
| `history_length`, `history` | Dynamic positive-click history before this event |
| `clicked_news`, `clicked_interests` | All positive clicks in the impression |
| `novel_interests` | Clicked interests absent from dynamic history |
| `repeated_interests` | Clicked interests already present in dynamic history |
| `has_new_interest` | Whether `novel_interests` is non-empty |

Observed full-corpus statistics:

| Split | Impressions | New-interest impressions | Novel memberships | Repeated memberships |
|---|---:|---:|---:|---:|
| Train | 2,186,664 | 1,157,913 | 1,401,017 | 1,770,992 |
| Online Dev | 365,199 | 179,555 | 214,627 | 313,994 |

Compared with the static definition, dynamic replay removes 61,640 Train and
20,089 Dev impressions that were previously treated as containing a new
interest. The generated directory is ignored by Git; metadata records source
and output SHA-256 values.

## 3. Dynamic Balanced Generative Dataset

```bash
PYTHONPATH=src python -m \
  new_interest.data.build_dynamic_generative_dataset
```

Default output:

```text
data/training/generative_new_interest_dynamic_balanced_200_h50/
├── behavior_only/{train,dev}.parquet
├── interest_only/{train,dev}.parquet
├── selected_train_samples.parquet
├── retained_interests.parquet
├── statistics.json
└── metadata.json
```

Rules:

- Train selects one deterministic novel target per impression using seed
  `20260806`;
- after that selection, retain classes with at least 200 Train impressions;
- sample exactly 200 Train impressions per retained class;
- keep the most recent 50 dynamic clicks in Behavior/Interest prompts;
- Dev keeps every online-dynamic novel target in the retained Train label set.

Observed output:

```text
retained classes: 110
Train: 22,000 unique impressions / target rows per task
Dev: 214,355 target rows from 179,374 positive impressions per task
```

The selected 22K Train impressions originally contain a mean of 1.2577 novel
targets each. The training output nevertheless has one deterministic target
per impression. Online Dev retains all acceptable targets and averages 1.1950
retained novel targets over impressions with at least one retained target.
Exact count and rate distributions are stored under
`statistics.json -> novel_targets_per_impression`.

## 4. Legacy Static Base Dataset

Expected raw layout: `data/raw/MINDlarge_reclassified/{train,dev}/` with the
reclassified `news.tsv` and labeled `behaviors.tsv` files.

```bash
PYTHONPATH=src python -m new_interest.data.build_dataset \
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

## 5. Legacy Generative Data

The model receives no candidates and must answer
`New topic: <subcategory>`. `behavior_only` uses ordered news titles;
`interest_only` uses the corresponding ordered subcategories, including
repetitions. The full builder preserves complete histories:

```bash
PYTHONPATH=src python -m new_interest.data.build_generative_dataset
```

Each task contains `train.parquet` and `dev.parquet`, plus shared
`statistics.json` and `metadata.json`. Rows include `sample_id`,
`source_sample_id`, source identity fields, `task_type`, `history_length`,
`prompt`, `response`, `messages`, and `target_subcategory`. They intentionally
contain no top-level target category. Concrete prompts are in
[generative_data_examples.md](../generative_data_examples.md).

## 6. Legacy Balanced Recent-50 Data

Canonical training uses one deterministic Train target per impression, 200
examples for each class with at least 200 source examples, and the most recent
50 history items:

```bash
PYTHONPATH=src python -m new_interest.data.build_generative_dataset \
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

## 7. SFT Subsets

```bash
PYTHONPATH=src python -m new_interest.data.build_sft_subsets
```

This creates class-balanced, nested Train prefixes at 1K, 5K, 10K, and 22K in
`data/training/sft_experiments/train/`. It also creates grouped Dev files for
beam evaluation. Nested means every smaller subset is a prefix of the larger
selection, not an independent sample. Do not substitute expanded per-target
candidate files for grouped beam Dev files.

## 8. Evaluation And GRPO Branches

```bash
PYTHONPATH=src python -m new_interest.data.build_candidate_selection
PYTHONPATH=src python -m new_interest.data.build_grpo_dataset
```

Candidate selection creates paired `behavior_only` and `interest_only` K=10
rows from SFT evaluation data. GRPO creates 1K and 5K veRL-formatted CoT Train
sets from SFT subsets. Continue with [evaluation.md](../evaluation.md) or
[grpo.md](../grpo.md).

## 9. SFT Training-Data Similarity Analysis

The active SFT22K Train and grouped Dev10K can be audited without retraining a
model:

```bash
PYTHONPATH=src python -m \
  new_interest.experiments.analyze_sft_training_similarity \
  --config configs/analysis/sft_training_similarity_v1.json

PYTHONPATH=src python -m \
  new_interest.experiments.report_sft_training_similarity
```

The analyzer reconnects every SFT row to the processed source row and raw MIND
impression. It uses the model-visible H50, clicked target titles, all
same-impression `-0` controls, and a primary subset whose subcategory is absent
from H50. It writes row-level title, subcategory-name, parent-name, and exact
taxonomy features; pinned MiniLM title/category-name caches; 2,000-replicate
row-bootstrap intervals; split-level target-set permutations; prompt overlap
statistics; and deterministic case studies under
`artifacts/sft-training-data-analysis/v1/`.

The published artifact path is an atomic symlink to an immutable generation.
`--overwrite` swaps the symlink only after the new generation is complete, so
readers never observe a missing or mixed-generation directory. See the
[full bilingual report](../experiments/2026-08-13/sft_training_data_similarity_analysis.md).

## Reproducibility

- Builder metadata records rule versions, source/output hashes, and row counts.
- Statistics record class frequencies and history/target distributions.
- Selection manifests retain stable sample IDs and deterministic priorities.
- Existing outputs are refused unless `--overwrite` is supplied; inspect the
  destination before using it.
- Run `pytest -q tests/test_build_dynamic_dataset.py` for the dynamic builder.
- Run `pytest -q tests/test_build_dynamic_generative_dataset.py` for dynamic
  balancing and generative conversion.
- Run `pytest -q tests/test_build_dataset.py` for the legacy static builder.
- Run `pytest -q tests/test_sft_training_similarity_analysis.py
  tests/test_sft_training_similarity_report.py` for the SFT data audit.
