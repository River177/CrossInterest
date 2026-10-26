#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${SEMANTIC_ID_BEHAVIOR_INTEREST_CROSSFIT_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Behavior+Interest Semantic-ID class cross-fit smoke requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "qwen35-behavior-interest-semantic-id-class-crossfit-oof-smoke" ]]; then
  echo "Unexpected combined Cross-Fitted smoke job name: $AMLT_JOB_NAME" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME"

MODEL_NAME="Qwen/Qwen3.5-9B"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_21600.yaml"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_behavior_interest_full.json"
SEMANTIC_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1"
SEMANTIC_TRAIN="$SEMANTIC_ROOT/train/behavior_interest/21600.parquet"
EVAL_FILE="$SEMANTIC_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$SEMANTIC_ROOT/interest_token_mapping.parquet"
FOLD_MANIFEST="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1/fold_manifest.parquet"
SMOKE_ROOT="$AMLT_OUTPUT_DIR/semantic-id-class-crossfit-behavior-interest-smoke"
DATA_DIR="$SMOKE_ROOT/data/fold_0"
TEACHER_DIR="$SMOKE_ROOT/teacher"
SCORE_DIR="$SMOKE_ROOT/oof"
WEIGHT_DIR="$SMOKE_ROOT/weights"
STUDENT_DIR="$SMOKE_ROOT/student"
EVAL_DIR="$SMOKE_ROOT/eval"

for path in "$CONFIG" "$EVAL_CONFIG" "$SEMANTIC_TRAIN" "$EVAL_FILE" "$MAPPING_FILE" "$FOLD_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing smoke input: $path" >&2
    exit 1
  fi
done

python - <<'PY'
import json
import platform

import accelerate
import peft
import torch
import transformers

print(
    "COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_ENV_OK",
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "cuda_devices": torch.cuda.device_count(),
        },
        sort_keys=True,
    ),
)
if torch.cuda.device_count() != 4:
    raise RuntimeError("Expected exactly four visible GPUs")
PY

python - "$MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = sys.argv[1]


def validate_snapshot(value: str) -> Path:
    root = Path(value)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError("Qwen snapshot has no safetensors index")
    files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    missing = [name for name in sorted(files) if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Qwen snapshot is incomplete: {missing[:3]}")
    AutoConfig.from_pretrained(root, local_files_only=True)
    AutoTokenizer.from_pretrained(root, local_files_only=True)
    return root


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception:
    snapshot = validate_snapshot(snapshot_download(name))
    source = "downloaded-or-repaired"
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m new_interest.experiments.semantic_id_crossfit prepare \
  --semantic-train-file "$SEMANTIC_TRAIN" \
  --fold-manifest "$FOLD_MANIFEST" \
  --output-dir "$DATA_DIR" \
  --fold 0 \
  --max-per-class 2

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$DATA_DIR/train.parquet" \
  --output-dir "$TEACHER_DIR" \
  --checkpoint-dir /scratch/semantic-id-class-crossfit-behavior-interest-teacher-checkpoints \
  --run-name "qwen35-behavior-interest-semantic-id-class-crossfit-teacher-smoke" \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps 675 \
  --wandb-mode disabled

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$TEACHER_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$TEACHER_DIR/semantic_reload_reference.json" \
  --output "$TEACHER_DIR/semantic_adapter_reload_audit.json"

CUDA_VISIBLE_DEVICES=0 python -m new_interest.experiments.semantic_id_crossfit score \
  --model-name "$MODEL_NAME" \
  --adapter-path "$TEACHER_DIR/final_adapter" \
  --heldout-file "$DATA_DIR/heldout.parquet" \
  --mapping-file "$MAPPING_FILE" \
  --output-dir "$SCORE_DIR" \
  --fold 0 \
  --batch-size 8

python -m new_interest.experiments.semantic_id_crossfit class-weights \
  --score-files "$SCORE_DIR/oof_scores.parquet" \
  --train-file "$DATA_DIR/heldout.parquet" \
  --output-dir "$WEIGHT_DIR"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$DATA_DIR/heldout.parquet" \
  --sample-weight-file "$WEIGHT_DIR/sample_weights.parquet" \
  --output-dir "$STUDENT_DIR" \
  --checkpoint-dir /scratch/semantic-id-class-crossfit-behavior-interest-student-checkpoints \
  --run-name "qwen35-behavior-interest-semantic-id-class-crossfit-student-smoke" \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps 675 \
  --wandb-mode disabled

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$STUDENT_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$STUDENT_DIR/semantic_reload_reference.json" \
  --output "$STUDENT_DIR/semantic_adapter_reload_audit.json"

CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
  --config "$EVAL_CONFIG" \
  --adapter-path "$STUDENT_DIR/final_adapter" \
  --eval-file "$EVAL_FILE" \
  --interest-token-mapping-file "$MAPPING_FILE" \
  --output-dir "$EVAL_DIR" \
  --run-name "qwen35-behavior-interest-semantic-id-class-crossfit-eval-smoke" \
  --max-examples 64 \
  --num-shards 1 \
  --shard-index 0 \
  --batch-size 8

python - "$SMOKE_ROOT" "$WEIGHT_DIR/sample_weights.parquet" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
weight_path = Path(sys.argv[2])
split = json.loads((root / "data/fold_0/split_summary.json").read_text())
if split["classes"] != 108 or split["train_rows"] != 216 or split["heldout_rows"] != 216:
    raise RuntimeError(f"Smoke split identity mismatch: {split}")
if split["overlapping_source_ids"] != 0:
    raise RuntimeError("Teacher saw held-out smoke rows")

for name, weighted in (("teacher", False), ("student", True)):
    summary = json.loads((root / name / "run_summary.json").read_text())
    if summary["model_name"] != "Qwen/Qwen3.5-9B":
        raise RuntimeError(f"{name}: model identity mismatch")
    if summary["world_size"] != 4 or summary["train_samples"] != 216:
        raise RuntimeError(f"{name}: DDP/data identity mismatch")
    if summary["effective_global_batch_size"] != 32 or summary["expected_optimizer_steps"] != 14:
        raise RuntimeError(f"{name}: optimizer-step identity mismatch")
    if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
        raise RuntimeError(f"{name}: non-finite training loss")
    semantic = summary["model_audit"]["semantic_id_tokens"]
    if not semantic["enabled"] or semantic["count"] != 108:
        raise RuntimeError(f"{name}: Semantic-ID audit mismatch")
    expected_weight = weight_path.resolve() if weighted else None
    actual_weight = Path(summary["sample_weight_file"]) if summary["sample_weight_file"] else None
    if actual_weight != expected_weight:
        raise RuntimeError(f"{name}: sample-weight semantics mismatch")
    if not (root / name / "semantic_adapter_reload_audit.json").is_file():
        raise RuntimeError(f"{name}: adapter reload audit missing")

score = json.loads((root / "oof/score_summary.json").read_text())
if score["rows"] != 216 or score["classes"] != 108 or score["fold"] != 0:
    raise RuntimeError("OOF score identity mismatch")
if not math.isclose(float(score["fold_prior_sum"]), 1.0, abs_tol=1e-12):
    raise RuntimeError("Fold output prior is not normalized")

weights = json.loads((root / "weights/weight_summary.json").read_text())
if weights["rows"] != 216 or weights["classes"] != 108:
    raise RuntimeError("Weight identity mismatch")
if not math.isclose(float(weights["global_weight_mean"]), 1.0, abs_tol=1e-12):
    raise RuntimeError("Global weight mean is not one")
if not math.isclose(float(weights["global_weight_max_min_ratio"]), 2.0, abs_tol=1e-12):
    raise RuntimeError("Global weight max/min ratio is not two")
if weights["formula"]["class_normalization"] is not True:
    raise RuntimeError("Within-class normalization was not enabled")
weight_rows = pq.read_table(weight_path).to_pylist()
by_class = {}
for row in weight_rows:
    by_class.setdefault(str(row["target_subcategory"]), []).append(row)
if len(by_class) != 108 or {len(rows) for rows in by_class.values()} != {2}:
    raise RuntimeError("Smoke weights are not two rows for each class")
for rows in by_class.values():
    if {int(row["within_class_rank"]) for row in rows} != {0, 1}:
        raise RuntimeError("Smoke within-class ranks are not complete")
    mean = sum(float(row["sample_weight"]) for row in rows) / len(rows)
    if not math.isclose(mean, 1.0, abs_tol=1e-12):
        raise RuntimeError("Smoke class weight mean is not one")

eval_metrics = json.loads((root / "eval/metrics.json").read_text())
if eval_metrics["samples"] != 64 or eval_metrics["label_vocabulary_size"] != 108:
    raise RuntimeError("Evaluation smoke identity mismatch")
if eval_metrics["top_k"] != 20 or eval_metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("Evaluation smoke ranking semantics mismatch")
for key in ("recall_at_10", "ndcg_at_10", "interest_macro_recall_at_10"):
    if not math.isfinite(float(eval_metrics[key])):
        raise RuntimeError(f"Non-finite smoke metric: {key}")

checkpoint_files = [path for path in root.rglob("*") if path.is_file() and "checkpoints" in path.parts]
if checkpoint_files:
    raise RuntimeError(f"Unexpected smoke checkpoint files: {checkpoint_files[:3]}")

files = {}
for path in sorted(value for value in root.rglob("*") if value.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files[path.relative_to(root).as_posix()] = digest.hexdigest()
manifest = {
    "manifest_version": "semantic-id-class-crossfit-behavior-interest-smoke-v1",
    "analysis_scope": "qwen35-behavior-plus-interest",
    "weight_source_view": "behavior-plus-interest-semantic-id-oof",
    "weight_rank_scope": "within-target-class-over-all-oof-rows",
    "class_normalization_used": True,
    "shuffled_evaluation_used": False,
    "checkpoint_files": 0,
    "files": files,
}
(root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_SMOKE_GATE_OK", len(files))
PY
