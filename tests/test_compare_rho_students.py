from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments import compare_rho_students as comparison
from new_interest.experiments.sid_relation_bias import metrics


NAMES = ["A", "B", comparison.RARE_CLASS, *[f"Unused-{i}" for i in range(22)]]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def source(index, user, targets, split="dev"):
    prompt = f"history for {split} impression {index}"
    return {"source_sample_id": f"{split}-{index}", "impression_id": f"impression-{split}-{index}",
        "user_id": user, "prompt": prompt, "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "targets": targets, "history_subcategories": ["old history category"]}


def predict(row, index, arm):
    order = [None] * len(NAMES)
    placed = set()
    for target in row["targets"]:
        target_index = NAMES.index(target)
        if target == comparison.RARE_CLASS:
            rank = 21
        else:
            rank = {"uniform": {"A": 12, "B": 22}, "static": {"A": 8, "B": 15},
                    "dynamic": {"A": 1, "B": 5}}[arm][target] + index % 3
        order[rank - 1] = target_index
        placed.add(target_index)
    remaining = iter(index for index in range(len(NAMES)) if index not in placed)
    order = [next(remaining) if value is None else value for value in order]
    scores = [0.] * len(NAMES)
    for rank, class_index in enumerate(order):
        scores[class_index] = float(len(NAMES) - rank)
    return {"record_index": index, **{field: row[field] for field in comparison.PAIRED_FIELDS},
            "scores": scores, "ranking": order}


@pytest.fixture
def accepted(tmp_path):
    sources = {"validation": [source(i, f"v-user{i}", ["A" if i < 2 else "B" if i < 4 else comparison.RARE_CLASS], "validation")
                              for i in range(5)],
               "dev": [source(0, "u0", ["A"]), source(1, "u0", ["A", "B"]),
                       source(2, "u1", ["B"]), source(3, "u2", ["A"])]}
    source_paths = {split: tmp_path / "inputs" / f"{split}.parquet" for split in sources}
    for split, rows in sources.items():
        write_parquet(source_paths[split], rows)
    mapping = tmp_path / "inputs/mapping.parquet"
    write_parquet(mapping, [{"interest_index": i, "interest_token": f"<interest_{i:03d}>", "target_subcategory": name}
                            for i, name in enumerate(NAMES)])
    arms = {}
    for arm in comparison.ARMS:
        raw_adapter_sha = hashlib.sha256(arm.encode()).hexdigest()
        adapter_files = {"adapter_model.safetensors": raw_adapter_sha}
        adapter_sha = comparison.canonical(adapter_files)
        worker = {"passed": True, "identity": {"arm": arm, "phase": "full", "runtime_sha256": "runtime",
            "protocol_sha256": "protocol", "inputs_sha256": "inputs", "reference_acceptance_sha256": "reference"},
            "training": {"full": {"rows": 21600, "steps": 2025, "epochs": 3,
                "initial_state_sha256": "same-initial", "adapter_sha256": raw_adapter_sha,
                "student_summary": {"reference_file_sha256": "same-reference"}}}, "evaluations": {}}
        for split, original in sources.items():
            root = tmp_path / arm / split
            predictions = [predict(row, index, arm) for index, row in enumerate(original)]
            report = metrics(predictions, NAMES)
            if split == "validation":
                report["main_107"] = metrics(predictions, NAMES, min_support=2)
            write_parquet(root / "merged/predictions.parquet", predictions)
            write_json(root / "merged/metrics.json", report)
            artifacts = []
            for shard in range(4):
                identity = {"role": "heldout_interest_ranking", "split": split,
                    "input_sha256": comparison.sha(source_paths[split]), "protocol_sha256": "protocol",
                    "runtime_sha256": "runtime", "source_snapshot_sha256": "snapshot", "model_revision": "model-revision",
                    "mapping_sha256": comparison.sha(mapping), "adapter_files": adapter_files, "adapter_sha256": adapter_sha,
                    "kernel_selection": "stable_fla_v1", "kernel_selection_protocol": "fla-valid-candidates-stable-sha256-v1",
                    "kernel_selector_sha256": "selector", "projection": "native-effective-head-FP32",
                    "shard": shard, "shards": 4, "limit": None, "batch_size": 4}
                directory = root / f"shard-{shard}"
                count = len(range(shard, len(original), 4))
                snapshots = [{"stage": f"rows:{start}:{min(start + 256, count)}",
                    "kernels": {"fla.kernel": {"cache": {"shared-key": "BLOCK=32"}, "cache_results": False}}}
                    for start in range(0, count, 256)]
                kernels = {**{key: identity[key] for key in ("kernel_selection", "kernel_selection_protocol", "kernel_selector_sha256")},
                           "evaluation_identity": comparison.canonical(identity), "snapshots": snapshots,
                           "snapshots_sha256": comparison.canonical(snapshots)}
                write_json(directory / "kernel_selection.json", kernels)
                write_json(directory / "complete.json", {"identity": identity, "rows": count,
                    "kernel_selection_sha256": comparison.sha(directory / "kernel_selection.json")})
                artifacts.append({"directory": str(directory), "complete_sha256": comparison.sha(directory / "complete.json"), "files": []})
            evaluation = {"accepted": True, "split": split, "rows": len(original), "metrics": report,
                "source_sha256": comparison.sha(source_paths[split]), "protocol_sha256": "protocol", "limit": None,
                "adapter_sha256": adapter_sha, "prediction_sha256": comparison.sha(root / "merged/predictions.parquet"),
                "metrics_sha256": comparison.sha(root / "merged/metrics.json"), "kernel_configs_consistent": True,
                "artifacts": artifacts, "prediction_file": str(root / "merged/predictions.parquet")}
            write_json(root / "merged/evaluation_acceptance.json", evaluation)
            worker["evaluations"][split] = evaluation
        arms[arm] = {"worker": worker}
    joint = {"passed": True, "arms": arms, "same_initial_state": True, "same_reference_scores": True}
    acceptance = tmp_path / "three_arm_acceptance.json"
    write_json(acceptance, joint)
    return {"acceptance_path": acceptance, "expected_acceptance_sha256": comparison.sha(acceptance),
        "validation_file": source_paths["validation"], "dev_file": source_paths["dev"], "mapping_file": mapping,
        "output_file": tmp_path / "analysis/comparison.json", "bootstrap_draws": 200, "seed": 42,
        "production": False, "validation_min_support": 2, "joint": joint, "sources": sources}


def run(case):
    return comparison.compare_students(**{key: value for key, value in case.items() if key not in ("joint", "sources")})


def reseal(case, arm=None, split=None, *, metrics_changed=False):
    if arm is not None:
        report = case["joint"]["arms"][arm]["worker"]["evaluations"][split]
        prediction = Path(report["prediction_file"])
        report["prediction_sha256"] = comparison.sha(prediction)
        if metrics_changed:
            write_json(prediction.parent / "metrics.json", report["metrics"])
            report["metrics_sha256"] = comparison.sha(prediction.parent / "metrics.json")
        write_json(prediction.parent / "evaluation_acceptance.json", report)
    write_json(case["acceptance_path"], case["joint"])
    case["expected_acceptance_sha256"] = comparison.sha(case["acceptance_path"])


def test_complete_small_artifact_comparison_recomputes_macros_and_separates_rare_class(accepted):
    result = run(accepted)
    assert result["passed"]
    assert result["primary_comparison"]["contrast"] == "dynamic_minus_static"
    assert result["primary_comparison"]["metric"] == "macro_ndcg_at_20"
    assert result["results"]["validation"]["main_macro_classes"] == 2
    assert result["results"]["validation"]["rare_validation_class"]["support"] == 1
    assert not result["results"]["validation"]["rare_validation_class"]["included_in_primary_macro"]
    assert result["results"]["dev"]["records"] == 4 and result["results"]["dev"]["users"] == 3
    assert result["results"]["dev"]["target_occurrences"] == 5
    assert result["results"]["dev"]["cross_arm_kernel_configs_consistent"]
    assert result["results"]["dev"]["kernel_cache_keys_checked"] == 1
    assert result["results"]["dev"]["means"]["dynamic"]["macro_recall_at_10"] == 1
    assert result["results"]["dev"]["comparisons"]["dynamic_minus_static"]["delta"]["macro_recall_at_10"] == .5
    assert result["family"]["intervals_are_marginal_not_multiplicity_adjusted"]
    assert comparison.read(accepted["output_file"])["acceptance_sha256"] == accepted["expected_acceptance_sha256"]


def test_cluster_bootstrap_matches_manual_repeated_impression_user_resampling():
    source_rows = [source(0, "u0", ["A"]), source(1, "u0", ["A", "B"]), source(2, "u1", ["B"]), source(3, "u2", ["A"])]
    arms = {arm: [predict(row, i, arm) for i, row in enumerate(source_rows)] for arm in comparison.ARMS}
    draws = 300
    result = comparison.paired_user_bootstrap(arms, NAMES, ["A", "B"], draws=draws, seed=42, batch_size=17)
    rng = np.random.default_rng(42)
    sampled = rng.integers(0, 3, size=(draws, 3), dtype=np.int64)
    expected = {new + "_minus_" + old: [] for new, old in comparison.CONTRASTS}
    missing = 0
    for indices in sampled:
        records = {arm: [row for index in indices for row in arms[arm] if row["user_id"] == f"u{index}"] for arm in comparison.ARMS}
        per_arm = {arm: metrics(rows, NAMES) for arm, rows in records.items()}
        missing += per_arm["uniform"]["supported_classes"] < 2
        for new, old in comparison.CONTRASTS:
            expected[new + "_minus_" + old].append([per_arm[new][key] - per_arm[old][key] for key in comparison.METRICS])
    for contrast, samples in expected.items():
        intervals = np.quantile(samples, [.025, .975], axis=0)
        for i, key in enumerate(comparison.METRICS):
            assert result["comparisons"][contrast]["ci95"][key] == pytest.approx(intervals[:, i], abs=1e-12)
    assert result["bootstrap"]["observed_users"] == result["bootstrap"]["sampled_users_per_draw"] == 3
    assert result["bootstrap"]["draws_missing_any_eligible_class"] == missing
    assert result["bootstrap"]["draw_indices_sha256"] == hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest()
    differently_batched = comparison.paired_user_bootstrap(arms, NAMES, ["A", "B"], draws=draws, seed=42, batch_size=64)
    assert differently_batched["bootstrap"]["draw_indices_sha256"] == result["bootstrap"]["draw_indices_sha256"]


def test_same_arm_has_exact_zero_delta_and_zero_intervals():
    rows = [predict(source(i, f"u{i // 2}", ["A" if i % 2 else "B"]), i, "uniform") for i in range(6)]
    result = comparison.paired_user_bootstrap({arm: copy.deepcopy(rows) for arm in comparison.ARMS}, NAMES, ["A", "B"], draws=200)
    for contrast in result["comparisons"].values():
        assert all(value == 0 for value in contrast["delta"].values())
        assert all(value == [0., 0.] for value in contrast["ci95"].values())


@pytest.mark.parametrize("mutation", ["prediction_hash", "source_id", "target", "tie_order", "macro_denominator", "source_hash", "model_identity"])
def test_wrong_artifact_identity_rank_or_macro_denominator_is_rejected(accepted, mutation):
    report = accepted["joint"]["arms"]["static"]["worker"]["evaluations"]["dev"]
    path = Path(report["prediction_file"])
    if mutation in ("prediction_hash", "source_id", "target", "tie_order"):
        rows = pq.read_table(path).to_pylist()
        if mutation in ("prediction_hash", "source_id"):
            rows[0]["source_sample_id"] = "wrong-source"
        elif mutation == "target":
            rows[0]["targets"] = ["B"]
        else:
            rows[0]["scores"] = [0.] * len(NAMES)
            rows[0]["ranking"] = list(reversed(range(len(NAMES))))
        write_parquet(path, rows)
        if mutation != "prediction_hash":
            reseal(accepted, "static", "dev")
    elif mutation == "macro_denominator":
        report["metrics"]["supported_classes"] = 999
        reseal(accepted, "static", "dev", metrics_changed=True)
    elif mutation == "source_hash":
        report["source_sha256"] = "0" * 64
        reseal(accepted, "static", "dev")
    else:
        complete = Path(report["artifacts"][0]["directory"]) / "complete.json"
        done = comparison.read(complete)
        done["identity"]["model_revision"] = "another model"
        write_json(complete, done)
        report["artifacts"][0]["complete_sha256"] = comparison.sha(complete)
        reseal(accepted, "static", "dev")
    with pytest.raises(ValueError):
        run(accepted)
    assert not accepted["output_file"].exists()


def test_duplicate_target_memberships_rejected_in_original_source(accepted):
    rows = accepted["sources"]["validation"]
    rows[0]["targets"] = ["A", "A"]
    write_parquet(accepted["validation_file"], rows)
    with pytest.raises(ValueError, match="unique nonempty"):
        run(accepted)


@pytest.mark.parametrize("mutation", ["unbound_hash", "different_config", "empty_cache", "nonstring_config", "timing_writes"])
def test_actual_kernel_receipts_and_cross_arm_selected_configs_are_verified(accepted, mutation):
    report = accepted["joint"]["arms"]["static"]["worker"]["evaluations"]["dev"]
    artifact = report["artifacts"][0]
    directory = Path(artifact["directory"])
    receipt = comparison.read(directory / "kernel_selection.json")
    kernel = receipt["snapshots"][0]["kernels"]["fla.kernel"]
    if mutation in ("unbound_hash", "different_config"):
        kernel["cache"]["shared-key"] = "BLOCK=64"
    elif mutation == "empty_cache":
        kernel["cache"] = {}
    elif mutation == "nonstring_config":
        kernel["cache"]["shared-key"] = None
    elif mutation == "timing_writes":
        kernel["cache_results"] = True
    receipt["snapshots_sha256"] = comparison.canonical(receipt["snapshots"])
    write_json(directory / "kernel_selection.json", receipt)
    if mutation != "unbound_hash":
        done = comparison.read(directory / "complete.json")
        done["kernel_selection_sha256"] = comparison.sha(directory / "kernel_selection.json")
        write_json(directory / "complete.json", done)
        artifact["complete_sha256"] = comparison.sha(directory / "complete.json")
        reseal(accepted, "static", "dev")
    with pytest.raises(ValueError, match="kernel"):
        run(accepted)


def test_empty_eligible_draws_are_counted_and_omitted_without_replacement():
    rows = [predict(source(0, "u0", ["A"]), 0, "uniform"), predict(source(1, "u1", ["B"]), 1, "uniform")]
    result = comparison.paired_user_bootstrap({arm: rows for arm in comparison.ARMS}, NAMES, ["A"], draws=300)
    audit = result["bootstrap"]
    assert 0 < audit["zero_supported_class_draws"] < 300
    assert audit["usable_draws"] + audit["zero_supported_class_draws"] == 300
    assert audit["draws_missing_any_eligible_class"] == audit["zero_supported_class_draws"]


def test_actual_pairing_not_just_equal_row_count_required():
    rows = [predict(source(i, f"u{i}", ["A"]), i, "uniform") for i in range(4)]
    arms = {arm: copy.deepcopy(rows) for arm in comparison.ARMS}
    arms["dynamic"][0]["user_id"] = "other-user"
    with pytest.raises(ValueError, match="exactly paired"):
        comparison.paired_user_bootstrap(arms, NAMES, ["A"], draws=20)


def test_production_bootstrap_and_macro_profile_are_frozen(accepted):
    accepted["production"] = True
    with pytest.raises(ValueError, match="2000 draws"):
        run(accepted)
    accepted["bootstrap_draws"] = 2000
    accepted["validation_min_support"] = 20
    with pytest.raises(ValueError, match="Mapping vocabulary"):
        run(accepted)
