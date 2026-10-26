"""CPU-only runner contracts; never load a pretrained model or use CUDA."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments import oi_ni_meta as runner


def test_record_normalization_preserves_grouped_truth_and_source_identity() -> None:
    single = {"source_sample_id": "train:10:2", "user_id": "u1",
              "prompt": "History", "target_subcategory": "B"}
    normalized = runner.as_eval_record(single)
    assert normalized["impression_id"] == "train:10:2"
    assert normalized["acceptable_subcategories"] == ["B"]
    assert normalized["user_id"] == "u1"
    assert "acceptable_subcategories" not in single
    grouped = {"impression_id": "dev:1", "user_id": "u2",
               "acceptable_subcategories": ["A", "B"], "prompt": "History"}
    assert runner.as_eval_record(grouped) == grouped


def test_metrics_have_calibration_and_full_dev_topology_fields() -> None:
    records = [runner.as_eval_record({"source_sample_id": "a", "target_subcategory": "A"}),
               runner.as_eval_record({"source_sample_id": "b", "target_subcategory": "B"})]
    ranked = [[{"label": "A", "score": 1.0}, {"label": "B", "score": 0.0}],
              [{"label": "B", "score": 1.0}, {"label": "A", "score": 0.0}]]
    metrics, _ = runner.evaluate_ranked_outputs(records, ranked)
    assert metrics["samples"] == 2
    assert metrics["supported_interest_count"] == 2
    assert metrics["interest_macro_recall_at_10"] == 1.0
    assert metrics["interest_macro_ndcg_at_10"] == 1.0


@pytest.fixture
def cpu_prediction(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(108, dtype=torch.float32))
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            # All equal scores exercise stable SID-index tie breaking.
            return self.weight[None, :].expand(len(input_ids), -1) * 0

    monkeypatch.setattr(runner, "batch_inputs", lambda sequences, pad: {
        "input_ids": torch.tensor(sequences, dtype=torch.long, device="cpu")})
    model = Model()
    records = [{"impression_id": str(i), "user_id": f"u{i}", "prompt": f"H{i}",
                "acceptable_subcategories": [f"C{i}"]} for i in range(3)]
    return model, [[1], [2], [3]], records, SimpleNamespace(pad_token_id=0), [f"C{i}" for i in range(108)]


def test_prediction_ranks_ties_without_parameter_adaptation_and_reuses_shards(
    tmp_path: Path, cpu_prediction,
) -> None:
    model, tokens, records, tokenizer, labels = cpu_prediction
    before = runner.parameter_digest(model)
    metrics = runner.predict(model, tokens, records, tokenizer, labels, tmp_path, "run-identity",
                             batch_size=1, shard_size=2)
    assert metrics["no_test_time_adaptation"] is True
    assert metrics["projection_dtype"] == "float32"
    assert metrics["samples"] == 3
    assert metrics["interest_macro_recall_at_10"] == 1.0
    assert before == runner.parameter_digest(model)
    assert model.training is True
    first = pq.read_table(tmp_path / "predictions-000000.parquet").to_pylist()[0]
    assert first["ranked_indices"] == list(range(20))
    assert first["ranked_labels"] == labels[:20]
    assert len(first["all_sid_log_probabilities"]) == 108
    assert np.isclose(np.exp(first["all_sid_log_probabilities"]).sum(), 1.0)
    calls = model.calls
    again = runner.predict(model, tokens, records, tokenizer, labels, tmp_path, "run-identity",
                           batch_size=1, shard_size=2)
    assert model.calls == calls
    assert again == metrics


def test_prediction_rejects_committed_run_or_source_mismatch(tmp_path: Path, cpu_prediction) -> None:
    model, tokens, records, tokenizer, labels = cpu_prediction
    runner.predict(model, tokens, records, tokenizer, labels, tmp_path, "original", shard_size=2)
    with pytest.raises(ValueError, match="identity"):
        runner.predict(model, tokens, records, tokenizer, labels, tmp_path, "different", shard_size=2)
    records[0] = {**records[0], "acceptable_subcategories": ["C99"]}
    with pytest.raises(ValueError, match="record mismatch"):
        runner.predict(model, tokens, records, tokenizer, labels, tmp_path, "original", shard_size=2)


def test_prediction_rejects_stale_history_even_if_impression_and_target_match(
    tmp_path: Path, cpu_prediction,
) -> None:
    model, tokens, records, tokenizer, labels = cpu_prediction
    runner.predict(model, tokens, records, tokenizer, labels, tmp_path, "original", shard_size=2)
    changed = [{**row, "prompt": "Different history"} for row in records]
    with pytest.raises(ValueError, match="(?i)(input|source|record|identity)"):
        runner.predict(model, [[91], [92], [93]], changed, tokenizer, labels,
                       tmp_path, "original", shard_size=2)


def test_publish_manifest_is_immutable_and_detects_mutated_payload(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "checkpoint"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    runner.publish(source, destination, "identity")
    runner.publish(source, destination, "identity")
    assert runner.verify_manifest(destination, "identity")["files"] == runner.inventory(source)
    with pytest.raises(ValueError, match="identity"):
        runner.publish(source, destination, "other-identity")
    (destination / "weights.bin").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash"):
        runner.verify_manifest(destination, "identity")


def test_committed_manifest_rejects_untracked_loader_visible_files(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "checkpoint"
    source.mkdir()
    (source / "adapter_config.json").write_text("{}")
    runner.publish(source, destination, "identity")
    (destination / "pytorch_model.bin").write_bytes(b"stale untracked weight file")
    with pytest.raises(ValueError, match="(?i)(file|inventory|untracked|manifest)"):
        runner.verify_manifest(destination, "identity")


def test_latest_checkpoint_ignores_uncommitted_directories(tmp_path: Path) -> None:
    source = tmp_path / "scratch"
    source.mkdir()
    (source / "training_state.pt").write_bytes(b"state")
    committed = tmp_path / "checkpoints" / "checkpoint-000100"
    runner.publish(source, committed, "identity")
    (tmp_path / "checkpoints" / "checkpoint-000200").mkdir()
    assert runner.latest_checkpoint(tmp_path, "identity") == committed


def test_resume_at_final_step_completes_validation_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    root, data = tmp_path / "run", tmp_path / "data"
    checkpoint = root / "checkpoints" / "checkpoint-000001"
    data.mkdir()
    checkpoint.mkdir(parents=True)
    config = {"seed": 42, "max_seq_length": 16, "outer_learning_rate": 1e-3,
              "weight_decay": 0.0, "warmup_ratio": 0.1, "evaluation_batch_size": 1}
    for name in ("fit_oi", "fit_ni", "validation_ni"):
        rows = [{"prompt": "History", "no_history_prompt": "Empty history",
                 "target_subcategory": "A", "interest_index": 0,
                 "source_sample_id": name, "impression_id": name, "user_id": name}]
        pq.write_table(pa.Table.from_pylist(rows), data / f"{name}.parquet")
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, "support_indices": [0], "query_indices": [0]}]),
                   data / "episodes.parquet")
    runner.write_json(checkpoint / "reload_probe.json", {"input_ids": [[1]], "scores": [[0.0]]})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.ones(1))
            self.backend_audit = {}

    reference = Model()
    initial = Model()
    with torch.no_grad():
        initial.lora_A.zero_()
    runner.write_json(root / "model_audit.json", {"initial_trainable_sha256": runner.parameter_digest(initial)})
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0.0)
    reference_scheduler = torch.optim.lr_scheduler.LambdaLR(reference_optimizer, lambda _: 1.0)
    monkeypatch.setattr(runner, "latest_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(runner, "load_interest_token_mapping", lambda *args: [{"target_subcategory": "A"}])
    monkeypatch.setattr(runner, "make_model", lambda *args, **kwargs: (Model(), SimpleNamespace(pad_token_id=0), None, {}))
    monkeypatch.setattr(runner, "encode_records", lambda tokenizer, rows, **kwargs: [[1] for _ in rows])
    monkeypatch.setattr(runner, "capture_probe", lambda *args: [[0.0]])
    monkeypatch.setattr(runner, "restore_rng", lambda state: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU test double")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=0))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def load_state(*args, **kwargs):
        identity = json.loads((root / "resolved_config.json").read_text())["identity"]
        return {"step": 1, "identity": identity, "optimizer": reference_optimizer.state_dict(),
                "scheduler": reference_scheduler.state_dict(), "rng": {}, "parameter_names": ["lora_A"]}

    monkeypatch.setattr(torch, "load", load_state)
    validations = []
    expected_metrics = {"interest_macro_recall_at_10": 0.25, "interest_macro_ndcg_at_10": 0.125}

    def predict(*args, **kwargs):
        validations.append(args[2])
        return expected_metrics

    monkeypatch.setattr(runner, "predict", predict)
    success = runner.run_training(config, "unused", data, root, mode="meta", alpha=1e-3,
                                  steps=1, phase="calibrate", source_identity="frozen")
    assert len(validations) == 1
    assert success["validation_metrics"] == expected_metrics
    assert json.loads((root / "success.json").read_text())["validation_metrics"] == expected_metrics


@pytest.fixture
def frozen_data(tmp_path: Path):
    data = tmp_path / "frozen"
    data.mkdir()
    for name, count, prefix in (("fit_oi", 17037, "O"), ("fit_ni", 19440, "N"),
                                ("validation_ni", 2160, "V")):
        pq.write_table(pa.Table.from_pylist([
            {"row_index": i, "user_id": f"{prefix}{i}", "interest_index": i % 108}
            for i in range(count)]), data / f"{name}.parquet")
    pq.write_table(pa.Table.from_pylist([{"user_id": "DEV-user"}]), data / "dev_full_grouped.parquet")
    episodes = []
    for step in range(540):
        classes = [(step * 8 + j) % 108 for j in range(8)]
        indices = [i for c in classes for i in (c, c + 108)]
        episodes.append({"episode_index": step, "class_indices": classes,
                         "support_indices": indices, "query_indices": indices})
    pq.write_table(pa.Table.from_pylist(episodes), data / "episodes.parquet")

    def bind_metadata():
        runner.write_json(data / "metadata.json", {"artifacts": {
            p.name: {"sha256": runner.sha256_file(p)} for p in data.glob("*.parquet")}})
        return {"data_metadata_sha256": runner.sha256_file(data / "metadata.json"), "pilot_steps": 540}

    return data, bind_metadata


def test_frozen_data_validates_all_bound_files_and_rejects_changed_bytes(frozen_data) -> None:
    data, bind = frozen_data
    config = bind()
    hashes = runner.validate_frozen_data(config, data)
    assert hashes["metadata.json"] == config["data_metadata_sha256"]
    table = pq.read_table(data / "validation_ni.parquet")
    pq.write_table(table.append_column("unbound_change", pa.array([1] * len(table))),
                   data / "validation_ni.parquet")
    with pytest.raises(ValueError, match="Frozen data hash differs"):
        runner.validate_frozen_data(config, data)


@pytest.mark.parametrize("fault", ["validation_user", "episode_label_marginal"])
def test_frozen_data_checks_semantics_after_hash_validation(frozen_data, fault: str) -> None:
    data, bind = frozen_data
    if fault == "validation_user":
        file = data / "validation_ni.parquet"
        rows = pq.read_table(file).to_pylist()
        rows[0]["user_id"] = "O0"
    else:
        file = data / "episodes.parquet"
        rows = pq.read_table(file).to_pylist()
        rows[0]["support_indices"][0] = 50
    pq.write_table(pa.Table.from_pylist(rows), file)
    config = bind()
    with pytest.raises(ValueError, match="(?i)(leakage|marginals)"):
        runner.validate_frozen_data(config, data)


@pytest.fixture
def calibration_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib.metadata
    torch = pytest.importorskip("torch")
    config_path, root = tmp_path / "config.json", tmp_path / "runs"
    config = {"data_dir": str(tmp_path / "unused-data"), "inner_alpha_candidates": [1e-4, 1e-3, 1e-2],
              "calibration_steps": 2, "pilot_steps": 540, "max_compute_multiplier": 3}
    runner.write_json(config_path, config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "fake-cuda")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: f"test-{name}")
    monkeypatch.setattr(runner, "validate_frozen_data", lambda *args: {"frozen-data": "trusted-hash"})
    monkeypatch.setattr(runner, "prepare_snapshot", lambda *args: "unused-snapshot")
    calls = []

    def train(*args, **kwargs):
        calls.append(kwargs)
        # Tie the two larger alphas, so deterministic lower-alpha tie breaking
        # must select 1e-3, using validation and never a Dev outcome.
        return {"alpha": kwargs["alpha"], "validation_metrics": {
            "interest_macro_recall_at_10": 0.1 if kwargs["alpha"] == 1e-4 else 0.2,
            "interest_macro_ndcg_at_10": 0.1}, "source_identity": kwargs["source_identity"]}

    monkeypatch.setattr(runner, "run_training", train)

    def invoke(*extra: str):
        monkeypatch.setattr(sys, "argv", ["oi_ni_meta", "--config", str(config_path), "--root", str(root), *extra])
        runner.main()

    invoke("--phase", "calibrate")
    return root, calls, invoke


def test_alpha_selection_uses_validation_and_pilot_binds_selection_receipt(calibration_cli) -> None:
    root, calls, invoke = calibration_cli
    selection = root / "alpha_selection.json"
    payload = json.loads(selection.read_text())
    assert payload["chosen_alpha"] == 1e-3
    assert payload["validation_only"] is True
    assert len(calls) == 3
    assert all(not call.get("eval_dev", False) for call in calls)
    invoke("--phase", "pilot", "--mode", "meta_no_history", "--selection-file", str(selection))
    assert calls[-1]["alpha"] == 1e-3
    assert calls[-1]["selection_sha256"] == runner.sha256_file(selection)
    assert calls[-1]["eval_dev"] is True
    invoke("--phase", "pilot", "--mode", "mixed", "--selection-file", str(selection))
    assert calls[-1]["alpha"] == 0.0


@pytest.mark.parametrize("fault", ["different_source", "not_validation_only", "unselected_alpha"])
def test_pilot_rejects_invalid_alpha_selection_receipt(calibration_cli, fault: str) -> None:
    root, calls, invoke = calibration_cli
    selection = root / "alpha_selection.json"
    payload = json.loads(selection.read_text())
    extra = []
    if fault == "different_source":
        payload["source_identity"] = "wrong-source"
    elif fault == "not_validation_only":
        payload["validation_only"] = False
    else:
        extra = ["--alpha", "0.01"]
    runner.write_json(selection, payload)
    with pytest.raises(ValueError, match="(?i)(selection|alpha)"):
        invoke("--phase", "pilot", "--mode", "meta", "--selection-file", str(selection), *extra)
    assert len(calls) == 3


@pytest.mark.parametrize("steps", [540, 1620])
def test_compute_matched_accepts_closed_budget_boundaries_and_binds_receipt(calibration_cli, steps: int) -> None:
    root, calls, invoke = calibration_cli
    selection = root / "alpha_selection.json"
    selected = json.loads(selection.read_text())
    budget = root / "compute_budget.json"
    runner.write_json(budget, {"source_identity": selected["source_identity"], "steps": steps})
    invoke("--phase", "compute_matched", "--mode", "mixed", "--selection-file", str(selection),
           "--budget-file", str(budget))
    assert len(calls) == 4
    assert calls[-1]["phase"] == "compute_matched"
    assert calls[-1]["mode"] == "mixed"
    assert calls[-1]["alpha"] == 0.0
    assert calls[-1]["steps"] == steps
    assert calls[-1]["budget_sha256"] == runner.sha256_file(budget)
    assert calls[-1]["selection_sha256"] == runner.sha256_file(selection)
    assert calls[-1]["eval_dev"] is True
    assert (root / "mixed_compute_success.json").is_file()


@pytest.mark.parametrize("fault", ["below_minimum", "above_maximum", "wrong_source", "wrong_mode", "missing_receipt"])
def test_compute_matched_rejects_invalid_budget_before_dispatch(calibration_cli, fault: str) -> None:
    root, calls, invoke = calibration_cli
    selection = root / "alpha_selection.json"
    selected = json.loads(selection.read_text())
    payload = {"source_identity": selected["source_identity"], "steps": 1080}
    mode = "mixed"
    if fault == "below_minimum":
        payload["steps"] = 539
    elif fault == "above_maximum":
        payload["steps"] = 1621
    elif fault == "wrong_source":
        payload["source_identity"] = "different-protocol"
    elif fault == "wrong_mode":
        mode = "meta"
    budget = root / "compute_budget.json"
    runner.write_json(budget, payload)
    extra = [] if fault == "missing_receipt" else ["--budget-file", str(budget)]
    with pytest.raises(ValueError, match="(?i)(compute|budget)"):
        invoke("--phase", "compute_matched", "--mode", mode,
               "--selection-file", str(selection), *extra)
    assert len(calls) == 3
