import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments import rho_student_experiment as experiment
from new_interest.evaluation.evaluate_beam import canonical_sha256
from new_interest.evaluation import score_rho_reference


def fixture_artifacts(tmp_path):
    project = tmp_path / "project"
    inputs = project / "inputs"
    inputs.mkdir(parents=True)
    (project / "protocol.json").write_text(json.dumps({"version": experiment.VERSION, "model_revision": "frozen-revision"}))
    (project / "source_snapshot.json").write_text(json.dumps({"model_revision": "frozen-revision"}))
    (project / "source_manifest.json").write_text(json.dumps({"runtime_sha256": "a" * 64,
        "files": {"src/new_interest/experiments/rho_student_experiment.py": experiment.sha(Path(experiment.__file__)),
                  "src/new_interest/evaluation/score_rho_reference.py": experiment.sha(Path(score_rho_reference.__file__))}}))
    pq.write_table(pa.Table.from_pylist([{"interest_index": i, "target_subcategory": f"class{i}"} for i in range(108)]),
                   inputs / "interest_token_mapping.parquet")
    original = [{"source_sample_id": f"s{i}", "user_id": f"u{i}", "impression_id": str(i),
                 "input_sha256": f"hash{i}", "targets": [f"class{i % 2}"], "history_subcategories": ["class3"]}
                for i in range(8)]
    pq.write_table(pa.Table.from_pylist(original), inputs / "dev.parquet")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    directories = []
    for shard in range(4):
        directory = tmp_path / f"shard-{shard}"
        directory.mkdir()
        directories.append(directory)
        identity = experiment.evaluation_identity(project, adapter, "dev", shard, None)
        indices = list(range(shard, 8, 4))
        records = [{"record_index": i, **original[i], "scores": [-float(j) for j in range(108)],
                    "ranking": list(range(108))} for i in indices]
        output = directory / "part-00000000.parquet"
        pq.write_table(pa.Table.from_pylist(records), output)
        experiment.write(output.with_suffix(".json"), {"identity": identity, "indices": indices, "sha256": experiment.sha(output)})
        experiment.write(directory / "loading_audit.json", {"adapter_tensor_reload": {"accepted": True},
                         "pretrained_text_loaded": True, "missing_text_keys": [], "unexpected_text_keys": []})
        snapshots = [{"stage": "rows:0:2", "kernels": {"fla.kernel": {"cache": {"shape": "config1"}, "cache_results": False}}}]
        kernel = {k: identity[k] for k in ("kernel_selection", "kernel_selection_protocol", "kernel_selector_sha256")}
        kernel.update(evaluation_identity=canonical_sha256(identity), snapshots=snapshots,
                      snapshots_sha256=canonical_sha256(snapshots))
        experiment.write(directory / "kernel_selection.json", kernel)
        experiment.write(directory / "complete.json", {"identity": identity, "rows": 2,
            "kernel_selection_sha256": experiment.sha(directory / "kernel_selection.json"),
            "loading_audit_sha256": experiment.sha(directory / "loading_audit.json")})
    return project, directories, adapter


def test_independent_reader_verifies_real_shards_and_metrics(tmp_path):
    project, dirs, adapter = fixture_artifacts(tmp_path)
    result = experiment.read_evaluation(project, dirs, tmp_path / "readback", adapter, "dev")
    assert result["accepted"] and result["rows"] == 8
    assert result["metrics"]["macro_recall_at_20"] == 1
    assert result["metrics"]["supported_classes"] == 2


@pytest.mark.parametrize("mutation", ["target", "ranking", "duplicate_index"])
def test_reader_rejects_semantic_corruption_even_with_recomputed_file_hash(tmp_path, mutation):
    project, dirs, adapter = fixture_artifacts(tmp_path)
    p = dirs[0] / "part-00000000.parquet"
    records = pq.read_table(p).to_pylist()
    if mutation == "target":
        records[0]["targets"] = ["class99"]
    elif mutation == "ranking":
        records[0]["ranking"][:2] = [1, 0]
    else:
        records[1]["record_index"] = records[0]["record_index"]
    pq.write_table(pa.Table.from_pylist(records), p)
    marker = experiment.read(p.with_suffix(".json"))
    marker["sha256"] = experiment.sha(p)
    experiment.write(p.with_suffix(".json"), marker)
    with pytest.raises(ValueError):
        experiment.read_evaluation(project, dirs, tmp_path / "readback", adapter, "dev")


def test_reader_rejects_kernel_choice_conflict(tmp_path):
    project, dirs, adapter = fixture_artifacts(tmp_path)
    p = dirs[1] / "kernel_selection.json"
    k = experiment.read(p)
    k["snapshots"][0]["kernels"]["fla.kernel"]["cache"]["shape"] = "config2"
    k["snapshots_sha256"] = canonical_sha256(k["snapshots"])
    experiment.write(p, k)
    marker = experiment.read(dirs[1] / "complete.json")
    marker["kernel_selection_sha256"] = experiment.sha(p)
    experiment.write(dirs[1] / "complete.json", marker)
    with pytest.raises(ValueError, match="kernel selections"):
        experiment.read_evaluation(project, dirs, tmp_path / "readback", adapter, "dev")


def test_reader_rejects_rehashed_empty_kernel_evidence(tmp_path):
    project, dirs, adapter = fixture_artifacts(tmp_path)
    p = dirs[0] / "kernel_selection.json"
    receipt = experiment.read(p)
    receipt["snapshots"][0]["kernels"] = {}
    receipt["snapshots_sha256"] = canonical_sha256(receipt["snapshots"])
    experiment.write(p, receipt)
    marker = experiment.read(dirs[0] / "complete.json")
    marker["kernel_selection_sha256"] = experiment.sha(p)
    experiment.write(dirs[0] / "complete.json", marker)
    with pytest.raises(ValueError, match="no selected kernels"):
        experiment.read_evaluation(project, dirs, tmp_path / "readback", adapter, "dev")


def test_standalone_evaluator_rejects_a_different_source_binding(tmp_path):
    project, dirs, adapter = fixture_artifacts(tmp_path)
    source = experiment.read(project / "source_manifest.json")
    source["files"]["src/new_interest/experiments/rho_student_experiment.py"] = "0" * 64
    experiment.write(project / "source_manifest.json", source)
    with pytest.raises(ValueError, match="frozen runtime"):
        experiment.read_evaluation(project, dirs, tmp_path / "readback", adapter, "dev")
