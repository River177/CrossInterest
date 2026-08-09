from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    (
        "source_job",
        "adapter_relative",
        "expected_condition",
    ),
    [
        ("qwen3-interest-22k", "sft/final_adapter", "sft22k"),
        (
            "grpo-1k-qwen3-interest-22k",
            "grpo/final_adapter",
            "old-grpo1k",
        ),
        (
            "cot-sft-1k-qwen3-interest-22k",
            "cot_sft/final_adapter",
            "cot-sft1k",
        ),
        (
            "correctness-grpo-1k-qwen3-interest-22k",
            "grpo/final_adapter",
            "new-grpo1k",
        ),
    ],
)
def test_accuracy_map_resolves_condition(
    tmp_path: Path,
    source_job: str,
    adapter_relative: str,
    expected_condition: str,
) -> None:
    code_dir = tmp_path / "code"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_path = tmp_path / "arguments.txt"

    adapter = input_dir / adapter_relative
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    eval_file = (
        code_dir
        / "data/training/sft_experiments/eval/interest_only"
        / "dev_10k_grouped.parquet"
    )
    eval_file.parent.mkdir(parents=True)
    eval_file.write_bytes(b"eval")
    labels_file = (
        code_dir
        / "data/training/generative_new_interest_balanced_200_h50"
        / "retained_interests.parquet"
    )
    labels_file.parent.mkdir(parents=True)
    labels_file.write_bytes(b"labels")
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ARGUMENTS_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = (
        Path(__file__).parents[1]
        / "scripts/amlt/evaluation/run_qwen_cot_accuracy_map.sh"
    )
    subprocess.run(
        ["bash", str(script), "cot", "2", "4", "full"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "ARGUMENTS_PATH": str(arguments_path),
            "AMLT_MAP_INPUT_JOB_NAME": source_job,
            "AMLT_MAP_INPUT_DIR": str(input_dir),
            "AMLT_CODE_DIR": str(code_dir),
            "AMLT_OUTPUT_DIR": str(output_dir),
        },
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert str(adapter) in arguments
    assert arguments[arguments.index("--start-index") + 1] == "5000"
    assert arguments[arguments.index("--max-examples") + 1] == "2500"
    assert (
        arguments[arguments.index("--run-name") + 1]
        == f"qwen3-{expected_condition}-interest-cot-dev10k-s2of4"
    )
