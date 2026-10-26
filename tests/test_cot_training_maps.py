from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_cot_sft_map_resolves_behavior_inputs(
    tmp_path: Path,
) -> None:
    code_dir = tmp_path / "code"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_path = tmp_path / "arguments.txt"

    config = (
        code_dir
        / "configs/sft/experiments/qwen3_behavior_cot_1k.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text("model_name: Qwen/Qwen3-8B\n", encoding="utf-8")
    train_file = (
        code_dir
        / "data/training/cot_sft_experiments/behavior_only/1k.parquet"
    )
    train_file.parent.mkdir(parents=True)
    train_file.write_bytes(b"train")
    adapter = input_dir / "sft/final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ARGUMENTS_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = (
        Path(__file__).parents[1]
        / "scripts/archive/amlt/2026-10-26/sft/run_qwen_cot_sft_map.sh"
    )
    subprocess.run(
        ["bash", str(script), "1k", "smoke"],
        check=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "ARGUMENTS_PATH": str(arguments_path),
            "AMLT_MAP_INPUT_JOB_NAME": "qwen3-behavior-22k",
            "AMLT_MAP_INPUT_DIR": str(input_dir),
            "AMLT_CODE_DIR": str(code_dir),
            "AMLT_OUTPUT_DIR": str(output_dir),
        },
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert str(config) in arguments
    assert str(train_file) in arguments
    assert str(adapter) in arguments
    assert arguments[arguments.index("--max-samples") + 1] == "32"
    assert (
        arguments[arguments.index("--run-name") + 1]
        == "qwen3-behavior-sft22k-cot-sft1k-smoke32"
    )
