import importlib.util
import json
from pathlib import Path

import pytest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('grpo_builder',REPO/'scripts/analysis/prepare_short_think_grpo.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_actual_installer_manifest_argument_is_required(tmp_path):
    shell=tmp_path/'scripts/amlt/common/install_qwen35_fast_kernels.sh'
    shell.parent.mkdir(parents=True)
    shell.write_text('python install.py \\\n  --manifest configs/experiments/kernels.json \\\n  --output result.json\n')
    with pytest.raises(FileNotFoundError,match='kernels.json'):
        builder.validate_environment_inputs(tmp_path)
    manifest=tmp_path/'configs/experiments/kernels.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'install_wheels':[{}, {}, {}, {}]}))
    assert builder.validate_environment_inputs(tmp_path)=='configs/experiments/kernels.json'
    manifest.write_text(json.dumps({'install_wheels':[{}]}))
    with pytest.raises(ValueError,match='Incomplete'):
        builder.validate_environment_inputs(tmp_path)
