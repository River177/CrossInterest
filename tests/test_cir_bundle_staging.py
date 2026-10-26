import importlib.util
from pathlib import Path

import pytest

from new_interest.data.build_cir_dataset import sha256

script = Path(__file__).resolve().parents[1] / "scripts/analysis/prepare_cir_bundle.py"
spec = importlib.util.spec_from_file_location("cir_bundle_prepare", script)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_upload_control_file_absent_on_worker(tmp_path):
    (tmp_path / ".amltignore").write_text("*.pyc\n")
    (tmp_path / "train.py").write_text("actual training implementation")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/cir_bundle.json").write_text("old manifest")
    payload = prepare.bundle_payload(tmp_path)
    assert set(payload) == {"train.py"}
    (tmp_path / ".amltignore").unlink()
    assert all(sha256(tmp_path / name) == digest for name, digest in payload.items())
    (tmp_path / "train.py").write_text("tampered")
    assert sha256(tmp_path / "train.py") != payload["train.py"]
    (tmp_path / "train.py").unlink()
    with pytest.raises(FileNotFoundError):
        sha256(tmp_path / "train.py")
