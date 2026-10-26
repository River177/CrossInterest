import hashlib
import importlib.util
from pathlib import Path

script = Path(__file__).resolve().parents[1] / "scripts/analysis/observe_cir_ni_matched.py"
spec = importlib.util.spec_from_file_location("matched_observer", script)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def test_cache_requires_remote_etag_size_and_local_content_hash(tmp_path):
    receipt_path = tmp_path / "previous/receipt.json"
    data = b'{"status":"passed"}'
    f = receipt_path.parent / "compact/gates/accepted.json"
    f.parent.mkdir(parents=True)
    f.write_bytes(data)
    previous = [(receipt_path, {"compact_files": {"gates/accepted.json": {
        "etag": "etag1", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}})]
    assert observer.cached_payload(previous, "gates/accepted.json", "etag1", len(data))[0] == data
    assert observer.cached_payload(previous, "gates/accepted.json", "etag2", len(data)) == (None, None)
    assert observer.cached_payload(previous, "gates/accepted.json", "etag1", len(data)+1) == (None, None)
    f.write_bytes(b'x'*len(data))
    assert observer.cached_payload(previous, "gates/accepted.json", "etag1", len(data)) == (None, None)
