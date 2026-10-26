"""Exercise installed AMLT's real code uploader against local capturing storage only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
REQUIRED_CLI_ENV = {"AMLT_SKIP_GIT_INFO": "1", "AMLT_CODE_CHECKSUM_SECS": "0"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): {"sha256": digest(path), "size": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()}


def map_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CapturingStorage:
    """No storage SDK: capture exact file bytes' hashes at the upload boundary."""

    def __init__(self):
        self.uploads = []
        self.extra_blobs = []
        self.transmitted = {}

    def upload(self, source, destination):
        assert not self.uploads, "Unexpected duplicate staging upload"
        source_files = files(Path(source))
        self.uploads.append({"destination": destination, "files": source_files})
        self.transmitted.update(source_files)

    def create_blob_from_bytes(self, destination, data, *, delete_first):
        assert self.uploads, "Extra blob preceded the real staging upload"
        prefix = self.uploads[0]["destination"]
        assert destination.startswith(prefix)
        relative = destination[len(prefix):]
        if isinstance(data, str):
            data = data.encode()
        record = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        self.extra_blobs.append({"relative_path": relative, "delete_first": delete_first, **record})
        self.transmitted[relative] = record


class CapturingSession:
    def __init__(self, *, existing=None, forbid_queries=False):
        self.existing = existing
        self.forbid_queries = forbid_queries
        self.queries = []
        self.added = []

    def query(self, model):
        if self.forbid_queries:
            raise AssertionError("Checksum-disabled upload queried an existing snapshot")
        record = {"model": model.__name__}
        self.queries.append(record)
        return CapturingQuery(self, record)

    @contextmanager
    def enable_writing(self):
        yield

    def add(self, model):
        self.added.append({"uuid": model.uuid, "remote_dir": model.remote_dir})


class CapturingQuery:
    def __init__(self, session, record):
        self.session = session
        self.record = record

    def filter_by(self, **values):
        self.record["filter"] = values
        return self

    def one(self):
        assert self.session.existing is not None
        self.record["returned_simulated_polluted_snapshot"] = True
        return self.session.existing


def make_uploader(CodeUploader, session):
    # Passing no project avoids constructing any Azure storage client. Only the
    # external storage and database boundaries are replaced; uploader methods,
    # package staging, git inspection and checksum logic remain the installed code.
    uploader = CodeUploader()
    uploader._project = SimpleNamespace(uuid="local-transport-proof", path="local-transport-proof", session=session)
    uploader._storage_client = CapturingStorage()
    return uploader


def verify(package: Path, expected_manifest: str) -> dict:
    from amlt.api.run import CodeUploader
    from amlt.config import CodeConfig
    import amlt

    package = package.resolve()
    assert digest(package / "recovery_manifest.json") == expected_manifest
    manifest = json.loads((package / "recovery_manifest.json").read_text())
    expected = {name: {"sha256": value, "size": (package / name).stat().st_size}
                for name, value in manifest["files"].items()}
    expected["recovery_manifest.json"] = {"sha256": expected_manifest, "size": (package / "recovery_manifest.json").stat().st_size}
    assert len(expected) == 204
    before = files(package)
    for name, record in expected.items():
        assert before[name] == record

    report = {"version": "rho-amlt-code-transport-boundary-v1", "passed": False,
              "real_amlt_methods": ["CodeUploader.prepare", "CodeUploader._get_git_info", "CodeUploader._upload_package", "CodeUploader.get_existing_code_package", "CodeUploader.upload", "CodePackage.checksum"],
              "mocked_boundaries": ["storage upload/create_blob_from_bytes", "project session query/add"],
              "amlt_version": amlt.__version__, "amlt_uploader_source": inspect.getsourcefile(CodeUploader),
              "amlt_uploader_source_sha256": digest(Path(inspect.getsourcefile(CodeUploader))),
              "package": str(package), "package_manifest_sha256": expected_manifest,
              "expected_file_count": len(expected), "expected_file_map_sha256": map_digest(expected),
              "required_submitter_cli_environment": REQUIRED_CLI_ENV,
              "network_attempts": [], "real_uploads": False, "real_database_writes": False, "submitted": False}

    def deny_network(event, values):
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
            report["network_attempts"].append(event)
            raise AssertionError("Network access forbidden in local uploader proof")

    sys.addaudithook(deny_network)
    config = CodeConfig(local_dir=str(package), ignore=[])

    with patch.dict(os.environ, {"AMLT_SKIP_GIT_INFO": "0", "AMLT_CODE_CHECKSUM_SECS": "2"}):
        old = make_uploader(CodeUploader, CapturingSession())
        old_package = old.prepare(config)
        try:
            assert files(Path(old_package.staging_dir)) == expected
            assert old_package.git_diff, "Current actual repository must reproduce its real repo.diff"
            assert old_package.checksum, "Legacy checksum unexpectedly timed out"
            old._upload_package(old_package)
            extras = sorted(set(old._storage_client.transmitted) - set(expected))
            assert "repo.diff" in extras and set(extras) <= {"repo.diff", "staged.diff"}
            assert all(old._storage_client.transmitted[name] == record for name, record in expected.items())
            checksum = old_package.checksum
            polluted = SimpleNamespace(uuid=checksum, remote_dir=old_package.remote_dir)
            report["legacy_upload_red"] = {"staged_file_count": len(expected), "staged_files_exact": True,
                "transmitted_file_count": len(old._storage_client.transmitted), "extra_paths": extras,
                "extra_blobs": old._storage_client.extra_blobs, "source_checksum": checksum,
                "remote_dir_uses_checksum": old_package.remote_dir.endswith(checksum),
                "expected_file_bytes_unchanged": True, "git_diff_from_actual_repository": True}
        finally:
            old_package.cleanup()

    # This separate negative control explains why skipping Git alone is not enough.
    with patch.dict(os.environ, {"AMLT_SKIP_GIT_INFO": "1", "AMLT_CODE_CHECKSUM_SECS": "2"}):
        stale_session = CapturingSession(existing=polluted)
        stale = make_uploader(CodeUploader, stale_session)
        stale_package = stale.prepare(config)
        try:
            assert stale_package.git_diff is None and stale_package.git_staged is None
            assert stale_package.checksum == checksum
            selected = stale.upload(stale_package)
            assert selected is polluted and len(stale_session.queries) == 1
            assert stale._storage_client.uploads == [] and stale_session.added == []
            report["skip_git_only_cache_red"] = {"git_metadata_absent": True, "same_source_checksum": True,
                "actual_existing_snapshot_lookup": stale_session.queries,
                "returned_simulated_polluted_snapshot": True, "staging_upload_count": 0}
        finally:
            stale_package.cleanup()

    with patch.dict(os.environ, REQUIRED_CLI_ENV):
        green_session = CapturingSession(existing=polluted, forbid_queries=True)
        green = make_uploader(CodeUploader, green_session)
        green_package = green.prepare(config)
        try:
            assert files(Path(green_package.staging_dir)) == expected
            assert green_package.checksum is None
            assert all(getattr(green_package, name) is None for name in ("git_diff", "git_staged", "git_sha", "git_branch", "git_repo_dir"))
            fresh_uuid, remote_dir = green_package.uuid, green_package.remote_dir
            assert uuid.UUID(fresh_uuid).version == 4
            assert remote_dir.endswith(fresh_uuid) and remote_dir != polluted.remote_dir
            selected = green.upload(green_package)
            assert selected is not polluted and selected.uuid == fresh_uuid
            assert green_session.queries == []
            assert green_session.added == [{"uuid": fresh_uuid, "remote_dir": remote_dir}]
            assert len(green._storage_client.uploads) == 1
            assert green._storage_client.extra_blobs == []
            assert green._storage_client.transmitted == expected
            report["skip_git_and_checksum_zero_green"] = {"transmitted_file_count": len(expected),
                "transmitted_file_map_sha256": map_digest(green._storage_client.transmitted),
                "exact_manifest_files_and_bytes": True, "extra_paths": [], "missing_paths": [], "changed_paths": [],
                "git_metadata_absent": True, "checksum_is_none": True,
                "actual_get_existing_code_package_called_by_upload": True,
                "existing_snapshot_database_lookup_count": len(green_session.queries),
                "polluted_snapshot_not_reused": True, "fresh_uuid": fresh_uuid, "fresh_remote_dir": remote_dir,
                "fresh_remote_dir_uses_uuid": True, "captured_database_additions": green_session.added,
                "staging_upload_count": 1, "additional_blob_upload_count": 0}
        finally:
            green_package.cleanup()

    after = files(package)
    assert before == after
    assert report["network_attempts"] == []
    report.update(passed=True, source_package_unchanged=True,
                  source_package_before_sha256=map_digest(before), source_package_after_sha256=map_digest(after))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = verify(args.package, args.manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output), "sha256": digest(args.output),
                      "old_extra_paths": report["legacy_upload_red"]["extra_paths"],
                      "new_transmitted_file_count": report["skip_git_and_checksum_zero_green"]["transmitted_file_count"]}))


if __name__ == "__main__":
    main()
