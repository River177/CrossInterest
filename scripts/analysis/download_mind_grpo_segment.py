"""Collect a completion-bound raw segment, reusing only SHA-identical local files.

This verifies the complete raw artifact manifest, not evaluation semantics or
the compressed archive. Run the frozen independent reader after collection.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import uuid


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or name == '.':
        raise ValueError('Unsafe or noncanonical manifest path')
    return p


def main():
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--prefix', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reuse-roots', type=Path, nargs='*', default=[])
    a = p.parse_args()
    if not re.fullmatch(r'projects/sft4safety/amlt-results/[A-Za-z0-9.-]+', a.prefix):
        raise ValueError('Invalid results prefix')
    if not re.fullmatch(r'\d{8}T\d{6}Z-[a-f0-9]{8}', a.attempt):
        raise ValueError('Invalid attempt')
    identity = sha(a.bundle / 'engine_contract.json')
    dest = a.output_dir
    meta = dest.with_name(dest.name + '.download-meta')
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        container = BlobServiceClient(
            'https://csnewsandfeeds4150361735.blob.core.windows.net',
            credential=AzureCliCredential(), connection_timeout=10, read_timeout=25,
            retry_total=2, max_single_get_size=4 * 1024 * 1024,
            max_chunk_get_size=4 * 1024 * 1024,
        ).get_container_client('amulet')
        base = a.prefix + '/attempts/' + a.attempt + '/'
        package = json.loads(container.download_blob(base + 'package.json').readall())
        marker = json.loads(container.download_blob(a.prefix + '/completed-' + a.attempt + '.json').readall())
        if package != marker or package['attempt'] != a.attempt or package['contract_sha256'] != identity:
            raise ValueError('Completion identity binding differs')
        if not 0 < package['bytes'] <= 16_000_000_000:
            raise ValueError('Unexpected archive size')
        body = container.download_blob(base + 'artifact_manifest.json').readall()
        if hashlib.sha256(body).hexdigest() != package['manifest_sha256']:
            raise ValueError('Artifact manifest binding differs')
        manifest = json.loads(body)
        if manifest['engine_identity'] != identity or not 0 < len(manifest['files']) < 100_000:
            raise ValueError('Invalid artifact manifest')
        for name, digest in manifest['files'].items():
            safe_name(name)
            if name == 'artifact_manifest.json' or not re.fullmatch('[a-f0-9]{64}', digest):
                raise ValueError('Invalid manifest member')
        binding = {'prefix': a.prefix, 'attempt': a.attempt, 'identity': identity, 'package': package}
        bound = meta / 'identity.json'
        if bound.exists():
            if json.loads(bound.read_text()) != binding:
                raise ValueError('Existing destination belongs to another artifact')
        else:
            if dest.exists():
                raise FileExistsError('Unbound destination already exists')
            write(bound, binding)
        dest.mkdir(parents=True, exist_ok=True)
        mf = dest / 'artifact_manifest.json'
        if mf.exists() and mf.read_bytes() != body:
            raise ValueError('Existing manifest differs')
        if not mf.exists():
            mf.write_bytes(body)

        def fetch(item):
            name, expected = item
            target = dest / name
            if target.exists():
                if not target.is_file() or sha(target) != expected:
                    raise ValueError(f'Existing artifact differs: {name}')
                return 'existing', target.stat().st_size
            target.parent.mkdir(parents=True, exist_ok=True)
            for root in a.reuse_roots:
                source = root / name
                if source.is_file() and sha(source) == expected:
                    # Both paths are immutable accepted artifacts; never edit
                    # either path in place. Recheck through the destination.
                    os.link(source, target)
                    if sha(target) != expected:
                        raise ValueError(f'Reused artifact differs: {name}')
                    return 'reused', target.stat().st_size
            temporary = None
            try:
                blob = container.get_blob_client(base + name)
                if blob.get_blob_properties().size > 2_000_000_000:
                    raise ValueError('Unexpected individual artifact size')
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-', delete=False) as stream:
                    temporary = Path(stream.name)
                    blob.download_blob(max_concurrency=2).readinto(stream)
                if sha(temporary) != expected:
                    raise ValueError(f'Downloaded artifact differs: {name}')
                size = temporary.stat().st_size
                os.replace(temporary, target)
                temporary = None
                return 'downloaded', size
            except Exception as error:
                failed = meta / 'failed' / (name + '.' + uuid.uuid4().hex[:8])
                failed.parent.mkdir(parents=True, exist_ok=True)
                if temporary is not None and temporary.exists():
                    os.replace(temporary, failed)
                write(failed.with_name(failed.name + '.json'), {'file': name, 'expected_sha256': expected, 'error_type': type(error).__name__})
                raise

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fetch, manifest['files'].items()))
        receipt = {'passed': True, 'kind': 'completion_bound_raw_artifact_readback',
                   'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                   **binding, 'files': len(results), 'bytes': sum(n for _, n in results),
                   'counts': dict(Counter(k for k, _ in results)),
                   'artifact_manifest_sha256': sha(mf), 'archive_downloaded': False,
                   'independent_evaluation_acceptance_pending': True}
        write(meta / 'receipt.json', receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
