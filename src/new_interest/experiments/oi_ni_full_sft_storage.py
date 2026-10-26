"""Node-local journal and strict parent binding for storage-only recovery."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil

from new_interest.experiments import oi_ni_meta as core


class LocalTrainingJournal:
    def __init__(self, directory: Path, prefix: list[dict]):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'training.jsonl'
        self.steps = len(prefix)
        if [r['step'] for r in prefix] != list(range(1,self.steps+1)):
            raise ValueError('Journal prefix has missing or duplicate steps')
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in prefix))

    def append(self, row: dict):
        if row['step'] != self.steps+1:
            raise ValueError('Journal append is not the next global step')
        with self.path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        self.steps += 1

    def snapshot(self, destination: Path):
        """Call only while peer ranks wait outside NCCL in primary_phase."""
        temporary = destination.with_name(destination.name+f'.tmp-{os.getpid()}')
        shutil.copyfile(self.path,temporary)
        if core.sha256_file(temporary) != core.sha256_file(self.path):
            raise ValueError('Journal snapshot hash differs')
        os.replace(temporary,destination)


def hydrate_recovery_parent(binding: dict, config: dict, mode: str, scratch: Path):
    """Validate immutable old identity and bytes before changing run identity."""
    if binding['mode'] != mode:
        raise ValueError('Recovery condition differs')
    parent = Path(binding['checkpoint_path'])
    attempt = parent.parent.parent
    if core.sha256_file(parent/'manifest.json') != binding['checkpoint_manifest_sha256']:
        raise ValueError('Recovery parent manifest differs')
    if core.sha256_file(attempt/'source_manifest.json') != binding['source_manifest_sha256']:
        raise ValueError('Recovery source manifest differs')
    resolved = json.loads((attempt/'resolved_config.json').read_text())
    if resolved['identity'] != binding['identity'] or resolved['mode'] != mode or resolved['config'] != config:
        raise ValueError('Recovery scientific configuration or identity differs')
    expected_identity = hashlib.sha256(json.dumps({'source':binding['source_manifest_sha256'],
        'config':config,'mode':mode},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if expected_identity != binding['identity'] or resolved['source_sha256'] != binding['source_manifest_sha256']:
        raise ValueError('Recovery source/config identity binding differs')
    manifest = json.loads((parent/'manifest.json').read_text())
    if manifest['identity'] != binding['identity']:
        raise ValueError('Recovery checkpoint identity differs')
    destination = scratch/'recovery-parent'
    destination.mkdir()
    for relative,expected in manifest['files'].items():
        path = (parent/relative).resolve()
        if not path.is_relative_to(parent.resolve()):
            raise ValueError('Recovery manifest path escapes parent')
        target = destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        if core.sha256_file(target) != expected:
            raise ValueError('Recovery payload bytes differ: '+relative)
    shutil.copyfile(parent/'manifest.json',destination/'manifest.json')
    core.verify_manifest(destination,binding['identity'])
    state = json.loads((destination/'state.json').read_text())
    if state['step'] != binding['step'] or state['tensor_sha256'] != binding['tensor_sha256']:
        raise ValueError('Recovery checkpoint cursor or tensor digest differs')
    return destination
