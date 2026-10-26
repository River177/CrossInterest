"""Freeze a small two-fold reference run; never include model/adapter weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate-acceptance', type=Path)
    parser.add_argument('--recovery-parent', type=Path)
    a = parser.parse_args()
    dest = a.output.resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    original = REPO / 'artifacts/mind-short-think-sft-20260913/bundle-full-v1'
    files = [p.relative_to(REPO).as_posix() for p in sorted((REPO / 'src').rglob('*.py'))]
    files += ['scripts/amlt/sft/run_rho_twofold_reference.sh',
              'scripts/amlt/common/install_qwen35_fast_kernels.sh',
              'scripts/amlt/common/install_qwen35_fast_kernels.py',
              'configs/experiments/qwen35_fast_kernel_install_v1.json']
    for name in files:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = REPO / name
        if not source.exists() and name in (
                'scripts/amlt/common/install_qwen35_fast_kernels.sh',
                'scripts/amlt/common/install_qwen35_fast_kernels.py',
                'configs/experiments/qwen35_fast_kernel_install_v1.json'):
            source = original / name
        shutil.copyfile(source, target)
    for name, source in [('requirements-worker.txt', original / 'requirements-short-think-worker.txt'),
                         ('source_snapshot.json', original / 'source_snapshot.json'),
                         ('protocol.json', REPO / 'configs/experiments/mind_rho_twofold_reference_v1.json')]:
        shutil.copyfile(source, dest / name)
    shutil.copytree(REPO / 'data/training/mind_rho_twofold_v1', dest / 'inputs')
    shutil.copyfile(REPO / 'artifacts/rho-twofold-reference-20260915/token_audit.json', dest / 'token_audit.json')
    if a.gate_acceptance:
        shutil.copyfile(a.gate_acceptance, dest / 'gate_acceptance.json')
    if a.recovery_parent:
        shutil.copyfile(a.recovery_parent, dest / 'recovery_parent.json')
        shutil.copyfile(a.recovery_parent.parent / 'reused_token_gradient_audit.json', dest / 'reused_token_gradient_audit.json')
    inventory = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    if any(p.endswith(('.safetensors', '.pt', '.bin')) for p in inventory):
        raise ValueError('Model/adapter weights must not be packaged')
    runtime = {k: v for k, v in inventory.items() if k.startswith(('src/', 'scripts/', 'configs/')) or k == 'requirements-worker.txt'}
    runtime_sha = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    if a.gate_acceptance:
        gate = json.loads(a.gate_acceptance.read_text())
        if not gate['passed'] or gate['runtime_sha256'] != runtime_sha or gate['identity']['protocol_sha256'] != inventory['protocol.json']:
            raise ValueError('Source/protocol differs from accepted gate')
    manifest = {'runtime_sha256': runtime_sha, 'files': inventory,
                'source_file_count': len(inventory), 'source_bytes': sum((dest / k).stat().st_size for k in inventory)}
    (dest / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}))


if __name__ == '__main__':
    main()
