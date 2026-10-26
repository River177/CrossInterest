"""Finite authorized gate -> two reference teachers -> compact remote acceptance."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'artifacts/rho-twofold-reference-20260915'
AMLT = '/Users/river/miniforge3/envs/amlt/bin/amlt'
GATE_EXP = 'mind-rho-twofold-gate-v1-0915'
GATE_BUNDLE = 'bundle-gate-v1'
GATE_CONFIG = 'mind_rho_twofold_gate_0915_v1.yaml'
FULL_EXP = 'mind-rho-twofold-reference-v1-0915'
GATE_JOB = 'qwen35-2b-rho-reference-gate'
FULL_JOBS = ['qwen35-2b-rho-reference-fold0', 'qwen35-2b-rho-reference-fold1']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def command(args, log, timeout=180):
    try:
        result = subprocess.run(args, cwd=REPO, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        out = result.stdout
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('Command timed out: ' + Path(log).name) from error
    out = re.sub(r'\x1b\[[0-9;]*m', '', out)
    out = re.sub(r'https?://[^\s]+\?[^\s]*(?:sig=|token=)[^\s]*', '[SIGNED_URL_REDACTED]', out)
    Path(log).write_text(out)
    if result.returncode:
        raise RuntimeError('Command failed; see ' + str(log))
    return out


def statuses(exp, jobs):
    try:
        out = command([AMLT, 'status', exp, '--hide-urls'], ROOT / f'{exp}-status.txt')
    except (TimeoutError, RuntimeError) as error:
        raise TimeoutError('Retryable status transport: ' + str(error)) from error
    found = {}
    for job in jobs:
        lines = [line for line in out.splitlines() if re.match(r'^\s*(?::\d+\s+)?:?' + re.escape(job) + r'\s', line)]
        if len(lines) != 1:
            raise ValueError('Missing/ambiguous exact job in status')
        match = re.search(r'\b(queued|preparing|running|pass|completed|failed|killed|canceled)\b', lines[0])
        if not match:
            raise ValueError('Unrecognized job state')
        found[job] = match[1]
    return found


def collect(exp, job, label):
    dest = ROOT / 'compact' / label
    dest.mkdir(parents=True, exist_ok=True)
    # AMLT/AzCopy supports basename include-patterns, not wildcard URL paths.
    names = ['environment.json', 'snapshot_audit.json', 'training_acceptance.json',
             'reference_acceptance.json', 'worker_success.json']
    if label == 'gate':
        names += ['token_gradient_audit.json', 'gate_acceptance.json', 'single_recovery_acceptance.json']
    includes = [value for name in names for value in ('--include', name)]
    try:
        command([AMLT, 'results', 'download', exp, ':' + job,
                 *includes, '--output', str(dest), '--n-workers', '4'],
                ROOT / f'collect-{label}.log', timeout=300)
    except (TimeoutError, RuntimeError) as error:
        raise TimeoutError('Retryable compact-report transport: ' + str(error)) from error
    candidates = list(dest.rglob('worker_success.json'))
    if len(candidates) != 1:
        raise ValueError('Missing/ambiguous worker acceptance')
    folder = candidates[0].parent
    success = read(candidates[0])
    acceptance = folder / 'reference_acceptance.json'
    if not success['passed'] or not read(acceptance)['accepted']:
        raise ValueError('Worker/independent remote acceptance rejected')
    if sha(acceptance) != success['reference_acceptance_sha256']:
        raise ValueError('Compact independent acceptance hash differs')
    gate_source = read(ROOT / GATE_BUNDLE / 'source_manifest.json')
    inputs = {k: v for k, v in gate_source['files'].items()
              if k.startswith('inputs/') or k in ('source_snapshot.json', 'token_audit.json',
                                                 'recovery_parent.json', 'reused_token_gradient_audit.json')}
    inputs_sha = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if (success['runtime_sha256'] != gate_source['runtime_sha256']
            or success['identity']['protocol_sha256'] != gate_source['files']['protocol.json']
            or success['identity']['experiment_inputs_sha256'] != inputs_sha):
        raise ValueError('Downloaded compact report has wrong source/protocol')
    reference = read(acceptance)
    if (reference['teacher_train_sha256'] != success['train_file_sha256']
            or reference['score_file_sha256'] != success['score_file_sha256']
            or reference['metrics']['rows'] != success['expected_oof_rows']):
        raise ValueError('Independent reference report differs from worker identity')
    return success, folder


def prepare_full(gate_folder):
    dest = ROOT / 'bundle-full-v1'
    if dest.exists():
        raise ValueError('Unexpected existing formal bundle; inspect state before retry')
    shutil.copytree(ROOT / GATE_BUNDLE, dest)
    shutil.copyfile(gate_folder / 'gate_acceptance.json', dest / 'gate_acceptance.json')
    source = read(dest / 'source_manifest.json')
    source['files']['gate_acceptance.json'] = sha(dest / 'gate_acceptance.json')
    source['source_file_count'] = len(source['files'])
    source['source_bytes'] = sum((dest / p).stat().st_size for p in source['files'])
    write(dest / 'source_manifest.json', source)
    cfg = yaml.safe_load((REPO / 'configs/amlt' / GATE_CONFIG).read_text())
    cfg['code']['local_dir'] = '$CONFIG_DIR/../../artifacts/rho-twofold-reference-20260915/bundle-full-v1'
    cfg['description'] = 'Authorized twofold reference teachers: disjoint10800 users each, native Qwen3.5-2B SFT3epochs, opposite-half full-vocabulary SID NLL, remote-only artifacts.'
    template = cfg['jobs'][0]
    cfg['jobs'] = []
    for fold, name in enumerate(FULL_JOBS):
        job = json.loads(json.dumps(template))
        job['name'] = name
        job['command'][-1] = f'bash scripts/amlt/sft/run_rho_twofold_reference.sh full {fold}'
        cfg['jobs'].append(job)
    path = REPO / 'configs/amlt/mind_rho_twofold_reference_0915_v1.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command([AMLT, 'run', str(path), FULL_EXP, '--dump'], ROOT / 'formal_dump.yaml')
    return path


def main():
    global GATE_EXP, GATE_BUNDLE, GATE_CONFIG
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--gate-version', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    if args.gate_version != 1:
        GATE_EXP = f'mind-rho-twofold-gate-v{args.gate_version}-0915'
        GATE_BUNDLE = f'bundle-gate-v{args.gate_version}'
        GATE_CONFIG = f'mind_rho_twofold_gate_0915_v{args.gate_version}.yaml'
    lock = (ROOT / 'pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Reference pipeline already running')
        return
    state_path = ROOT / 'pipeline_state.json'
    state = read(state_path) if state_path.exists() else {'phase': 'gate_wait', 'formal_submission_attempted': False}
    write(ROOT / 'pipeline_process.json', {'pid': os.getpid(), 'started_unix': time.time(), 'script': str(Path(__file__))})
    terminal = {'pass', 'completed', 'failed', 'killed', 'canceled'}
    while state['phase'] not in {'complete', 'needs_attention'}:
        try:
            if state['phase'] == 'gate_wait':
                observed = statuses(GATE_EXP, [GATE_JOB])
                state['gate_status'] = observed[GATE_JOB]
                if observed[GATE_JOB] in {'failed', 'killed', 'canceled'}:
                    raise ValueError('Gate ended without acceptance; inspect native failure before any retry')
                if observed[GATE_JOB] in {'pass', 'completed'}:
                    state['phase'] = 'gate_readback'
            if state['phase'] == 'gate_readback':
                success, folder = collect(GATE_EXP, GATE_JOB, 'gate')
                if success['identity']['phase'] != 'gate' or success['expected_oof_rows'] != 64:
                    raise ValueError('Wrong gate scope')
                if read(folder / 'gate_acceptance.json') != success:
                    raise ValueError('Gate marker differs from verified worker receipt')
                shutil.copyfile(folder / 'gate_acceptance.json', ROOT / 'gate_acceptance.json')
                state.update(gate_acceptance=str(ROOT / 'gate_acceptance.json'), phase='formal_prepare',
                             gate_compact_folder=str(folder))
                write(state_path, state)
            if state['phase'] == 'formal_prepare':
                path = prepare_full(Path(state['gate_compact_folder']))
                state.update(formal_config=str(path), phase='formal_submit')
                write(state_path, state)
            if state['phase'] == 'formal_submit':
                if state['formal_submission_attempted']:
                    raise ValueError('Ambiguous prior formal submission; never automatically duplicate')
                state['formal_submission_attempted'] = True
                write(state_path, state)
                command([AMLT, 'run', state['formal_config'], FULL_EXP, '-d',
                         'Twofold Qwen3.5-2B reference models from untouched pretrained weights. Each trains10800 users for3epochs and scores the other10800. All large artifacts and independent verification remain remote.', '-y'],
                        ROOT / 'submit-formal.log', timeout=900)
                state['phase'] = 'formal_wait'
            if state['phase'] == 'formal_wait':
                observed = statuses(FULL_EXP, FULL_JOBS)
                state['formal_status'] = observed
                if any(x in {'failed', 'killed', 'canceled'} for x in observed.values()):
                    raise ValueError('A reference worker failed; preserve other job and inspect native logs')
                if all(x in {'pass', 'completed'} for x in observed.values()):
                    state['phase'] = 'formal_readback'
            if state['phase'] == 'formal_readback':
                successes = []
                for fold, job in enumerate(FULL_JOBS):
                    success, folder = collect(FULL_EXP, job, f'fold{fold}')
                    if (success['identity']['fold'] != fold or success['identity']['phase'] != 'full'
                            or success['expected_oof_rows'] != 10800 or success['training']['full']['steps'] != 1014):
                        raise ValueError('Formal reference scope mismatch')
                    successes.append(success)
                a, b = successes
                if (a['train_file_sha256'] != b['score_file_sha256'] or b['train_file_sha256'] != a['score_file_sha256']
                        or a['training']['full']['initial_state_sha256'] != b['training']['full']['initial_state_sha256']
                        or a['identity']['experiment_inputs_sha256'] != b['identity']['experiment_inputs_sha256']):
                    raise ValueError('Twofold directions/initialization/inputs differ')
                write(ROOT / 'twofold_acceptance.json', {'passed': True, 'rows': 21600,
                    'independent_remote_readback_per_fold': True, 'opposite_halves_exact': True,
                    'same_initialization': True, 'folds': successes, 'large_artifact_policy': 'remote_only'})
                state.update(phase='complete', acceptance=str(ROOT / 'twofold_acceptance.json'),
                             completion_scope='Two reference teachers and all21600 OOF scores accepted; no student training launched.')
            state.pop('transport_error', None)
        except TimeoutError as error:
            if state['phase'] == 'formal_submit':
                state.update(phase='needs_attention', reason='Formal submission outcome ambiguous; inspect before retry')
            else:
                state['transport_error'] = str(error)
        except Exception as error:
            state.update(phase='needs_attention', reason=type(error).__name__ + ': ' + str(error))
        state['observed_unix'] = time.time()
        write(state_path, state)
        print(json.dumps(state), flush=True)
        if args.once or state['phase'] in {'complete', 'needs_attention'}:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
