"""Finish the authorized checkpoint-only repair, with compact local receipts."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from run_rho_twofold_reference_pipeline import AMLT, REPO, command, statuses

ROOT = REPO / 'artifacts/rho-twofold-reference-20260915'
DIAGNOSTIC = 'mind-rho-fold1-reload-diagnostic-v1-0916'
DIAGNOSTIC_JOB = 'qwen35-2b-rho-fold1-reload-diagnostic'
RESCORE = 'mind-rho-twofold-stable-rescore-v1-0916'
JOBS = ['qwen35-2b-rho-stable-score-fold0', 'qwen35-2b-rho-stable-score-fold1']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def compact(exp, job, label, names):
    if any(any(c in x for c in '/\\*?[]') or not x.endswith('.json') for x in names):
        raise ValueError('Only exact compact JSON basenames are allowed')
    dest = ROOT / 'compact' / label
    dest.mkdir(parents=True, exist_ok=True)
    include = [x for name in names for x in ('--include', name)]
    try:
        command([AMLT, 'results', 'download', exp, ':' + job, *include,
                 '--output', str(dest), '--n-workers', '4'], ROOT / f'collect-{label}.log', timeout=300)
    except (RuntimeError, TimeoutError) as error:
        raise TimeoutError('Compact report transport: ' + str(error)) from error
    found = {}
    for name in names:
        paths = list(dest.rglob(name))
        if len(paths) != 1:
            raise ValueError('Missing/ambiguous compact artifact: ' + name)
        found[name] = paths[0]
    return found


def check_diagnostic(report, expected_contract_sha):
    if (report.get('contract_sha256') != expected_contract_sha or report.get('model_updates') != 0
            or report.get('parent_endpoint', {}).get('global_step') != 1014
            or report.get('original_training_reference_equivalence_established') is not False):
        raise ValueError('Diagnostic identity or original-reference boundary differs')
    for key in ('diagnostic_completed', 'read_only_integrity_passed', 'fixed_controls_passed'):
        if report.get(key) is not True:
            raise ValueError('Diagnostic control failed: ' + key)
    if report.get('fixed_cross_condition_max_abs') != 0:
        raise ValueError('Fixed kernel conditions were not exactly equal')


def check_fold(files, fold, contract, expected_runtime, expected_contract_sha):
    success = read(files['worker_success.json'])
    ref = read(files['reference_acceptance.json'])
    probe = read(files['fresh_probe_acceptance.json'])
    checkpoint = read(files['checkpoint_verification.json'])
    parent = contract['parents'][str(fold)]
    opposite = contract['parents'][str(1-fold)]
    if (not success.get('passed') or success.get('fold') != fold or success.get('scored_rows') != 10800
            or success.get('new_training_updates') != 0 or success.get('parent_files_mutated') is not False
            or success.get('historical_training_reload_parity_claimed') is not False
            or success['identity']['fold'] != fold
            or success['identity']['runtime_sha256'] != expected_runtime
            or success['identity']['contract_sha256'] != expected_contract_sha
            or success['original_training_runtime_sha256'] != contract['original_training_runtime_sha256']
            or success['original_observed_initial_state_sha256'] != contract['expected_initial_state_sha256']):
        raise ValueError('Rescore worker identity/scope differs')
    if (success['parent_marker_sha256'] != parent['checkpoint_marker_sha256']
            or success['parent_adapter_sha256'] != parent['adapter_sha256']):
        raise ValueError('Rescore did not use the exact completed weights')
    for name, field in [('reference_acceptance.json', 'reference_acceptance_sha256'),
                        ('fresh_probe_acceptance.json', 'fresh_probe_acceptance_sha256')]:
        if sha(files[name]) != success[field]:
            raise ValueError('Independent acceptance hash differs: ' + name)
    if (not ref.get('accepted') or ref['metrics']['rows'] != 10800
            or ref['metrics']['users'] != 10800 or ref['teacher_score_user_overlap'] != 0
            or len(ref['metrics']['class_support']) != 108
            or set(ref['metrics']['class_support'].values()) != {100}
            or ref['teacher_train_sha256'] != parent['checkpoint_identity']['train_file_sha256']
            or ref['score_file_sha256'] != opposite['checkpoint_identity']['train_file_sha256']
            or ref.get('kernel_selection') != 'stable_fla_v1'):
        raise ValueError('Incomplete or inconsistent opposite-fold scores')
    if (not probe.get('accepted') or not probe.get('native_and_fp32_exact')
            or not probe.get('production_batch4_fp32_exact') or probe['model_updates'] != 0
            or probe.get('independent_processes') != 2 or probe.get('probe_rows') != 4
            or probe.get('overlapping_kernel_cache_keys_consistent') is not True
            or probe.get('historical_training_parity_claimed') is not False):
        raise ValueError('Actual scoring/reload numerical probes failed')
    if (checkpoint.get('all_parent_file_hashes_verified') is not True
            or checkpoint['new_training_updates'] != 0 or checkpoint['optimizer_state_entries'] != 373
            or checkpoint['parent_marker_sha256'] != parent['checkpoint_marker_sha256']
            or checkpoint['identity'] != parent['checkpoint_identity']
            or checkpoint['adapter_sha256'] != parent['adapter_sha256']
            or checkpoint['optimizer_step_min'] != 1014 or checkpoint['optimizer_step_max'] != 1014
            or checkpoint['scheduler_last_epoch'] != 1014):
        raise ValueError('Checkpoint bytes/cursors not accepted')
    choices = ref['selected_kernel_configs']
    if not choices or canonical(choices) != ref['selected_kernel_configs_sha256']:
        raise ValueError('Selected kernel configuration digest differs')
    for key, expected in success['identity']['kernel'].items():
        if ref.get(key) != expected:
            raise ValueError('Scoring kernel identity differs from worker')
    return {'worker': success, 'reference': ref, 'probe': probe,
            'compact_files': {name: {'path': str(path), 'sha256': sha(path)} for name, path in files.items()}}


def combine(folds, diagnostic):
    common = {}
    for fold in folds:
        for item in fold['reference']['selected_kernel_configs']:
            key = (item['kernel'], item['cache_key'])
            if key in common and common[key] != item['config']:
                raise ValueError('Same kernel key chose different configurations across folds')
            common[key] = item['config']
    if folds[0]['reference']['teacher_train_sha256'] != folds[1]['reference']['score_file_sha256']:
        raise ValueError('Fold0 direction differs')
    if folds[1]['reference']['teacher_train_sha256'] != folds[0]['reference']['score_file_sha256']:
        raise ValueError('Fold1 direction differs')
    for key in ('kernel_selection', 'kernel_selection_protocol', 'kernel_selector_sha256', 'loss_definition', 'mapping_sha256'):
        if folds[0]['reference'][key] != folds[1]['reference'][key]:
            raise ValueError('Two folds use different scoring definitions: ' + key)
    return {'passed': True, 'rows': 21600, 'users': 21600, 'folds': folds,
        'twofold_union_exact_from_verified_complementary_inputs': True,
        'kernel_selection': 'stable_fla_v1', 'common_kernel_configs_consistent': True,
        'kernel_cache_keys_checked': len(common), 'new_training_updates': 0,
        'original_fold0_native_scores_preserved': True,
        'original_fold1_training_reference_equivalence_established': False,
        'diagnostic': diagnostic,
        'verification_scope': 'Both completed checkpoints byte-verified remotely; two independent fresh native/FP32 and real-batch4 probes per fold exact; all21600 OOF scores independently reread/recomputed remotely; common kernel choices cross-checked locally from compact metadata.',
        'large_artifact_policy': 'remote_only', 'student_training_started': False}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'repair.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Repair finisher already active')
        return
    state_path = ROOT / 'repair_state.json'
    state = read(state_path)
    write(ROOT / 'repair_process.json', {'pid': os.getpid(), 'started_unix': time.time(), 'script': str(Path(__file__))})
    while state['phase'] not in ('complete', 'needs_attention'):
        try:
            if state['phase'] == 'diagnostic_submitted':
                observed = statuses(DIAGNOSTIC, [DIAGNOSTIC_JOB])[DIAGNOSTIC_JOB]
                state['diagnostic_status'] = observed
                if observed in ('failed', 'killed', 'canceled'):
                    raise ValueError('Diagnostic failed; inspect preserved reports')
                if observed in ('pass', 'completed'):
                    state['phase'] = 'diagnostic_readback'
            if state['phase'] == 'diagnostic_readback':
                files = compact(DIAGNOSTIC, DIAGNOSTIC_JOB, 'reload-diagnostic',
                                ['diagnostic_summary.json', 'environment.json', 'parent_validation.json'])
                report = read(files['diagnostic_summary.json'])
                check_diagnostic(report, sha(ROOT / 'bundle-reload-diagnostic-v1/diagnostic_contract.json'))
                state.update(diagnostic_report=str(files['diagnostic_summary.json']), phase='rescore_prepare')
            if state['phase'] == 'rescore_prepare':
                ready = ROOT / 'rescore_bundle_ready.json'
                if ready.exists():
                    prepared = read(ready)
                    if sha(Path(prepared['bundle']) / 'source_manifest.json') != prepared['source_manifest_sha256']:
                        raise ValueError('Frozen rescore bundle changed')
                    state.update(rescore_config=prepared['config'], phase='rescore_submit')
                    write(state_path, state)
            if state['phase'] == 'rescore_submit':
                if state.get('rescoring_submission_attempted'):
                    raise ValueError('Prior submission outcome ambiguous; never duplicate')
                state['rescoring_submission_attempted'] = True
                write(state_path, state)
                command([AMLT, 'run', state['rescore_config'], RESCORE, '-d',
                    'Complete user-authorized twofold scoring from immutable1014-step checkpoints. No training updates. Both folds use identical stable FLA choices, exact independent reload/batch4 probes and remote full OOF readback.', '-y'],
                    ROOT / 'submit-stable-rescore.log', timeout=900)
                state['phase'] = 'rescore_wait'
            if state['phase'] == 'rescore_wait':
                observed = statuses(RESCORE, JOBS)
                state['rescore_status'] = observed
                if any(x in ('failed', 'killed', 'canceled') for x in observed.values()):
                    raise ValueError('A rescore worker failed; preserve successful peer and inspect')
                if all(x in ('pass', 'completed') for x in observed.values()):
                    state['phase'] = 'rescore_readback'
            if state['phase'] == 'rescore_readback':
                prepared = read(ROOT / 'rescore_bundle_ready.json')
                bundle = Path(prepared['bundle'])
                source = read(bundle / 'source_manifest.json')
                contract = read(bundle / 'rescore_contract.json')
                accepted = []
                for fold, job in enumerate(JOBS):
                    files = compact(RESCORE, job, f'stable-fold{fold}', ['worker_success.json',
                        'reference_acceptance.json', 'fresh_probe_acceptance.json', 'checkpoint_verification.json', 'environment.json'])
                    accepted.append(check_fold(files, fold, contract, source['runtime_sha256'], sha(bundle / 'rescore_contract.json')))
                total = combine(accepted, read(state['diagnostic_report']))
                write(ROOT / 'twofold_acceptance.json', total)
                state.update(phase='complete', acceptance=str(ROOT / 'twofold_acceptance.json'), scored_rows=21600)
                original = read(ROOT / 'pipeline_state.json')
                original.pop('reason', None)
                original.pop('transport_error', None)
                original.update(phase='complete', recovery_acceptance=state['acceptance'], scored_rows=21600,
                    completion_scope='Both immutable reference models scored opposite halves under stable FLA, independent remote acceptance complete; historical originalfold1 reload failure retained.')
                write(ROOT / 'pipeline_state.json', original)
            state.pop('transport_error', None)
        except TimeoutError as error:
            if state['phase'] == 'rescore_submit':
                state.update(phase='needs_attention', reason='Rescore submission outcome ambiguous; inspect exact experiment')
            else:
                state['transport_error'] = str(error)
        except Exception as error:
            state.update(phase='needs_attention', reason=type(error).__name__ + ': ' + str(error))
        state['observed_unix'] = time.time()
        write(state_path, state)
        print(json.dumps(state), flush=True)
        if args.once or state['phase'] in ('complete', 'needs_attention'):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
