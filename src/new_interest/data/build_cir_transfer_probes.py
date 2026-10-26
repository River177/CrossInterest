"""Freeze non-training SID mapping and held-out category reconstruction probes."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import pyarrow as pa
import pyarrow.parquet as pq
from new_interest.infrastructure.paths import portable_path
from new_interest.data.build_cir_dataset import PROMPT, history_from_record, render_history, select_category, sha256, write_json

VERSION = 'cir-parameter-transfer-probes-v1'


def reconstruction_view(row, mapping):
    body = history_from_record(row)
    history = []
    for line in body.splitlines():
        match = re.fullmatch(r'\d+\. \[([^\]]+)\] (.+)', line)
        if not match:
            raise ValueError('Invalid frozen history serialization')
        history.append({'subcategory': match[1], 'title': match[2]})
    unit = json.dumps([row['user_id'], hashlib.sha256(body.encode()).hexdigest()], separators=(',', ':'))
    target, _ = select_category(history, set(mapping), unit, 42)
    if target is None:
        return None
    retained = [x for x in history if x['subcategory'] != target]
    return {'impression_id': 'cir-probe:' + str(row['impression_id']), 'user_id': row['user_id'],
            'prompt': PROMPT.format(history=render_history(retained)), 'history_length': len(retained),
            'history_subcategories': [x['subcategory'] for x in retained],
            'acceptable_subcategories': [target], 'acceptable_responses': [mapping[target]],
            'source_impression_id': str(row['impression_id']),
            'source_history_sha256': hashlib.sha256(body.encode()).hexdigest()}


def build(ni_file: Path, dev_file: Path, mapping_file: Path, output: Path, *, maximum=512):
    if output.exists():
        raise FileExistsError(output)
    mapping = {x['target_subcategory']: x['interest_token'] for x in pq.read_table(mapping_file).to_pylist()}
    if len(mapping) != 108:
        raise ValueError('Expected fixed 108 SID mapping')
    train_users = set(pq.read_table(ni_file, columns=['user_id'])['user_id'].to_pylist())
    dev = pq.read_table(dev_file).to_pylist()
    if train_users.intersection(x['user_id'] for x in dev):
        raise ValueError('Probe users overlap training users')
    unique = {}
    for row in sorted(dev, key=lambda x: (str(x['user_id']), str(x['impression_id']))):
        if row['user_id'] not in unique:
            unique[row['user_id']] = row
    eligible = [x for row in unique.values() if (x := reconstruction_view(row, mapping)) is not None]
    eligible.sort(key=lambda x: hashlib.sha256((VERSION + ':42:' + str(x['user_id'])).encode()).digest())
    selected = eligible[:maximum]
    if len(selected) != maximum:
        raise ValueError('Too few eligible independent probe users')
    names = [{'impression_id': 'sid-name:' + token, 'user_id': 'category:' + token,
              'prompt': f'Topic category: {label}\n\nRespond with only the learned interest token assigned to this topic category, in the exact format <interest_NNN>. Do not output any explanation.',
              'history_length': 0, 'history_subcategories': [], 'acceptable_subcategories': [label],
              'acceptable_responses': [token]} for label, token in sorted(mapping.items(), key=lambda x: x[1])]
    output.mkdir(parents=True)
    for name, rows in [('sid_name', names), ('heldout_cir', selected)]:
        pq.write_table(pa.Table.from_pylist(rows), output/f'{name}.parquet', compression='zstd')
    report = {'version': VERSION, 'seed': 42, 'training_use': False, 'future_targets_used': False,
              'train_user_overlap': 0, 'unique_dev_users': len(unique), 'eligible_reconstruction_users': len(eligible),
              'reconstruction_users': len(selected), 'reconstruction_class_counts': dict(sorted(Counter(x['acceptable_subcategories'][0] for x in selected).items())),
              'sid_name_rows': 108, 'class_balancing': False, 'source_sha256': {portable_path(p): sha256(p) for p in [ni_file, dev_file, mapping_file]},
              'files': {p.name: sha256(p) for p in output.glob('*.parquet')},
              'selection': 'one deterministic visible history per independent Dev user; category selection uses only history; fixed hash selection of 512 eligible users'}
    write_json(output/'metadata.json', report)
    return report
