"""Freeze complete-data epochs shared by Mixed and exact OI/NI meta SFT."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


def epoch_batches(rows, seed, epoch, batch_size=32):
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    # Same-user OI/NI examples cannot be support and query in one episode.
    # Swap only within a role, preserving the full permutation and all exposure.
    for bi, batch in enumerate(batches):
        for j, index in enumerate(batch):
            others = [rows[k] for k in batch[:j]]
            conflict = any(r['user_id'] == rows[index]['user_id'] and
                           r['training_task'] != rows[index]['training_task'] for r in others)
            if not conflict:
                continue
            found = False
            for bj in range(bi + 1, len(batches)):
                for q, candidate in enumerate(batches[bj]):
                    if rows[candidate]['training_task'] != rows[index]['training_task']:
                        continue
                    if any(rows[k]['user_id'] == rows[candidate]['user_id'] and
                           rows[k]['training_task'] != rows[candidate]['training_task'] for k in batch):
                        continue
                    if any(rows[k]['user_id'] == rows[index]['user_id'] and
                           rows[k]['training_task'] != rows[index]['training_task'] for k in batches[bj]):
                        continue
                    batch[j], batches[bj][q] = candidate, index
                    found = True
                    break
                if found:
                    break
            if not found:
                raise ValueError('Could not resolve support/query user collision without repeating data')
    validate_epoch(rows, batches)
    return batches


def validate_epoch(rows, batches):
    if sorted(i for batch in batches for i in batch) != list(range(len(rows))):
        raise ValueError('Epoch must contain each row exactly once')
    for batch in batches:
        oi = [rows[i] for i in batch if rows[i]['training_task'] == 'old_interest']
        ni = [rows[i] for i in batch if rows[i]['training_task'] == 'new_interest']
        if not oi or not ni or len(oi) + len(ni) != len(batch):
            raise ValueError('Every batch requires both OI and NI')
        if {r['user_id'] for r in oi} & {r['user_id'] for r in ni}:
            raise ValueError('Support and query users overlap')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = pq.read_table(args.mixed).to_pylist()
    if Counter(r['training_task'] for r in rows) != {'old_interest': 17098, 'new_interest': 21600}:
        raise ValueError('Original full Mixed dataset counts differ')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample identity')
    episodes = []
    for epoch in range(3):
        for batch in epoch_batches(rows, 42, epoch):
            episodes.append({'step': len(episodes) + 1, 'epoch': epoch,
                'row_indices': batch,
                'support_indices': [i for i in batch if rows[i]['training_task'] == 'old_interest'],
                'query_indices': [i for i in batch if rows[i]['training_task'] == 'new_interest']})
    if len(episodes) != 3630:
        raise ValueError('Expected three complete epochs and 3630 updates')
    stage = args.output.with_name(args.output.name + '.staging')
    stage.mkdir(parents=True, exist_ok=False)
    for source, name in ((args.mixed, 'mixed.parquet'), (args.mapping, 'interest_token_mapping.parquet'),
                         (args.dev, 'dev_full_grouped.parquet')):
        shutil.copyfile(source, stage / name)
    pq.write_table(pa.Table.from_pylist(episodes), stage / 'episodes.parquet')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir() if p.is_file()}
    metadata = {'version': 'oi-ni-full-sft-data-v1', 'files': files, 'rows': len(rows),
        'epochs': 3, 'steps': len(episodes), 'seed': 42, 'global_batch_size': 32,
        'role_rows': dict(Counter(r['training_task'] for r in rows)),
        'role_presentations': {'old_interest': 51294, 'new_interest': 64800},
        'ordering': 'Python Random(seed+epoch) shuffle; deterministic same-role swap for cross-role user collision',
        'strict_class_matched_sampling': False, 'sample_replacement': False,
        'support_query_users_disjoint': True, 'same_order_for_both_modes': True,
        'additional_ni_fast_forward_is_meta_compute_not_an_extra_dataset_epoch': True}
    (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    stage.rename(args.output)
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
