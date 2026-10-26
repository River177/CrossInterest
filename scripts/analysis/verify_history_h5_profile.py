"""Verify the original H5 reference in a fresh process using a bound FLA profile."""
import argparse
import json
from pathlib import Path

from new_interest.training.fla_norm_profile import apply_fla_norm_profile
from new_interest.training.verify_semantic_id_token_adapter import verify_semantic_id_token_adapter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--profile', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
receipt = apply_fla_norm_profile(args.profile, adapter_path=args.source / 'adapter',
                                 reference_file=args.source / 'reference.json')
(args.output / 'profile_applied.json').write_text(json.dumps(receipt, indent=2) + '\n')
verify_semantic_id_token_adapter('Qwen/Qwen3.5-2B', args.source / 'adapter',
    args.source / 'adapter/interest_token_mapping.parquet', args.source / 'reference.json',
    args.output / 'reload_audit.json', model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc', atol=0, rtol=0)
print('H5_ORIGINAL_REFERENCE_PROFILE_VERIFIED', flush=True)
