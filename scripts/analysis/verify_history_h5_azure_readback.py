"""Recompute acceptance from the archive downloaded onto GCR from Azure."""
from pathlib import Path, PurePosixPath
import tarfile

from new_interest.experiments.history_length_ablation import bind, read, write, require, sha256, subset_metrics
import pyarrow.parquet as pq

base = Path('/task/azure-readback')
receipt = read('/task/azure_publication_receipt.json')
require(sha256(base / 'publication.tar.gz') == receipt['files']['publication.tar.gz']['sha256'], 'Azure archive hash mismatch')
manifest = read(base / 'artifact_manifest.json')
allowed = set(manifest['files']) | {'complete.json','artifact_manifest.json'}
destination = base / 'extracted'
require(not destination.exists(), 'Use a fresh archive verification directory')
destination.mkdir()
with tarfile.open(base / 'publication.tar.gz','r:gz') as archive:
    for member in archive:
        name = PurePosixPath(member.name)
        require(not name.is_absolute() and '..' not in name.parts and name.parts[0] == 'publication', 'Unsafe archive path')
        require(member.isfile() or member.isdir(), 'Unexpected special archive member')
        if member.isfile():
            require(str(name.relative_to('publication')) in allowed, 'Unknown archive file')
        archive.extract(member, destination)
root = destination / 'publication'
for relative, item in manifest['files'].items():
    require(sha256(root / relative) == item['sha256'], 'Downloaded artifact hash mismatch')
baseline, contract = bind(root / 'runtime',5)
evaluation = baseline.audit_evaluation(root / 'runtime','2b',root,'full')
require(evaluation == read(root / 'evaluation_readback.json'), 'Archived metric recomputation differs')
require(read(root / 'complete.json')['artifact_manifest_sha256'] == sha256(root / 'artifact_manifest.json'), 'Complete/manifest mismatch')
cohort = set(read(root / 'runtime/data/training/mind_history_length_ablation_20260915_v1/dev_h50_cohort.json')['impression_ids'])
cohort_metrics = subset_metrics(pq.read_table(root / 'runtime' / baseline.DEV).to_pylist(),
    pq.read_table(root / 'real_eval/merged/id_token_predictions.parquet').to_pylist(),cohort)
require(cohort_metrics == read(root / 'cohort_metrics.json'), 'Archived cohort recomputation differs')
acceptance = {'status':'azure_downloaded_archive_full_predictions_and_metrics_recomputed',
    'archive_sha256':sha256(base / 'publication.tar.gz'), 'artifact_manifest_sha256':sha256(root / 'artifact_manifest.json'),
    'files_verified':len(manifest['files']), 'evaluation':evaluation,'cohort':cohort_metrics,
    'adapter':baseline.audit_adapter(root / 'runtime','2b',root / 'sft'),
    'reload':baseline.audit_reload(root / 'runtime','2b',root / 'sft')}
write('/task/azure_archive_acceptance.json',acceptance)
print('H5_AZURE_DOWNLOADED_ARCHIVE_INDEPENDENT_ACCEPTANCE_PASSED',flush=True)
