"""Select the largest same-user cohort with equal CIR and NI label histograms.

Requires networkx for the integral minimum-cost circulation solver. No labels,
histories or prompts are changed. Cohort selection uses both training labels;
it does not use held-out labels or model performance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256, write_json
from new_interest.data.build_cir_ni_direct_repeated import select_pairs
from new_interest.infrastructure.paths import portable_path

VERSION = "mind-cir-ni-direct-balanced-20260929-v1"


def balanced_indices(pairs):
    import networkx as nx

    groups = defaultdict(list)
    seen = set()
    for i,p in enumerate(pairs):
        if p["source_sample_id"] in seen:
            raise ValueError("Duplicate pair source")
        seen.add(p["source_sample_id"])
        groups[(p["cir_target"],p["ni_target"])].append(i)
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted({label for edge in groups for label in edge}), demand=0)
    for (cir,ni),indices in sorted(groups.items()):
        graph.add_edge(cir,ni,capacity=len(indices),weight=-1)
    # Each retained user contributes one CIR label and one NI label. Flow
    # conservation makes their label counts equal; cost -1 maximizes users.
    cost,flow = nx.network_simplex(graph)
    selected = []
    for (cir,ni),indices in sorted(groups.items()):
        ordered = sorted(indices,key=lambda i:(hashlib.sha256(
            (VERSION+":"+pairs[i]["source_sample_id"]).encode()).digest(),pairs[i]["source_sample_id"]))
        selected.extend(ordered[:flow[cir][ni]])
    selected.sort()
    if not selected:
        raise ValueError("No nonempty equal-label same-user cohort")
    counts = Counter(pairs[i]["cir_target"] for i in selected)
    if counts != Counter(pairs[i]["ni_target"] for i in selected) or len(selected) != -cost:
        raise ValueError("Label balance or objective verification failed")
    return selected, {"solver":"networkx.network_simplex", "networkx_version":nx.__version__,
        "objective":"maximum retained distinct source pairs with per-label flow conservation",
        "optimal_retained_pairs":len(selected),"class_counts":dict(sorted(counts.items()))}


def build(source: Path, output: Path):
    if output.exists():raise FileExistsError(output)
    meta=json.loads((source/"metadata.json").read_text())
    for name,digest in meta["files"].items():
        if sha256(source/name)!=digest:raise ValueError("Source hash differs: "+name)
    tables={n:pq.read_table(source/f"{n}.parquet") for n in ("cir_as_ni","original_ni_paired","pairs")}
    rows={n:t.to_pylist() for n,t in tables.items()}
    repeated,_=select_pairs(rows['cir_as_ni'],rows['original_ni_paired'],rows['pairs'])
    if len(repeated)!=len(rows['pairs']):raise ValueError("Source includes singleton CIR targets")
    indices,solver=balanced_indices(rows['pairs']);chosen=set(indices)
    pairs=[{**rows['pairs'][i],"repeated_pair_index":i,"pair_index":j} for j,i in enumerate(indices)]
    if len({p['user_id'] for p in pairs})!=len(pairs):raise ValueError("User identity not unique")
    allowed=set(pq.read_table(source/'interest_token_mapping.parquet',columns=['target_subcategory']).column(0).to_pylist())
    stats={"version":VERSION,"status":"candidate_data_only","source_pairs":len(rows['pairs']),
        "paired_rows":len(pairs),"paired_users":len(pairs),"training_classes":len(solver['class_counts']),
        "class_counts_each_arm":solver['class_counts'],"missing_training_classes_both":sorted(allowed-set(solver['class_counts'])),
        "same_users":True,"same_source_order":True,"same_label_histogram":True,
        "targets_relabelled":False,"oversampling":False,"CIR_target_frequency_minimum":2,
        "selection_uses_both_training_labels":True,"selection_uses_heldout_or_performance":False,
        "training_submitted":False,"solver":solver}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cir-ni-balanced-',dir=output.parent) as tmp:
        staging=Path(tmp)/'data';staging.mkdir()
        for name in ('cir_as_ni','original_ni_paired'):
            table=tables[name].take(pa.array(indices));pq.write_table(table,staging/f'{name}.parquet',compression='zstd')
            if not pq.read_table(staging/f'{name}.parquet').equals(table):raise ValueError('Training readback mismatch')
        for name,records in [('pairs',pairs),('selection_manifest',[{'source_sample_id':p['source_sample_id'],
            'repeated_pair_index':i,'kept':i in chosen} for i,p in enumerate(rows['pairs'])])]:
            pq.write_table(pa.Table.from_pylist(records),staging/f'{name}.parquet',compression='zstd')
            if pq.read_table(staging/f'{name}.parquet').to_pylist()!=records:raise ValueError('Sidecar readback mismatch')
        for name in ('dev_full_grouped.parquet','interest_token_mapping.parquet'):
            shutil.copyfile(source/name,staging/name)
            if sha256(source/name)!=sha256(staging/name):raise ValueError('Eval companion mismatch')
        write_json(staging/'statistics.json',stats)
        write_json(staging/'metadata.json',{'version':VERSION,'prompt_template':meta['prompt_template'],
            'parent_dataset':portable_path(source),'builder_sha256':sha256(Path(__file__)),
            'source_files':[{'path':portable_path(source/n),'sha256':sha256(source/n)} for n in [*meta['files'],'metadata.json']],
            'files':{p.name:sha256(p) for p in staging.iterdir() if p.is_file()},'scope':'candidate_data_only_no_training'})
        staging.rename(output)
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/training/mind_cir_ni_direct_repeated_20260929_v1'))
    p.add_argument('--output',type=Path,default=Path('data/training/mind_cir_ni_direct_balanced_20260929_v1'))
    a=p.parse_args();print(json.dumps(build(a.source,a.output),ensure_ascii=False,indent=2))
