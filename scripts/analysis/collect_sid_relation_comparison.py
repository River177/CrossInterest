"""Paired uncertainty and prespecified screening from independently accepted scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


METRICS=['macro_recall_at_10','macro_recall_at_20','macro_ndcg_at_10','macro_ndcg_at_20']


def contributions(rows,names):
    index={c:i for i,c in enumerate(names)};users={u:i for i,u in enumerate(sorted({r['user_id'] for r in rows}))}
    classes=[];who=[];values=[];record=[]
    for j,r in enumerate(rows):
        positions={c:k+1 for k,c in enumerate(r['ranking'])}
        for c in r['targets']:
            rank=positions[index[c]];classes.append(index[c]);who.append(users[r['user_id']]);record.append(j)
            values.append([rank<=10,rank<=20,1/np.log2(rank+1) if rank<=10 else 0,
                           1/np.log2(rank+1) if rank<=20 else 0])
    return np.array(classes),np.array(who),np.asarray(values,dtype=float),np.array(record)


def compare(a,b,names,stratified,parents=None,seed=20260915):
    if len(a)!=len(b):raise ValueError('Paired cohorts differ')
    for left,right in zip(a,b):
        if any(left[k]!=right[k] for k in ['record_index','user_id','targets','input_sha256']):
            raise ValueError('Paired identity differs')
    classes,users,av,row_index=contributions(a,names);_,_,bv,_=contributions(b,names)
    delta=bv-av;counts=np.bincount(classes,minlength=len(names))
    selected=np.flatnonzero(counts>=20 if stratified else counts>0)
    point=np.mean([delta[classes==c].mean(0) for c in selected],axis=0)
    rng=np.random.default_rng(seed);replicates=[];coverage=[]
    if stratified:
        if len(set(users))!=len(users):raise ValueError('Validation stratification assumes one target per user')
        arrays=[delta[classes==c] for c in selected]
        if len({len(x) for x in arrays})!=1:raise ValueError('Unexpected unequal supported validation strata')
        data=np.stack(arrays);n=data.shape[1]
        for _ in range(20):
            indices=rng.integers(0,n,size=(100,len(selected),n))
            values=data[np.arange(len(selected))[None,:,None],indices]
            replicates.extend(values.mean(2).mean(1))
        coverage=[len(selected)]*2000
        policy='within-class user resampling; fixed107 classes'
    else:
        n=int(users.max())+1
        for _ in range(2000):
            weight=np.bincount(rng.integers(0,n,size=n),minlength=n)[users]
            den=np.bincount(classes,weights=weight,minlength=len(names))
            supported=selected[den[selected]>0];coverage.append(len(supported))
            means=[]
            for m in range(4):
                sums=np.bincount(classes,weights=weight*delta[:,m],minlength=len(names))
                means.append(np.mean(sums[supported]/den[supported]))
            replicates.append(means)
        policy='whole-user resampling; classes absent in a replicate omitted symmetrically for both models'
    intervals=np.quantile(np.asarray(replicates),[.025,.975],axis=0)
    slices={}
    histories=np.asarray([len(a[i]['history_subcategories']) for i in row_index])
    distinct=np.asarray([len(set(a[i]['history_subcategories'])) for i in row_index])
    definitions={'history_1':histories==1,'history_2_4':(histories>=2)&(histories<=4),
        'history_5_20':(histories>=5)&(histories<=20),'history_21_50':histories>=21,
        'unique_1':distinct==1,'unique_2_4':(distinct>=2)&(distinct<=4),'unique_5plus':distinct>=5}
    definitions['has_history_only_class']=np.array([any(c not in names for c in a[i]['history_subcategories']) for i in row_index])
    definitions['no_history_only_class']=~definitions['has_history_only_class']
    if parents is not None:
        same=np.array([parents[names[c]] in {parents[h] for h in a[i]['history_subcategories']}
                       for c,i in zip(classes,row_index)])
        definitions['target_parent_seen']=same;definitions['target_parent_unseen']=~same
    for key,mask in definitions.items():
        if not mask.any():continue
        cc=np.bincount(classes[mask],minlength=len(names));supported=np.flatnonzero(cc>=5)
        macro=np.mean([delta[mask&(classes==c)].mean(0) for c in supported],axis=0) if len(supported) else None
        slices[key]={'target_occurrences':int(mask.sum()),'macro_classes_min5':len(supported),
            'target_occurrence_delta':dict(zip([k.removeprefix('macro_') for k in METRICS],delta[mask].mean(0).tolist())),
            'macro_delta':dict(zip(METRICS,macro.tolist())) if macro is not None else None}
    return {'paired_delta':dict(zip(METRICS,point.tolist())),
        'ci95':{k:[float(intervals[0,j]),float(intervals[1,j])] for j,k in enumerate(METRICS)},
        'bootstrap_repeats':2000,'bootstrap_seed':seed,'bootstrap_unit':'user','bootstrap_policy':policy,
        'bootstrap_supported_class_range':[min(coverage),max(coverage)],'slices':slices}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    state=json.loads((root/'pipeline_state.json').read_text());project=Path(state['formal_bundle'])
    names=[r['target_subcategory'] for r in sorted(pq.read_table(project/'inputs/interest_token_mapping.parquet').to_pylist(),key=lambda r:r['interest_index'])]
    baseline=Path(state['baseline_worker_root']);relation=Path(state['relation_worker_root'])
    parents=json.loads((root/'taxonomy_parent_map.json').read_text())['parents']
    result={}
    for split in ['validation','dev']:
        b=pq.read_table(baseline/f'evaluation/{split}/merged/predictions.parquet').to_pylist()
        r=pq.read_table(relation/f'evaluation/{split}/merged/predictions.parquet').to_pylist()
        f=pq.read_table(baseline/f'frozen_reference/{split}/merged/predictions.parquet').to_pylist()
        result[split]={'relation_vs_continuation':compare(b,r,names,split=='validation',parents),
                       'relation_vs_frozen':compare(f,r,names,split=='validation',parents)}
    v=result['validation']['relation_vs_continuation'];f=result['validation']['relation_vs_frozen']
    tests={'r10_gain_at_least_half_pp':v['paired_delta'][METRICS[0]]>=.005,
        'r10_ci_lower_positive':v['ci95'][METRICS[0]][0]>0,
        'ndcg10_noninferiority_point':v['paired_delta'][METRICS[2]]>=-.001,
        'r10_not_below_frozen':f['paired_delta'][METRICS[0]]>=0}
    result['next_stage_screen']={'criteria':tests,'passed':all(tests.values()),'automatic_extra_training':False}
    result['interpretation']='Fixed-parent single-seed continuation pilot; bootstrap is sample uncertainty, not seed variability; official Dev is exploratory.'
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['next_stage_screen'],indent=2))


if __name__=='__main__':main()
