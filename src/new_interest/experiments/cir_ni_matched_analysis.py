"""Remote-only independent metrics and paired-user comparison for the two arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from new_interest.data.build_cir_dataset import sha256, write_json

ARMS = ("ni_prefix", "cir_length_matched")
CUTOFFS = (1, 5, 10, 20)


def compare(records, predictions, allowed_labels, *, draws=2000, seed=42):
    if set(predictions) != set(ARMS) or not records:
        raise ValueError("Exactly two nonempty matched arms required")
    ids = [str(r["impression_id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated reference impression IDs")
    users = sorted({str(r["user_id"]) for r in records})
    classes = sorted({c for r in records for c in r["acceptable_subcategories"]})
    if not set(classes) <= set(allowed_labels):
        raise ValueError("Ground truth outside frozen vocabulary")
    user_map, class_map = {u:i for i,u in enumerate(users)}, {c:i for i,c in enumerate(classes)}
    ui, ci = [], []
    ranks = {arm: [] for arm in ARMS}
    for arm in ARMS:
        rows = predictions[arm]
        if len(rows) != len(records):
            raise ValueError("Wrong prediction coverage")
        for record, prediction in zip(records, rows):
            labels = prediction["ranked_labels"]
            if str(prediction["impression_id"]) != str(record["impression_id"]):
                raise ValueError("Misaligned paired prediction")
            if len(labels) != 20 or len(set(labels)) != 20 or not set(labels) <= set(allowed_labels):
                raise ValueError("Illegal Top20 ranking")
    for i, record in enumerate(records):
        maps = {arm: {c:r for r,c in enumerate(predictions[arm][i]["ranked_labels"],1)} for arm in ARMS}
        for c in record["acceptable_subcategories"]:
            ui.append(user_map[str(record["user_id"])]); ci.append(class_map[c])
            for arm in ARMS:
                ranks[arm].append(maps[arm].get(c,21))
    ui, ci = np.array(ui), np.array(ci)
    support = np.bincount(ci,minlength=len(classes))
    values, metrics = {}, {}
    for arm in ARMS:
        rank = np.array(ranks[arm]); values[arm] = {}; metrics[arm] = {}
        for k in CUTOFFS:
            for kind in ("recall", "ndcg"):
                key = f"interest_macro_{kind}_at_{k}"
                score = (rank<=k).astype(float)
                if kind == "ndcg": score = score / np.log2(rank+1)
                values[arm][key] = score
                metrics[arm][key] = float(np.mean(np.bincount(ci,weights=score,minlength=len(classes))/support))
    keys = [f"interest_macro_{kind}_at_{k}" for k in (10,20) for kind in ("recall","ndcg")]
    delta = {k: values[ARMS[1]][k]-values[ARMS[0]][k] for k in keys}
    rng = np.random.default_rng(seed); samples=np.empty((draws,len(keys)))
    for d in range(draws):
        uw = rng.exponential(size=len(users))
        while (uw<=0).any(): uw[uw<=0]=rng.exponential(size=int((uw<=0).sum()))
        weights=uw[ui]; den=np.bincount(ci,weights=weights,minlength=len(classes))
        for j,k in enumerate(keys):
            samples[d,j]=np.mean(np.bincount(ci,weights=weights*delta[k],minlength=len(classes))/den)
    difference={k:{"difference_pp":100*(metrics[ARMS[1]][k]-metrics[ARMS[0]][k]),
                   "paired_user_bayesian_95_interval_pp":(100*np.quantile(samples[:,j],[.025,.975])).tolist()}
                for j,k in enumerate(keys)}
    return {"status":"passed", "records":len(records), "users":len(users),
            "target_occurrences":len(ci), "supported_classes":len(classes),
            "class_support":dict(zip(classes,map(int,support))), "metrics":metrics,
            "comparison":"cir_length_matched_minus_ni_prefix", "differences":difference,
            "uncertainty":{"method":"paired-user-Exp(1)-Bayesian-bootstrap", "draws":draws,"seed":seed,
                           "training_seed_uncertainty_included":False}}


def analyze(dev, mapping, roots, output):
    records=pq.read_table(dev).to_pylist()
    allowed=set(pq.read_table(mapping,columns=["target_subcategory"])["target_subcategory"].to_pylist())
    predictions, original_metrics, sources = {}, {}, {}
    for arm, root in roots.items():
        folder=root/"formal/real_eval/merged"
        m=json.loads((folder/"metrics.json").read_text())
        manifest=json.loads((folder/"merge_manifest.json").read_text())
        pred=folder/"id_token_predictions.parquet"
        if m["eval_sha256"]!=sha256(dev) or m["mapping_sha256"]!=sha256(mapping):
            raise ValueError("Evaluation input identity differs")
        if manifest["metrics_sha256"]!=sha256(folder/"metrics.json") or manifest["merged_prediction_sha256"]!=sha256(pred):
            raise ValueError("Committed evaluation hash differs")
        if m["label_vocabulary_size"]!=108 or m["history_prediction_filtering"] or m["top_k"]!=20:
            raise ValueError("Evaluation candidate protocol differs")
        predictions[arm]=pq.read_table(pred).to_pylist();original_metrics[arm]=m
        sources[arm]={"prediction_sha256":sha256(pred),"metrics_sha256":sha256(folder/"metrics.json"),
                      "evaluation_identity":m["evaluation_identity"]}
    result=compare(records,predictions,allowed)
    if tuple(result[k] for k in ("records","users","target_occurrences","supported_classes"))!=(31392,27082,37728,104):
        raise ValueError("Frozen Full Dev scope differs")
    for arm in ARMS:
        for key,value in result["metrics"][arm].items():
            if abs(value-original_metrics[arm][key])>1e-12:
                raise ValueError("Independent recomputation differs")
    result["sources"]=sources;result["analysis_source_sha256"]=sha256(Path(__file__))
    write_json(output,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dev","mapping","ni-prefix-root","cir-length-matched-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(analyze(a.dev,a.mapping,{"ni_prefix":a.ni_prefix_root,
                     "cir_length_matched":a.cir_length_matched_root},a.output),indent=2))
