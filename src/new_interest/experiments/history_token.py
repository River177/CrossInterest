"""Two cold NI conditions on a shared 125-token vocabulary and 108 candidates.

The frozen bundle supplies the accepted CIR numerical runtime. Only the data,
vocabulary boundary, and corresponding shape/identity checks are specialized.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pyarrow.parquet as pq

from new_interest.experiments import cir_warmup as core


def schedule(project):
    value = core.read(project / core.CONTRACT)["direct_training"]
    core.require(value == {"samples":21600,"epochs":3,"optimizer_steps":2025,
                           "presentations":64800,"parent":None}, "NI budget changed")
    return value


def stage_specs(project, root, *, gates):
    schedule(project)
    entries = [("single_ni",32,2,1,1,8), ("ddp_ni",41,1,4,2,32)] if gates else [("ni",21600,3,4,2,32)]
    return [{"key":key,"task":"ni","parent":None,"samples":n,"epochs":epochs,
             "world_size":world,"per_device_train_batch_size":micro,
             "global_batch_size":batch,"accumulation":batch,
             "train_file":root/"gate_data"/f"ni-{n}.parquet" if gates else project/core.DATA/"ni.parquet"}
            for key,n,epochs,world,micro,batch in entries]


def representative_messages(train_file, expected_tokens):
    """Real rows cover all supervised outputs and every history token in the arm."""
    rows = pq.read_table(train_file, columns=["response","messages","prompt"]).to_pylist()
    ordered = sorted(enumerate(rows), key=lambda item:(len(item[1]["prompt"]),item[0]))
    targets = {row["response"] for row in rows}
    core.require(targets <= expected_tokens and len(targets)==108, "Gradient target coverage differs")
    input_tokens = {t for row in rows for t in re.findall(r"<interest_\d{3}>",row["prompt"])}
    core.require(not input_tokens or input_tokens == expected_tokens, "Partial history tokenization")
    remaining_targets, remaining_inputs = set(targets), set(input_tokens)
    selected=[]
    for _,row in ordered:
        seen=set(re.findall(r"<interest_\d{3}>",row["prompt"]))
        if row["response"] in remaining_targets or seen & remaining_inputs:
            selected.append((row["response"],row["messages"]))
            remaining_targets.discard(row["response"])
            remaining_inputs.difference_update(seen)
        if not remaining_targets and not remaining_inputs:
            break
    core.require(not remaining_targets and not remaining_inputs, "Incomplete gradient representatives")
    return selected


def gradient(project, output):
    core.reference_kernels()
    from new_interest.training import audit_semantic_id_token_gradients as audit
    audit._representative_messages = representative_messages
    config=core.read(project/"configs/sft/mind_cir_base_v1.json")
    result=audit.audit_semantic_id_token_gradients(config,output,batch_size=1)
    core.require(result["interest_token_count"]==125 and result["all_token_rows_finite_and_nonzero"], "125-row gradient gate failed")
    print("HISTORY_TOKEN_GRADIENT_ACCEPTED", flush=True)


def worker(project, root):
    core.stage_specs=stage_specs
    core.formal_schedule=schedule
    # The first cold smoke establishes this expanded-vocabulary initialization.
    # Every subsequent gate and formal stage must reproduce all trainable tensors.
    original_stage_config=core.stage_config
    def stage_config(p,r,s):
        cfg=original_stage_config(p,r,s)
        established=root/"gates/single_ni/initial_adapter_state.json"
        if established.exists():
            cfg["expected_initial_adapter_state_sha256"]=core.read(established)["state_sha256"]
        return cfg
    core.stage_config=stage_config
    core.worker(project,root)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker','gradient'])
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='gradient':
        gradient(args.project,args.output)
    else:
        worker(args.project,args.root)
