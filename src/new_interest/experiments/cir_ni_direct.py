"""Standalone CIR-as-NI vs original-NI training using the frozen CIR runtime.

Only the stage schedule changes: one cold SFT stage per arm, no warm-up parent
and no subsequent NI fine-tuning. Numerical Trainer and evaluator are reused.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from new_interest.experiments import cir_warmup as core


def schedule(project):
    cfg = core.read(project / core.CONTRACT)
    value = cfg["direct_training"]
    core.require(value == {"samples": 1055, "epochs": 3, "optimizer_steps": 99,
                           "presentations": 3165, "parent": None}, "Direct training budget changed")
    return value


def stage_specs(project, root, *, gates):
    schedule(project)
    entries = [("single_ni",32,2,1,1,8), ("ddp_ni",41,1,4,2,32)] if gates else [("ni",1055,3,4,2,32)]
    return [{"key":key, "task":"ni", "parent":None, "samples":n, "epochs":epochs,
             "world_size":world, "per_device_train_batch_size":micro,
             "global_batch_size":batch, "accumulation":batch,
             "train_file":root/"gate_data"/f"ni-{n}.parquet" if gates else project/core.DATA/"ni.parquet"}
            for key,n,epochs,world,micro,batch in entries]


def worker(project, root):
    core.require(math.ceil(1055 / 32)*3 == 99, "Incorrect batch arithmetic")
    # These are workflow-level hooks only. Child train/reload/infer commands use
    # the unchanged accepted cir_warmup numerical entrypoints.
    core.stage_specs = stage_specs
    core.formal_schedule = schedule
    core.worker(project, root)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();worker(a.project,a.root)
