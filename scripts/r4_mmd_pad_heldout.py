#!/usr/bin/env python3
"""R4 item 4: score the five detectors on pad_heldout for the MMD arm (and its ERM reference), from the saved
best checkpoints, with the unchanged Phase 15 code (collect_all, detector_block; fit on ISIC train only).
pad_full is recomputed as a pipeline check against the stored summary.json value.

Usage: r4_mmd_pad_heldout.py TASK_ID   (indexes RUNS)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import train_phase15_2_objectives as T
import train_phase2_pad_holdout as p2
from phase15_common import detector_block
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.utils.seed import seed_everything

RES = T.PAPERB / "results" / "paperB"
OUT = RES / "r4" / "mmd"
RUNS = (["phase15/objectives/erm_w0_s{}".format(s) for s in T.SEEDS]
        + ["phase15/objectives/mmd_w{}_s{}".format(w, s) for w in (1, 10, 100) for s in T.SEEDS]
        + ["phase15b/mmd/mmd_w{}_s{}".format(w, s) for w in (300, 1000, 3000, 10000) for s in T.SEEDS])


def main():
    rel = RUNS[int(sys.argv[1])]
    run = Path(rel).name
    out = OUT / "{}.json".format(run)
    if out.exists():
        print("skip_done", out)
        return
    summ = json.loads((RES / rel / "summary.json").read_text())
    seed_everything(int(summ["seed"]))
    spec = json.loads(T.P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, _val, test = p2.isic_splits(isic)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    if len(pad_hold) != 716:
        raise RuntimeError("pad_heldout size {} != 716".format(len(pad_hold)))

    lit = T.InvariantLightning.load_from_checkpoint(summ["best_checkpoint"], strict=False, map_location="cpu")
    device = torch.device("cpu")
    net = lit.net.to(device).eval()
    tf = build_val_transform_robust()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=tf), 64, 4, False)

    with torch.no_grad():
        tr = T.collect_all(net, loader(train), device)
        idp = T.collect_all(net, loader(test), device)
        full = T.collect_all(net, loader(pad), device)
        hold = T.collect_all(net, loader(pad_hold), device)
    blk = {}
    for name, pk in (("pad_full", full), ("pad_heldout", hold)):
        blk[name] = detector_block(tr["z"], tr["labels"], idp["z"], pk["z"], idp["logits"], pk["logits"], n_classes=8)
        blk[name]["n_ood"] = int(len(pk["z"]))
    stored = summ["ood_pad_full"]
    check = {k: float(blk["pad_full"][k] - stored[k]) for k in stored if isinstance(stored[k], (int, float))}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run": run, "source": rel, "objective": summ["objective"], "weight": summ["weight"], "seed": summ["seed"],
        "checkpoint": summ["best_checkpoint"], "id_balanced_acc": summ["id_balanced_acc"], "id_acc": summ["id_acc"],
        "leakage_bal_acc": summ["leakage"]["bal_acc_mean"],
        "ood": blk, "pad_full_minus_stored": check,
        "max_abs_pad_full_diff": float(np.max(np.abs(list(check.values())))),
    }, indent=1) + "\n")
    print("saved", out, "maha heldout", blk["pad_heldout"]["mahalanobis_classcond"], "check", check)


if __name__ == "__main__":
    main()
