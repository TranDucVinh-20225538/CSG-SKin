#!/usr/bin/env python3
"""Re-score Phase 15.2 ERM checkpoints on pad_heldout (inference only)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = object
    _tf.AutoTokenizer = object
    _cfg = types.ModuleType("transformers.configuration_utils")

    class _PC:
        def __eq__(self, other):
            return False

    _cfg.PretrainedConfig = _PC
    _tf.configuration_utils = _cfg
    sys.modules["transformers"] = _tf
    sys.modules["transformers.configuration_utils"] = _cfg

import numpy as np
import torch

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")

import train_phase2_pad_holdout as p2
from phase15_common import detector_block
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from train_phase15_1_single_dann import SingleDannLightning, collect_all

P2_SPLIT = ROOT / "results/paperB/phase2_pad_holdout/pad_patient_split_paths.json"
OUT = ROOT / "results/paperB/cleanup"
SEEDS = (42, 52, 62)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"]
    pad = df[df.domain == "pad_ufes"]
    train, val, test = p2.isic_splits(isic)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    eval_tf = build_val_transform_robust()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, 4, False)

    rows = []
    for seed in SEEDS:
        ckpt = ROOT / f"checkpoints/phase15/objectives/erm_w0_s{seed}"
        best = sorted(ckpt.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
        lit = SingleDannLightning.load_from_checkpoint(str(best), strict=False)
        net = lit.net.to(device).eval()
        train_p = collect_all(net, loader(train), device)
        id_p = collect_all(net, loader(test), device)
        hold_p = collect_all(net, loader(pad_hold), device)
        det = detector_block(
            train_p["z"], train_p["labels"], id_p["z"], hold_p["z"], id_p["logits"], hold_p["logits"], 8
        )
        summ_path = ROOT / f"results/paperB/phase15/objectives/erm_w0_s{seed}/summary.json"
        old = json.loads(summ_path.read_text()) if summ_path.exists() else {}
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(best),
                "maha_pad_heldout": float(det["mahalanobis_classcond"]),
                "maha_pad_full_stored": old.get("ood_pad_full", {}).get("mahalanobis_classcond"),
                "knn_pad_heldout": float(det["knn_k50"]),
            }
        )
    maha = [r["maha_pad_heldout"] for r in rows]
    doc = {
        "method": "erm_w0 Phase15.2",
        "partition": "pad_heldout",
        "per_seed": rows,
        "mean": float(np.mean(maha)),
        "std": float(np.std(maha, ddof=1)),
        "n_seeds": len(rows),
    }
    (OUT / "erm_pad_heldout_rescore.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("ERM pad_heldout Maha", doc["mean"], "±", doc["std"])
    for r in rows:
        print(r["seed"], r["maha_pad_heldout"], "was pad_full", r["maha_pad_full_stored"])


if __name__ == "__main__":
    main()
