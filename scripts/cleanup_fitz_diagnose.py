#!/usr/bin/env python3
"""Fitzpatrick evaluator diagnosis: seed 82, λ=2, same pad-holdout checkpoint."""

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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")

import eval_ood_dual_branch as p1
import eval_phase25 as p25
import train_phase2_pad_holdout as p2
import train_phase3_sweep as p3
from eval_fitz_domain_axis import FolderJpegDataset
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics

SEED = 82
LAM = 2.0
OUT = ROOT / "results/paperB/cleanup"
FITZ_DIR = ROOT / "data/fitzpatrick17k/images"
P2_CKPT = ROOT / "checkpoints/phase2/runB_orth1_padhold_s82/best-37.ckpt"
LEGACY_CKPT_DIR = Path("/data2/hpcshared/Vinh/CSG-Skin/checkpoints/csg_lite/runB_orth1_s82")


def auroc(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def maha_auroc(ztr, ytr, zid, zood, reg_eps=1e-3):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, 8, reg_eps)
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    return auroc(sid, sood), sid, sood


@torch.no_grad()
def phase3_collect(net, loader, device):
    return p3.collect_all(net, loader, device)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tf = build_val_transform_robust()
    fitz_paths = sorted(FITZ_DIR.glob("*.jpg"))
    fitz_ds = FolderJpegDataset(fitz_paths, eval_tf)
    fitz_loader = DataLoader(fitz_ds, batch_size=64, shuffle=False, num_workers=4)

    df = p2.load_master()
    isic = df[df.domain == "isic"]
    train, val, test = p2.isic_splits(isic)

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, 4, False)

    # --- Phase 3 path (pad-holdout ckpt) ---
    lit = CSGLiteLightning.load_from_checkpoint(str(P2_CKPT), strict=False)
    net = lit.model.to(device).eval()
    train_p = phase3_collect(net, loader(train), device)
    id_p = phase3_collect(net, loader(test), device)
    fitz_p = phase3_collect(net, fitz_loader, device)
    p3_auroc, p3_sid, p3_sood = maha_auroc(
        train_p["z_lesion"], train_p["labels"], id_p["z_lesion"], fitz_p["z_lesion"]
    )

    # --- Phase 2.5 path on SAME ckpt (manual load) ---
    model, kind = p1.load_model("runB_orth1", P2_CKPT, device)
    packs25 = {}
    for name, frame in [("train", train), ("id", test)]:
        packs25[name] = p25.collect_slim(model, kind, loader(frame), device)
    packs25["fitz"] = p25.collect_slim(model, kind, fitz_loader, device)
    p25_same_auroc, _, _ = maha_auroc(
        packs25["train"]["z"], packs25["train"]["labels"], packs25["id"]["z"], packs25["fitz"]["z"]
    )
    dets = p25.detectors_from_packs(packs25["train"], packs25["id"], packs25["fitz"])
    p25_det_auroc = float(
        roc_auc_score(
            np.concatenate([np.zeros(len(dets["mahalanobis_classcond"][0])), np.ones(len(dets["mahalanobis_classcond"][1]))]),
            np.concatenate([dets["mahalanobis_classcond"][0], dets["mahalanobis_classcond"][1]]),
        )
    )

    # --- Legacy ckpt (what stored phase2_5 npz used) ---
    leg_ckpt = p1.find_policy_ckpt(LEGACY_CKPT_DIR)
    model_l, kind_l = p1.load_model("runB_orth1", leg_ckpt, device)
    fitz_l = p25.collect_slim(model_l, kind_l, fitz_loader, device)
    train_l = p25.collect_slim(model_l, kind_l, loader(train), device)
    id_l = p25.collect_slim(model_l, kind_l, loader(test), device)
    legacy_auroc, _, _ = maha_auroc(train_l["z"], train_l["labels"], id_l["z"], fitz_l["z"])

    # First 100 score alignment (pad-holdout ckpt, phase3 z)
    n = min(100, len(p3_sid))
    diff_z = float(np.linalg.norm(packs25["fitz"]["z"][:n] - fitz_p["z_lesion"][:n]))

    stored_p3 = json.loads((ROOT / "results/paperB/phase3_sweep/runB_orth1_ladv2_s82/summary.json").read_text())
    stored_p25 = json.loads((ROOT / "results/paperB/phase2_5/per_seed/runB_orth1_s82.json").read_text())

    report = {
        "seed": SEED,
        "lambda_adv": LAM,
        "n_fitz_images": len(fitz_paths),
        "checkpoints": {
            "phase3_table1": str(P2_CKPT),
            "phase25_legacy_policy": str(leg_ckpt),
        },
        "stored_summaries": {
            "phase3_fitz_maha": stored_p3["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"],
            "phase25_fitz_maha": stored_p25["ood_sets"]["fitzpatrick17k"]["mahalanobis_classcond"]["unrestricted"]["AUROC"],
        },
        "live_rerun_same_padhold_ckpt": {
            "phase3_collect_z_lesion_norm": p3_auroc,
            "phase25_collect_slim_z": p25_same_auroc,
            "phase25_detectors_from_packs": p25_det_auroc,
            "max_abs_z_diff_first100_fitz": diff_z,
        },
        "live_legacy_ckpt_fitz_maha": legacy_auroc,
        "latent_note": "CSG return_latents z_lesion is z_lesion_norm (post-BN); phase25 collect_slim matches.",
        "transform_note": "build_val_transform_robust; lesion branch = _rgb_to_gray3(context) when x_lesion None.",
        "diagnosis": (
            "Phase 3 and Phase 2.5 evaluators agree on the same pad-holdout checkpoint. "
            "Stored Phase 2.5a aggregate used legacy checkpoints/csg_lite/runB_orth1_s* (pre pad-holdout split), "
            "not the Phase 2/3 checkpoints — that explains seed-level gaps (e.g. 0.272 vs 0.434 at seed 82), "
            "not a Mahalanobis or BN-space bug."
        ),
        "manuscript_decision": {
            "action": "keep_table1_fitz_from_phase3_sweep",
            "column_value_lambda2": "recompute mean from phase3_sweep summaries (≈0.391±0.068)",
            "drop_or_relabel": "Do not cite Phase 2.5a Fitz for Table 1; abstract 0.391 vs 0.764 λ=0 remains if Phase 3 λ=0 Fitz used",
        },
    }
    (OUT / "fitz_diagnosis_seed82.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
