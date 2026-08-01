#!/usr/bin/env python3
"""CBM revision OOD benchmarks on z_lesion space (where applicable)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule, SkinDataset
from src.datasets.splits import build_id_ood_test_dataloaders, load_filtered_master
from src.models.baseline import BaselineResNet50
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.seed import seed_everything


RESULT_DIR = ROOT / "results" / "cbm_revision"
OUT_CSV = RESULT_DIR / "ood_comparison.csv"
OUT_JSON = RESULT_DIR / "ood_comparison.json"
METADATA = ROOT / "data" / "master_metadata_lesion_only_soft.csv"
SEEDS = [42, 52, 62, 72, 82]


def _method_ckpt_dir(method_key, seed):
    if method_key == "Baseline Soft":
        return ROOT / "checkpoints" / "baseline" / f"baseline_soft_s{seed}"
    if method_key == "Run A / GRL":
        return ROOT / "checkpoints" / "csg_lite" / f"runA_grl_s{seed}"
    if method_key == "Run B (orth=1.0)":
        return ROOT / "checkpoints" / "csg_lite" / f"runB_orth1_s{seed}"
    raise ValueError(method_key)


def _find_ckpt(dir_path):
    if not dir_path.exists():
        return None
    best = sorted(dir_path.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = sorted(dir_path.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return last[0] if last else None


def _fpr95(y_true, scores):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) if idx.size > 0 else 1.0


def _aupr_in(y_true, scores):
    # y_true: 1 for ID, 0 for OOD
    return float(average_precision_score(y_true, scores))


def _aupr_out(y_true_ood, scores_ood):
    # y_true_ood: 1 for OOD, 0 for ID
    return float(average_precision_score(y_true_ood, scores_ood))


@torch.no_grad()
def _collect_baseline_features(model, loader, device):
    logits, feats, labels = [], [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        out, f = model(images, return_features=True)
        logits.append(out.cpu())
        feats.append(f.cpu())
        labels.append(y.cpu())
    return (
        torch.cat(logits, dim=0).numpy(),
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
    )


@torch.no_grad()
def _collect_csg_features(model, loader, device):
    logits, zles, labels = [], [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        out, _dctx, _dadv, z_l, _zctx = model(images, x_lesion=None, return_latents=True)
        logits.append(out.cpu())
        zles.append(z_l.cpu())
        labels.append(y.cpu())
    return (
        torch.cat(logits, dim=0).numpy(),
        torch.cat(zles, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
    )


def _build_isic_train_loader(dm):
    df = load_filtered_master(dm.metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    isic_train_val, _ = train_test_split(
        isic_df,
        test_size=dm.split_config.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=dm.split_config.random_state,
    )
    isic_train, _ = train_test_split(
        isic_train_val,
        test_size=dm.split_config.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=dm.split_config.random_state,
    )
    ds = SkinDataset(isic_train, transform=dm.eval_transform)
    return DataLoader(
        ds,
        batch_size=dm.batch_size,
        shuffle=False,
        num_workers=dm.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=dm.num_workers > 0,
    )


def _compute_scores(logits_id, logits_ood, ztr, ytr, zid, zood):
    # Build binary labels
    y_ood = np.concatenate([np.zeros(len(zid), dtype=np.int64), np.ones(len(zood), dtype=np.int64)])
    y_id = 1 - y_ood

    # MSP (OOD larger score => negative max prob)
    p_id = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
    s_id_msp = -p_id.max(axis=1)
    s_ood_msp = -p_ood.max(axis=1)

    # Energy
    s_id_en = (-torch.logsumexp(torch.from_numpy(logits_id), dim=1)).numpy()
    s_ood_en = (-torch.logsumexp(torch.from_numpy(logits_ood), dim=1)).numpy()

    # Cosine prototype on z_lesion-like space
    n_classes = int(logits_id.shape[1])
    prot = np.zeros((n_classes, ztr.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        m = ytr == c
        if not np.any(m):
            continue
        prot[c] = ztr[m].mean(axis=0)
    prot = prot / (np.linalg.norm(prot, axis=1, keepdims=True) + 1e-12)
    zidn = zid / (np.linalg.norm(zid, axis=1, keepdims=True) + 1e-12)
    zoodn = zood / (np.linalg.norm(zood, axis=1, keepdims=True) + 1e-12)
    s_id_cos = -(zidn @ prot.T).max(axis=1)
    s_ood_cos = -(zoodn @ prot.T).max(axis=1)

    # Mahalanobis on z_lesion-like space
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, num_classes=n_classes, reg_eps=1e-3)
    s_id_maha = ood_metrics.mahalanobis_min_squared_distances(zid, means, precision)
    s_ood_maha = ood_metrics.mahalanobis_min_squared_distances(zood, means, precision)

    out = {}
    for name, sid, sood in [
        ("MSP", s_id_msp, s_ood_msp),
        ("Energy", s_id_en, s_ood_en),
        ("Cosine", s_id_cos, s_ood_cos),
        ("Mahalanobis", s_id_maha, s_ood_maha),
    ]:
        s = np.concatenate([sid, sood])
        out[name] = {
            "AUROC": float(roc_auc_score(y_ood, s)),
            "AUPR_IN": _aupr_in(y_id, -s),
            "AUPR_OUT": _aupr_out(y_ood, s),
            "FPR95": _fpr95(y_ood, s),
        }
    return out


def main():
    seed_everything(42)
    pl.seed_everything(42, workers=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    dm = SkinDataModule(metadata_csv=METADATA, batch_size=128, num_workers=8)
    dm.setup()
    id_loader, ood_loader = build_id_ood_test_dataloaders(dm)
    train_loader = _build_isic_train_loader(dm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    methods = ["Baseline Soft", "Run A / GRL", "Run B (orth=1.0)"]
    per_seed = {}
    rows_for_csv = []

    for method in methods:
        per_seed[method] = {}
        for seed in SEEDS:
            ckpt = _find_ckpt(_method_ckpt_dir(method, seed))
            if ckpt is None:
                continue
            if method == "Baseline Soft":
                lit = BaselineResNet50.load_from_checkpoint(str(ckpt))
                model = lit.to(device)
                logits_tr, z_tr, y_tr = _collect_baseline_features(model, train_loader, device)
                logits_id, z_id, _ = _collect_baseline_features(model, id_loader, device)
                logits_ood, z_ood, _ = _collect_baseline_features(model, ood_loader, device)
                model_name = "baseline(backbone-as-latent)"
            else:
                lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), strict=False)
                model = lit.model.to(device)
                logits_tr, z_tr, y_tr = _collect_csg_features(model, train_loader, device)
                logits_id, z_id, _ = _collect_csg_features(model, id_loader, device)
                logits_ood, z_ood, _ = _collect_csg_features(model, ood_loader, device)
                model_name = "csg(z_lesion)"

            scores = _compute_scores(logits_id, logits_ood, z_tr, y_tr, z_id, z_ood)
            per_seed[method][str(seed)] = {"checkpoint": str(ckpt), "scores": scores}

        # aggregate over available seeds
        score_types = ["MSP", "Energy", "Cosine", "Mahalanobis"]
        for st in score_types:
            vals = {"AUROC": [], "AUPR_IN": [], "AUPR_OUT": [], "FPR95": []}
            for s, obj in per_seed[method].items():
                if st not in obj["scores"]:
                    continue
                for k in vals:
                    vals[k].append(float(obj["scores"][st][k]))
            if not vals["AUROC"]:
                continue
            rows_for_csv.append(
                {
                    "Method": method,
                    "Model": "z_lesion",
                    "ScoreType": st,
                    "AUROC": float(np.mean(vals["AUROC"])),
                    "AUPR_IN": float(np.mean(vals["AUPR_IN"])),
                    "AUPR_OUT": float(np.mean(vals["AUPR_OUT"])),
                    "FPR95": float(np.mean(vals["FPR95"])),
                }
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Model", "ScoreType", "AUROC", "AUPR_IN", "AUPR_OUT", "FPR95"])
        writer.writeheader()
        for r in rows_for_csv:
            writer.writerow(r)
    OUT_JSON.write_text(json.dumps({"per_seed": per_seed, "aggregate_rows": rows_for_csv}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()

