#!/usr/bin/env python3
"""Phase 1.5b: z_context / z_lesion score distributions and PCA-AUROC. One seed is enough for geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_ood_dual_branch as p1

from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.utils import ood_metrics

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_5")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", type=str, default="runB_orth1")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--output_dir", type=Path, default=OUT)
    return p.parse_args()


def describe(x, name):
    x = np.asarray(x, dtype=np.float64)
    return {
        "name": name,
        "n": int(len(x)),
        "min": float(x.min()) if len(x) else None,
        "max": float(x.max()) if len(x) else None,
        "mean": float(x.mean()) if len(x) else None,
        "std": float(x.std()) if len(x) else None,
        "p01": float(np.percentile(x, 1)) if len(x) else None,
        "p99": float(np.percentile(x, 99)) if len(x) else None,
    }


def gap_report(id_s, ood_s):
    id_s = np.asarray(id_s, dtype=np.float64)
    ood_s = np.asarray(ood_s, dtype=np.float64)
    # higher = more OOD
    overlap = float(np.minimum(id_s.max(), ood_s.max()) - np.maximum(id_s.min(), ood_s.min()))
    return {
        "max_id": float(id_s.max()),
        "min_ood": float(ood_s.min()),
        "gap_minOOD_minus_maxID": float(ood_s.min() - id_s.max()),
        "overlap_range": overlap if overlap > 0 else 0.0,
        "perfect_sep_no_overlap": bool(ood_s.min() > id_s.max()),
    }


def pca_auroc(ztr, zid, zood, ks):
    rows = []
    d = ztr.shape[1]
    y = np.concatenate([np.zeros(len(zid)), np.ones(len(zood))])
    for k in ks:
        kk = min(int(k), d, len(ztr) - 1)
        if kk < 1:
            continue
        pca = PCA(n_components=kk, random_state=42)
        tr = pca.fit_transform(ztr)
        idp = pca.transform(zid)
        oodp = pca.transform(zood)
        # class-agnostic Maha in PCA space (and also use first coord as 1d score)
        mu = tr.mean(axis=0)
        c = tr - mu
        cov = (c.T @ c) / max(len(tr) - 1, 1) + 1e-3 * np.eye(kk)
        prec = np.linalg.inv(cov)
        def maha(z):
            dlt = z - mu
            return np.einsum("ni,ij,nj->n", dlt, prec, dlt)
        s = np.concatenate([maha(idp), maha(oodp)])
        auroc = float(roc_auc_score(y, s))
        # 1d projection: PC1 signed so OOD mean > ID mean
        pc1_id = idp[:, 0]
        pc1_ood = oodp[:, 0]
        if pc1_ood.mean() < pc1_id.mean():
            pc1_id, pc1_ood = -pc1_id, -pc1_ood
        auroc_pc1 = float(roc_auc_score(y, np.concatenate([pc1_id, pc1_ood]))) if k == 1 else None
        rows.append({"k": kk, "requested_k": int(k), "auroc_maha": auroc, "auroc_pc1_if_k1": auroc_pc1, "var_explained": float(pca.explained_variance_ratio_.sum())})
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    splits = p1.build_split_frames(p1.METADATA)
    tf = build_val_transform_robust()
    max_b = 2 if args.dry_run else None

    def loader(frame):
        ds = SkinDataset(frame, transform=tf)
        if args.dry_run:
            ds.data = ds.data.iloc[: max_b * args.batch_size].reset_index(drop=True)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    ckpt = p1.find_policy_ckpt(p1.method_ckpt_dir(args.method, args.seed))
    print("ckpt", ckpt)
    model, kind = p1.load_model(args.method, ckpt, device)
    assert kind == "csg"
    packs = {}
    for name, frame in [("train", splits["isic_train"]), ("id", splits["isic_test"]), ("ood", splits["pad"])]:
        packs[name] = p1.collect_csg(model, loader(frame), device, max_batches=max_b)
        print(name, "z_context", packs[name]["z_context"].shape)

    out = {"method": args.method, "seed": args.seed, "checkpoint": str(ckpt), "spaces": {}}
    for space in ["z_context", "z_lesion"]:
        ztr, ytr = packs["train"][space], packs["train"]["labels"]
        zid, zood = packs["id"][space], packs["ood"][space]
        n_classes = 8
        means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, num_classes=n_classes, reg_eps=1e-3)
        sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
        sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
        y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
        auroc = float(roc_auc_score(y, np.concatenate([sid, sood])))
        ks = [1, 2, 4, 8, 16, 32, 64]
        out["spaces"][space] = {
            "dim": int(ztr.shape[1]),
            "maha_auroc": auroc,
            "id_scores": describe(sid, "id"),
            "ood_scores": describe(sood, "ood"),
            "separation": gap_report(sid, sood),
            "pca": pca_auroc(ztr, zid, zood, ks),
            "supervision": "domain-supervised" if space == "z_context" else "unsupervised",
        }
        print(space, "auroc", auroc, "gap", out["spaces"][space]["separation"])
        np.savez_compressed(args.output_dir / f"scores_{space}_s{args.seed}.npz", id=sid, ood=sood)

    dest = args.output_dir / ("separation_margin_dryrun.json" if args.dry_run else "separation_margin.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
