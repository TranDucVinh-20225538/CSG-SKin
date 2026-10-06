#!/usr/bin/env python3
"""Phase 1.5a: trivial (hand-crafted) OOD detectors. No GPU, no retraining.

Protocol matches Phase 1: class-conditional Mahalanobis is fit on ISIC train
only, scored on ISIC test (ID) vs PAD (OOD). Logistic domain probe is a
70/30 stratified ISIC-vs-PAD fit on the leakage probe set (ISIC test + PAD).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.constants import LABEL_TO_INDEX
from src.utils import ood_metrics

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_5")
SOFT_META = REPO / "data" / "master_metadata_lesion_only_soft.csv"
RAW_META = REPO / "data" / "master_metadata.csv"
LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def remap_path(p: str) -> str:
    s = str(p)
    if s.startswith("/mnt/data2/Vinh/"):
        return "/data2/hpcshared/Vinh/" + s[len("/mnt/data2/Vinh/") :]
    return s


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
    df["path"] = df["path"].map(remap_path)
    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    df["exists"] = df["path"].map(os.path.isfile)
    return df.reset_index(drop=True)


def splits(df: pd.DataFrame):
    isic = df[df["domain"] == "isic"].copy()
    pad = df[df["domain"] == "pad_ufes"].copy()
    tv, test = train_test_split(isic, test_size=0.2, stratify=isic["label_idx"], random_state=42)
    train, val = train_test_split(tv, test_size=0.2, stratify=tv["label_idx"], random_state=42)
    return {
        "isic_train": train.reset_index(drop=True),
        "isic_val": val.reset_index(drop=True),
        "isic_test": test.reset_index(drop=True),
        "pad": pad.reset_index(drop=True),
    }


def eval_transform():
    # Same geometry as build_val_transform_robust, without ImageNet norm.
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


def raw_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def _skew_kurt(x: np.ndarray):
    x = x.astype(np.float64).ravel()
    n = max(len(x), 1)
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0, 0.0
    z = (x - m) / s
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)
    return skew, kurt


def features_from_tensor(t):
    """t: 3x224x224 in [0,1]."""
    x = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if x.ndim != 3:
        raise ValueError(x.shape)
    rgb = np.transpose(x, (1, 2, 0))  # HWC
    out = {}
    means = rgb.reshape(-1, 3).mean(axis=0)
    stds = rgb.reshape(-1, 3).std(axis=0)
    out["rgb_moments"] = np.concatenate([means, stds]).astype(np.float32)

    hist = []
    for c in range(3):
        h, _ = np.histogram(rgb[:, :, c], bins=16, range=(0.0, 1.0), density=False)
        h = h.astype(np.float64)
        h = h / max(h.sum(), 1.0)
        hist.append(h)
    out["color_hist"] = np.concatenate(hist).astype(np.float32)

    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    small = np.array(
        Image.fromarray((gray * 255).clip(0, 255).astype(np.uint8)).resize((16, 16), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    out["gray_downsample"] = small.ravel()

    # HSV
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    h = np.zeros_like(mx)
    mask = df > 1e-8
    # hue
    idx = mask & (mx == r)
    h[idx] = np.mod(((g - b)[idx] / df[idx]), 6.0) / 6.0
    idx = mask & (mx == g)
    h[idx] = (((b - r)[idx] / df[idx]) + 2.0) / 6.0
    idx = mask & (mx == b)
    h[idx] = (((r - g)[idx] / df[idx]) + 4.0) / 6.0
    s = np.zeros_like(mx)
    s[mx > 1e-8] = df[mx > 1e-8] / mx[mx > 1e-8]
    v = mx
    hsv = np.stack([h, s, v], axis=2)
    out["hsv_moments"] = np.concatenate([hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0)]).astype(np.float32)

    # Laplacian on gray
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from numpy.lib.stride_tricks import as_strided

    g64 = gray.astype(np.float64)
    # pad
    gp = np.pad(g64, 1, mode="edge")
    lap = (
        k[0, 1] * gp[0:-2, 1:-1]
        + k[1, 0] * gp[1:-1, 0:-2]
        + k[1, 1] * gp[1:-1, 1:-1]
        + k[1, 2] * gp[1:-1, 2:]
        + k[2, 1] * gp[2:, 1:-1]
    )
    sk, ku = _skew_kurt(lap)
    out["laplacian_stats"] = np.array([lap.mean(), lap.std(), sk, ku], dtype=np.float32)
    return out


def extract_row(path, tf):
    img = Image.open(path).convert("RGB")
    t = tf(img)
    return features_from_tensor(t)


def _worker(item):
    path, label_idx, domain, mode = item
    if mode == "eval":
        tf = eval_transform()
    else:
        tf = raw_transform()
    f = extract_row(path, tf)
    return f, int(label_idx), int(domain)


def extract_frame(frame, mode, limit=None, workers=8):
    n = len(frame) if limit is None else min(limit, len(frame))
    items = []
    for i, row in enumerate(frame.itertuples(index=False)):
        if i >= n:
            break
        items.append((row.path, int(row.label_idx), 0 if row.domain == "isic" else 1, mode))
    feats = {k: [] for k in ["rgb_moments", "color_hist", "gray_downsample", "hsv_moments", "laplacian_stats"]}
    labels = []
    domains = []
    ok = 0
    fail = 0
    if workers <= 1 or n <= 32:
        for it in items:
            try:
                f, y, d = _worker(it)
                for k, v in f.items():
                    feats[k].append(v)
                labels.append(y)
                domains.append(d)
                ok += 1
            except Exception:
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, it) for it in items]
            for i, fut in enumerate(as_completed(futs), start=1):
                try:
                    f, y, d = fut.result()
                    for k, v in f.items():
                        feats[k].append(v)
                    labels.append(y)
                    domains.append(d)
                    ok += 1
                except Exception:
                    fail += 1
                if i % 2000 == 0:
                    print("  {}/{} ok={} fail={}".format(i, n, ok, fail), flush=True)
    packed = {k: np.stack(v, axis=0) if v else np.zeros((0, 1), np.float32) for k, v in feats.items()}
    return packed, np.asarray(labels, np.int64), np.asarray(domains, np.int64), ok, fail


def fpr95(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) if idx.size else 1.0


def maha_metrics(ztr, ytr, zid, zood, n_classes=8, reg_eps=1e-3):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, num_classes=n_classes, reg_eps=reg_eps)
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    y = np.concatenate([np.zeros(len(sid), np.int64), np.ones(len(sood), np.int64)])
    s = np.concatenate([sid, sood])
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPR_IN": float(average_precision_score(1 - y, -s)),
        "AUPR_OUT": float(average_precision_score(y, s)),
        "FPR95": fpr95(y, s),
        "n_train": int(len(ztr)),
        "n_id": int(len(sid)),
        "n_ood": int(len(sood)),
        "feat_dim": int(ztr.shape[1]),
    }


def logistic_probe(z_id, z_ood, seeds=(42, 52, 62, 72, 82)):
    x = np.concatenate([z_id, z_ood], axis=0)
    y = np.concatenate([np.zeros(len(z_id), np.int64), np.ones(len(z_ood), np.int64)])
    accs, bals, aurocs = [], [], []
    for s in seeds:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=s, stratify=y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
        clf.fit(xtr, ytr)
        pred = clf.predict(xte)
        proba = clf.predict_proba(xte)[:, 1]
        accs.append(float(accuracy_score(yte, pred)))
        bals.append(float(balanced_accuracy_score(yte, pred)))
        aurocs.append(float(roc_auc_score(yte, proba)))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "bal_acc_mean": float(np.mean(bals)),
        "bal_acc_std": float(np.std(bals)),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "acc_per_seed": accs,
        "majority": float(max((y == 0).mean(), (y == 1).mean())),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=OUT)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Cap rows per split (0=all).")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def run_mode(name, df, tf, out_dir, limit, workers=8):
    print("=== mode {} ===".format(name), flush=True)
    present = df[df["exists"]].copy()
    print("rows with files: {} / {}".format(len(present), len(df)))
    print(present.groupby("domain").size().to_string())
    if present[present.domain == "pad_ufes"].empty:
        print("BLOCKED: no readable PAD images in this mode")
        return {"mode": name, "blocked": True, "reason": "PAD images unreadable"}

    sp = splits(present)
    for k, v in sp.items():
        missing = (~v["path"].map(os.path.isfile)).sum()
        print("  {} n={} missing_files={}".format(k, len(v), missing))
    assert (sp["isic_train"]["domain"] == "isic").all()

    packs = {}
    for split_name in ["isic_train", "isic_test", "pad"]:
        print("extract", name, split_name, flush=True)
        packed, y, d, ok, fail = extract_frame(
            sp[split_name],
            mode="eval" if name.startswith("processed") else "raw",
            limit=limit,
            workers=1 if (limit is not None and limit <= 32) else workers,
        )
        packs[split_name] = {"feat": packed, "y": y, "d": d, "ok": ok, "fail": fail}
        print("  ok={} fail={}".format(ok, fail), flush=True)

    ytr = packs["isic_train"]["y"]
    rows = []
    probes = []
    for fname in ["rgb_moments", "color_hist", "gray_downsample", "hsv_moments", "laplacian_stats"]:
        ztr = packs["isic_train"]["feat"][fname]
        zid = packs["isic_test"]["feat"][fname]
        zood = packs["pad"]["feat"][fname]
        print("maha", name, fname, ztr.shape, flush=True)
        try:
            mets = maha_metrics(ztr, ytr, zid, zood)
            err = ""
        except Exception as e:
            mets = {"AUROC": float("nan"), "AUPR_IN": float("nan"), "AUPR_OUT": float("nan"), "FPR95": float("nan")}
            err = str(e)
        for metric in ["AUROC", "AUPR_IN", "AUPR_OUT", "FPR95"]:
            rows.append(
                {
                    "mode": name,
                    "feature_set": fname,
                    "detector": "mahalanobis_classcond",
                    "metric": metric,
                    "value": mets.get(metric, float("nan")),
                    "feat_dim": int(ztr.shape[1]),
                    "error": err,
                }
            )
        probe = logistic_probe(zid, zood)
        probes.append({"mode": name, "feature_set": fname, "feat_dim": int(ztr.shape[1]), **probe})
        print(
            "  {} Maha AUROC={:.4f} logistic_auroc={:.4f} bal_acc={:.4f}".format(
                fname, mets.get("AUROC", float("nan")), probe["auroc_mean"], probe["bal_acc_mean"]
            ),
            flush=True,
        )
    return {"mode": name, "blocked": False, "maha_rows": rows, "probes": probes, "counts": {k: packs[k]["ok"] for k in packs}}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    limit = 16 if args.dry_run else (args.limit or None)
    if args.dry_run:
        print("dry_run limit=16")

    soft = load_master(SOFT_META)
    raw = load_master(RAW_META)
    print("soft exists", int(soft.exists.sum()), "raw exists", int(raw.exists.sum()))
    print("raw by domain\n", raw.groupby(["domain", "exists"]).size())

    results = []
    results.append(run_mode("processed_evalgeom", soft, eval_transform(), args.output_dir, limit, workers=args.workers))
    results.append(run_mode("raw_resize224", raw, raw_transform(), args.output_dir, limit, workers=args.workers))

    maha_rows = []
    probe_rows = []
    for r in results:
        if r.get("blocked"):
            continue
        maha_rows.extend(r["maha_rows"])
        probe_rows.extend(r["probes"])

    maha_csv = args.output_dir / ("trivial_maha_dryrun.csv" if args.dry_run else "trivial_maha.csv")
    probe_csv = args.output_dir / ("trivial_logistic_dryrun.csv" if args.dry_run else "trivial_logistic.csv")
    if maha_rows:
        with maha_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(maha_rows[0].keys()))
            w.writeheader()
            w.writerows(maha_rows)
    if probe_rows:
        with probe_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(probe_rows[0].keys()))
            w.writeheader()
            w.writerows(probe_rows)
    (args.output_dir / ("trivial_summary_dryrun.json" if args.dry_run else "trivial_summary.json")).write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    print("wrote", maha_csv, probe_csv)
    if args.dry_run:
        (args.output_dir / "trivial_dry_run_ok.json").write_text(json.dumps({"ok": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
