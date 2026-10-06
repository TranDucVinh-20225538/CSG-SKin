#!/usr/bin/env python3
"""Phase 4.5a/b: re-score Phase 4 checkpoints. Inference only. No GPU claimed.

4.5a: bootstrap 95% CI over test samples; per-seed values; paired diffs.
4.5b: split 4a OOD into DF-only and VASC-only.
Detectors: Maha class-cond, MSP, Energy, cosine, kNN (Phase 1 suite minus ODIN).
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase4_semantic_ood as p4
from eval_ood_dual_branch import cosine_ood_scores, find_policy_ckpt, fit_class_means, fit_knn, knn_scores

from src.datasets.skin_dataset import SkinDataset
from src.models.baseline import BaselineResNet50
from src.models.csg_lightning import CSGLiteLightning
from src.models.effb3_single import EffB3SingleLightning
from src.utils import ood_metrics

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = PAPERB / "results" / "paperB" / "phase4_5"
CKPT_ROOT = PAPERB / "checkpoints" / "phase4"
CONFIGS = ("4a", "4b")
METHODS = ("baseline", "effb3", "runB_orth1")
SEEDS = (42, 52, 62)
JOBS = [(c, m, s) for c in CONFIGS for m in METHODS for s in SEEDS]
assert len(JOBS) == 18
N_BOOT = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS), default=None)
    p.add_argument("--method", choices=list(METHODS), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--n_boot", type=int, default=N_BOOT)
    return p.parse_args()


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def fpr95(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) if idx.size else 1.0


def auroc(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return float(roc_auc_score(y, s))


def bootstrap_auroc(sid, sood, n_boot, rng):
    vals = np.empty(n_boot, dtype=np.float64)
    n_id, n_ood = len(sid), len(sood)
    for b in range(n_boot):
        i = rng.integers(0, n_id, n_id)
        j = rng.integers(0, n_ood, n_ood)
        vals[b] = roc_auc_score(
            np.concatenate([np.zeros(n_id), np.ones(n_ood)]),
            np.concatenate([sid[i], sood[j]]),
        )
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n_boot": int(n_boot)}


def score_block(sid, sood, n_boot, rng):
    return {
        "AUROC": auroc(sid, sood),
        "FPR95": fpr95(sid, sood),
        "n_id": int(len(sid)),
        "n_ood": int(len(sood)),
        "bootstrap_auroc": bootstrap_auroc(sid, sood, n_boot, rng),
    }


def detectors(ztr, ytr, zid, zood, logits_id, logits_ood, n_classes):
    out = {}
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(
        ztr, ytr, num_classes=n_classes, reg_eps=1e-3
    )
    out["mahalanobis_classcond"] = (
        ood_metrics.mahalanobis_min_squared_distances(zid, means, prec),
        ood_metrics.mahalanobis_min_squared_distances(zood, means, prec),
    )
    cmeans, present = fit_class_means(ztr, ytr, n_classes)
    out["cosine_max"] = (
        cosine_ood_scores(zid, cmeans, present),
        cosine_ood_scores(zood, cmeans, present),
    )
    knn = fit_knn(ztr, k=50)
    out["knn_k50"] = (knn_scores(knn, zid), knn_scores(knn, zood))
    p_id = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
    out["MSP"] = (-p_id.max(axis=1), -p_ood.max(axis=1))
    e_id = torch.logsumexp(torch.from_numpy(logits_id).float(), dim=1).numpy()
    e_ood = torch.logsumexp(torch.from_numpy(logits_ood).float(), dim=1).numpy()
    out["Energy_T1"] = (-e_id, -e_ood)
    return out


def load_net(method, ckpt, n_classes, device):
    if method == "baseline":
        lit = BaselineResNet50.load_from_checkpoint(str(ckpt), strict=False)
        net = lit.net
    elif method == "effb3":
        lit = EffB3SingleLightning.load_from_checkpoint(str(ckpt), strict=False)
        net = lit.net
    else:
        lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), strict=False)
        net = lit.model
    return net.to(device).eval()


def spaces_of(method, pack):
    if method == "runB_orth1":
        return {"z_lesion": pack["z_lesion"], "z_context": pack["z_context"]}
    return {"backbone_raw": pack["backbone_raw"]}


def run_one(args, config, method, seed, device):
    dest = OUT / "per_run" / "{}_{}_s{}.json".format(config, method, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_done and dest.exists() and not args.dry_run:
        print("skip_done", dest, flush=True)
        return json.loads(dest.read_text())

    cfg = p4.CONFIGS[config]
    hold = cfg["hold"]
    all_labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    keep = [x for x in all_labels if x not in hold]
    n_classes = len(keep)
    isic = p4.load_isic()
    train8, val8, test8 = p4.isic_splits(isic)
    train, _ = p4.remap_kept(train8, keep)
    id_test, _ = p4.remap_kept(test8, keep)
    ood = isic[isic["label"].isin(hold)].copy().reset_index(drop=True)
    # keep original 8-class names for 4.5b; label_idx unused for OOD scoring
    ckpt_dir = CKPT_ROOT / "{}_{}_s{}".format(config, method, seed)
    ckpt = find_policy_ckpt(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(ckpt_dir)
    print("ckpt", ckpt, flush=True)

    if method == "baseline":
        eval_tf = p4.baseline_eval_tf()
    else:
        from src.datasets.skin_dataset import build_val_transform_robust

        eval_tf = build_val_transform_robust()

    max_b = 2 if args.dry_run else None

    def loader(frame):
        ds = SkinDataset(frame, transform=eval_tf)
        if args.dry_run:
            ds.data = ds.data.iloc[: args.batch_size * 2].reset_index(drop=True)
        return p4.make_loader(ds, args.batch_size, 0 if args.dry_run else args.num_workers, False)

    net = load_net(method, ckpt, n_classes, device)
    ptr = p4.collect(method, net, loader(train), device, max_batches=max_b)
    pid = p4.collect(method, net, loader(id_test), device, max_batches=max_b)
    pood = p4.collect(method, net, loader(ood), device, max_batches=max_b)
    ood_names = ood["label"].to_numpy()
    if args.dry_run:
        ood_names = ood_names[: len(pood["labels"])]

    rng = np.random.default_rng(seed)
    rec = {
        "config": config,
        "method": method,
        "seed": seed,
        "n_classes": n_classes,
        "keep": keep,
        "hold_out": hold,
        "ckpt": str(ckpt),
        "spaces": {},
    }
    for space, ztr in spaces_of(method, ptr).items():
        zid = spaces_of(method, pid)[space]
        zood = spaces_of(method, pood)[space]
        try:
            dets = detectors(ztr, ptr["labels"], zid, zood, pid["logits"], pood["logits"], n_classes)
        except ValueError as exc:
            print("  detectors skipped", space, exc, flush=True)
            rec["spaces"][space] = {"error": str(exc)}
            continue
        block = {}
        for det, (sid, sood) in dets.items():
            if np.isnan(np.asarray(sid)).any() or np.isnan(np.asarray(sood)).any():
                continue
            try:
                entry = {"pooled_holdout": score_block(sid, sood, args.n_boot, rng)}
                if config == "4a":
                    for cls in ("DF", "VASC"):
                        m = ood_names == cls
                        if m.sum() == 0:
                            continue
                        entry[cls] = score_block(sid, sood[m], args.n_boot, rng)
                block[det] = entry
            except ValueError as exc:
                print("  skip", det, exc, flush=True)
        rec["spaces"][space] = block

    if not args.dry_run:
        dest.write_text(json.dumps(rec, indent=2, default=json_default) + "\n")
        # tiny score dump for paired bootstrap later
        npz = OUT / "scores" / "{}_{}_s{}.npz".format(config, method, seed)
        npz.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ood_label": ood_names}
        for space, ztr in spaces_of(method, ptr).items():
            dets = detectors(
                ztr,
                ptr["labels"],
                spaces_of(method, pid)[space],
                spaces_of(method, pood)[space],
                pid["logits"],
                pood["logits"],
                n_classes,
            )
            payload["{}_maha_id".format(space)] = dets["mahalanobis_classcond"][0]
            payload["{}_maha_ood".format(space)] = dets["mahalanobis_classcond"][1]
        np.savez_compressed(npz, **payload)
    summary = {}
    for sp, blk in rec["spaces"].items():
        if isinstance(blk, dict) and "mahalanobis_classcond" in blk:
            summary[sp] = blk["mahalanobis_classcond"]["pooled_holdout"]["AUROC"]
    print("done", config, method, seed, summary, flush=True)
    return rec


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device", device, flush=True)
    jobs = JOBS
    if args.config or args.method or args.seed is not None:
        jobs = [
            (c, m, s)
            for c, m, s in JOBS
            if (args.config is None or c == args.config)
            and (args.method is None or m == args.method)
            and (args.seed is None or s == args.seed)
        ]
    for c, m, s in jobs:
        print("===", c, m, s, flush=True)
        run_one(args, c, m, s, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
