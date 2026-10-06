#!/usr/bin/env python3
"""Phase 2.5a/b: Fitzpatrick OOD + class-composition controls. Inference only.

Fit statistics on ISIC train only. Do not retune detectors.
Scoring space: z_lesion (CSG), 16-d z (EffB3), backbone_raw (ResNet-50 baseline).
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_ood_dual_branch as p1
from eval_fitz_domain_axis import FolderJpegDataset

from src.datasets.constants import INDEX_TO_LABEL, LABELS
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.utils import ood_metrics

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase2_5")
FITZ_DIR = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/data/fitzpatrick17k/images")
SEEDS = (42, 52, 62, 72, 82)
METHODS = ("runB_orth1", "baseline_soft", "effb3_control")
# PAD has MEL,NV,BCC,AK,BKL,SCC — not DF (5) or VASC (6).
PAD_CLASSES = (0, 1, 2, 3, 4, 7)
REG_EPS = 1e-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_extract", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--method", type=str, default=None, choices=list(METHODS))
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@torch.no_grad()
def collect_slim(model, kind, loader, device, max_batches=None):
    """z_lesion / EffB3 16-d z / baseline backbone_raw + logits. No fat caches."""
    logits_l, z_l, y_l = [], [], []
    model.eval()
    for b_idx, (images, y) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        if kind == "csg":
            x_les = model._rgb_to_gray3(images)
            feat_les = model.extract_lesion_features(x_les)
            z = model.lesion_bn(model.lesion_projector(feat_les))
            logits = model.lesion_classifier(z)
        elif kind == "effb3":
            x_in = model._rgb_to_gray3(images)
            backbone = model.backbone(x_in)
            z = model.bn(model.projector(backbone))
            logits = model.classifier(z)
        else:
            logits, z = model(images, return_features=True)
        logits_l.append(logits.cpu())
        z_l.append(z.cpu())
        y_l.append(y)
    return {
        "z": torch.cat(z_l).numpy(),
        "logits": torch.cat(logits_l).numpy(),
        "labels": torch.cat(y_l).numpy(),
    }


def detectors_from_packs(train, idp, ood, n_classes=8):
    ztr, ytr = train["z"], train["labels"]
    zid, zood = idp["z"], ood["z"]
    out = {}
    means, present = p1.fit_class_means(ztr, ytr, n_classes)
    out["cosine_max"] = (
        p1.cosine_ood_scores(zid, means, present),
        p1.cosine_ood_scores(zood, means, present),
    )
    dim = ztr.shape[1]
    maha_eps = 1e-3 if dim >= 64 else REG_EPS
    try:
        cmeans, prec = p1.fit_maha_classcond(ztr, ytr, n_classes, maha_eps)
        out["mahalanobis_classcond"] = (
            ood_metrics.mahalanobis_min_squared_distances(zid, cmeans, prec),
            ood_metrics.mahalanobis_min_squared_distances(zood, cmeans, prec),
        )
    except ValueError as exc:
        # Dry-run batches can miss classes; the real run fits on full ISIC train (all 8).
        print("  maha_classcond skipped:", exc, flush=True)
        nan_id = np.full(len(zid), np.nan)
        nan_ood = np.full(len(zood), np.nan)
        out["mahalanobis_classcond"] = (nan_id, nan_ood)
    mu, prec_a = p1.fit_maha_agnostic(ztr, maha_eps)
    out["mahalanobis_agnostic"] = (
        p1.maha_agnostic_scores(zid, mu, prec_a),
        p1.maha_agnostic_scores(zood, mu, prec_a),
    )
    knn = p1.fit_knn(ztr, k=50)
    out["knn_k50"] = (p1.knn_scores(knn, zid), p1.knn_scores(knn, zood))
    p_id = torch.softmax(torch.from_numpy(idp["logits"]), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(ood["logits"]), dim=1).numpy()
    out["MSP"] = (-p_id.max(axis=1), -p_ood.max(axis=1))
    e_id = torch.logsumexp(torch.from_numpy(idp["logits"]).float(), dim=1).numpy()
    e_ood = torch.logsumexp(torch.from_numpy(ood["logits"]).float(), dim=1).numpy()
    out["Energy_T1"] = (-e_id, -e_ood)
    return out


def weighted_auroc(sid, sood, y_id, pad_p, isic_p):
    w_id = np.zeros(len(sid), dtype=np.float64)
    for c, p_pad in pad_p.items():
        p_isic = isic_p.get(c, 0.0)
        if p_isic <= 0:
            continue
        w_id[y_id == c] = p_pad / p_isic
    w = np.concatenate([w_id, np.ones(len(sood), dtype=np.float64)])
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    if w[y == 0].sum() <= 0 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, s, sample_weight=w))


def class_dist_table(sid, y_id, sood, y_ood):
    rows = []
    for c in range(8):
        name = INDEX_TO_LABEL[c]
        id_c = sid[y_id == c]
        ood_c = sood[y_ood == c] if y_ood is not None else np.array([])

        def blk(x):
            if len(x) == 0:
                return {"n": 0, "median": None, "iqr": None, "mean": None}
            q25, q75 = np.percentile(x, [25, 75])
            return {
                "n": int(len(x)),
                "median": float(np.median(x)),
                "iqr": float(q75 - q25),
                "mean": float(x.mean()),
            }

        rows.append(
            {
                "class": name,
                "class_idx": c,
                "in_pad": c in PAD_CLASSES,
                "id": blk(id_c),
                "ood": blk(ood_c),
            }
        )
    return rows


def extract_method(method, seed, splits, fitz_loader, device, bs, workers, dry_run, feat_dir):
    kind = "csg" if p1.is_csg(method) else ("effb3" if method == "effb3_control" else "baseline")
    npz = feat_dir / "{}_s{}.npz".format(method, seed)
    tf = build_val_transform_robust()
    max_b = 2 if dry_run else None

    def loader(frame):
        ds = SkinDataset(frame, transform=tf)
        if dry_run:
            ds.data = ds.data.iloc[: max_b * bs].reset_index(drop=True)
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0 if dry_run else workers)

    ckpt = p1.find_policy_ckpt(p1.method_ckpt_dir(method, seed))
    print("  ckpt", ckpt, flush=True)
    model, kind = p1.load_model(method, ckpt, device)
    packs = {}
    for name, frame in [("train", splits["isic_train"]), ("id", splits["isic_test"]), ("ood_pad", splits["pad"])]:
        packs[name] = collect_slim(model, kind, loader(frame), device, max_batches=max_b)
        print("   ", name, packs[name]["z"].shape, flush=True)
    packs["ood_fitz"] = collect_slim(model, kind, fitz_loader, device, max_batches=max_b)
    print("    ood_fitz", packs["ood_fitz"]["z"].shape, flush=True)
    if not dry_run:
        np.savez_compressed(
            npz,
            train_z=packs["train"]["z"],
            train_y=packs["train"]["labels"],
            train_logits=packs["train"]["logits"],
            id_z=packs["id"]["z"],
            id_y=packs["id"]["labels"],
            id_logits=packs["id"]["logits"],
            pad_z=packs["ood_pad"]["z"],
            pad_y=packs["ood_pad"]["labels"],
            pad_logits=packs["ood_pad"]["logits"],
            fitz_z=packs["ood_fitz"]["z"],
            fitz_logits=packs["ood_fitz"]["logits"],
            kind=np.array(kind),
        )
    return packs, kind


def load_npz(npz):
    d = np.load(npz, allow_pickle=True)
    kind = str(d["kind"])

    def pack(prefix, has_y=True):
        out = {"z": d[prefix + "_z"], "logits": d[prefix + "_logits"]}
        if has_y:
            out["labels"] = d[prefix + "_y"]
        else:
            out["labels"] = np.full(len(out["logits"]), -1)
        return out

    return {
        "train": pack("train"),
        "id": pack("id"),
        "ood_pad": pack("pad"),
        "ood_fitz": pack("fitz", has_y=False),
    }, kind


def analyse(method, seed, packs, kind, pad_p, isic_p):
    y_id = packs["id"]["labels"]
    y_pad = packs["ood_pad"]["labels"]
    six = np.isin(y_id, PAD_CLASSES)
    rec = {
        "method": method,
        "seed": seed,
        "kind": kind,
        "feat_dim": int(packs["train"]["z"].shape[1]),
        "n_id": int(len(y_id)),
        "n_id_6class": int(six.sum()),
        "n_pad": int(len(y_pad)),
        "n_fitz": int(len(packs["ood_fitz"]["z"])),
        "ood_sets": {},
    }
    for ood_name, ood_pack in [("pad", packs["ood_pad"]), ("fitzpatrick17k", packs["ood_fitz"])]:
        dets = detectors_from_packs(packs["train"], packs["id"], ood_pack)
        block = {}
        for det, (sid, sood) in dets.items():
            if np.isnan(sid).any() or np.isnan(sood).any():
                mets = {"AUROC": None, "AUPR_IN": None, "AUPR_OUT": None, "FPR95": None}
                mets6 = mets
            else:
                mets = p1.binary_ood_metrics(sid, sood)
                mets6 = p1.binary_ood_metrics(sid[six], sood) if six.any() else mets
            try:
                mets_w = weighted_auroc(sid, sood, y_id, pad_p, isic_p)
            except Exception:
                mets_w = None
            block[det] = {
                "unrestricted": mets,
                "id_6class_restricted": mets6,
                "id_reweighted_to_pad": mets_w,
            }
        rec["ood_sets"][ood_name] = block
        if ood_name == "pad":
            sid = dets["mahalanobis_classcond"][0]
            sood = dets["mahalanobis_classcond"][1]
            rec["per_class_maha"] = class_dist_table(sid, y_id, sood, y_pad)
    return rec


def run_one(args, method, seed, splits, fitz_loader, pad_p, isic_p, device, feat_dir, seed_dir):
    dest = seed_dir / "{}_s{}.json".format(method, seed)
    if args.skip_done and dest.exists() and not args.dry_run:
        print("skip_done", dest, flush=True)
        return json.loads(dest.read_text())
    npz = feat_dir / "{}_s{}.npz".format(method, seed)
    if args.skip_extract and npz.exists() and not args.dry_run:
        packs, kind = load_npz(npz)
    else:
        packs, kind = extract_method(
            method, seed, splits, fitz_loader, device, args.batch_size, args.num_workers, args.dry_run, feat_dir
        )
    rec = analyse(method, seed, packs, kind, pad_p, isic_p)
    maha = rec["ood_sets"]["fitzpatrick17k"]["mahalanobis_classcond"]["unrestricted"]["AUROC"]
    maha_pad = rec["ood_sets"]["pad"]["mahalanobis_classcond"]["unrestricted"]["AUROC"]
    maha6 = rec["ood_sets"]["pad"]["mahalanobis_classcond"]["id_6class_restricted"]["AUROC"]
    print("  fitz Maha", maha, "pad Maha", maha_pad, "pad 6class", maha6, flush=True)
    if not args.dry_run:
        dest.write_text(json.dumps(rec, indent=2, default=json_default) + "\n")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    feat_dir = OUT / "features"
    seed_dir = OUT / "per_seed"
    feat_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device", device, flush=True)
    splits = p1.build_split_frames(p1.METADATA)
    y_id_full = splits["isic_test"]["label_idx"].to_numpy()
    y_pad_full = splits["pad"]["label_idx"].to_numpy()
    isic_p = {c: float((y_id_full == c).mean()) for c in range(8)}
    pad_p = {c: float((y_pad_full == c).mean()) for c in range(8)}
    (OUT / "class_priors.json").write_text(
        json.dumps({"isic_test": isic_p, "pad": pad_p, "pad_classes": list(PAD_CLASSES), "labels": LABELS}, indent=2)
        + "\n"
    )

    fitz_paths = sorted(FITZ_DIR.glob("*.jpg"))
    print("fitz images", len(fitz_paths), flush=True)
    tf = build_val_transform_robust()
    fds = FolderJpegDataset(fitz_paths, tf)
    if args.dry_run:
        fds.paths = fds.paths[: args.batch_size * 2]
    fitz_loader = DataLoader(
        fds, batch_size=args.batch_size, shuffle=False, num_workers=0 if args.dry_run else args.num_workers
    )

    methods = [args.method] if args.method else list(METHODS)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    rows = []
    for method in methods:
        for seed in seeds:
            print("===", method, seed, flush=True)
            rec = run_one(args, method, seed, splits, fitz_loader, pad_p, isic_p, device, feat_dir, seed_dir)
            rows.append(rec)

    if args.dry_run:
        (OUT / "phase25_dryrun.json").write_text(json.dumps(rows, indent=2, default=json_default) + "\n")
        print("dry_run ok", len(rows))


if __name__ == "__main__":
    main()
