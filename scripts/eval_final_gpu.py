#!/usr/bin/env python3
"""Paper B final package — GPU inference.

Phase 6: 6-class-restricted zero-shot transfer on pad_heldout only
         (716 images / 412 patients). Never pad_full / pad_adv.

Phase 3.5: z_lesion centroid / kNN / norm geometry at λ ∈ {0, 0.25, 2, 8}.

PAD labels never entered L_cls (ignore_index=-1). This is zero-shot
cross-domain transfer. Do not retune anything.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_ood_dual_branch as p1
import train_phase2_pad_holdout as p2
import train_phase3_sweep as p3
from eval_fitz_domain_axis import FolderJpegDataset
from src.datasets.constants import INDEX_TO_LABEL, LABELS
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.models.csg_lightning import CSGLiteLightning
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
P3_ROOT = PAPERB / "results" / "paperB" / "phase3_sweep"
FITZ_DIR = PAPERB / "data" / "fitzpatrick17k" / "images"
XFER_DIR = PAPERB / "results" / "paperB" / "phase6_xfer"
MECH_DIR = PAPERB / "results" / "paperB" / "phase35_mech"

PAD_CLASSES = np.array([0, 1, 2, 3, 4, 7], dtype=int)  # MEL NV BCC AK BKL SCC
PAD_NAMES = [LABELS[i] for i in PAD_CLASSES]
MECH_LAMBDAS = {0.0, 0.25, 2.0, 8.0}
N_BOOT = 2000

CSG_JOBS = list(p3.JOBS)  # 27
SEEDS = (42, 52, 62, 72, 82)

JOBS = []
for lam, seed in CSG_JOBS:
    JOBS.append({"kind": "csg", "lambda_adv": float(lam), "seed": int(seed)})
for seed in SEEDS:
    JOBS.append({"kind": "baseline_soft", "lambda_adv": None, "seed": int(seed)})
for seed in SEEDS:
    JOBS.append({"kind": "effb3_control", "lambda_adv": None, "seed": int(seed)})
assert len(JOBS) == 37


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def dist_summary(arr):
    arr = np.asarray(arr, dtype=np.float64)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "median": float(med),
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
    }


def bootstrap_auroc(sid, sood, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n_id, n_ood = len(sid), len(sood)
    y = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = np.concatenate([sid[rng.integers(0, n_id, n_id)], sood[rng.integers(0, n_ood, n_ood)]])
        vals[i] = roc_auc_score(y, s)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point": float(roc_auc_score(y, np.concatenate([sid, sood]))),
        "boot_mean": float(vals.mean()),
        "ci95": [float(lo), float(hi)],
        "n_boot": int(n),
        "n_id": int(n_id),
        "n_ood": int(n_ood),
    }


def bootstrap_xy(y, pred_or_score, fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n_s = len(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, n_s, n_s)
        try:
            v = fn(y[i], pred_or_score[i])
        except ValueError:
            continue
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        vals.append(float(v))
    if not vals:
        return {"point": float("nan"), "boot_mean": float("nan"), "ci95": [None, None], "n_boot": 0}
    arr = np.asarray(vals)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    try:
        point = float(fn(y, pred_or_score))
    except ValueError:
        point = float(arr.mean())
    return {
        "point": point,
        "boot_mean": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "n_boot": int(len(arr)),
        "n": int(n_s),
    }


def melanoma_sens_at_spec(y, score, spec_target=0.85):
    y_bin = (np.asarray(y) == 0).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    out = {
        "spec_target": spec_target,
        "n_pos_mel": n_pos,
        "n_neg": n_neg,
        "note": "pad_heldout melanoma n=9; do not overclaim",
    }
    if n_pos == 0 or n_neg == 0:
        out["sens"] = None
        return out
    fpr, tpr, thr = roc_curve(y_bin, score)
    spec = 1.0 - fpr
    ok = spec >= spec_target
    if not np.any(ok):
        out["sens"] = float("nan")
        out["achieved_spec"] = float(spec.max())
        out["threshold"] = None
        return out
    idx = int(np.where(ok)[0][-1])
    out["sens"] = float(tpr[idx])
    out["achieved_spec"] = float(spec[idx])
    out["threshold"] = float(thr[idx]) if idx < len(thr) else None
    return out


def six_class_metrics(logits, labels, boot_seed=0):
    """PAD has 6 of 8 ISIC classes. Softmax over those 6; report the restriction."""
    y = np.asarray(labels).astype(int)
    logits = np.asarray(logits)
    logits6 = logits[:, PAD_CLASSES]
    P = softmax_np(logits6)
    pred = PAD_CLASSES[P.argmax(1)]
    y6 = np.array([{c: i for i, c in enumerate(PAD_CLASSES)}[int(v)] for v in y], dtype=int)
    rec = recall_score(y, pred, labels=PAD_CLASSES, average=None, zero_division=0)
    rec_map = {PAD_NAMES[i]: float(rec[i]) for i in range(len(PAD_CLASSES))}
    try:
        macro_auc = float(roc_auc_score(y6, P, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    mel = melanoma_sens_at_spec(y, P[:, 0], 0.85)
    acc_boot = bootstrap_xy(y, pred, lambda a, b: float(accuracy_score(a, b)), seed=boot_seed)
    bal_boot = bootstrap_xy(y, pred, lambda a, b: float(balanced_accuracy_score(a, b)), seed=boot_seed + 1)

    return {
        "subset": "pad_heldout",
        "n_images": int(len(y)),
        "n_patients": 412,
        "classes_evaluated": PAD_NAMES,
        "classes_missing_from_PAD": ["DF", "VASC"],
        "restriction": "6-class: softmax and argmax over MEL,NV,BCC,AK,BKL,SCC (indices 0,1,2,3,4,7). DF and VASC are absent from PAD.",
        "zero_shot": "PAD labels never entered L_cls (ignore_index=-1)",
        "class_counts": {PAD_NAMES[i]: int((y == PAD_CLASSES[i]).sum()) for i in range(len(PAD_CLASSES))},
        "accuracy": acc_boot,
        "balanced_accuracy": bal_boot,
        "macro_auc_ovr": {
            "point": macro_auc,
            "n": int(len(y)),
        },
        "per_class_recall": rec_map,
        "melanoma_sens_at_spec_0.85": mel,
        "pred_class_counts": {PAD_NAMES[i]: int((pred == PAD_CLASSES[i]).sum()) for i in range(len(PAD_CLASSES))},
    }


def nearest_centroid_l2(z, means, present):
    prot = means[present]
    d = np.linalg.norm(z[:, None, :].astype(np.float64) - prot[None, :, :].astype(np.float64), axis=2)
    return d.min(axis=1)


def load_csg(ckpt, device):
    try:
        lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), map_location="cpu")
        net = lit.model.to(device).eval()
        return net
    except Exception as e:
        print("load_from_checkpoint failed, falling back to state_dict:", e, flush=True)
        net, _kind = p1.load_model("runB_orth1", Path(ckpt), device)
        return net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.task_id < 0 or args.task_id >= len(JOBS):
        raise SystemExit("task_id out of range")
    job = JOBS[args.task_id]
    seed_everything(job["seed"])
    XFER_DIR.mkdir(parents=True, exist_ok=True)
    MECH_DIR.mkdir(parents=True, exist_ok=True)

    if job["kind"] == "csg":
        run_name = "runB_orth1_ladv{}_s{}".format(p3.lam_tag(job["lambda_adv"]), job["seed"])
        tag = run_name
    else:
        tag = "{}_s{}".format(job["kind"], job["seed"])

    xfer_path = XFER_DIR / "{}.json".format(tag)
    mech_path = MECH_DIR / "{}.json".format(tag)
    need_mech = job["kind"] == "csg" and job["lambda_adv"] in MECH_LAMBDAS
    need_mech_json = job["kind"] == "csg"
    if args.skip_done and xfer_path.exists() and (not need_mech_json or mech_path.exists()):
        print("skip_done", tag, flush=True)
        return

    if args.dry_run:
        print("dry_run", job, "xfer", str(xfer_path), "mech", need_mech)
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device", device, "job", job, flush=True)

    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, _val, test = p2.isic_splits(isic)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    if len(pad_hold) != 716:
        raise RuntimeError("pad_heldout size {} != 716".format(len(pad_hold)))
    eval_tf = build_val_transform_robust()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), args.batch_size, args.num_workers, False)

    if job["kind"] == "csg":
        summary_path = P3_ROOT / tag / "summary.json"
        if not summary_path.exists():
            raise RuntimeError("missing Phase 3 summary {}".format(summary_path))
        summary = json.loads(summary_path.read_text())
        ckpt = Path(summary["best_checkpoint"])
        if not ckpt.exists():
            raise RuntimeError("missing ckpt {}".format(ckpt))
        net = load_csg(ckpt, device)
        hold_pack = p3.collect_all(net, loader(pad_hold), device)
        xfer = six_class_metrics(hold_pack["logits"], hold_pack["labels"], boot_seed=job["seed"])
        xfer_out = {
            "kind": "csg",
            "run_name": tag,
            "seed": job["seed"],
            "lambda_adv": job["lambda_adv"],
            "checkpoint": str(ckpt),
            "reused_phase2_ckpt": bool(summary.get("reused_phase2_ckpt")),
            "pad_heldout_6class": xfer,
        }
        xfer_path.write_text(json.dumps(xfer_out, indent=2) + "\n")
        print("saved", xfer_path, "bal", xfer["balanced_accuracy"]["point"], flush=True)

        train_pack = p3.collect_all(net, loader(train), device)
        id_pack = p3.collect_all(net, loader(test), device)
        ztr, ytr = train_pack["z_lesion"], train_pack["labels"]
        zid = id_pack["z_lesion"]
        zhold = hold_pack["z_lesion"]

        sid, shold = p3.maha_scores(ztr, ytr, zid, zhold)
        strain, _ = p3.maha_scores(ztr, ytr, ztr, zhold[:1])
        means, present = p1.fit_class_means(ztr, ytr, 8)
        euc_tr = nearest_centroid_l2(ztr, means, present)
        euc_id = nearest_centroid_l2(zid, means, present)
        euc_hold = nearest_centroid_l2(zhold, means, present)
        knn = p1.fit_knn(ztr, k=50)
        knn_tr = p1.knn_scores(knn, ztr)
        knn_id = p1.knn_scores(knn, zid)
        knn_hold = p1.knn_scores(knn, zhold)
        norm_tr = np.linalg.norm(ztr, axis=1)
        norm_id = np.linalg.norm(zid, axis=1)
        norm_hold = np.linalg.norm(zhold, axis=1)

        groups = {
            "isic_train": {"maha": strain, "euc": euc_tr, "knn": knn_tr, "norm": norm_tr},
            "isic_test": {"maha": sid, "euc": euc_id, "knn": knn_id, "norm": norm_id},
            "pad_heldout": {"maha": shold, "euc": euc_hold, "knn": knn_hold, "norm": norm_hold},
        }

        fitz_pack = None
        if need_mech:
            fitz_paths = sorted(FITZ_DIR.glob("*.jpg"))
            fitz_ds = FolderJpegDataset(fitz_paths, eval_tf)
            fitz_loader = DataLoader(
                fitz_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            fitz_pack = p3.collect_all(net, fitz_loader, device)
            zf = fitz_pack["z_lesion"]
            _, sf = p3.maha_scores(ztr, ytr, zid, zf)
            euc_f = nearest_centroid_l2(zf, means, present)
            knn_f = p1.knn_scores(knn, zf)
            norm_f = np.linalg.norm(zf, axis=1)
            groups["fitzpatrick17k"] = {"maha": sf, "euc": euc_f, "knn": knn_f, "norm": norm_f}

        six = np.isin(id_pack["labels"], PAD_CLASSES)
        mech = {
            "kind": "csg",
            "run_name": tag,
            "seed": job["seed"],
            "lambda_adv": job["lambda_adv"],
            "checkpoint": str(ckpt),
            "feature": "z_lesion",
            "fit": "class centroids / Maha / kNN bank fit on ISIC train only",
            "distance_definitions": {
                "maha": "min squared Mahalanobis to class-conditional centroids (same as OOD detector)",
                "euc": "Euclidean L2 to nearest class mean (independent of covariance)",
                "knn": "mean distance to k=50 nearest ISIC-train neighbours, cosine-normalised (same as Phase 1 knn_k50)",
                "norm": "L2 feature norm of z_lesion (scaling control)",
            },
            "ood_auroc_pad_heldout": {
                "maha": bootstrap_auroc(sid, shold, seed=job["seed"]),
                "euc": bootstrap_auroc(euc_id, euc_hold, seed=job["seed"] + 11),
                "knn": bootstrap_auroc(knn_id, knn_hold, seed=job["seed"] + 17),
            },
            "ood_auroc_pad_heldout_id6class": {
                "maha": bootstrap_auroc(sid[six], shold, seed=job["seed"] + 3) if six.any() else None
            },
            "id_balanced_acc": bootstrap_xy(
                id_pack["labels"],
                id_pack["logits"].argmax(1),
                lambda a, b: float(balanced_accuracy_score(a, b)),
                seed=job["seed"] + 5,
            ),
            "groups": {g: {k: dist_summary(v) for k, v in d.items()} for g, d in groups.items()},
            "ratios_vs_isic_test": {},
        }
        if fitz_pack is not None:
            mech["ood_auroc_fitz"] = {
                "maha": bootstrap_auroc(sid, groups["fitzpatrick17k"]["maha"], seed=job["seed"] + 7)
            }
            _, sctx_f = p3.maha_scores(
                train_pack["z_context"], train_pack["labels"], id_pack["z_context"], fitz_pack["z_context"]
            )
            sctx_id, sctx_hold = p3.maha_scores(
                train_pack["z_context"], train_pack["labels"], id_pack["z_context"], hold_pack["z_context"]
            )
            mech["z_context_ood_auroc_pad_heldout"] = bootstrap_auroc(sctx_id, sctx_hold, seed=job["seed"] + 9)
            mech["z_context_ood_auroc_fitz"] = bootstrap_auroc(sctx_id, sctx_f, seed=job["seed"] + 13)

        id_sum = mech["groups"]["isic_test"]
        for gname, g in mech["groups"].items():
            if gname == "isic_test":
                continue
            ratios = {}
            for metric in ("maha", "euc", "knn", "norm"):
                med_id = id_sum[metric]["median"]
                iqr_id = id_sum[metric]["iqr"]
                ratios[metric] = {
                    "median_ratio": float(g[metric]["median"] / med_id) if med_id > 0 else None,
                    "iqr_ratio": float(g[metric]["iqr"] / iqr_id) if iqr_id > 0 else None,
                }
            mech["ratios_vs_isic_test"][gname] = ratios

        npz_path = MECH_DIR / "{}_dists.npz".format(tag)
        npz = {
            "maha_isic_train": groups["isic_train"]["maha"],
            "maha_isic_test": groups["isic_test"]["maha"],
            "maha_pad_heldout": groups["pad_heldout"]["maha"],
            "euc_isic_train": groups["isic_train"]["euc"],
            "euc_isic_test": groups["isic_test"]["euc"],
            "euc_pad_heldout": groups["pad_heldout"]["euc"],
            "knn_isic_train": groups["isic_train"]["knn"],
            "knn_isic_test": groups["isic_test"]["knn"],
            "knn_pad_heldout": groups["pad_heldout"]["knn"],
            "norm_isic_train": groups["isic_train"]["norm"],
            "norm_isic_test": groups["isic_test"]["norm"],
            "norm_pad_heldout": groups["pad_heldout"]["norm"],
        }
        if "fitzpatrick17k" in groups:
            npz["maha_fitzpatrick17k"] = groups["fitzpatrick17k"]["maha"]
            npz["euc_fitzpatrick17k"] = groups["fitzpatrick17k"]["euc"]
            npz["knn_fitzpatrick17k"] = groups["fitzpatrick17k"]["knn"]
            npz["norm_fitzpatrick17k"] = groups["fitzpatrick17k"]["norm"]
        np.savez_compressed(npz_path, **npz)
        mech["dists_npz"] = str(npz_path)
        # Always write a mech json for CSG (ratios + bootstrap); full Fitz only at MECH_LAMBDAS.
        mech["full_mechanism"] = bool(need_mech)
        mech_path.write_text(json.dumps(mech, indent=2) + "\n")
        print("saved", mech_path, flush=True)
        return

    # baseline / EffB3: pad_heldout logits only
    ckpt = p1.find_policy_ckpt(p1.method_ckpt_dir(job["kind"], job["seed"]))
    if ckpt is None:
        raise RuntimeError("missing {} seed {} ckpt".format(job["kind"], job["seed"]))
    net, kind = p1.load_model(job["kind"], ckpt, device)
    pack = p1.collect_single_encoder(net, loader(pad_hold), device, kind)
    xfer = six_class_metrics(pack["logits"], pack["labels"], boot_seed=job["seed"])
    out = {
        "kind": job["kind"],
        "run_name": tag,
        "seed": job["seed"],
        "lambda_adv": None,
        "checkpoint": str(ckpt),
        "pad_heldout_6class": xfer,
    }
    xfer_path.write_text(json.dumps(out, indent=2) + "\n")
    print("saved", xfer_path, "bal", xfer["balanced_accuracy"]["point"], flush=True)


if __name__ == "__main__":
    main()
