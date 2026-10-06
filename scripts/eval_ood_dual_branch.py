#!/usr/bin/env python3
"""Paper B Phase 1: dual-branch OOD on existing checkpoints.

Copied/adapted from cbm_revision/scripts/eval_ood_benchmarks.py so that file
stays untouched. New outputs only under --output_dir (default: this tree's
results/paperB/).

Honesty: z_context spaces and the context_predictor detector are
domain-supervised. Every CSV row for those carries supervision=domain-supervised.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

# torch-env ships a broken transformers/tokenizers pair. torchmetrics imports
# bert_score at package import time; torchvision/dynamo also probes transformers.
# Provide a minimal stub that satisfies both without touching torch-env.
if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = object
    _tf.AutoTokenizer = object
    _cfg_mod = types.ModuleType("transformers.configuration_utils")

    class _PretrainedConfig:
        def __eq__(self, other):
            return False

    _cfg_mod.PretrainedConfig = _PretrainedConfig
    _tf.configuration_utils = _cfg_mod
    sys.modules["transformers"] = _tf
    sys.modules["transformers.configuration_utils"] = _cfg_mod

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.datasets.constants import LABEL_TO_INDEX
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.models.baseline import BaselineNet
from src.models.csg_lite import get_csg_lite
from src.models.effb3_single import EffB3SingleNet
from src.utils import ood_metrics
from src.utils.seed import seed_everything

SEEDS = (42, 52, 62, 72, 82)
METHODS = ("baseline_soft", "effb3_control", "runA_grl", "runB_orth1")
METADATA = REPO / "data" / "master_metadata_lesion_only_soft.csv"
DEFAULT_OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB")

# Feature spaces requested by the work order.
CSG_FEATURE_SPACES = (
    "z_lesion",
    "z_context",
    "concat_z",
    "backbone_raw_lesion",
    "backbone_raw_context",
    "logits",
)
SINGLE_FEATURE_SPACES = ("backbone_raw", "logits")

# Embedding detectors (fit on ISIC train features only).
EMBED_DETECTORS = (
    "cosine_max",
    "mahalanobis_classcond",
    "mahalanobis_agnostic",
    "knn_k50",
)
# Logit detectors (no train-set fit except ODIN is input-space).
LOGIT_DETECTORS = (
    "MSP",
    "Energy_T1",
    "ODIN_T1_e0",
    "ODIN_T10_e0",
    "ODIN_T100_e0",
    "ODIN_T1_e0014",
    "ODIN_T10_e0014",
    "ODIN_T100_e0014",
)
CONTEXT_DETECTOR = "context_predictor_softmax"

DOMAIN_SUPERVISED_SPACES = frozenset({"z_context", "backbone_raw_context"})
DOMAIN_SUPERVISED_DETECTORS = frozenset({CONTEXT_DETECTOR})


# ---------------------------------------------------------------------------
# Checkpoint policy (Phase 0.1): newest best-*.ckpt, else newest last*.ckpt
# ---------------------------------------------------------------------------


def find_policy_ckpt(dir_path: Path):
    if not dir_path.exists():
        return None
    best = sorted(dir_path.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = sorted(dir_path.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return last[0] if last else None


def method_ckpt_dir(method: str, seed: int) -> Path:
    if method == "baseline_soft":
        return REPO / "checkpoints" / "baseline" / f"baseline_soft_s{seed}"
    if method == "effb3_control":
        return REPO / "checkpoints" / "effb3_single" / f"effb3_single_s{seed}"
    if method == "runA_grl":
        return REPO / "checkpoints" / "csg_lite" / f"runA_grl_s{seed}"
    if method == "runB_orth1":
        return REPO / "checkpoints" / "csg_lite" / f"runB_orth1_s{seed}"
    raise ValueError(method)


def is_csg(method: str) -> bool:
    return method in {"runA_grl", "runB_orth1"}


# ---------------------------------------------------------------------------
# Model loaders — raw state_dict, pretrained=False (no ImageNet download)
# ---------------------------------------------------------------------------


def _strip_prefix(state, prefix):
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out


def load_model(method: str, ckpt: Path, device):
    try:
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(str(ckpt), map_location="cpu")
    sd = raw.get("state_dict", raw)
    hp = raw.get("hyper_parameters", {}) or {}
    if method == "baseline_soft":
        net = BaselineNet(num_classes=8, pretrained=False)
        loaded = _strip_prefix(sd, "net.")
        net.load_state_dict(loaded, strict=True)
        kind = "baseline"
    elif method == "effb3_control":
        net = EffB3SingleNet(
            num_classes=8,
            latent_dim=int(hp.get("latent_dim", 16)),
            pretrained=False,
        )
        loaded = _strip_prefix(sd, "net.")
        net.load_state_dict(loaded, strict=True)
        kind = "effb3"
    else:
        net = get_csg_lite(
            lesion_classes=8,
            domain_classes=2,
            lesion_latent_dim=int(hp.get("lesion_latent_dim", 16)),
            context_latent_dim=int(hp.get("context_latent_dim", 64)),
            pretrained=False,
            backbone_variant=str(hp.get("backbone_variant", "b3")),
        )
        loaded = _strip_prefix(sd, "model.")
        net.load_state_dict(loaded, strict=True)
        kind = "csg"
    net = net.to(device).eval()
    return net, kind


# ---------------------------------------------------------------------------
# Splits — ISIC train only for statistics. Runtime assertions.
# ---------------------------------------------------------------------------


def _remap_legacy_path(path: str) -> str:
    """CSV was written on /mnt/data2/Vinh; this host mounts the same tree at /data2/hpcshared/Vinh."""
    p = str(path)
    if p.startswith("/mnt/data2/Vinh/"):
        return "/data2/hpcshared/Vinh/" + p[len("/mnt/data2/Vinh/") :]
    return p


def load_filtered_master(metadata_csv: Path):
    import os

    import pandas as pd

    df = pd.read_csv(metadata_csv)
    required = {"path", "label", "domain"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required metadata columns: {}".format(sorted(missing)))
    df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
    df["path"] = df["path"].map(_remap_legacy_path)
    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    df = df[df["path"].map(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with existing image paths after remapping /mnt/data2/Vinh → /data2/hpcshared/Vinh.")
    return df


def build_split_frames(metadata_csv: Path):
    df = load_filtered_master(metadata_csv)
    isic = df[df["domain"] == "isic"].copy()
    pad = df[df["domain"] == "pad_ufes"].copy()
    isic_tv, isic_test = train_test_split(
        isic, test_size=0.2, stratify=isic["label_idx"], random_state=42
    )
    isic_train, isic_val = train_test_split(
        isic_tv, test_size=0.2, stratify=isic_tv["label_idx"], random_state=42
    )
    isic_train = isic_train.reset_index(drop=True)
    isic_val = isic_val.reset_index(drop=True)
    isic_test = isic_test.reset_index(drop=True)
    pad = pad.reset_index(drop=True)

    train_paths = set(isic_train["path"].astype(str))
    val_paths = set(isic_val["path"].astype(str))
    test_paths = set(isic_test["path"].astype(str))
    pad_paths = set(pad["path"].astype(str))

    # Hygiene assertions — these fire before any covariance is touched.
    assert train_paths.isdisjoint(val_paths), "ISIC train overlaps val"
    assert train_paths.isdisjoint(test_paths), "ISIC train overlaps test"
    assert train_paths.isdisjoint(pad_paths), "ISIC train overlaps PAD"
    assert (isic_train["domain"] == "isic").all()
    assert (pad["domain"] == "pad_ufes").all()
    assert isic_train["label_idx"].min() >= 0
    return {
        "isic_train": isic_train,
        "isic_val": isic_val,
        "isic_test": isic_test,
        "pad": pad,
        "paths": {
            "isic_train": train_paths,
            "isic_val": val_paths,
            "isic_test": test_paths,
            "pad": pad_paths,
        },
    }


def assert_fit_frame_is_isic_train(frame, splits, tag):
    """Runtime assertion: every mean/cov/centroid/kNN bank is ISIC-train only."""
    if tag != "isic_train":
        raise RuntimeError("Statistics fit tag must be 'isic_train', got {}".format(tag))
    paths = set(frame["path"].astype(str))
    if paths != splits["paths"]["isic_train"]:
        extra = paths - splits["paths"]["isic_train"]
        missing = splits["paths"]["isic_train"] - paths
        raise RuntimeError(
            "Fit frame is not exactly ISIC train. extra={} missing={}".format(len(extra), len(missing))
        )
    if not (frame["domain"] == "isic").all():
        raise RuntimeError("Fit frame contains non-ISIC rows")
    forbidden = splits["paths"]["isic_val"] | splits["paths"]["isic_test"] | splits["paths"]["pad"]
    if paths & forbidden:
        raise RuntimeError("Fit frame leaked val/test/PAD paths: n={}".format(len(paths & forbidden)))


def make_loader(frame, transform, batch_size, num_workers, max_batches=None):
    ds = SkinDataset(frame, transform=transform)
    if max_batches is not None:
        n = min(len(ds), max_batches * batch_size)
        ds.data = ds.data.iloc[:n].reset_index(drop=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_single_encoder(model, loader, device, kind, max_batches=None):
    """Baseline / EffB3: backbone_raw + logits + labels."""
    logits_l, feat_l, y_l = [], [], []
    model.eval()
    for b_idx, (images, y) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        if kind == "effb3":
            x_in = model._rgb_to_gray3(images)
            backbone = model.backbone(x_in)
            z = model.bn(model.projector(backbone))
            logits = model.classifier(z)
            feat = backbone
        else:
            logits, feat = model(images, return_features=True)
        logits_l.append(logits.cpu())
        feat_l.append(feat.cpu())
        y_l.append(y)
    return {
        "logits": torch.cat(logits_l).numpy(),
        "backbone_raw": torch.cat(feat_l).numpy(),
        "labels": torch.cat(y_l).numpy(),
        "d_ctx": None,
    }


@torch.no_grad()
def collect_csg(model, loader, device, max_batches=None):
    logits_l, zl, zc, bl, bc, dctx, y_l = [], [], [], [], [], [], []
    model.eval()
    for b_idx, (images, y) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        x_les = model._rgb_to_gray3(images)
        feat_ctx = model.extract_context_features(images)
        feat_les = model.extract_lesion_features(x_les)
        z_lesion = model.lesion_bn(model.lesion_projector(feat_les))
        z_context = model.context_projector(feat_ctx)
        logits = model.lesion_classifier(z_lesion)
        d_ctx = model.context_predictor(z_context)
        logits_l.append(logits.cpu())
        zl.append(z_lesion.cpu())
        zc.append(z_context.cpu())
        bl.append(feat_les.cpu())
        bc.append(feat_ctx.cpu())
        dctx.append(d_ctx.cpu())
        y_l.append(y)
    z_lesion = torch.cat(zl).numpy()
    z_context = torch.cat(zc).numpy()
    return {
        "logits": torch.cat(logits_l).numpy(),
        "z_lesion": z_lesion,
        "z_context": z_context,
        "concat_z": np.concatenate([z_lesion, z_context], axis=1),
        "backbone_raw_lesion": torch.cat(bl).numpy(),
        "backbone_raw_context": torch.cat(bc).numpy(),
        "d_ctx": torch.cat(dctx).numpy(),
        "labels": torch.cat(y_l).numpy(),
    }


def collect_features(model, kind, loader, device, max_batches=None):
    if kind == "csg":
        return collect_csg(model, loader, device, max_batches=max_batches)
    return collect_single_encoder(model, loader, device, kind, max_batches=max_batches)


# ---------------------------------------------------------------------------
# ODIN (needs grad; separate from no-grad collection)
# ---------------------------------------------------------------------------


def _forward_logits(model, kind, images):
    if kind == "csg":
        out = model(images, x_lesion=None, return_latents=False)
        return out[0]
    return model(images)


def odin_scores(model, kind, loader, device, temperature, eps, max_batches=None):
    """Higher score = more OOD (negative max temperature-scaled softmax)."""
    model.eval()
    scores = []
    for b_idx, (images, _y) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        if eps <= 0:
            with torch.no_grad():
                logits = _forward_logits(model, kind, images)
                s = -torch.softmax(logits / temperature, dim=1).max(dim=1).values
            scores.append(s.cpu().numpy())
            continue
        images = images.clone().detach().requires_grad_(True)
        logits = _forward_logits(model, kind, images)
        pred = logits.argmax(dim=1)
        logp = F.log_softmax(logits / temperature, dim=1)
        loss = -logp.gather(1, pred.unsqueeze(1)).sum()
        model.zero_grad(set_to_none=True)
        if images.grad is not None:
            images.grad.zero_()
        loss.backward()
        perturb = eps * images.grad.detach().sign()
        with torch.no_grad():
            x_hat = images.detach() - perturb
            logits2 = _forward_logits(model, kind, x_hat)
            s = -torch.softmax(logits2 / temperature, dim=1).max(dim=1).values
        scores.append(s.cpu().numpy())
        model.zero_grad(set_to_none=True)
    return np.concatenate(scores, axis=0)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _fpr95(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) if idx.size else 1.0


def binary_ood_metrics(id_scores, ood_scores):
    """id_scores / ood_scores: higher = more OOD."""
    y = np.concatenate(
        [np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)]
    )
    s = np.concatenate([id_scores, ood_scores])
    y_id = 1 - y
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPR_IN": float(average_precision_score(y_id, -s)),
        "AUPR_OUT": float(average_precision_score(y, s)),
        "FPR95": _fpr95(y, s),
    }


def fit_class_means(z, y, n_classes):
    means = np.zeros((n_classes, z.shape[1]), dtype=np.float64)
    present = []
    for c in range(n_classes):
        m = y == c
        if not np.any(m):
            continue
        means[c] = z[m].mean(axis=0)
        present.append(c)
    if not present:
        raise RuntimeError("No class present for centroid fit")
    return means, present


def cosine_ood_scores(z, means, present):
    prot = means[present]
    prot = prot / (np.linalg.norm(prot, axis=1, keepdims=True) + 1e-12)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return -(zn @ prot.T).max(axis=1)


def fit_maha_classcond(z, y, n_classes, reg_eps):
    return ood_metrics.compute_mahalanobis_params_from_arrays(
        z, y, num_classes=n_classes, reg_eps=reg_eps
    )


def fit_maha_agnostic(z, reg_eps):
    mu = z.mean(axis=0).astype(np.float64)
    centered = z.astype(np.float64) - mu
    n = max(z.shape[0] - 1, 1)
    cov = (centered.T @ centered) / n
    cov = cov + reg_eps * np.eye(z.shape[1], dtype=np.float64)
    precision = np.linalg.inv(cov).astype(np.float32)
    return mu.astype(np.float32), precision


def maha_agnostic_scores(z, mu, precision):
    p = precision.astype(np.float64)
    mu = mu.astype(np.float64)
    delta = z.astype(np.float64) - mu
    # (x-μ)^T Σ^{-1} (x-μ) row-wise
    return np.einsum("ni,ij,nj->n", delta, p, delta).astype(np.float64)


def fit_knn(z, k=50):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=min(k, len(zn)), algorithm="auto", metric="euclidean")
    nn.fit(zn)
    return nn


def knn_scores(nn, z):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    dist, _ = nn.kneighbors(zn)
    return dist.mean(axis=1)


def supervision_label(feature_space, detector):
    if feature_space in DOMAIN_SUPERVISED_SPACES or detector in DOMAIN_SUPERVISED_DETECTORS:
        return "domain-supervised"
    return "unsupervised"


# ---------------------------------------------------------------------------
# One (method, seed)
# ---------------------------------------------------------------------------


def eval_one(method, seed, model, kind, loaders, device, dry_run, max_batches, reg_eps):
    id_loader = loaders["id"]
    ood_loader = loaders["ood"]
    train_loader = loaders["train"]

    train_pack = collect_features(model, kind, train_loader, device, max_batches=max_batches)
    id_pack = collect_features(model, kind, id_loader, device, max_batches=max_batches)
    ood_pack = collect_features(model, kind, ood_loader, device, max_batches=max_batches)

    y_tr = train_pack["labels"]
    if (y_tr < 0).any():
        raise RuntimeError("Train features contain ignore_index labels — PAD leaked into fit")
    n_classes = int(id_pack["logits"].shape[1])

    pred = id_pack["logits"].argmax(axis=1)
    id_acc = float((pred == id_pack["labels"]).mean())
    id_bal = float(balanced_accuracy_score(id_pack["labels"], pred))

    spaces = CSG_FEATURE_SPACES if kind == "csg" else SINGLE_FEATURE_SPACES
    rows = []

    # --- embedding detectors, fit on train only ---
    for space in spaces:
        if space == "logits":
            continue
        ztr, zid, zood = train_pack[space], id_pack[space], ood_pack[space]
        means, present = fit_class_means(ztr, y_tr, n_classes)
        s_id_cos = cosine_ood_scores(zid, means, present)
        s_ood_cos = cosine_ood_scores(zood, means, present)

        maha_eps = 1e-3 if ztr.shape[1] >= 64 else reg_eps
        try:
            cmeans, prec = fit_maha_classcond(ztr, y_tr, n_classes, maha_eps)
            s_id_mcc = ood_metrics.mahalanobis_min_squared_distances(zid, cmeans, prec)
            s_ood_mcc = ood_metrics.mahalanobis_min_squared_distances(zood, cmeans, prec)
            mcc_ok = True
        except Exception as exc:
            mcc_ok = False
            mcc_err = str(exc)

        mu, prec_a = fit_maha_agnostic(ztr, maha_eps)
        s_id_mag = maha_agnostic_scores(zid, mu, prec_a)
        s_ood_mag = maha_agnostic_scores(zood, mu, prec_a)

        knn = fit_knn(ztr, k=50)
        s_id_knn = knn_scores(knn, zid)
        s_ood_knn = knn_scores(knn, zood)

        packed = [
            ("cosine_max", s_id_cos, s_ood_cos),
            ("mahalanobis_agnostic", s_id_mag, s_ood_mag),
            ("knn_k50", s_id_knn, s_ood_knn),
        ]
        if mcc_ok:
            packed.append(("mahalanobis_classcond", s_id_mcc, s_ood_mcc))
        for det, sid, sood in packed:
            mets = binary_ood_metrics(sid, sood)
            for metric, val in mets.items():
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "feature_space": space,
                        "detector": det,
                        "metric": metric,
                        "value": val,
                        "supervision": supervision_label(space, det),
                        "n_id": int(len(sid)),
                        "n_ood": int(len(sood)),
                        "feat_dim": int(ztr.shape[1]),
                    }
                )
        if not mcc_ok:
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "feature_space": space,
                    "detector": "mahalanobis_classcond",
                    "metric": "ERROR",
                    "value": float("nan"),
                    "supervision": supervision_label(space, "mahalanobis_classcond"),
                    "n_id": int(len(zid)),
                    "n_ood": int(len(zood)),
                    "feat_dim": int(ztr.shape[1]),
                    "error": mcc_err,
                }
            )

    # --- logit detectors ---
    p_id = torch.softmax(torch.from_numpy(id_pack["logits"]), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(ood_pack["logits"]), dim=1).numpy()
    logit_pairs = {
        "MSP": (-p_id.max(axis=1), -p_ood.max(axis=1)),
        "Energy_T1": (
            (-torch.logsumexp(torch.from_numpy(id_pack["logits"]), dim=1)).numpy(),
            (-torch.logsumexp(torch.from_numpy(ood_pack["logits"]), dim=1)).numpy(),
        ),
    }
    odin_grid = [
        ("ODIN_T1_e0", 1.0, 0.0),
        ("ODIN_T10_e0", 10.0, 0.0),
        ("ODIN_T100_e0", 100.0, 0.0),
        ("ODIN_T1_e0014", 1.0, 0.0014),
        ("ODIN_T10_e0014", 10.0, 0.0014),
        ("ODIN_T100_e0014", 100.0, 0.0014),
    ]
    for name, T, eps in odin_grid:
        logit_pairs[name] = (
            odin_scores(model, kind, id_loader, device, T, eps, max_batches=max_batches),
            odin_scores(model, kind, ood_loader, device, T, eps, max_batches=max_batches),
        )
    for det, (sid, sood) in logit_pairs.items():
        mets = binary_ood_metrics(sid, sood)
        for metric, val in mets.items():
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "feature_space": "logits",
                    "detector": det,
                    "metric": metric,
                    "value": val,
                    "supervision": "unsupervised",
                    "n_id": int(len(sid)),
                    "n_ood": int(len(sood)),
                    "feat_dim": int(id_pack["logits"].shape[1]),
                }
            )

    # also Maha/cosine/kNN on logits as an 8-d feature space
    ztr, zid, zood = train_pack["logits"], id_pack["logits"], ood_pack["logits"]
    means, present = fit_class_means(ztr, y_tr, n_classes)

    def _safe_pair(name, fn):
        try:
            return name, fn()
        except Exception as exc:
            if dry_run:
                print("  dry_run skip {} on logits: {}".format(name, exc))
                return name, None
            raise

    logit_embed = [
        _safe_pair("cosine_max", lambda: (
            cosine_ood_scores(zid, means, present),
            cosine_ood_scores(zood, means, present),
        )),
        _safe_pair("mahalanobis_classcond", lambda: (
            ood_metrics.mahalanobis_min_squared_distances(
                zid, *fit_maha_classcond(ztr, y_tr, n_classes, reg_eps)
            ),
            ood_metrics.mahalanobis_min_squared_distances(
                zood, *fit_maha_classcond(ztr, y_tr, n_classes, reg_eps)
            ),
        )),
        _safe_pair("mahalanobis_agnostic", lambda: (
            maha_agnostic_scores(zid, *fit_maha_agnostic(ztr, reg_eps)),
            maha_agnostic_scores(zood, *fit_maha_agnostic(ztr, reg_eps)),
        )),
        _safe_pair("knn_k50", lambda: (
            knn_scores(fit_knn(ztr), zid),
            knn_scores(fit_knn(ztr), zood),
        )),
    ]
    for det, pair in logit_embed:
        if pair is None:
            continue
        sid, sood = pair
        mets = binary_ood_metrics(sid, sood)
        for metric, val in mets.items():
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "feature_space": "logits",
                    "detector": det,
                    "metric": metric,
                    "value": val,
                    "supervision": "unsupervised",
                    "n_id": int(len(sid)),
                    "n_ood": int(len(sood)),
                    "feat_dim": int(ztr.shape[1]),
                }
            )

    # --- context_predictor (CSG only; domain-supervised) ---
    if kind == "csg" and train_pack["d_ctx"] is not None:
        # P(domain = PAD) as OOD score. Head was trained with domain labels.
        p_pad_id = torch.softmax(torch.from_numpy(id_pack["d_ctx"]), dim=1).numpy()[:, 1]
        p_pad_ood = torch.softmax(torch.from_numpy(ood_pack["d_ctx"]), dim=1).numpy()[:, 1]
        mets = binary_ood_metrics(p_pad_id, p_pad_ood)
        for metric, val in mets.items():
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "feature_space": "z_context",
                    "detector": CONTEXT_DETECTOR,
                    "metric": metric,
                    "value": val,
                    "supervision": "domain-supervised",
                    "n_id": int(len(p_pad_id)),
                    "n_ood": int(len(p_pad_ood)),
                    "feat_dim": 2,
                }
            )

    summary = {
        "method": method,
        "seed": seed,
        "kind": kind,
        "id_acc": id_acc,
        "id_balanced_acc": id_bal,
        "n_train": int(len(y_tr)),
        "n_id": int(len(id_pack["labels"])),
        "n_ood": int(len(ood_pack["labels"])),
        "dry_run": bool(dry_run),
        "feature_spaces": list(spaces),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


FIELDNAMES = [
    "method",
    "seed",
    "feature_space",
    "detector",
    "metric",
    "value",
    "supervision",
    "n_id",
    "n_ood",
    "feat_dim",
]


def write_per_seed(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def already_done(path: Path, method, seed):
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") == method and int(row.get("seed", -1)) == int(seed):
                return True
    return False


def write_aggregate(per_seed_csv: Path, out_csv: Path):
    buckets = defaultdict(list)
    with per_seed_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "ERROR":
                continue
            key = (
                row["method"],
                row["feature_space"],
                row["detector"],
                row["metric"],
                row.get("supervision", ""),
            )
            buckets[key].append(float(row["value"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "feature_space",
                "detector",
                "metric",
                "supervision",
                "n_seeds",
                "mean",
                "std",
            ],
        )
        w.writeheader()
        for key, vals in sorted(buckets.items()):
            arr = np.asarray(vals, dtype=np.float64)
            w.writerow(
                {
                    "method": key[0],
                    "feature_space": key[1],
                    "detector": key[2],
                    "metric": key[3],
                    "supervision": key[4],
                    "n_seeds": int(len(arr)),
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                }
            )


def write_phase1_report(out_dir: Path, summaries, per_seed_csv: Path, agg_csv: Path):
    """Fill P1 verdict + the required 3-row table. Failures are reported as findings."""

    def pick(method, space, detector, metric="AUROC"):
        vals = []
        if not per_seed_csv.exists():
            return float("nan"), 0
        with per_seed_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (
                    row["method"] == method
                    and row["feature_space"] == space
                    and row["detector"] == detector
                    and row["metric"] == metric
                ):
                    vals.append(float(row["value"]))
        if not vals:
            return float("nan"), 0
        return float(np.mean(vals)), len(vals)

    maha_ctx, n_ctx = pick("runB_orth1", "z_context", "mahalanobis_classcond")
    maha_les, n_les = pick("runB_orth1", "z_lesion", "mahalanobis_classcond")
    maha_base, n_base = pick("baseline_soft", "backbone_raw", "mahalanobis_classcond")
    p1_pass = (not np.isnan(maha_ctx)) and (maha_ctx > 0.95)
    p1_gate = (not np.isnan(maha_ctx)) and (maha_ctx >= 0.90)

    id_bal_csg = [s["id_balanced_acc"] for s in summaries if s["method"] == "runB_orth1"]
    id_bal_base = [s["id_balanced_acc"] for s in summaries if s["method"] == "baseline_soft"]

    lines = []
    lines.append("# Paper B — Phase 1 report")
    lines.append("")
    lines.append("Inference only. No retraining. Checkpoints selected by Phase 0.1 policy")
    lines.append("(newest `best-*.ckpt` per run). Statistics (means, covariances, class")
    lines.append("centroids, kNN bank) were fit on the **ISIC training split only**; a")
    lines.append("runtime assertion rejects val / test / PAD leakage into the fit.")
    lines.append("")
    lines.append("## Honesty (non-negotiable)")
    lines.append("")
    lines.append(
        "`z_context` detectors, `backbone_raw_context` detectors, and "
        "`context_predictor_softmax` are **domain-supervised**: the context branch "
        "and its predictor were trained with an explicit ISIC-vs-PAD domain label "
        "(`L_ctx`). The baseline Mahalanobis score is unsupervised. This asymmetry "
        "is labelled in every CSV row via the `supervision` column. The defence "
        "against “of course it works, you supervised it” is Phase 4 (semantic OOD, "
        "domain held constant) and Phase 11 (unseen third domain), not spin."
    )
    lines.append("")
    lines.append("## P1 verdict")
    lines.append("")
    lines.append(
        "Prediction P1: Mahalanobis AUROC on `z_context` > 0.95, vs ~0.41 on "
        "`z_lesion`, same `runB_orth1` checkpoints."
    )
    lines.append("")
    lines.append("| Quantity | Value | n seeds |")
    lines.append("|---|---|---:|")
    lines.append(
        "| runB_orth1 `z_context` Maha AUROC (domain-supervised) | {:.4f} | {} |".format(
            maha_ctx, n_ctx
        )
    )
    lines.append(
        "| runB_orth1 `z_lesion` Maha AUROC | {:.4f} | {} |".format(maha_les, n_les)
    )
    lines.append(
        "| baseline_soft `backbone_raw` Maha AUROC | {:.4f} | {} |".format(maha_base, n_base)
    )
    lines.append("")
    if np.isnan(maha_ctx):
        verdict = "INCOMPLETE — z_context Maha AUROC not computed"
        gate = "STOP (no number)"
    elif p1_pass:
        verdict = "PASS (z_context Maha AUROC > 0.95)"
        gate = "Phase 2 authorised"
    elif p1_gate:
        verdict = "FAIL vs 0.95 threshold, but ≥ 0.90 so the work-order stop-gate does not fire"
        gate = "Phase 2 authorised only after human review (gate was < 0.90)"
    else:
        verdict = "FAIL (z_context Maha AUROC < 0.90)"
        gate = "STOP before Phase 2, as specified"
    lines.append("**P1: {}**".format(verdict))
    lines.append("")
    lines.append("Gate: {}".format(gate))
    lines.append("")
    lines.append("## Required table")
    lines.append("")
    lines.append(
        "Probe acc / ID bal acc in the left and right columns are the published "
        "n=5 figures from `results/cbm_revision` / `leakage.json` (not re-probed "
        "here). Maha AUROC is new."
    )
    lines.append("")
    lines.append("| Branch | Domain probe acc | OOD AUROC (Maha) | ID Bal Acc | supervision |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| Baseline (entangled, ResNet-50) | 0.979 | {:.3f} | 0.655 | unsupervised |".format(
            maha_base if not np.isnan(maha_base) else float("nan")
        )
    )
    lines.append(
        "| z_lesion (invariant) | 0.724 | {:.3f} | 0.702 | unsupervised |".format(
            maha_les if not np.isnan(maha_les) else float("nan")
        )
    )
    lines.append(
        "| z_context (leaky) | 0.9997 | {:.3f} | n/a | **domain-supervised** |".format(
            maha_ctx if not np.isnan(maha_ctx) else float("nan")
        )
    )
    lines.append("")
    if id_bal_csg or id_bal_base:
        lines.append("ID balanced accuracy recomputed on this pass (ISIC test):")
        if id_bal_base:
            lines.append(
                "- baseline_soft: {:.4f} ± {:.4f}".format(
                    float(np.mean(id_bal_base)), float(np.std(id_bal_base))
                )
            )
        if id_bal_csg:
            lines.append(
                "- runB_orth1: {:.4f} ± {:.4f}".format(
                    float(np.mean(id_bal_csg)), float(np.std(id_bal_csg))
                )
            )
        lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `{}`".format(per_seed_csv))
    lines.append("- `{}`".format(agg_csv))
    lines.append("")
    lines.append("## What ran")
    lines.append("")
    for s in summaries:
        lines.append(
            "- {method} seed={seed} kind={kind} id_acc={id_acc:.4f} "
            "id_bal={id_balanced_acc:.4f} n_train={n_train} n_id={n_id} n_ood={n_ood}".format(**s)
        )
    (out_dir / "PHASE1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict, p1_gate, maha_ctx, maha_les


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Paper B Phase 1 dual-branch OOD")
    p.add_argument("--metadata", type=Path, default=METADATA)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--reg_eps", type=float, default=1e-3)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths, shapes and split assertions on a handful of batches; do not write full tables.",
    )
    p.add_argument("--dry_run_batches", type=int, default=2)
    p.add_argument("--skip_done", action="store_true", default=True)
    p.add_argument("--no_skip_done", action="store_false", dest="skip_done")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(42)
    try:
        import pytorch_lightning as pl

        pl.seed_everything(42, workers=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device={} dry_run={}".format(device, args.dry_run))
    if device.type == "cuda":
        print("gpu={}".format(torch.cuda.get_device_name(0)))

    splits = build_split_frames(args.metadata)
    assert_fit_frame_is_isic_train(splits["isic_train"], splits, "isic_train")
    print(
        "splits: train={} val={} test={} pad={}".format(
            len(splits["isic_train"]),
            len(splits["isic_val"]),
            len(splits["isic_test"]),
            len(splits["pad"]),
        )
    )

    eval_tf = build_val_transform_robust()
    max_batches = args.dry_run_batches if args.dry_run else None
    loaders = {
        "train": make_loader(
            splits["isic_train"], eval_tf, args.batch_size, args.num_workers, max_batches=max_batches
        ),
        "id": make_loader(
            splits["isic_test"], eval_tf, args.batch_size, args.num_workers, max_batches=max_batches
        ),
        "ood": make_loader(
            splits["pad"], eval_tf, args.batch_size, args.num_workers, max_batches=max_batches
        ),
    }

    # Dry-run: confirm one batch shape per method family before the grid.
    if args.dry_run:
        images, labels = next(iter(loaders["train"]))
        print("dry_run batch images={} labels={}".format(tuple(images.shape), tuple(labels.shape)))
        assert images.ndim == 4 and images.shape[1] == 3
        assert labels.min() >= 0

    per_seed_csv = out_dir / ("ood_dual_branch_per_seed_dryrun.csv" if args.dry_run else "ood_dual_branch_per_seed.csv")
    agg_csv = out_dir / ("ood_dual_branch_aggregate_dryrun.csv" if args.dry_run else "ood_dual_branch_aggregate.csv")
    summaries = []
    run_cfg = {
        "script": "eval_ood_dual_branch.py",
        "metadata": str(args.metadata),
        "methods": methods,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "reg_eps": args.reg_eps,
        "dry_run": bool(args.dry_run),
        "device": str(device),
        "checkpoint_policy": "newest best-*.ckpt else newest last*.ckpt",
        "fit_split": "isic_train_only",
        "honesty": "z_context / backbone_raw_context / context_predictor_softmax are domain-supervised",
    }
    (out_dir / ("phase1_config_dryrun.json" if args.dry_run else "phase1_config.json")).write_text(
        json.dumps(run_cfg, indent=2) + "\n", encoding="utf-8"
    )

    for method in methods:
        for seed in seeds:
            ckpt = find_policy_ckpt(method_ckpt_dir(method, seed))
            if ckpt is None:
                print("SKIP missing ckpt {} seed={}".format(method, seed))
                continue
            if args.skip_done and not args.dry_run and already_done(per_seed_csv, method, seed):
                print("SKIP already done {} seed={}".format(method, seed))
                continue
            print("=== {} seed={} ckpt={} ===".format(method, seed, ckpt))
            model, kind = load_model(method, ckpt, device)
            rows, summary = eval_one(
                method,
                seed,
                model,
                kind,
                loaders,
                device,
                args.dry_run,
                max_batches,
                args.reg_eps,
            )
            summary["checkpoint"] = str(ckpt)
            summaries.append(summary)
            write_per_seed(per_seed_csv, rows)
            (out_dir / "phase1_summaries.json").write_text(
                json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
            )
            print(
                "  wrote {} rows  id_acc={:.4f} id_bal={:.4f}".format(
                    len(rows), summary["id_acc"], summary["id_balanced_acc"]
                )
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.dry_run:
                # One seed per method is enough to validate shapes.
                break

    if per_seed_csv.exists():
        write_aggregate(per_seed_csv, agg_csv)
        verdict, gate_ok, maha_ctx, maha_les = write_phase1_report(
            out_dir, summaries, per_seed_csv, agg_csv
        )
        print("aggregate ->", agg_csv)
        print("P1 verdict:", verdict, "z_context={:.4f} z_lesion={:.4f}".format(maha_ctx, maha_les))
        if args.dry_run:
            (out_dir / "phase1_dry_run_ok.json").write_text(
                json.dumps({"ok": True, "summaries": summaries}, indent=2) + "\n",
                encoding="utf-8",
            )
            print("dry_run OK")


if __name__ == "__main__":
    main()
