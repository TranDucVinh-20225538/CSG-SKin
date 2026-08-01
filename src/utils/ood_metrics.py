# src/utils/ood_metrics.py
"""
OOD scoring: MSP, Energy, Mahalanobis (class-conditional means + pooled covariance).

Style aligned with Ban_sao_datn/src/utils/scoring.py (sectioned helpers).
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


# ====================== ROC / FPR@95 ======================


def fpr_at_95_tpr(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    reached = np.where(tpr >= 0.95)[0]
    if reached.size == 0:
        return 1.0
    return float(fpr[reached[0]])


# ====================== Mahalanobis (shared Σ, per-class μ) ======================


def compute_mahalanobis_params_from_arrays(features, labels, num_classes=8, reg_eps=1e-5):
    """
    Class-conditional means + shared precision (inverse pooled within-class covariance).
    Σ = (1/(N-K)) sum_i (x_i - μ_{y_i})(x_i - μ_{y_i})^T
    """
    n_samples, feat_dim = features.shape
    if n_samples < num_classes + 1:
        raise ValueError("Not enough samples to estimate a shared covariance.")

    class_means = np.zeros((num_classes, feat_dim), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            raise ValueError("No samples for class {}; cannot fit Mahalanobis.".format(c))
        class_means[c] = features[mask].mean(axis=0)

    centered = features - class_means[labels.astype(np.int64)]
    denom = max(n_samples - num_classes, 1)
    cov = (centered.T @ centered) / denom
    cov = cov + reg_eps * np.eye(feat_dim, dtype=np.float64)
    precision = np.linalg.inv(cov)
    return class_means.astype(np.float32), precision.astype(np.float32)


def mahalanobis_min_squared_distances(features, class_means, precision):
    """Min over classes of squared Mahalanobis distance (higher → more OOD-friendly)."""
    n_samples = features.shape[0]
    n_classes = class_means.shape[0]
    mins = np.empty(n_samples, dtype=np.float64)
    p = precision.astype(np.float64)
    for i in range(n_samples):
        x = features[i].astype(np.float64)
        best = np.inf
        for c in range(n_classes):
            delta = x - class_means[c].astype(np.float64)
            d2 = float(delta @ p @ delta)
            if d2 < best:
                best = d2
        mins[i] = best
    return mins.astype(np.float32)


def mahalanobis_auroc_fpr95(id_features, ood_features, class_means, precision):
    id_scores = mahalanobis_min_squared_distances(id_features, class_means, precision)
    ood_scores = mahalanobis_min_squared_distances(ood_features, class_means, precision)
    y_true = np.concatenate(
        [np.zeros(id_scores.shape[0], dtype=int), np.ones(ood_scores.shape[0], dtype=int)]
    )
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(y_true, scores))
    fpr95 = fpr_at_95_tpr(y_true, scores)
    return auroc, fpr95


# ====================== Feature / logit collection ======================


@torch.no_grad()
def collect_logits_labels_features_baseline(model, loader, device):
    """Baseline Lightning module: logits + backbone features (pre-FC)."""
    model.eval()
    logits_list = []
    labels_list = []
    features_list = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())
        features_list.append(features.cpu())

    logits_np = torch.cat(logits_list, dim=0).numpy()
    labels_np = torch.cat(labels_list, dim=0).numpy()
    features_np = torch.cat(features_list, dim=0).numpy()
    return logits_np, labels_np, features_np


@torch.no_grad()
def collect_z_lesion_labels_csg(model, loader, device):
    """CSG-lite nn.Module: z_lesion from lesion encoder. Keeps only rows with lesion label >= 0 (drops PAD)."""
    model.eval()
    z_list = []
    y_list = []
    for batch in loader:
        if len(batch) == 5:
            _images_ctx, images_lesion, labels, _y_supcon, _domain = batch
            images = images_lesion
        elif len(batch) == 4:
            _images_ctx, images_lesion, labels, _domain = batch
            images = images_lesion
        elif len(batch) == 3:
            images, labels, _domain = batch
        else:
            images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        z = model.encode_z_lesion(images)
        mask = labels >= 0
        if mask.any():
            z_list.append(z[mask].cpu())
            y_list.append(labels[mask].cpu())
    if not z_list:
        raise ValueError("No ISIC (label>=0) rows in loader for z_lesion collection.")
    return torch.cat(z_list, dim=0).numpy(), torch.cat(y_list, dim=0).numpy()


def compute_mahalanobis_params_baseline(model, train_loader, device, num_classes=8, reg_eps=1e-5):
    _l, labels, features = collect_logits_labels_features_baseline(model, train_loader, device)
    return compute_mahalanobis_params_from_arrays(features, labels, num_classes=num_classes, reg_eps=reg_eps)


# ====================== Full baseline table (MSP / Energy / Maha) ======================


def print_baseline_id_ood_table(model, id_loader, ood_loader, device, class_means, precision):
    id_logits, id_labels, id_features = collect_logits_labels_features_baseline(model, id_loader, device)
    ood_logits, _y_ood, ood_features = collect_logits_labels_features_baseline(model, ood_loader, device)

    id_preds = id_logits.argmax(axis=1)
    id_acc = float((id_preds == id_labels).mean())

    pred_counts = np.bincount(id_preds.astype(np.int64), minlength=id_logits.shape[1])
    true_counts = np.bincount(id_labels.astype(np.int64), minlength=id_logits.shape[1])
    print("ID test: pred argmax counts per class: {}".format(pred_counts.tolist()))
    print("ID test: true label counts per class:  {}".format(true_counts.tolist()))

    id_probs = torch.softmax(torch.from_numpy(id_logits), dim=1).numpy()
    ood_probs = torch.softmax(torch.from_numpy(ood_logits), dim=1).numpy()
    id_msp = id_probs.max(axis=1)
    ood_msp = ood_probs.max(axis=1)
    id_msp_score = -id_msp
    ood_msp_score = -ood_msp

    id_energy = (-torch.logsumexp(torch.from_numpy(id_logits), dim=1)).numpy()
    ood_energy = (-torch.logsumexp(torch.from_numpy(ood_logits), dim=1)).numpy()

    id_maha = mahalanobis_min_squared_distances(id_features, class_means, precision)
    ood_maha = mahalanobis_min_squared_distances(ood_features, class_means, precision)

    y_true = np.concatenate([np.zeros_like(id_msp_score), np.ones_like(ood_msp_score)]).astype(int)
    msp_scores = np.concatenate([id_msp_score, ood_msp_score])
    energy_scores = np.concatenate([id_energy, ood_energy])
    maha_scores = np.concatenate([id_maha, ood_maha])

    msp_auroc = float(roc_auc_score(y_true, msp_scores))
    energy_auroc = float(roc_auc_score(y_true, energy_scores))
    maha_auroc = float(roc_auc_score(y_true, maha_scores))
    msp_fpr95 = fpr_at_95_tpr(y_true, msp_scores)
    energy_fpr95 = fpr_at_95_tpr(y_true, energy_scores)
    maha_fpr95 = fpr_at_95_tpr(y_true, maha_scores)

    print("\n=== Final Evaluation ===")
    print("ID Test Accuracy (ISIC): {:.4f}".format(id_acc))
    print("\nOOD Detection (ID=ISIC test, OOD=PAD-UFES)")
    print("{:<14} {:>8} {:>8}".format("Method", "AUROC", "FPR@95"))
    print("{:<14} {:>8.4f} {:>8.4f}".format("MSP", msp_auroc, msp_fpr95))
    print("{:<14} {:>8.4f} {:>8.4f}".format("Energy", energy_auroc, energy_fpr95))
    print("{:<14} {:>8.4f} {:>8.4f}".format("Mahalanobis", maha_auroc, maha_fpr95))


def find_checkpoint(path_or_dir):
    if path_or_dir is None:
        return None
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    ckpts = list(p.glob("*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: x.stat().st_mtime)
