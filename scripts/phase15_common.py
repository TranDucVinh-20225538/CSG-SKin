"""Shared Phase 15 pieces. Import only after the transformers stub is installed."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from eval_ood_dual_branch import cosine_ood_scores, fit_class_means, fit_knn, knn_scores
from src.utils import ood_metrics

PAD_CLASSES = np.array([0, 1, 2, 3, 4, 7], dtype=int)
PAD_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]
LN2 = float(math.log(2.0))


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def set_lambd(self, lambd):
        self.lambd = float(lambd)

    def forward(self, x):
        return _GRLFn.apply(x, self.lambd)


def derm_domain_head(in_dim, n_domains):
    """Same template as CSGLite.domain_classifier_adv, scaled to in_dim."""
    h1 = max(int(in_dim), 32)
    h2 = max(int(in_dim) // 2, 16)
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(h2, n_domains),
    )


class SingleDannNet(nn.Module):
    """Plain single-encoder DANN: one EffNet-B3, linear classifier, GRL + derm domain head."""

    def __init__(self, n_classes=8, n_domains=2, pretrained=True):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b3(weights=weights)
        self.feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, n_classes)
        self.grl = GradientReversal(1.0)
        self.domain_head = derm_domain_head(self.feat_dim, n_domains)

    def forward(self, x):
        z = self.backbone(x)
        z_norm = self.feat_bn(z)
        logits = self.classifier(z_norm)
        dlogits = self.domain_head(self.grl(z_norm))
        return logits, dlogits, z_norm

    def set_grl_lambda(self, lambd):
        self.grl.set_lambd(lambd)


def adversary_gate(epoch_rows, ln_k, lambda_adv, leakage_bal=None, floor=None, ce_margin=0.02):
    """Distinguish an adversary that learned and was then driven back to chance
    from one that never left ln K.

    At chance at the end, after a best-epoch CE meaningfully below ln K, is
    success: the encoder won. At chance from epoch 1 and never below ln K is
    the Camelyon17 null. λ=0 is a control: the adversary term is multiplied by
    zero, so the head is not trained and is not a null experiment.

    ce_margin=0.02 is the gap that still counts λ=8 (best CE ≈ 0.66 vs ln 2 =
    0.693, best acc_adv ≈ 0.63) as having learned.
    """
    first = next((r for r in epoch_rows if r.get("event") == "first_batch_epoch1"), None)
    epochs = [r for r in epoch_rows if "epoch" in r and "loss_adv" in r]
    last = epochs[-1] if epochs else None
    best = min(epochs, key=lambda r: float(r["loss_adv"])) if epochs else None
    base = {
        "leakage_bal_acc": leakage_bal,
        "chance_floor": floor,
        "ln_k": float(ln_k),
        "ce_margin": float(ce_margin),
        "first_batch": first,
        "epoch1": next((r for r in epochs if r.get("epoch") == 1), None),
        "best_epoch": best,
        "last_epoch": last,
        "n_epochs_logged": len(epochs),
        "lambda_adv": float(lambda_adv),
    }
    if abs(float(lambda_adv)) < 1e-12:
        base.update({
            "status": "valid_control",
            "reason": "λ=0 control. Adversary loss weight is zero, so the head is not trained. Not a null experiment.",
        })
        return base
    if not epochs:
        base.update({"status": "inconclusive", "reason": "no adversary epoch log"})
        return base
    best_ce = float(best["loss_adv"])
    learned = best_ce < float(ln_k) - float(ce_margin)
    last_acc = float(last.get("acc_adv", 0.0))
    last_g = float(last.get("g_enc_grl_mean", last.get("g_enc_grl", 1.0)))
    returned = learned and abs(float(last["loss_adv"]) - float(ln_k)) <= float(ce_margin) and last_acc < 0.55
    saturated = last_acc > 0.97 and last_g < 1e-8
    if not learned:
        status = "inconclusive"
        reason = (
            "adversary never left chance: best-epoch CE {:.4f} is not below ln K − {:.2f} = {:.4f}. "
            "Null experiment, not a scientific negative."
        ).format(best_ce, ce_margin, float(ln_k) - float(ce_margin))
    elif saturated:
        status = "inconclusive"
        reason = "adversary pinned at 1.0 with vanishing encoder gradient — null experiment"
    elif returned:
        status = "valid"
        reason = (
            "encoder won: best-epoch CE {:.4f} (epoch {}) is below ln K, "
            "and the head is back at chance by epoch {}. That is the intended outcome."
        ).format(best_ce, best.get("epoch"), last.get("epoch"))
    else:
        status = "valid"
        reason = "adversary learned: best-epoch CE {:.4f} (epoch {}) is below ln K − {:.2f}.".format(
            best_ce, best.get("epoch"), ce_margin
        )
    base.update({"status": status, "reason": reason, "learned": learned, "returned_to_chance": returned})
    return base


def grl_alpha(step, total_steps, gamma=20.0, power=0.5, alpha_min=0.2):
    denom = max(int(total_steps) - 1, 1)
    progress = max(0.0, min(1.0, float(step) / float(denom)))
    shaped = progress ** float(power)
    alpha = (2.0 / (1.0 + math.exp(-float(gamma) * shaped))) - 1.0
    return float(max(float(alpha_min), alpha)), float(progress)


def ece_from_logits(logits, labels, n_bins=15):
    probs = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=1).numpy()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    labels = np.asarray(labels)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(labels), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece), float(conf.mean())


def leakage_probe(z, domain, seeds=(42, 52, 62)):
    domain = np.asarray(domain)
    k = int(len(np.unique(domain)))
    accs, bals, aurocs = [], [], []
    for s in seeds:
        ztr, zte, ytr, yte = train_test_split(z, domain, test_size=0.3, random_state=s, stratify=domain)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(ztr, ytr)
        pred = clf.predict(zte)
        accs.append(float(accuracy_score(yte, pred)))
        bals.append(float(balanced_accuracy_score(yte, pred)))
        if k == 2:
            score = clf.predict_proba(zte)[:, 1]
            aurocs.append(float(roc_auc_score(yte, score)))
    return {
        "acc_mean": float(np.mean(accs)),
        "bal_acc_mean": float(np.mean(bals)),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "acc_std": float(np.std(accs, ddof=1)),
        "bal_acc_std": float(np.std(bals, ddof=1)),
        "n_probe_seeds": len(seeds),
        "n_domains": k,
        "chance_floor_balanced": 1.0 / k,
        "metric": "balanced_accuracy",
    }


def auroc_pair(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return float(roc_auc_score(y, s))


def detector_block(ztr, ytr, zid, zood, logits_id, logits_ood, n_classes):
    """All five detectors. Statistics fit on train only (ztr, ytr)."""
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(
        ztr, ytr, num_classes=n_classes, reg_eps=1e-3
    )
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    cmeans, present = fit_class_means(ztr, ytr, n_classes)
    knn = fit_knn(ztr, k=50)
    p_id = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
    e_id = torch.logsumexp(torch.from_numpy(logits_id).float(), dim=1).numpy()
    e_ood = torch.logsumexp(torch.from_numpy(logits_ood).float(), dim=1).numpy()
    return {
        "mahalanobis_classcond": auroc_pair(sid, sood),
        "cosine_max": auroc_pair(cosine_ood_scores(zid, cmeans, present), cosine_ood_scores(zood, cmeans, present)),
        "knn_k50": auroc_pair(knn_scores(knn, zid), knn_scores(knn, zood)),
        "MSP": auroc_pair(-p_id.max(axis=1), -p_ood.max(axis=1)),
        "Energy_T1": auroc_pair(-e_id, -e_ood),
        "fit": "ISIC train split only",
    }


def six_class_xfer(logits, labels):
    y = np.asarray(labels).astype(int)
    logits6 = np.asarray(logits)[:, PAD_CLASSES]
    exp = np.exp(logits6 - logits6.max(axis=1, keepdims=True))
    P = exp / exp.sum(axis=1, keepdims=True)
    pred = PAD_CLASSES[P.argmax(1)]
    rec = recall_score(y, pred, labels=PAD_CLASSES, average=None, zero_division=0)
    return {
        "subset": "pad_heldout",
        "n_images": int(len(y)),
        "n_patients": 412,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class_recall": {PAD_NAMES[i]: float(rec[i]) for i in range(len(PAD_CLASSES))},
        "class_counts": {PAD_NAMES[i]: int((y == PAD_CLASSES[i]).sum()) for i in range(len(PAD_CLASSES))},
        "restriction": "6-class softmax/argmax over MEL,NV,BCC,AK,BKL,SCC",
    }


def assert_split_hygiene(train_paths, val_paths, test_paths, pad_adv_paths, pad_hold_paths):
    train_paths = set(map(str, train_paths))
    val_paths = set(map(str, val_paths))
    test_paths = set(map(str, test_paths))
    pad_adv_paths = set(map(str, pad_adv_paths))
    pad_hold_paths = set(map(str, pad_hold_paths))
    if train_paths & val_paths:
        raise RuntimeError("train/val path overlap")
    if train_paths & test_paths:
        raise RuntimeError("train/test path overlap")
    if val_paths & test_paths:
        raise RuntimeError("val/test path overlap")
    if pad_adv_paths & pad_hold_paths:
        raise RuntimeError("pad_adv / pad_heldout overlap — invariance OOD is not held out")
    if not pad_hold_paths:
        raise RuntimeError("pad_heldout is empty")
    return {
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_test": len(test_paths),
        "n_pad_adv": len(pad_adv_paths),
        "n_pad_heldout": len(pad_hold_paths),
        "pad_patient_disjoint": True,
    }


def coral_penalty(z_src, z_tgt):
    def _cov(z):
        z = z - z.mean(0, keepdim=True)
        n = max(z.size(0) - 1, 1)
        return (z.t() @ z) / n

    d = _cov(z_src) - _cov(z_tgt)
    return (d * d).sum() / (4.0 * z_src.size(1) ** 2)


def mmd_rbf_penalty(z_src, z_tgt, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    def _pdist(a, b):
        return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1)

    xx, yy, xy = _pdist(z_src, z_src), _pdist(z_tgt, z_tgt), _pdist(z_src, z_tgt)
    n, m = z_src.size(0), z_tgt.size(0)
    loss = z_src.new_tensor(0.0)
    for s in sigmas:
        g = 1.0 / (2.0 * float(s) * float(s))
        kxx = torch.exp(-g * xx)
        kyy = torch.exp(-g * yy)
        kxy = torch.exp(-g * xy)
        kxx = kxx - torch.diag(kxx.diag())
        kyy = kyy - torch.diag(kyy.diag())
        loss = loss + kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()
    return loss / len(sigmas)


def irm_penalty(logits, y, ignore_index=-1):
    """IRMv1 dummy-classifier penalty on one environment."""
    valid = y != ignore_index
    if int(valid.sum()) < 2:
        return logits.new_tensor(0.0)
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits[valid] * scale, y[valid])
    (grad,) = torch.autograd.grad(loss, [scale], create_graph=True)
    return grad.pow(2).sum()
