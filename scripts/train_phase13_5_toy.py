#!/usr/bin/env python3
"""Phase 13.5 — pre-registered toy for the sub-chance inversion.

Grid is locked in results/paperB/phase13_5/GRID_LOCKED.json.
Do not retune toward a target curve. Negative result is a result.
Runs on CPU. Do not submit behind Phase 12.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(1)

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase13_5")
FIG = OUT / "figures"
GRID_PATH = OUT / "GRID_LOCKED.json"
WONG = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "gray": "#888888",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}
DERM_AUROC = {
    0.0: 0.863,
    0.25: 0.508,
    0.5: 0.449,
    1.0: 0.439,
    2.0: 0.413,
    4.0: 0.443,
    8.0: 0.481,
}


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def derm_adv_head(m, n_out=2):
    h1 = max(m, 32)
    h2 = max(m // 2, 16)
    return nn.Sequential(
        nn.Linear(m, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(h2, n_out),
    )


class ToyNet(nn.Module):
    def __init__(self, D, m, K, encoder, hidden, bn, adversary):
        super().__init__()
        if encoder == "linear":
            self.enc = nn.Linear(D, m)
        elif encoder == "mlp":
            h1, h2 = hidden
            self.enc = nn.Sequential(nn.Linear(D, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, m))
        else:
            raise ValueError(encoder)
        self.bn = nn.BatchNorm1d(m) if bn else nn.Identity()
        self.cls = nn.Linear(m, K)
        if adversary == "linear":
            self.adv = nn.Linear(m, 2)
        elif adversary == "mlp_derm":
            self.adv = derm_adv_head(m)
        else:
            raise ValueError(adversary)
        self.encoder_kind = encoder

    def encode(self, x):
        return self.bn(self.enc(x))

    def forward(self, x, lambd):
        z = self.encode(x)
        return self.cls(z), self.adv(_GRL.apply(z, lambd)), z


def orthonormal_basis(rng, D, n):
    A = rng.normal(size=(D, n))
    Q, _ = np.linalg.qr(A)
    return Q[:, :n]


def make_world(cfg, seed):
    rng = np.random.default_rng(1000 + int(seed))
    D, K = cfg["D"], cfg["K"]
    Q = orthonormal_basis(rng, D, K + 2)
    label_basis = Q[:, :K]
    u = Q[:, K]
    u2 = Q[:, K + 1]
    mu = cfg["label_scale"] * label_basis.T  # K x D
    return {"mu": mu, "u": u, "u2": u2}


def sample_domain(world, n, classes, beta, tau, seed, tag):
    tag_id = {"d0": 17, "d1": 29, "d2": 41}.get(tag, 53)
    rng = np.random.default_rng(seed * 1009 + tag_id)
    y = rng.choice(np.asarray(classes, dtype=int), size=n)
    eps = rng.normal(size=(n, world["mu"].shape[1]))
    x = world["mu"][y] + beta * world["u"][None, :] + tau * eps
    return x.astype(np.float32), y.astype(np.int64)


def sample_heldout_domain(world, n, classes, beta, tau, seed):
    rng = np.random.default_rng(9000 + seed)
    y = rng.choice(np.asarray(classes, dtype=int), size=n)
    eps = rng.normal(size=(n, world["mu"].shape[1]))
    x = world["mu"][y] + beta * world["u2"][None, :] + tau * eps
    return x.astype(np.float32), y.astype(np.int64)


def auroc_pair(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def fit_maha(ztr, ytr, n_classes, eps=1e-3):
    ztr = np.asarray(ztr, np.float64)
    ytr = np.asarray(ytr, int)
    # assert fit is ID-only is the caller's job
    means = np.zeros((n_classes, ztr.shape[1]), np.float64)
    present = []
    for c in range(n_classes):
        m = ytr == c
        if not np.any(m):
            continue
        means[c] = ztr[m].mean(0)
        present.append(c)
    present = np.asarray(present, int)
    centered = ztr - means[ytr]
    cov = (centered.T @ centered) / max(len(ztr) - len(present), 1)
    cov = cov + eps * np.eye(cov.shape[0])
    prec = np.linalg.pinv(cov)
    return means, prec, present


def maha_scores(z, means, prec, present):
    z = np.asarray(z, np.float64)
    mu = means[present]
    zp = z @ prec
    zpz = np.einsum("nd,nd->n", zp, z)
    mp = mu @ prec
    mpm = np.einsum("cd,cd->c", mp, mu)
    d2 = zpz[:, None] + mpm[None, :] - 2.0 * (zp @ mu.T)
    return d2.min(1)


def participation_ratio(z):
    z = np.asarray(z, np.float64)
    zc = z - z.mean(0)
    ev = np.linalg.eigvalsh((zc.T @ zc) / max(len(z) - 1, 1))
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    if tot <= 0:
        return float("nan")
    return float((tot ** 2) / max(float((ev ** 2).sum()), 1e-18))


def evaluate(z_tr, y_tr, z_id, y_id, logits_id, z_ood, y_ood, logits_ood, z_d2, logits_d2, K, shared, u, W_lin):
    assert int((y_tr >= 0).sum()) == len(y_tr)
    means, prec, present = fit_maha(z_tr, y_tr, K)
    sid = maha_scores(z_id, means, prec, present)
    sood = maha_scores(z_ood, means, prec, present)
    sd2 = maha_scores(z_d2, means, prec, present)
    out = {"maha": auroc_pair(sid, sood), "maha_domain2": auroc_pair(sid, sd2)}

    def softmax(lg):
        x = lg - lg.max(1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(1, keepdims=True)

    p_id, p_ood = softmax(logits_id), softmax(logits_ood)
    out["MSP"] = auroc_pair(-p_id.max(1), -p_ood.max(1))
    out["Energy"] = auroc_pair(-np.logaddexp.reduce(logits_id, 1), -np.logaddexp.reduce(logits_ood, 1))
    prot = means[present]
    prot_n = prot / (np.linalg.norm(prot, axis=1, keepdims=True) + 1e-12)

    def cos_sc(z):
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        return -(zn @ prot_n.T).max(1)

    out["cosine"] = auroc_pair(cos_sc(z_id), cos_sc(z_ood))
    zntr = z_tr / (np.linalg.norm(z_tr, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=min(50, len(zntr)), algorithm="auto", metric="euclidean")
    nn.fit(zntr)

    def knn_sc(z):
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        d, _ = nn.kneighbors(zn)
        return d.mean(1)

    out["knn50"] = auroc_pair(knn_sc(z_id), knn_sc(z_ood))

    shared = np.asarray(shared, int)
    m_id_s = np.isin(y_id, shared)
    out["maha_id_shared_classes"] = auroc_pair(sid[m_id_s], sood) if m_id_s.any() else float("nan")

    # reweight ID to OOD class mix among shared classes
    ood_counts = np.array([(y_ood == c).sum() for c in shared], dtype=np.float64)
    ood_p = ood_counts / max(ood_counts.sum(), 1)
    rng = np.random.default_rng(0)
    idx_by = {c: np.where(y_id == c)[0] for c in shared}
    n_rep = min(len(y_id), 4000)
    chosen = []
    for _ in range(n_rep):
        c = int(rng.choice(shared, p=ood_p))
        if len(idx_by[c]) == 0:
            continue
        chosen.append(int(rng.choice(idx_by[c])))
    out["maha_id_reweighted_to_ood_mix"] = auroc_pair(sid[np.asarray(chosen)], sood) if chosen else float("nan")

    out["var_ood_over_id"] = float(z_ood.var(0).sum() / max(z_id.var(0).sum(), 1e-18))
    gmu = z_tr.mean(0)
    out["med_nn_centroid_id"] = float(np.median(np.linalg.norm(z_id[:, None, :] - means[present][None, :, :], axis=2).min(1)))
    out["med_nn_centroid_ood"] = float(np.median(np.linalg.norm(z_ood[:, None, :] - means[present][None, :, :], axis=2).min(1)))
    out["mean_dist_global_id"] = float(np.linalg.norm(z_id - gmu, axis=1).mean())
    out["mean_dist_global_ood"] = float(np.linalg.norm(z_ood - gmu, axis=1).mean())
    out["pr_id_train"] = participation_ratio(z_tr)
    if W_lin is not None:
        out["Wu_norm"] = float(np.linalg.norm(W_lin @ u))
    else:
        shifts = []
        for c in shared:
            a, b = z_id[y_id == c], z_ood[y_ood == c]
            if len(a) and len(b):
                shifts.append(np.linalg.norm(a.mean(0) - b.mean(0)))
        out["Wu_norm"] = float(np.mean(shifts)) if shifts else float("nan")
        out["Wu_norm_is"] = "mean class-conditional latent domain shift (MLP analogue)"

    zprobe = np.concatenate([z_id, z_ood])
    yprobe = np.concatenate([np.zeros(len(z_id)), np.ones(len(z_ood))])
    bals = []
    for s in (0, 1, 2):
        ztrp, zte, ytrp, yte = train_test_split(zprobe, yprobe, test_size=0.3, random_state=s, stratify=yprobe)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
        clf.fit(ztrp, ytrp)
        bals.append(float(balanced_accuracy_score(yte, clf.predict(zte))))
    out["leakage_bal_acc"] = float(np.mean(bals))
    out["fit_n"] = int(len(z_tr))
    out["fit_domains"] = "domain0_train_only"
    return out


def run_cell(payload):
    cell_name, cfg, lam, seed = payload
    torch.manual_seed(seed)
    np.random.seed(seed)
    world = make_world(cfg, seed)
    id_classes = list(range(cfg["K"]))
    ood_classes = list(range(cfg["K_ood"]))
    x0, y0 = sample_domain(world, cfg["n0"], id_classes, 0.0, cfg["tau0"], seed, "d0")
    x1, y1 = sample_domain(world, cfg["n1"], ood_classes, cfg["delta"], cfg["tau1"], seed, "d1")
    x2, y2 = sample_heldout_domain(world, cfg["n1"], ood_classes, cfg["delta2"], cfg["tau0"], seed)

    idx = np.arange(len(x0))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    nte = max(int(0.2 * len(idx)), 1)
    te, tr = idx[:nte], idx[nte:]
    x0tr, y0tr = x0[tr], y0[tr]
    x0te, y0te = x0[te], y0[te]

    net = ToyNet(cfg["D"], cfg["m"], cfg["K"], cfg["encoder"], tuple(cfg["hidden"]), cfg["bn"], cfg["adversary"])
    opt = torch.optim.AdamW(
        [
            {"params": list(net.enc.parameters()) + list(net.bn.parameters()) + list(net.cls.parameters()), "lr": cfg["lr"]},
            {"params": list(net.adv.parameters()), "lr": cfg["lr"] * cfg["adv_lr_mult"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    x1t, y1t = torch.from_numpy(x1), torch.from_numpy(y1)
    x0trt, y0trt = torch.from_numpy(x0tr), torch.from_numpy(y0tr)
    n_steps = max(len(x0tr) // (cfg["batch"] // 2), 1)
    half = cfg["batch"] // 2
    net.train()
    for _ep in range(int(cfg["epochs"])):
        perm0 = rng.integers(0, len(x0tr), size=(n_steps, half))
        perm1 = rng.integers(0, len(x1), size=(n_steps, half))
        for s in range(n_steps):
            xb = torch.cat([x0trt[perm0[s]], x1t[perm1[s]]])
            yb = torch.cat([y0trt[perm0[s]], y1t[perm1[s]]])
            db = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
            logits, dlog, _z = net(xb, lam)
            if cfg["cls_domain0_only"]:
                loss_cls = F.cross_entropy(logits[:half], yb[:half])
            else:
                loss_cls = F.cross_entropy(logits, yb)
            loss = loss_cls + float(lam) * F.cross_entropy(dlog, db)
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        def pack(x):
            lg, _d, z = net(torch.from_numpy(x), 0.0)
            return z.numpy(), lg.numpy()

        ztr, _ = pack(x0tr)
        zid, lid = pack(x0te)
        zood, lood = pack(x1)
        zd2, ld2 = pack(x2)
    assert len(ztr) == len(x0tr)
    W_lin = None
    if cfg["encoder"] == "linear":
        W_lin = net.enc.weight.detach().numpy().astype(np.float64)
    ev = evaluate(ztr, y0tr, zid, y0te, lid, zood, y1, lood, zd2, ld2, cfg["K"], ood_classes, world["u"], W_lin)
    ev.update({"cell": cell_name, "lambda_adv": float(lam), "seed": int(seed), "m": int(cfg["m"]), "encoder": cfg["encoder"]})
    return ev


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def fmt(m, s):
    if not np.isfinite(m):
        return "—"
    return "{:.3f} ± {:.3f}".format(m, s)


def validate_facts(by_lam):
    def g(lam, key):
        recs = by_lam.get(lam) or []
        if not recs:
            return float("nan")
        return mean_std([r[key] for r in recs])[0]

    a0, a025, a2 = g(0.0, "maha"), g(0.25, "maha"), g(2.0, "maha")
    early = a0 - a025
    late = a025 - a2
    F1 = bool(early > 2 * max(late, 0.0) and a0 > 0.70 and a025 < 0.60)
    F2 = bool(a2 < 0.48)
    dets = [g(2.0, k) for k in ("maha", "MSP", "Energy", "cosine", "knn50")]
    F3 = bool(np.nanmax(dets) - np.nanmin(dets) < 0.08)
    F4 = bool(abs(g(2.0, "maha") - g(2.0, "maha_id_shared_classes")) < 0.03)
    F5 = bool(g(2.0, "maha_id_reweighted_to_ood_mix") < g(2.0, "maha") - 0.03)
    F7 = bool(g(2.0, "maha_domain2") < 0.48)
    return {
        "F1_cliff": {"pass": F1, "auc0": a0, "auc025": a025, "auc2": a2},
        "F2_subchance": {"pass": F2, "auc2": a2},
        "F3_together": {"pass": F3, "detectors_at_2": dict(zip(("maha", "MSP", "Energy", "cosine", "knn50"), dets))},
        "F4_shared_classes": {"pass": F4, "auc_all": g(2.0, "maha"), "auc_shared": g(2.0, "maha_id_shared_classes")},
        "F5_reweight": {"pass": F5, "auc_all": g(2.0, "maha"), "auc_reweight": g(2.0, "maha_id_reweighted_to_ood_mix")},
        "F7_heldout_domain2": {"pass": F7, "auc_domain2": g(2.0, "maha_domain2")},
        "n_pass": int(sum([F1, F2, F3, F4, F5, F7])),
        "n_tested": 6,
    }


def write_report(grid, rows, facts_full, ablation_summary):
    lines = []
    lines.append("# Phase 13.5 — toy model for the sub-chance inversion")
    lines.append("")
    lines.append("Grid locked in `GRID_LOCKED.json` before any fit. Not retuned. Failures are findings.")
    lines.append("")
    lines.append("## Analytical baseline (linear encoder, equal-covariance Gaussians)")
    lines.append("")
    lines.append("Let \(x \\mid y=k,\\, d \\sim \\mathcal{N}(\\mu_k + \\beta_d u,\\, \\tau^2 I)\) and \(z = Wx\). Then")
    lines.append("")
    lines.append("\\[ z \\mid y=k,\\, d \\;\\sim\\; \\mathcal{N}\\big(W\\mu_k + \\beta_d\\, Wu,\\; \\tau^2 WW^\\top\\big). \\]")
    lines.append("")
    lines.append("A perfect adversary equalises the domain-conditional latents. With a shared covariance this requires the class-conditional means to match across domains for every \(k\):")
    lines.append("")
    lines.append("\\[ W\\mu_k + \\beta_0 Wu = W\\mu_k + \\beta_1 Wu \\quad\\Rightarrow\\quad (\\beta_1-\\beta_0)\\,Wu = 0. \\]")
    lines.append("")
    lines.append("If \(\\beta_1 \\neq \\beta_0\), necessarily \(Wu \\to 0\). At \(Wu=0\) the two domains have **identical** class-conditional Gaussians. A class-conditional Mahalanobis score fit on domain 0 then has identical score distributions on both domains, so **AUROC \(= 0.5\) exactly** when class priors match.")
    lines.append("")
    lines.append("Two consequences:")
    lines.append("")
    lines.append("1. A linear encoder on equal-covariance Gaussian data **cannot** produce sub-chance AUROC except through class-composition differences. F4 and F5 have already excluded composition in the real data (6-class restriction 0.414 vs 0.409; reweighting *strengthens* the inversion to 0.240).")
    lines.append("2. Therefore F2 (robust AUROC \(< 0.5\)) requires an ingredient outside the linear-Gaussian picture. Identifying the minimal such ingredient is the result of this phase.")
    lines.append("")
    lines.append("If instead \(\\tau_1 < \\tau_0\), even a linear map yields \(\\Sigma_d = \\tau_d^2 WW^\\top\), so OOD is more concentrated and Mahalanobis inverts **at \(\\lambda=0\)**. The derm number at \(\\lambda=0\) is 0.863, so a static input-variance gap is already excluded. The `tau1_half` cell exists only to confirm the toy reproduces that exclusion.")
    lines.append("")
    lines.append("## Locked full configuration")
    lines.append("")
    lines.append("MLP encoder (128–128 ReLU), \(m=16\), BatchNorm on, \(L_\\mathrm{cls}\) on domain 0 only, derm-template adversary, `adv_lr_multiplier=30`, \(n_0=20000\), \(n_1=2000\), \(K=8\), \(K'=6\), \(\\tau_0=\\tau_1=1\), 40 epochs, 5 seeds, \(\\lambda\\in\\{0,0.05,0.1,0.25,0.5,1,2,4,8\\}\). Ablations move one knob.")
    lines.append("")

    def cell_rows(name):
        return [r for r in rows if r["cell"] == name]

    def table_lambda(name):
        recs = cell_rows(name)
        by = {}
        for r in recs:
            by.setdefault(r["lambda_adv"], []).append(r)
        return by

    lines.append("## Full configuration vs λ")
    lines.append("")
    lines.append("| λ | Maha | MSP | Energy | cosine | kNN-50 | leak bal | var OOD/ID | PR |")
    lines.append("|---:|---|---|---|---|---|---|---|---|")
    by_full = table_lambda("full")
    for lam in grid["lams"]:
        recs = by_full.get(lam, [])
        def g(k):
            return fmt(*mean_std([r[k] for r in recs]))
        lines.append("| {:g} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            lam, g("maha"), g("MSP"), g("Energy"), g("cosine"), g("knn50"),
            g("leakage_bal_acc"), g("var_ood_over_id"), g("pr_id_train"),
        ))
    lines.append("")
    npass = facts_full["n_pass"] if facts_full else 0
    lines.append("## F1–F7 validation on the full configuration")
    lines.append("")
    if facts_full:
        lines.append("**{}/{}** independent facts reproduced (F6 is the never-trained domain-2 inversion, labelled F7 in the toy).".format(npass, facts_full["n_tested"]))
        lines.append("")
        lines.append("| Fact | Pass? | Detail |")
        lines.append("|---|---|---|")
        for key, lab in (
            ("F1_cliff", "F1 cliff"),
            ("F2_subchance", "F2 sub-chance"),
            ("F3_together", "F3 detectors together"),
            ("F4_shared_classes", "F4 shared-class restriction"),
            ("F5_reweight", "F5 reweight to OOD mix"),
            ("F7_heldout_domain2", "F6/F7 held-out domain"),
        ):
            blob = facts_full[key]
            lines.append("| {} | {} | {} |".format(lab, "yes" if blob["pass"] else "no", json.dumps({k: v for k, v in blob.items() if k != "pass"})))
        lines.append("")
        if npass <= 1:
            lines.append("Reproducing F2 alone is weak. This count is the evidence, not a fitted headline.")
        elif npass >= 4:
            lines.append("F2 plus several independent facts, with no extra fitting, is the strongest evidence this construction can give.")
    lines.append("")
    lines.append("## Ablation table (sub-chance at λ=2)")
    lines.append("")
    lines.append("| Cell | Maha λ=0 | Maha λ=2 | sub-chance at 2? | var OOD/ID at 2 | leak at 2 |")
    lines.append("|---|---|---|---|---|---|")
    for name, blob in ablation_summary.items():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name,
            fmt(*blob["auc0"]),
            fmt(*blob["auc2"]),
            "yes" if blob["subchance"] else "no",
            fmt(*blob["var2"]),
            fmt(*blob["leak2"]),
        ))
    lines.append("")
    necessary = [n for n, b in ablation_summary.items() if n != "full" and ablation_summary["full"]["subchance"] and not b["subchance"]]
    sufficient_note = "Sub-chance on full died when: {}.".format(", ".join(necessary) if necessary else "no single ablation killed it")
    if not ablation_summary.get("full", {}).get("subchance"):
        sufficient_note = (
            "Full configuration did **not** produce sub-chance. No minimal reproducing config exists on this grid. "
            "The effect requires something absent from the model (image statistics, a pretrained backbone, or optimisation at scale). "
            "Not retuned."
        )
    lines.append(sufficient_note)
    lines.append("")
    lin = ablation_summary.get("linear_encoder", {})
    if lin:
        if lin.get("subchance"):
            lines.append("Linear encoder **did** go sub-chance. Non-linearity is not necessary on this grid.")
        elif ablation_summary["full"]["subchance"]:
            lines.append("Linear saturates at or above chance; only the MLP went below. Non-linearity is necessary. Clean finding.")
        else:
            lines.append("Linear did not invert, and neither did the MLP full config.")
    lines.append("")
    tau = ablation_summary.get("tau1_half", {})
    if tau:
        lines.append(
            "`tau1_half` at λ=0: Maha {}. If this is already < 0.5 while full λ=0 is not, the toy confirms that a static input-variance gap inverts without any adversary — and is therefore not the derm mechanism.".format(
                fmt(*tau["auc0"])
            )
        )
        lines.append("")
    lines.append("## Mechanism statement")
    lines.append("")
    hyp = (
        "Leading hypothesis: a capacity-constrained encoder matches the ID mode, not the tails; "
        "label-asymmetric CE spreads domain 0 into class clusters while OOD is packed into the core."
    )
    if ablation_summary.get("full", {}).get("subchance"):
        var2 = ablation_summary["full"]["var2"][0]
        both = ablation_summary.get("ce_both_domains", {})
        if var2 < 1.0 and both and not both.get("subchance"):
            lines.append(
                hyp + " Supported on this grid: Var(z|d=1)/Var(z|d=0) = {:.3f} at λ=2, and removing label asymmetry abolished the inversion.".format(var2)
            )
        elif var2 < 1.0:
            lines.append(hyp + " Partially supported: variance ratio {:.3f} at λ=2, but label-asymmetry was not necessary or not decisive.".format(var2))
        else:
            lines.append(hyp + " Not supported: variance ratio did not fall below 1. Sub-chance occurred by another route. Do not invent a story.")
    else:
        lines.append(
            "None found. The controlled synthetic reproduction failed. The derm paper reports a robust phenomenon, "
            "four refuted explanations, and an explicit statement that this toy did not invert. That is the publishable position."
        )
    lines.append("")
    lines.append("Figures: `phase13_5/figures/fig_13_5_*.{png,pdf}`.")
    (OUT / "PHASE13_5_REPORT.md").write_text("\n".join(lines) + "\n")


def write_figures(grid, rows):
    FIG.mkdir(parents=True, exist_ok=True)
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], {}).setdefault(r["lambda_adv"], []).append(r)

    def series(cell, key):
        xs, ys, es = [], [], []
        for lam in grid["lams"]:
            recs = by_cell.get(cell, {}).get(lam, [])
            m, s = mean_std([r[key] for r in recs])
            xs.append(lam)
            ys.append(m)
            es.append(s)
        return xs, ys, es

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x, y, e = series("full", "maha")
    ax.errorbar(x, y, yerr=e, color=WONG["orange"], marker="o", lw=1.8, label="toy full Maha")
    derm_x = [k for k in DERM_AUROC if k in grid["lams"] or k in DERM_AUROC]
    ax.plot(list(DERM_AUROC.keys()), list(DERM_AUROC.values()), color=WONG["blue"], marker="s", lw=1.4, ls="--", label="derm Maha (Phase 3)")
    if "linear_encoder" in by_cell:
        x, y, e = series("linear_encoder", "maha")
        ax.errorbar(x, y, yerr=e, color=WONG["gray"], marker="^", lw=1.2, label="toy linear")
    ax.axhline(0.5, color=WONG["gray"], ls=":", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("Mahalanobis AUROC")
    ax.set_title("13.5: toy vs derm — not retuned")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_5_auroc.{}".format(ext), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x, y, e = series("full", "var_ood_over_id")
    ax.errorbar(x, y, yerr=e, color=WONG["purple"], marker="o", lw=1.8, label="Var(z|OOD) / Var(z|ID)")
    ax.axhline(1.0, color=WONG["gray"], ls="--", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("variance ratio")
    ax.set_title("13.5: does OOD collapse into the core?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_5_var_ratio.{}".format(ext), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_jobs", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    grid = json.loads(GRID_PATH.read_text())
    full = grid["full"]
    cells = {"full": deepcopy(full)}
    for name, patch in grid["ablations_one_at_a_time"].items():
        cfg = deepcopy(full)
        cfg.update(patch)
        cells[name] = cfg
    jobs = []
    for name, cfg in cells.items():
        for lam in grid["lams"]:
            for seed in grid["seeds"]:
                jobs.append((name, cfg, float(lam), int(seed)))
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    (OUT / "job_list.json").write_text(json.dumps({"n": len(jobs), "cells": list(cells)}, indent=2) + "\n")
    if args.dry_run:
        print("dry_run n_jobs", len(jobs), "cells", list(cells))
        return

    rows = []
    done_path = OUT / "all_rows.jsonl"
    if done_path.exists():
        done_path.unlink()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            rows.append(rec)
            with done_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if i % 20 == 0 or i == len(jobs):
                print("done", i, "/", len(jobs), rec["cell"], rec["lambda_adv"], rec["seed"], "maha", rec["maha"], flush=True)

    (OUT / "all_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    by_full = {}
    for r in rows:
        if r["cell"] == "full":
            by_full.setdefault(r["lambda_adv"], []).append(r)
    facts = validate_facts(by_full) if by_full else None
    ablation = {}
    for name in cells:
        recs = [r for r in rows if r["cell"] == name]
        a0 = mean_std([r["maha"] for r in recs if abs(r["lambda_adv"] - 0.0) < 1e-12])
        a2 = mean_std([r["maha"] for r in recs if abs(r["lambda_adv"] - 2.0) < 1e-12])
        v2 = mean_std([r["var_ood_over_id"] for r in recs if abs(r["lambda_adv"] - 2.0) < 1e-12])
        l2 = mean_std([r["leakage_bal_acc"] for r in recs if abs(r["lambda_adv"] - 2.0) < 1e-12])
        ablation[name] = {"auc0": a0, "auc2": a2, "var2": v2, "leak2": l2, "subchance": bool(a2[0] < 0.48)}
    (OUT / "facts_full.json").write_text(json.dumps(facts, indent=2) + "\n")
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2) + "\n")
    write_figures(grid, rows)
    write_report(grid, rows, facts, ablation)
    print("n_pass", None if facts is None else facts["n_pass"])
    print("wrote", OUT / "PHASE13_5_REPORT.md")


if __name__ == "__main__":
    main()
