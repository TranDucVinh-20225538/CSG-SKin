#!/usr/bin/env python3
"""Phase 13.5 — small analysable model of the invariance / OOD inversion.

Do not retune toward sub-chance AUROC. If the toy cannot invert, that is the result.
CPU minutes. Outputs under results/paperB/phase13/toy/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase13/toy")
FIG = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase13/figures")
WONG = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "gray": "#888888"}


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class Encoder(nn.Module):
    def __init__(self, d_in, d_z, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(d_in, d_z, bias=True)
        elif kind == "mlp":
            h = max(d_z * 2, 32)
            self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_z))
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x)


def make_data(n_per, n_classes, d_label, d_domain, d_noise, seed, mu1=2.0, sig=0.4):
    rng = np.random.default_rng(seed)
    # orthonormal-ish axes
    d = d_label + d_domain + d_noise
    y0 = rng.integers(0, n_classes, size=n_per)
    y1 = rng.integers(0, n_classes, size=n_per)
    x0 = rng.normal(0.0, sig, size=(n_per, d))
    x1 = rng.normal(0.0, sig, size=(n_per, d))
    # label occupies first d_label dims via one-hot-ish
    for i in range(n_per):
        x0[i, y0[i] % d_label] += 1.6
        x1[i, y1[i] % d_label] += 1.6
    # domain occupies next d_domain dims
    x0[:, d_label : d_label + d_domain] += 0.0
    x1[:, d_label : d_label + d_domain] += mu1
    y = np.concatenate([y0, y1])
    dlab = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])
    x = np.concatenate([x0, x1]).astype(np.float32)
    return x, y.astype(np.int64), dlab


def maha_auroc(z_id_tr, y_id_tr, z_id_te, z_ood, n_classes, eps=1e-3):
    z_id_tr = np.asarray(z_id_tr, dtype=np.float64)
    z_id_te = np.asarray(z_id_te, dtype=np.float64)
    z_ood = np.asarray(z_ood, dtype=np.float64)
    y_id_tr = np.asarray(y_id_tr, dtype=int)
    means = []
    centered = []
    for c in range(n_classes):
        m = y_id_tr == c
        if not np.any(m):
            continue
        mu = z_id_tr[m].mean(axis=0)
        means.append(mu)
        centered.append(z_id_tr[m] - mu)
    means = np.stack(means)
    c = np.concatenate(centered)
    cov = (c.T @ c) / max(len(c) - len(means), 1)
    cov = cov + eps * np.eye(cov.shape[0])
    prec = np.linalg.pinv(cov)

    def sc(z):
        zp = z @ prec
        zpz = np.einsum("nd,nd->n", zp, z)
        mp = means @ prec
        mpm = np.einsum("cd,cd->c", mp, means)
        d2 = zpz[:, None] + mpm[None, :] - 2.0 * (zp @ means.T)
        return d2.min(axis=1)

    sid, sood = sc(z_id_te), sc(z_ood)
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def train_once(kind, d_z, lam, seed, n_classes=8, d_label=8, d_domain=4, d_noise=20, n_per=2000, epochs=25, lr=1e-2, cls_both=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y, dom = make_data(n_per, n_classes, d_label, d_domain, d_noise, seed)
    d_in = x.shape[1]
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    dt = torch.from_numpy(dom)
    enc = Encoder(d_in, d_z, kind)
    clf = nn.Linear(d_z, n_classes)
    adv = nn.Sequential(nn.Linear(d_z, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()) + list(adv.parameters()), lr=lr)
    for _ in range(epochs):
        z = enc(xt)
        logits = clf(z)
        if cls_both:
            loss_cls = F.cross_entropy(logits, yt)
        else:
            m0 = dt == 0
            loss_cls = F.cross_entropy(logits[m0], yt[m0])
        dlog = adv(_GRL.apply(z, 1.0))
        loss = loss_cls + float(lam) * F.cross_entropy(dlog, dt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = enc(xt).numpy()
    m0 = dom == 0
    m1 = dom == 1
    # hold out 30% of domain 0 as ID test
    rng = np.random.default_rng(seed + 7)
    idx0 = np.where(m0)[0]
    rng.shuffle(idx0)
    nte = max(int(0.3 * len(idx0)), 1)
    te, tr = idx0[:nte], idx0[nte:]
    auc = maha_auroc(z[tr], y[tr], z[te], z[m1], n_classes)
    return {
        "kind": kind,
        "d_z": int(d_z),
        "lambda_adv": float(lam),
        "seed": int(seed),
        "cls_both_domains": bool(cls_both),
        "maha_auroc": auc,
        "z_var_id": float(z[m0].var(axis=0).sum()),
        "z_var_ood": float(z[m1].var(axis=0).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    grid = {
        "kinds": ("linear", "mlp"),
        "d_z": (4, 16, 64),
        "lams": (0.0, 0.25, 1.0, 2.0, 8.0),
        "seeds": (0, 1, 2),
        "cls_both": (False, True),
    }
    if args.dry_run:
        print("dry_run", json.dumps(grid))
        return

    rows = []
    for kind in grid["kinds"]:
        for dz in grid["d_z"]:
            for lam in grid["lams"]:
                for seed in grid["seeds"]:
                    for both in grid["cls_both"]:
                        rec = train_once(kind, dz, lam, seed, cls_both=both)
                        rows.append(rec)
                        print(rec, flush=True)

    # classify: does any setting reach AUROC < 0.5?
    sub = [r for r in rows if r["maha_auroc"] < 0.48 and not r["cls_both_domains"]]
    linear_sub = [r for r in sub if r["kind"] == "linear"]
    mlp_sub = [r for r in sub if r["kind"] == "mlp"]
    asym_sub = [r for r in sub if not r["cls_both_domains"]]
    both_sub = [r for r in rows if r["maha_auroc"] < 0.48 and r["cls_both_domains"]]
    if not sub and not both_sub:
        reading = (
            "The toy never produced sub-chance Mahalanobis AUROC. "
            "The derm inversion requires something absent from this construction "
            "(capacity, non-linearity, or the data model). Not retuned. Finding."
        )
        verdict = "NO_INVERSION"
    elif not linear_sub and mlp_sub:
        reading = (
            "Linear encoder saturates at chance; MLP reaches sub-chance. "
            "Inversion requires non-linearity (or extra capacity), not the GRL term alone."
        )
        verdict = "NONLINEARITY_REQUIRED"
    elif linear_sub:
        reading = (
            "A linear encoder already reaches sub-chance. Inversion can be produced by the "
            "objective + label-asymmetry without a compressed non-linear latent."
        )
        verdict = "LINEAR_SUFFICES"
    else:
        reading = "Sub-chance appeared only in mixed cells. See table; do not overclaim."
        verdict = "MIXED"
    if both_sub and not asym_sub:
        reading += " Label-asymmetry was necessary: L_cls on both domains never inverted."
    elif asym_sub and not both_sub:
        reading += " Label-asymmetry (L_cls on domain 0 only) was the setting that inverted."

    summary = {"verdict": verdict, "reading": reading, "n": len(rows), "rows": rows, "grid": grid}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # plot main cell: mlp, d_z=16, label-asymmetric
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for kind, color in (("linear", WONG["blue"]), ("mlp", WONG["orange"])):
        xs, ys, es = [], [], []
        for lam in grid["lams"]:
            vals = [
                r["maha_auroc"]
                for r in rows
                if r["kind"] == kind and r["d_z"] == 16 and abs(r["lambda_adv"] - lam) < 1e-12 and not r["cls_both_domains"]
            ]
            xs.append(lam)
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=es, color=color, marker="o", lw=1.8, label="{} d_z=16, L_cls on D0".format(kind))
    ax.axhline(0.5, color=WONG["gray"], ls="--", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("Mahalanobis AUROC (D1 vs held-out D0)")
    ax.set_title("13.5 toy — not retuned")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_5_toy.{}".format(ext), dpi=300)
    plt.close(fig)

    md = ["# Phase 13.5 — toy model", "", "**{}**".format(verdict), "", reading, "", "| kind | d_z | λ | L_cls | Maha AUROC |", "|---|---:|---:|---|---|"]
    # aggregate
    from collections import defaultdict

    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["kind"], r["d_z"], r["lambda_adv"], r["cls_both_domains"])].append(r["maha_auroc"])
    for key in sorted(bucket):
        a = np.asarray(bucket[key])
        md.append("| {} | {} | {:g} | {} | {:.3f} ± {:.3f} |".format(
            key[0], key[1], key[2], "both" if key[3] else "D0 only", a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
        ))
    md.append("")
    md.append("Not retuned toward inversion. Figure: `fig_13_5_toy.{png,pdf}`.")
    (OUT / "PHASE13_5_TOY.md").write_text("\n".join(md) + "\n")
    print(verdict)
    print(reading)


if __name__ == "__main__":
    main()
