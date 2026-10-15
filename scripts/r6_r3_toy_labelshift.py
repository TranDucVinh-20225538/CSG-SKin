#!/usr/bin/env python3
"""Paper B R6 R3: Gaussian-mixture toy with class-prior (label) shift between the two training domains
(decisions/paperB_r6_r3.md). CPU only.

What changes relative to Phase 13.5 v1/v2 (train_phase13_5_toy.py / _v2.py): those worlds had uniform class priors
(v1: source uniform over 8 classes, target uniform over 6; v2: 6 equal common + 2 rare source classes, target uniform)
— no class-prior shift among shared classes, no majority class — and the OOD set was the adversary's own target sample.
Here everything else is the locked v1 world (D = 50, K = 8, K_ood = 6, label_scale 2, δ = 2 along u, τ = 1, MLP
[128, 128] → m = 16, mlp_derm adversary, adversary LR ×30, 40 epochs, batch 256, AdamW 1e-3 / 1e-4, classifier on the
source only) plus:
  * source prior ISIC-like, majority class 0 at 0.50;
  * target prior q(s) = (1 − s)·q0 + s·q_PAD over the 6 shared classes, q0 = source prior restricted to them
    (majority 0.515), q_PAD = PAD-UFES-like mix (majority 0.12, class 4 "AK-like" 0.315); s ∈ {0, .25, .5, .75, 1};
  * target images grouped into patients (random patient offset, σ_p = 0.5); 30 % of patients held out: the adversary
    never sees them and they are the OOD evaluation set ("held-out target cluster"); a never-seen third domain (δ2 = 3
    along u2, prior q(s)) is the secondary OOD set;
  * GRL convention of the main model: gradient-reversal coefficient 1, adversary loss weighted by λ (v1 used λ in both).
Cells: primary = BN off × s grid; bn_on control (s ∈ {0, 1}); uniform-prior control (v1 priors, BN off); oracle =
adversary's source half-batch resampled to the target class mix (label-shift-corrected alignment), s ∈ {0.5, 1}.
Metrics per run: Mahalanobis AUROC (fit on source train; ID = source test; OOD = held-out target patients), the four
other detectors, domain-2 AUROC, majority-centroid share of OOD and of ID test, predicted-majority share on OOD,
class-4 recall on OOD, AUROC with ID reweighted to the OOD class mix, leakage probe.
Usage: r6_r3_toy_labelshift.py [--workers 32] [--cells primary bn_on uniform oracle] [--report_only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase13_5_toy as v1  # noqa: E402

torch.set_num_threads(1)
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r6/r3")
LOCKED = {"D": 50, "K": 8, "K_ood": 6, "n0": 20000, "n_t": 2900, "delta": 2.0, "delta2": 3.0, "label_scale": 2.0,
          "tau": 1.0, "encoder": "mlp", "hidden": [128, 128], "m": 16, "adversary": "mlp_derm", "adv_lr_mult": 30.0,
          "epochs": 40, "batch": 256, "lr": 1e-3, "weight_decay": 1e-4, "patient_sd": 0.5, "imgs_per_patient": 1.7,
          "heldout_patient_frac": 0.3}
P_SRC = np.array([0.50, 0.18, 0.13, 0.10, 0.035, 0.025, 0.015, 0.015])
Q_PAD = np.array([0.12, 0.025, 0.36, 0.10, 0.315, 0.08])
SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
SEEDS = (0, 1, 2, 3, 4)
MAJ, AKL = 0, 4
CELLS = {"primary": {"bn": False, "shifts": SHIFTS, "prior": "ISIC-like", "oracle": False},
         "bn_on": {"bn": True, "shifts": (0.0, 1.0), "prior": "ISIC-like", "oracle": False},
         "uniform": {"bn": False, "shifts": (None,), "prior": "uniform", "oracle": False},
         "oracle": {"bn": False, "shifts": (0.5, 1.0), "prior": "ISIC-like", "oracle": True}}


def priors(cell, s):
    if CELLS[cell]["prior"] == "uniform":
        return np.full(8, 1 / 8), np.full(6, 1 / 6)
    q0 = P_SRC[:6] / P_SRC[:6].sum()
    return P_SRC, (1 - s) * q0 + s * Q_PAD


def sample(world, n, prior, shift_vec, delta, rng, patients=False):
    y = rng.choice(len(prior), size=n, p=prior)
    x = world["mu"][y] + delta * shift_vec[None, :] + LOCKED["tau"] * rng.normal(size=(n, world["mu"].shape[1]))
    pid = None
    if patients:
        n_p = int(round(n / LOCKED["imgs_per_patient"]))
        pid = rng.integers(0, n_p, size=n)
        x = x + LOCKED["patient_sd"] * rng.normal(size=(n_p, x.shape[1]))[pid]
    return x.astype(np.float32), y.astype(np.int64), pid


def run_one(payload):
    cell, s, lam, seed = payload
    cfg = CELLS[cell]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed * 7919 + int(100 * (s or 0)) + 3)
    world = v1.make_world({"D": LOCKED["D"], "K": LOCKED["K"], "label_scale": LOCKED["label_scale"]}, seed)
    p_s, q_t = priors(cell, s)
    x0, y0, _ = sample(world, LOCKED["n0"], p_s, world["u"], 0.0, rng)
    xt, yt, pt = sample(world, LOCKED["n_t"], q_t, world["u"], LOCKED["delta"], rng, patients=True)
    x2, y2, _ = sample(world, 2000, q_t, world["u2"], LOCKED["delta2"], rng)
    held_p = rng.permutation(np.unique(pt))[: int(LOCKED["heldout_patient_frac"] * len(np.unique(pt)))]
    hmask = np.isin(pt, held_p)
    xa, ya = xt[~hmask], yt[~hmask]
    xo, yo, po = xt[hmask], yt[hmask], pt[hmask]
    idx = rng.permutation(len(x0))
    nte = int(0.2 * len(idx))
    te, tr = idx[:nte], idx[nte:]
    net = v1.ToyNet(LOCKED["D"], LOCKED["m"], LOCKED["K"], LOCKED["encoder"], tuple(LOCKED["hidden"]), cfg["bn"], LOCKED["adversary"])
    opt = torch.optim.AdamW([{"params": list(net.enc.parameters()) + list(net.bn.parameters()) + list(net.cls.parameters()), "lr": LOCKED["lr"]},
                             {"params": list(net.adv.parameters()), "lr": LOCKED["lr"] * LOCKED["adv_lr_mult"]}],
                            weight_decay=LOCKED["weight_decay"])
    X0, Y0, XA = torch.from_numpy(x0[tr]), torch.from_numpy(y0[tr]), torch.from_numpy(xa)
    half = LOCKED["batch"] // 2
    n_steps = max(len(tr) // half, 1)
    if cfg["oracle"]:
        w = np.zeros(8)
        w[:6] = q_t / p_s[:6]
        pw = w[y0[tr]] / w[y0[tr]].sum()
    net.train()
    for _ep in range(LOCKED["epochs"]):
        b0 = rng.integers(0, len(tr), size=(n_steps, half))
        ba = rng.integers(0, len(xa), size=(n_steps, half))
        bo = rng.choice(len(tr), size=(n_steps, half), p=pw) if cfg["oracle"] else b0
        for k in range(n_steps):
            z0 = net.encode(X0[b0[k]])
            loss = F.cross_entropy(net.cls(z0), Y0[b0[k]])
            if lam > 0:
                zs = z0 if not cfg["oracle"] else net.encode(X0[bo[k]])
                zd = torch.cat([zs, net.encode(XA[ba[k]])])
                dl = net.adv(v1._GRL.apply(zd, 1.0))
                db = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
                loss = loss + float(lam) * F.cross_entropy(dl, db)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        enc = lambda x: (net.encode(torch.from_numpy(x)).numpy(), net.cls(net.encode(torch.from_numpy(x))).numpy())  # noqa: E731
        ztr, _ = enc(x0[tr])
        zid, lid = enc(x0[te])
        zo, lo = enc(xo)
        z2, l2 = enc(x2)
    out = v1.evaluate(ztr, y0[tr], zid, y0[te], lid, zo, yo, lo, z2, l2, LOCKED["K"], list(range(6)), world["u"], None)
    means, prec, present = v1.fit_maha(ztr, y0[tr], LOCKED["K"])
    near = lambda z: present[(np.einsum("nd,nd->n", z @ prec, z)[:, None] - 2 * (z @ prec) @ means[present].T  # noqa: E731
                              + np.einsum("cd,cd->c", means[present] @ prec, means[present])[None]).argmin(1)]
    pred_o = lo.argmax(1)
    out.update({"cell": cell, "shift": s, "lambda": lam, "seed": seed, "bn": cfg["bn"], "oracle": cfg["oracle"],
                "p_src_major": float(p_s[MAJ]), "q_tgt_major": float(q_t[MAJ]), "tv_shared": float(0.5 * np.abs(p_s[:6] / p_s[:6].sum() - q_t).sum()),
                "n_ood": int(len(yo)), "n_ood_patients": int(len(np.unique(po))), "n_adv_target": int(len(ya)),
                "maj_share_ood": float((near(zo) == MAJ).mean()), "maj_share_id": float((near(zid) == MAJ).mean()),
                "maj_share_d2": float((near(z2) == MAJ).mean()), "pred_maj_ood": float((pred_o == MAJ).mean()),
                "recall_c4_ood": float((pred_o[yo == AKL] == AKL).mean()) if (yo == AKL).any() else float("nan"),
                "id_bal_acc": float(np.mean([(lid.argmax(1)[y0[te] == c] == c).mean() for c in range(8) if (y0[te] == c).any()]))})
    return out


def jobs(cells):
    return [(c, s, lam, seed) for c in cells for s in CELLS[c]["shifts"] for lam in LAMS for seed in SEEDS]


def fname(c, s, lam, seed):
    return "{}_s{}_lam{}_seed{}.json".format(c, "na" if s is None else "{:g}".format(s), "{:g}".format(lam), seed)


def report():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("*.json"))]
    df = pd.DataFrame(rows)
    df["shift_key"] = df["shift"].fillna(-1.0)
    keep = [c for c in df.columns if not isinstance(df[c].iloc[0], (dict, list))]
    df[keep].to_csv(OUT / "summary.csv", index=False)
    agg = df.groupby(["cell", "shift_key", "lambda"]).agg(
        n=("seed", "size"), maha=("maha", "mean"), maha_sd=("maha", "std"), k_below=("maha", lambda v: int((v < 0.5).sum())),
        maha_d2=("maha_domain2", "mean"), maj_ood=("maj_share_ood", "mean"), maj_id=("maj_share_id", "mean"),
        pred_maj=("pred_maj_ood", "mean"), rec4=("recall_c4_ood", "mean"), leak=("leakage_bal_acc", "mean"),
        rew=("maha_id_reweighted_to_ood_mix", "mean"), knn=("knn50", "mean"), msp=("MSP", "mean")).reset_index()
    agg.to_csv(OUT / "cells.csv", index=False)
    L = ["# Paper B R6 R3 — toy model with label shift", "",
         "Design: `decisions/paperB_r6_r3.md`. Mahalanobis AUROC (fit on source train) of source test vs held-out target patients; "
         "mean ± s.d. over 5 seeds, k = seeds below 0.5. s = label-shift level (0: target mix = source mix on shared classes, majority "
         "0.515; 1: PAD-like, majority 0.12).", ""]
    tex = []
    for cell in CELLS:
        a = agg[agg.cell == cell]
        if not len(a):
            continue
        L += ["## " + cell, "", "| s | λ | n | Maha AUROC | k<0.5 | domain-2 AUROC | maj-centroid share OOD / ID | pred. majority OOD | class-4 recall | leakage | AUROC ID→OOD mix |",
              "|---|---:|---:|---|---|---|---|---|---|---|---|"]
        for x in a.itertuples():
            L.append("| {} | {:g} | {} | {:.3f} ± {:.3f} | {}/{} | {:.3f} | {:.3f} / {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                "—" if x.shift_key < 0 else "{:g}".format(x.shift_key), x._3, x.n, x.maha, x.maha_sd, x.k_below, x.n, x.maha_d2, x.maj_ood,
                x.maj_id, x.pred_maj, x.rec4, x.leak, x.rew))
            if cell == "primary":
                tex.append("{:g} & {:g} & {:.3f}\\,$\\pm$\\,{:.3f} & {:.3f} & {:.3f} \\\\".format(x.shift_key, x._3, x.maha, x.maha_sd, x.maj_ood, x.leak))
        L.append("")
    p = agg[agg.cell == "primary"]
    z = p[p["lambda"] == 0]
    flags = ["s = {:g}: λ = 0 Mahalanobis {:.3f} < 0.6 (no headroom)".format(x.shift_key, x.maha) for x in z.itertuples() if x.maha < 0.6]
    flags += ["s = {:g}: λ = 0 leakage {:.3f} < 0.8 (domain shift too weak)".format(x.shift_key, x.leak) for x in z.itertuples() if x.leak < 0.8]
    L += ["## Gate flags (primary)", ""] + (["- " + f for f in flags] if flags else ["None."]) + [""]
    if len(p):
        mins = p.groupby("shift_key").maha.min()
        L += ["## Plain reading", "", "Primary (BN off): minimum over λ of the mean Mahalanobis AUROC, by label-shift level: " +
              "; ".join("s = {:g}: {:.3f}".format(k, v) for k, v in mins.items()) +
              ". At λ = 0 the AUROC is {:.3f}–{:.3f} across s.".format(p[p["lambda"] == 0].maha.min(), p[p["lambda"] == 0].maha.max())]
        u = agg[(agg.cell == "uniform")]
        if len(u):
            L.append("Uniform-prior control: minimum over λ {:.3f}.".format(u.maha.min()))
        o = agg[(agg.cell == "oracle")]
        if len(o):
            L.append("Oracle (adversary sees the source resampled to the target mix): minimum over λ " +
                     "; ".join("s = {:g}: {:.3f}".format(k, v) for k, v in o.groupby("shift_key").maha.min().items()) + ".")
    L += ["", "## LaTeX rows (s & λ & Maha AUROC & majority-centroid share OOD & leakage), primary", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--cells", nargs="*", default=list(CELLS))
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        (OUT / "config.json").write_text(json.dumps({"locked": LOCKED, "p_src": P_SRC.tolist(), "q_pad": Q_PAD.tolist(), "shifts": SHIFTS,
                                                     "lams": LAMS, "seeds": SEEDS, "cells": {k: {**v, "shifts": list(v["shifts"])} for k, v in CELLS.items()}}, indent=2))
        todo = [j for j in jobs(a.cells) if not (OUT / "runs" / fname(*j)).exists()]
        print(len(todo), "runs to do", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            fut = {ex.submit(run_one, j): j for j in todo}
            for f in as_completed(fut):
                j = fut[f]
                (OUT / "runs" / fname(*j)).write_text(json.dumps(f.result(), default=float))
                print("done", fname(*j), flush=True)
    report()


if __name__ == "__main__":
    main()
