#!/usr/bin/env python3
"""Phase 13.1–13.4 and 13.6 on cached Phase 3 features + summaries.

13.6 k=50 is available from Phase 3 JSON even before extraction.
k ∈ {10, 200} and 13.1–13.4 need results/paperB/phase13/features/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase3_sweep as p3
from eval_ood_dual_branch import cosine_ood_scores, fit_class_means

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
FEAT = PAPERB / "results" / "paperB" / "phase13" / "features"
OUT = PAPERB / "results" / "paperB" / "phase13"
FIG = OUT / "figures"
P3 = PAPERB / "results" / "paperB" / "phase3_sweep"
PAD_CLASSES = np.array([0, 1, 2, 3, 4, 7], dtype=int)
PAD_NAMES = ("MEL", "NV", "BCC", "AK", "BKL", "SCC")
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
EPS_SWEEP = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
HEADLINE_EPS = 1e-3
WONG = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "gray": "#888888",
    "black": "#000000",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
}


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def fmt_ms(m, s):
    if not np.isfinite(m):
        return "—"
    return "{:.3f} ± {:.3f}".format(m, s)


def auroc_pair(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def load_summaries():
    rows = []
    for lam, seed in p3.JOBS:
        tag = "runB_orth1_ladv{}_s{}".format(lam_tag(lam), seed)
        p = P3 / tag / "summary.json"
        rows.append(json.loads(p.read_text()))
    return rows


def has_features(tag):
    d = FEAT / tag
    return all((d / "{}.npz".format(s)).exists() for s in ("isic_train", "isic_test", "pad_full", "pad_heldout", "fitzpatrick17k"))


def load_split(tag, split):
    return dict(np.load(FEAT / tag / "{}.npz".format(split)))


def spectrum_of(z):
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(axis=0, keepdims=True)
    n, d = zc.shape
    if n > d:
        cov = (zc.T @ zc) / max(n - 1, 1)
        evals = np.linalg.eigvalsh(cov)
    else:
        s = np.linalg.svd(zc, compute_uv=False)
        evals = np.zeros(d, dtype=np.float64)
        evals[: len(s)] = (s ** 2) / max(n - 1, 1)
    evals = np.clip(np.sort(evals)[::-1], 0.0, None)
    tot = float(evals.sum())
    if tot <= 0:
        return {"evals": evals, "pr": float("nan"), "effective_rank": float("nan")}
    pr = float((tot ** 2) / max(float((evals ** 2).sum()), 1e-18))
    p = evals / tot
    ppos = p[p > 0]
    erank = float(np.exp(-(ppos * np.log(ppos)).sum()))
    csum = np.cumsum(p)
    def n_var(t):
        idx = np.searchsorted(csum, t)
        return int(min(idx + 1, d))
    pos = evals[evals > 0]
    kappa = float(pos[0] / pos[-1]) if pos.size else float("nan")
    return {
        "evals": evals,
        "pr": pr,
        "effective_rank": erank,
        "n90": n_var(0.90),
        "n95": n_var(0.95),
        "n99": n_var(0.99),
        "condition_number": kappa,
        "trace": tot,
        "d": int(d),
        "n": int(n),
    }


def pooled_centered(z, y, n_classes=8):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    means = np.zeros((n_classes, z.shape[1]), dtype=np.float64)
    present = []
    for c in range(n_classes):
        mask = y == c
        if not np.any(mask):
            continue
        means[c] = z[mask].mean(axis=0)
        present.append(c)
    centered = z - means[y]
    return means, centered, present


def maha_min_sq(z, means, precision):
    z = np.asarray(z, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    p = np.asarray(precision, dtype=np.float64)
    zp = z @ p
    zpz = np.einsum("nd,nd->n", zp, z)
    mp = means @ p
    mpm = np.einsum("cd,cd->c", mp, means)
    d2 = zpz[:, None] + mpm[None, :] - 2.0 * (zp @ means.T)
    return d2.min(axis=1)


def cov_kappa(cov):
    ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    pos = ev[ev > 0]
    if pos.size == 0:
        return float("nan")
    return float(pos[-1] / pos[0])


def maha_ridge(ztr, ytr, zid, zood, eps, n_classes=8):
    means, centered, present = pooled_centered(ztr, ytr, n_classes)
    denom = max(len(ztr) - len(present), 1)
    cov = (centered.T @ centered) / denom
    cov = cov + float(eps) * np.eye(cov.shape[0])
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    return {
        "auroc": auroc_pair(maha_min_sq(zid, means, prec), maha_min_sq(zood, means, prec)),
        "condition_number": cov_kappa(cov),
        "eps": float(eps),
        "estimator": "pooled + eps I",
    }


def maha_ledoit(ztr, ytr, zid, zood, n_classes=8):
    means, centered, _present = pooled_centered(ztr, ytr, n_classes)
    lw = LedoitWolf().fit(centered)
    cov = np.asarray(lw.covariance_, dtype=np.float64)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    return {
        "auroc": auroc_pair(maha_min_sq(zid, means, prec), maha_min_sq(zood, means, prec)),
        "condition_number": cov_kappa(cov),
        "shrinkage": float(lw.shrinkage_),
        "estimator": "LedoitWolf on pooled within-class residuals",
    }


def knn_auroc(ztr, zid, zood, k):
    zn = ztr / (np.linalg.norm(ztr, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=min(int(k), len(zn)), algorithm="auto", metric="euclidean")
    nn.fit(zn)

    def sc(z):
        zz = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        dist, _ = nn.kneighbors(zz)
        return dist.mean(axis=1)

    return auroc_pair(sc(zid), sc(zood))


def leakage_bal(zid, zood, seeds=(42, 52, 62)):
    z = np.concatenate([zid, zood])
    y = np.concatenate([np.zeros(len(zid)), np.ones(len(zood))])
    bals = []
    for s in seeds:
        ztr, zte, ytr, yte = train_test_split(z, y, test_size=0.3, random_state=s, stratify=y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(ztr, ytr)
        bals.append(float(balanced_accuracy_score(yte, clf.predict(zte))))
    return float(np.mean(bals)), float(np.std(bals, ddof=1))


def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def ece_from_probs(p, y, n_bins=15):
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def conf_pack(logits, labels=None, restrict=None):
    if restrict is not None:
        logits = logits[:, restrict]
        if labels is not None:
            labels = np.asarray(labels)
    p = softmax(logits)
    msp = p.max(axis=1)
    out = {
        "msp_mean": float(msp.mean()),
        "msp_std": float(msp.std(ddof=1) if len(msp) > 1 else 0.0),
        "msp_median": float(np.median(msp)),
        "entropy_mean": float(entropy(p).mean()),
        "n": int(len(msp)),
    }
    if labels is not None and restrict is not None:
        remap = {int(c): i for i, c in enumerate(restrict)}
        y = np.array([remap.get(int(v), -1) for v in labels], dtype=int)
        ok = y >= 0
        if ok.any():
            out["ece"] = ece_from_probs(p[ok], y[ok])
    elif labels is not None:
        out["ece"] = ece_from_probs(p, np.asarray(labels))
    return out, msp


def neural_collapse(z, y, classes):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    global_mu = z.mean(axis=0)
    means = []
    sw_tr = 0.0
    n_used = 0
    for c in classes:
        m = y == c
        if m.sum() < 2:
            continue
        mu = z[m].mean(axis=0)
        means.append(mu)
        sw_tr += float(((z[m] - mu) ** 2).sum())
        n_used += int(m.sum())
    if len(means) < 2:
        return None
    means = np.stack(means)
    sb_tr = float(((means - means.mean(axis=0)) ** 2).sum() * (n_used / len(means)))
    var = float(((z - global_mu) ** 2).sum() / max(len(z), 1))
    d_global = float(np.linalg.norm(z - global_mu, axis=1).mean())
    return {
        "within_over_between": float(sw_tr / max(sb_tr, 1e-18)),
        "variance_to_global": var,
        "mean_dist_to_global_centroid": d_global,
        "n": int(len(z)),
        "n_classes_used": int(len(means)),
    }


def item_13_6_from_summaries(rows):
    by = defaultdict(list)
    for r in rows:
        o = r["ood"]["pad_full"]["z_lesion"]
        by[r["lambda_adv"]].append(
            {
                "knn": o["knn_k50"],
                "maha": o["mahalanobis_classcond"]["unrestricted"],
                "msp": o["MSP"],
                "cosine": o["cosine_max"],
                "energy": o["Energy_T1"],
                "leak": r["leakage"]["z_lesion"]["bal_acc_mean"],
            }
        )
    table = []
    for lam in LAMS:
        recs = by[lam]
        row = {"lambda_adv": lam, "n": len(recs)}
        for k in ("knn", "maha", "msp", "cosine", "energy", "leak"):
            m, s = mean_std([x[k] for x in recs])
            row[k] = {"mean": m, "std": s}
        table.append(row)
    # collapse together? all five drop from λ=0 to λ=0.25 and sit near 0.4–0.55
    k025 = table[1]["knn"]["mean"]
    m025 = table[1]["maha"]["mean"]
    together = all(table[1][k]["mean"] < 0.60 for k in ("knn", "maha", "msp", "cosine", "energy"))
    together = together and all(table[0][k]["mean"] > 0.80 for k in ("knn", "maha", "msp", "cosine", "energy"))
    return {
        "space": "Phase 3 stored z_lesion = z_lesion_norm (post-BN 16-d)",
        "ood": "pad_full unrestricted",
        "k": 50,
        "table": table,
        "all_detectors_collapse_together": bool(together),
        "knn_at_lambda2": table[4]["knn"],
        "maha_at_lambda2": table[4]["maha"],
        "reading": (
            "kNN k=50 collapses with MSP/Energy/cosine/Mahalanobis. "
            "The derm 'all detectors collapse together' claim holds on the Phase 3 sweep. "
            "Camelyon17's Maha-only drop remains anomalous relative to this signature. "
            "k ∈ {10, 200} is reported once cached features exist."
        )
        if together
        else (
            "kNN k=50 does NOT track the other detectors. Rewrite the 'all detectors collapse together' claim. "
            "Camelyon17 Maha/kNN divergence is then not anomalous."
        ),
    }


def analyze_run(tag, lam, seed):
    tr = load_split(tag, "isic_train")
    idp = load_split(tag, "isic_test")
    pad = load_split(tag, "pad_full")
    hold = load_split(tag, "pad_heldout")
    fitz = load_split(tag, "fitzpatrick17k")
    out = {"run_name": tag, "lambda_adv": lam, "seed": seed}

    # 13.1 spectra on ISIC-train
    specs = {}
    for name in ("z_lesion_norm", "z_lesion", "z_context", "backbone_raw_lesion"):
        specs[name] = spectrum_of(tr[name])
        specs[name]["evals"] = specs[name]["evals"].tolist()
    out["spectrum"] = specs

    # 13.1 Maha robustness on the Phase 3 space (z_lesion_norm)
    ztr, ytr = tr["z_lesion_norm"], tr["labels"]
    zid, zood = idp["z_lesion_norm"], pad["z_lesion_norm"]
    ridge = {str(eps): maha_ridge(ztr, ytr, zid, zood, eps) for eps in EPS_SWEEP}
    lewo = maha_ledoit(ztr, ytr, zid, zood)
    out["maha_robustness_z_lesion_norm_pad_full"] = {
        "ridge": ridge,
        "ledoit_wolf": lewo,
        "headline_eps": HEADLINE_EPS,
        "headline_auroc": ridge[str(HEADLINE_EPS)]["auroc"],
        "headline_condition_number": ridge[str(HEADLINE_EPS)]["condition_number"],
    }

    # 13.2 three depths
    depths = {
        "backbone_raw_lesion": "backbone_raw_lesion",
        "z_lesion": "z_lesion",
        "z_lesion_norm": "z_lesion_norm",
    }
    depth_block = {}
    for key, fname in depths.items():
        zt, zi, zo = tr[fname], idp[fname], pad[fname]
        det = {}
        ridge_d = maha_ridge(zt, ytr, zi, zo, HEADLINE_EPS)
        det["mahalanobis_eps1e-3"] = ridge_d["auroc"]
        det["maha_condition_number"] = ridge_d["condition_number"]
        means, present = fit_class_means(zt, ytr, 8)
        det["cosine_max"] = auroc_pair(cosine_ood_scores(zi, means, present), cosine_ood_scores(zo, means, present))
        for k in (10, 50, 200):
            det["knn_k{}".format(k)] = knn_auroc(zt, zi, zo, k)
        det["MSP"] = auroc_pair(-softmax(idp["logits"]).max(1), -softmax(pad["logits"]).max(1))
        det["Energy_T1"] = auroc_pair(
            -np.logaddexp.reduce(idp["logits"], axis=1),
            -np.logaddexp.reduce(pad["logits"], axis=1),
        )
        leak_m, leak_s = leakage_bal(zi, zo)
        det["leakage_bal_acc_mean"] = leak_m
        det["leakage_bal_acc_std"] = leak_s
        depth_block[key] = det
    # logits MSP/Energy are depth-independent; leave them only under z_lesion_norm conceptually
    out["depths_pad_full"] = depth_block

    # 13.3 confidence
    id_c, id_msp = conf_pack(idp["logits"], idp["labels"])
    hold_c, hold_msp = conf_pack(hold["logits"], hold["labels"], restrict=PAD_CLASSES)
    fitz_c, fitz_msp = conf_pack(fitz["logits"])
    pad_c, pad_msp = conf_pack(pad["logits"], pad["labels"], restrict=PAD_CLASSES)
    out["confidence"] = {
        "isic_test": id_c,
        "pad_heldout": hold_c,
        "pad_full": pad_c,
        "fitzpatrick17k": fitz_c,
        "gap_id_minus_pad_heldout_msp": float(id_c["msp_mean"] - hold_c["msp_mean"]),
        "gap_id_minus_fitz_msp": float(id_c["msp_mean"] - fitz_c["msp_mean"]),
        "ood_more_confident_than_id_pad_heldout": bool(hold_c["msp_mean"] > id_c["msp_mean"]),
        "ood_more_confident_than_id_fitz": bool(fitz_c["msp_mean"] > id_c["msp_mean"]),
        "fitz_ece_note": "Fitzpatrick17k has no ISIC 8-class labels; ECE not computed.",
    }

    # 13.4 geometry on z_lesion_norm
    geo = {}
    for split, pack in (("isic_test", idp), ("pad_heldout", hold), ("pad_full", pad)):
        geo[split] = neural_collapse(pack["z_lesion_norm"], pack["labels"], PAD_CLASSES)
    geo["isic_train"] = neural_collapse(tr["z_lesion_norm"], tr["labels"], PAD_CLASSES)
    per_class = {}
    gmu = tr["z_lesion_norm"].mean(axis=0)
    for c, name in zip(PAD_CLASSES, PAD_NAMES):
        mid = idp["labels"] == c
        mpad = hold["labels"] == c
        rec = {"id_n": int(mid.sum()), "ood_n": int(mpad.sum())}
        if mid.sum() >= 2:
            rec["id_var"] = float(idp["z_lesion_norm"][mid].var(axis=0).sum())
            rec["id_dist_global"] = float(np.linalg.norm(idp["z_lesion_norm"][mid] - gmu, axis=1).mean())
        if mpad.sum() >= 2:
            rec["ood_var"] = float(hold["z_lesion_norm"][mpad].var(axis=0).sum())
            rec["ood_dist_global"] = float(np.linalg.norm(hold["z_lesion_norm"][mpad] - gmu, axis=1).mean())
        per_class[name] = rec
    out["geometry_z_lesion_norm"] = {"groups": geo, "per_class_pad_heldout": per_class}
    return out


def write_figures(per_run, knn_json):
    FIG.mkdir(parents=True, exist_ok=True)
    by_lam = defaultdict(list)
    for r in per_run:
        by_lam[r["lambda_adv"]].append(r)

    # 13.1 PR vs AUROC
    fig, ax1 = plt.subplots(figsize=(6.6, 4.2))
    lams = list(LAMS)
    pr_m, pr_s, auc_m, auc_s, leak_m, leak_s = [], [], [], [], [], []
    for lam in lams:
        recs = by_lam[lam]
        pr_m.append(mean_std([r["spectrum"]["z_lesion_norm"]["pr"] for r in recs])[0])
        pr_s.append(mean_std([r["spectrum"]["z_lesion_norm"]["pr"] for r in recs])[1])
        auc_m.append(mean_std([r["maha_robustness_z_lesion_norm_pad_full"]["headline_auroc"] for r in recs])[0])
        auc_s.append(mean_std([r["maha_robustness_z_lesion_norm_pad_full"]["headline_auroc"] for r in recs])[1])
        leak_m.append(mean_std([r["depths_pad_full"]["z_lesion_norm"]["leakage_bal_acc_mean"] for r in recs])[0])
        leak_s.append(mean_std([r["depths_pad_full"]["z_lesion_norm"]["leakage_bal_acc_mean"] for r in recs])[1])
    ax1.errorbar(lams, pr_m, yerr=pr_s, color=WONG["blue"], marker="o", lw=1.8, label="PR (z_lesion_norm)")
    ax1.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax1.set_ylabel("participation ratio", color=WONG["blue"])
    ax2 = ax1.twinx()
    ax2.errorbar(lams, auc_m, yerr=auc_s, color=WONG["orange"], marker="s", lw=1.8, label="Maha AUROC PAD")
    ax2.errorbar(lams, leak_m, yerr=leak_s, color=WONG["green"], marker="^", lw=1.4, label="leakage bal acc")
    ax2.set_ylabel("AUROC / leakage")
    ax2.axhline(0.5, color=WONG["gray"], ls="--", lw=0.8)
    ax1.set_title("13.1: dimensionality vs leakage vs OOD")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_1_pr_vs_auroc.{}".format(ext), dpi=300)
    plt.close(fig)

    # eigenspectrum overlay
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=False)
    spaces = (("z_lesion_norm", "z_lesion_norm 16-d"), ("z_context", "z_context 64-d"), ("backbone_raw_lesion", "backbone 1536-d"))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(lams)))
    for ax, (sp, title) in zip(axes, spaces):
        for lam, color in zip(lams, cmap):
            recs = by_lam[lam]
            evs = [np.asarray(r["spectrum"][sp]["evals"], dtype=np.float64) for r in recs]
            ev = np.mean(evs, axis=0)
            ev = np.clip(ev, 1e-12, None)
            ax.plot(np.arange(1, len(ev) + 1), ev, color=color, lw=1.2, label="{:g}".format(lam))
        ax.set_yscale("log")
        ax.set_xlabel("component")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("eigenvalue")
    axes[2].legend(title=r"$\lambda$", fontsize=7, frameon=False, ncol=2)
    fig.suptitle("13.1 eigenspectrum (ISIC-train, mean across seeds)", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_1_eigenspectrum.{}".format(ext), dpi=300)
    plt.close(fig)

    # 13.2 depths
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, color, mk in (
        ("backbone_raw_lesion", WONG["blue"], "o"),
        ("z_lesion", WONG["sky"], "s"),
        ("z_lesion_norm", WONG["orange"], "^"),
    ):
        ys, es = [], []
        for lam in lams:
            recs = by_lam[lam]
            m, s = mean_std([r["depths_pad_full"][key]["mahalanobis_eps1e-3"] for r in recs])
            ys.append(m)
            es.append(s)
        ax.errorbar(lams, ys, yerr=es, color=color, marker=mk, lw=1.8, label=key)
    ax.axhline(0.5, color=WONG["gray"], ls="--", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("Maha AUROC (PAD full)")
    ax.set_title("13.2: where the collapse lives")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_2_depths.{}".format(ext), dpi=300)
    plt.close(fig)

    # 13.3 confidence
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    series = (
        ("isic_test", WONG["green"], "ISIC test (ID)"),
        ("pad_heldout", WONG["orange"], "pad_heldout"),
        ("fitzpatrick17k", WONG["purple"], "Fitzpatrick17k"),
    )
    for key, color, lab in series:
        ys, es = [], []
        for lam in lams:
            recs = by_lam[lam]
            m, s = mean_std([r["confidence"][key]["msp_mean"] for r in recs])
            ys.append(m)
            es.append(s)
        ax.errorbar(lams, ys, yerr=es, color=color, marker="o", lw=1.8, label=lab)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("mean max-softmax")
    ax.set_title("13.3: confidence on unseen domains")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_3_confidence.{}".format(ext), dpi=300)
    plt.close(fig)

    # 13.6 from summaries (always)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, color, lab in (
        ("knn", WONG["blue"], "kNN k=50"),
        ("maha", WONG["orange"], "Mahalanobis"),
        ("msp", WONG["green"], "MSP"),
        ("cosine", WONG["purple"], "cosine"),
        ("energy", WONG["sky"], "Energy"),
    ):
        ys = [row[key]["mean"] for row in knn_json["table"]]
        es = [row[key]["std"] for row in knn_json["table"]]
        ax.errorbar(lams, ys, yerr=es, color=color, marker="o", lw=1.6, label=lab)
    ax.axhline(0.5, color=WONG["gray"], ls="--", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("PAD-full AUROC")
    ax.set_title("13.6: do all detectors collapse together?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_6_detectors.{}".format(ext), dpi=300)
    plt.close(fig)


def classify_13_1(per_run):
    # lockstep vs separable: Δ from λ=0 to 0.25
    def col(lam, fn):
        recs = [r for r in per_run if r["lambda_adv"] == lam]
        return mean_std([fn(r) for r in recs])[0]

    pr0 = col(0.0, lambda r: r["spectrum"]["z_lesion_norm"]["pr"])
    pr1 = col(0.25, lambda r: r["spectrum"]["z_lesion_norm"]["pr"])
    leak0 = col(0.0, lambda r: r["depths_pad_full"]["z_lesion_norm"]["leakage_bal_acc_mean"])
    leak1 = col(0.25, lambda r: r["depths_pad_full"]["z_lesion_norm"]["leakage_bal_acc_mean"])
    d_pr = pr0 - pr1
    d_leak = leak0 - leak1
    # "moved" = >15% relative for PR, >0.10 abs for leakage
    pr_moved = abs(d_pr) > 0.15 * max(abs(pr0), 1e-6)
    leak_moved = abs(d_leak) > 0.10
    if leak_moved and not pr_moved:
        sep = "Leakage dropped at λ=0.25 while PR did not. Dimensional collapse does not explain the OOD cliff."
        kind = "separated_leakage_only"
    elif pr_moved and not leak_moved:
        sep = "PR dropped while leakage did not. Unexpected; report as a finding."
        kind = "separated_pr_only"
    elif pr_moved and leak_moved:
        sep = (
            "Leakage and PR move together at the cliff. The two explanations are confounded in this data. "
            "The paper must say so."
        )
        kind = "lockstep"
    else:
        sep = "Neither PR nor leakage moved enough at λ=0.25 to decide. Report the numbers."
        kind = "neither"
    return {"kind": kind, "pr0": pr0, "pr025": pr1, "leak0": leak0, "leak025": leak1, "reading": sep}


def classify_13_2(per_run):
    def col(lam, depth):
        recs = [r for r in per_run if r["lambda_adv"] == lam]
        return mean_std([r["depths_pad_full"][depth]["mahalanobis_eps1e-3"] for r in recs])[0]

    bb0, bb1 = col(0.0, "backbone_raw_lesion"), col(0.25, "backbone_raw_lesion")
    z0, z1 = col(0.0, "z_lesion_norm"), col(0.25, "z_lesion_norm")
    if bb1 >= 0.70 and z1 <= 0.60:
        return {
            "outcome": "a",
            "reading": (
                "Backbone retains OOD detection ({:.3f}) while z_lesion_norm loses it ({:.3f}). "
                "Invariance is confined to the projection head. Practical fix: score OOD on pre-projection features."
            ).format(bb1, z1),
        }
    if bb1 <= 0.60:
        return {
            "outcome": "b",
            "reading": (
                "Backbone Maha also collapses ({:.3f} at λ=0.25, was {:.3f}). "
                "Invariance propagates through the encoder. No cheap fix. This is the stronger result."
            ).format(bb1, bb0),
        }
    return {
        "outcome": "mixed",
        "reading": (
            "Backbone {:.3f} → {:.3f}, z_lesion_norm {:.3f} → {:.3f}. Neither clean (a) nor (b)."
        ).format(bb0, bb1, z0, z1),
    }


def classify_maha_shift(per_run):
    recs2 = [r for r in per_run if r["lambda_adv"] == 2.0]
    if not recs2:
        recs2 = [r for r in per_run if r["lambda_adv"] == 0.25]
    head = mean_std([r["maha_robustness_z_lesion_norm_pad_full"]["headline_auroc"] for r in recs2])[0]
    lw = mean_std([r["maha_robustness_z_lesion_norm_pad_full"]["ledoit_wolf"]["auroc"] for r in recs2])[0]
    delta = abs(lw - head)
    return {
        "headline_at_ref": head,
        "ledoit_at_ref": lw,
        "abs_delta": delta,
        "survives": bool(delta < 0.03),
        "reading": (
            "0.41 survives Ledoit-Wolf (headline {:.3f}, LW {:.3f}). Robust; pre-empts the conditioning attack."
            if delta < 0.03
            else "Ledoit-Wolf moves the headline materially ({:.3f} → {:.3f}). Main-text caveat, not a footnote."
        ).format(head, lw),
    }


def write_reports(knn_json, per_run, toy=None):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase 13 — technical depth on the derm result")
    lines.append("")
    lines.append("No new CSG training. Phase 12 was not interrupted. Outputs only under `results/paperB/phase13/`.")
    lines.append("")
    lines.append("Phase 3 `z_lesion` is **`z_lesion_norm`** (post-BN 16-d). Pre-BN projector output is stored separately as `z_lesion`.")
    lines.append("Leakage is 2-class balanced accuracy; chance floor **0.5**.")
    lines.append("")

    # 13.6 first — verification
    lines.append("## 13.6 — do all detectors collapse together?")
    lines.append("")
    lines.append(knn_json["reading"])
    lines.append("")
    lines.append("| λ_adv | n | kNN k=50 | Maha | MSP | cosine | Energy | leakage bal |")
    lines.append("|---:|---:|---|---|---|---|---|---|")
    for row in knn_json["table"]:
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["lambda_adv"],
                row["n"],
                fmt_ms(row["knn"]["mean"], row["knn"]["std"]),
                fmt_ms(row["maha"]["mean"], row["maha"]["std"]),
                fmt_ms(row["msp"]["mean"], row["msp"]["std"]),
                fmt_ms(row["cosine"]["mean"], row["cosine"]["std"]),
                fmt_ms(row["energy"]["mean"], row["energy"]["std"]),
                fmt_ms(row["leak"]["mean"], row["leak"]["std"]),
            )
        )
    lines.append("")
    lines.append("k=50 is the Phase 3 cached number. k ∈ {10, 200} below if features were extracted.")
    lines.append("")

    if not per_run:
        lines.append("**Features not yet cached.** 13.1–13.4 and k-sweep wait on `extract_phase13_features.py`.")
        (OUT / "PHASE13_REPORT.md").write_text("\n".join(lines) + "\n")
        (OUT / "PHASE13_6_KNN.md").write_text("\n".join(lines[lines.index("## 13.6 — do all detectors collapse together?"):]) + "\n")
        return

    c11 = classify_13_1(per_run)
    c12 = classify_13_2(per_run)
    cmaha = classify_maha_shift(per_run)

    lines.append("## 13.1 — dimensional collapse")
    lines.append("")
    lines.append(c11["reading"])
    lines.append("")
    lines.append("PR(z_lesion_norm) λ=0 {:.3f} → λ=0.25 {:.3f}. Leakage {:.3f} → {:.3f}.".format(
        c11["pr0"], c11["pr025"], c11["leak0"], c11["leak025"]
    ))
    lines.append("")
    lines.append(cmaha["reading"])
    lines.append("")
    lines.append("| λ | n | PR z_norm | PR z_preBN | PR z_context | PR backbone | Maha eps=1e-3 | Maha LW | cond (eps=1e-3) |")
    lines.append("|---:|---:|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in per_run:
        by[r["lambda_adv"]].append(r)
    for lam in LAMS:
        recs = by[lam]
        def g(fn):
            return fmt_ms(*mean_std([fn(r) for r in recs]))
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam, len(recs),
                g(lambda r: r["spectrum"]["z_lesion_norm"]["pr"]),
                g(lambda r: r["spectrum"]["z_lesion"]["pr"]),
                g(lambda r: r["spectrum"]["z_context"]["pr"]),
                g(lambda r: r["spectrum"]["backbone_raw_lesion"]["pr"]),
                g(lambda r: r["maha_robustness_z_lesion_norm_pad_full"]["headline_auroc"]),
                g(lambda r: r["maha_robustness_z_lesion_norm_pad_full"]["ledoit_wolf"]["auroc"]),
                g(lambda r: r["maha_robustness_z_lesion_norm_pad_full"]["headline_condition_number"]),
            )
        )
    lines.append("")

    lines.append("## 13.2 — where in the network")
    lines.append("")
    lines.append("**Outcome ({})**. {}".format(c12["outcome"], c12["reading"]))
    lines.append("")
    lines.append("| λ | backbone Maha | z_lesion Maha | z_norm Maha | backbone leak | z_norm leak |")
    lines.append("|---:|---|---|---|---|---|")
    for lam in LAMS:
        recs = by[lam]
        def g(depth, key):
            return fmt_ms(*mean_std([r["depths_pad_full"][depth][key] for r in recs]))
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} |".format(
                lam,
                g("backbone_raw_lesion", "mahalanobis_eps1e-3"),
                g("z_lesion", "mahalanobis_eps1e-3"),
                g("z_lesion_norm", "mahalanobis_eps1e-3"),
                g("backbone_raw_lesion", "leakage_bal_acc_mean"),
                g("z_lesion_norm", "leakage_bal_acc_mean"),
            )
        )
    lines.append("")

    lines.append("## 13.3 — confidence on unseen domains")
    lines.append("")
    crossings = []
    for lam in LAMS:
        recs = by[lam]
        idm = mean_std([r["confidence"]["isic_test"]["msp_mean"] for r in recs])[0]
        hm = mean_std([r["confidence"]["pad_heldout"]["msp_mean"] for r in recs])[0]
        fm = mean_std([r["confidence"]["fitzpatrick17k"]["msp_mean"] for r in recs])[0]
        if hm > idm or fm > idm:
            crossings.append(lam)
    if crossings:
        lines.append(
            "The more invariant the model is made, the more confident it becomes on data it has never seen. "
            "Crossing (mean OOD MSP > mean ID MSP) first at λ ∈ {}.".format(crossings)
        )
    else:
        lines.append("No λ in this grid has mean OOD MSP exceeding mean ID MSP. The sentence is not earned. Finding.")
    lines.append("")
    lines.append("| λ | ID MSP | pad_heldout MSP | Fitz MSP | ID−hold gap | ID ECE | hold ECE |")
    lines.append("|---:|---|---|---|---|---|---|")
    for lam in LAMS:
        recs = by[lam]
        def gg(fn):
            return fmt_ms(*mean_std([fn(r) for r in recs]))
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} |".format(
                lam,
                gg(lambda r: r["confidence"]["isic_test"]["msp_mean"]),
                gg(lambda r: r["confidence"]["pad_heldout"]["msp_mean"]),
                gg(lambda r: r["confidence"]["fitzpatrick17k"]["msp_mean"]),
                gg(lambda r: r["confidence"]["gap_id_minus_pad_heldout_msp"]),
                gg(lambda r: r["confidence"]["isic_test"].get("ece", float("nan"))),
                gg(lambda r: r["confidence"]["pad_heldout"].get("ece", float("nan"))),
            )
        )
    lines.append("")
    lines.append("Fitz ECE is not computed (no compatible labels).")
    lines.append("")

    lines.append("## 13.4 — latent geometry")
    lines.append("")
    lines.append("| λ | ID var | pad_heldout var | ID dist-global | hold dist-global | ID Sw/Sb | hold Sw/Sb |")
    lines.append("|---:|---|---|---|---|---|---|")
    for lam in LAMS:
        recs = by[lam]
        def gg(fn):
            vals = []
            for r in recs:
                v = fn(r)
                if v is not None:
                    vals.append(v)
            return fmt_ms(*mean_std(vals)) if vals else "—"
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} |".format(
                lam,
                gg(lambda r: (r["geometry_z_lesion_norm"]["groups"]["isic_test"] or {}).get("variance_to_global")),
                gg(lambda r: (r["geometry_z_lesion_norm"]["groups"]["pad_heldout"] or {}).get("variance_to_global")),
                gg(lambda r: (r["geometry_z_lesion_norm"]["groups"]["isic_test"] or {}).get("mean_dist_to_global_centroid")),
                gg(lambda r: (r["geometry_z_lesion_norm"]["groups"]["pad_heldout"] or {}).get("mean_dist_to_global_centroid")),
                gg(lambda r: (r["geometry_z_lesion_norm"]["groups"]["isic_test"] or {}).get("within_over_between")),
                gg(lambda r: (r["geometry_z_lesion_norm"]["groups"]["pad_heldout"] or {}).get("within_over_between")),
            )
        )
    lines.append("")
    lines.append("Compression prediction: as λ rises, OOD variance and distance-to-centroid fall below ID of the same classes.")
    lines.append("")

    # k-sweep if present
    lines.append("## 13.6b — kNN k ∈ {10, 50, 200} on cached z_lesion_norm")
    lines.append("")
    lines.append("| λ | k=10 | k=50 (recomputed) | k=200 | Phase 3 k=50 |")
    lines.append("|---:|---|---|---|---|")
    for lam, prow in zip(LAMS, knn_json["table"]):
        recs = by[lam]
        def gg(key):
            return fmt_ms(*mean_std([r["depths_pad_full"]["z_lesion_norm"][key] for r in recs]))
        lines.append(
            "| {:g} | {} | {} | {} | {} |".format(
                lam,
                gg("knn_k10"),
                gg("knn_k50"),
                gg("knn_k200"),
                fmt_ms(prow["knn"]["mean"], prow["knn"]["std"]),
            )
        )
    lines.append("")
    if toy is not None:
        lines.append("## 13.5 — toy model")
        lines.append("")
        lines.append(toy.get("reading", "see PHASE13_5_TOY.md"))
        lines.append("")

    lines.append("Figures: `phase13/figures/fig_13_*.{png,pdf}`.")
    (OUT / "PHASE13_REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "PHASE13_6_KNN.md").write_text("# 13.6 kNN verification\n\n" + knn_json["reading"] + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_summaries()
    knn_json = item_13_6_from_summaries(rows)
    (OUT / "knn_from_phase3.json").write_text(json.dumps(knn_json, indent=2) + "\n")

    per_run = []
    missing = []
    for lam, seed in p3.JOBS:
        tag = "runB_orth1_ladv{}_s{}".format(lam_tag(lam), seed)
        if not has_features(tag):
            missing.append(tag)
            continue
        print("analyze", tag, flush=True)
        rec = analyze_run(tag, float(lam), int(seed))
        (OUT / "per_run").mkdir(parents=True, exist_ok=True)
        # drop huge eval lists from per-run json? keep them, 16-d is tiny; backbone 1536 is ok
        (OUT / "per_run" / "{}.json".format(tag)).write_text(json.dumps(rec, indent=2) + "\n")
        per_run.append(rec)

    toy_path = OUT / "toy" / "summary.json"
    toy = json.loads(toy_path.read_text()) if toy_path.exists() else None

    if per_run:
        write_figures(per_run, knn_json)
    write_reports(knn_json, per_run, toy=toy)
    dump = {
        "n_summaries": len(rows),
        "n_featured": len(per_run),
        "missing_features": missing,
        "knn": knn_json,
        "n13_1_kind": classify_13_1(per_run)["kind"] if per_run else None,
        "n13_2": classify_13_2(per_run) if per_run else None,
        "maha_shift": classify_maha_shift(per_run) if per_run else None,
    }
    (OUT / "aggregate.json").write_text(json.dumps(dump, indent=2) + "\n")
    print("featured", len(per_run), "missing", len(missing))
    print("13.6 together", knn_json["all_detectors_collapse_together"])
    print("wrote", OUT / "PHASE13_REPORT.md")


if __name__ == "__main__":
    main()
