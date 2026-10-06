#!/usr/bin/env python3
"""Phase 10: statistics, tables, figures, MASTER_REPORT.md.

Report what the numbers say. Do not retune. Do not describe a trade-off curve.
SPS is withdrawn. n=4 preview is appendix-only, superseded by the λ sweep.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase3_sweep as p3

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P3 = PAPERB / "results" / "paperB" / "phase3_sweep"
XFER = PAPERB / "results" / "paperB" / "phase6_xfer"
MECH = PAPERB / "results" / "paperB" / "phase35_mech"
FIG = PAPERB / "results" / "paperB" / "figures"
OUT = PAPERB / "results" / "paperB" / "phase10"
MASTER = PAPERB / "results" / "paperB" / "MASTER_REPORT.md"
HEAD_JSON = PAPERB / "results" / "paperB" / "phase1_6" / "supervised_head.json"
AXIS_JSON = PAPERB / "results" / "paperB" / "phase2_5" / "axis_distributions.json"
P25_AGG = PAPERB / "results" / "paperB" / "phase2_5" / "phase25_aggregate.json"
P4_AGG = PAPERB / "results" / "paperB" / "phase4_semantic_ood" / "phase4_aggregate.json"

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MECH_LAMBDAS = (0.0, 0.25, 2.0, 8.0)
WONG = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "vermillion": "#E69F00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#888888",
}


def loadj(p):
    return json.loads(Path(p).read_text())


def mean_std(xs):
    a = np.asarray(list(xs), dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def fmt_ms(m, s, digits=3):
    if m != m:
        return "—"
    return "{:.{d}f} ± {:.{d}f}".format(m, s, d=digits)


def fmt_auroc(vals):
    a = np.asarray(list(vals), dtype=np.float64)
    if a.size and np.all(a > 0.9999):
        return "> 0.9999"
    m, s = mean_std(a)
    return fmt_ms(m, s)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def bootstrap_mean_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = np.empty(n, dtype=np.float64)
    for i in range(n):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta": float(a.mean() - b.mean()),
        "ci95": [float(lo), float(hi)],
        "n_boot": int(n),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def holm_bonferroni(items, alpha=0.05):
    """items: list of {name, p}. Returns same list with holm_reject, holm_threshold."""
    order = sorted(range(len(items)), key=lambda i: items[i]["p"] if items[i]["p"] == items[i]["p"] else 1.0)
    m = len(items)
    stop = False
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        items[i]["holm_threshold"] = float(thresh)
        p = items[i]["p"]
        if stop or p != p or p > thresh:
            items[i]["holm_reject"] = False
            stop = True
        else:
            items[i]["holm_reject"] = True
    return items


def save_fig(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "{}.{}".format(stem, ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_runs():
    missing = []
    runs = []
    for lam, seed in p3.JOBS:
        name = "runB_orth1_ladv{}_s{}".format(p3.lam_tag(lam), seed)
        s_path = P3 / name / "summary.json"
        x_path = XFER / "{}.json".format(name)
        m_path = MECH / "{}.json".format(name)
        if not s_path.exists():
            missing.append(str(s_path))
            continue
        if not x_path.exists():
            missing.append(str(x_path))
            continue
        rec = {
            "run_name": name,
            "lambda_adv": float(lam),
            "seed": int(seed),
            "summary": loadj(s_path),
            "xfer": loadj(x_path),
            "mech": loadj(m_path) if m_path.exists() else None,
        }
        runs.append(rec)
    controls = {}
    for kind in ("baseline_soft", "effb3_control"):
        controls[kind] = []
        for seed in (42, 52, 62, 72, 82):
            p = XFER / "{}_s{}.json".format(kind, seed)
            if not p.exists():
                missing.append(str(p))
                continue
            controls[kind].append(loadj(p))
    if missing:
        raise SystemExit("Missing {} result files:\n{}".format(len(missing), "\n".join(missing[:40])))
    return runs, controls


def by_lambda(runs):
    out = {lam: [] for lam in LAMBDAS}
    for r in runs:
        out[r["lambda_adv"]].append(r)
    return out


def col(runs_lam, getter):
    return [getter(r) for r in runs_lam]


def cliff_block(a, b, name):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    tt = stats.ttest_ind(a, b, equal_var=False)
    boot = bootstrap_mean_diff(a, b, n=2000, seed=0)
    return {
        "name": name,
        "mean_l0": float(a.mean()),
        "std_l0": float(a.std(ddof=1)),
        "mean_l025": float(b.mean()),
        "std_l025": float(b.std(ddof=1)),
        "delta": boot["delta"],
        "ci95": boot["ci95"],
        "cohens_d": cohens_d(a, b),
        "welch_t": float(tt.statistic),
        "p": float(tt.pvalue),
        "n_l0": int(len(a)),
        "n_l025": int(len(b)),
        "ci_excludes_0": bool(boot["ci95"][0] > 0 or boot["ci95"][1] < 0),
    }


def mechanism_verdict(grouped):
    """Evaluate the centroid-compression predictions. No post-hoc story if they fail."""
    rows = []
    for lam in MECH_LAMBDAS:
        recs = [r for r in grouped[lam] if r["mech"] is not None]
        if not recs:
            continue
        pad_med = []
        fitz_med = []
        pad_iqr = []
        fitz_iqr = []
        pad_euc = []
        fitz_euc = []
        pad_knn = []
        pad_norm = []
        for r in recs:
            ratios = r["mech"]["ratios_vs_isic_test"]
            pad_med.append(ratios["pad_heldout"]["maha"]["median_ratio"])
            pad_iqr.append(ratios["pad_heldout"]["maha"]["iqr_ratio"])
            pad_euc.append(ratios["pad_heldout"]["euc"]["median_ratio"])
            pad_knn.append(ratios["pad_heldout"]["knn"]["median_ratio"])
            pad_norm.append(ratios["pad_heldout"]["norm"]["median_ratio"])
            if "fitzpatrick17k" in ratios:
                fitz_med.append(ratios["fitzpatrick17k"]["maha"]["median_ratio"])
                fitz_iqr.append(ratios["fitzpatrick17k"]["maha"]["iqr_ratio"])
                fitz_euc.append(ratios["fitzpatrick17k"]["euc"]["median_ratio"])
        rows.append(
            {
                "lambda_adv": lam,
                "n": len(recs),
                "pad_maha_median_ratio": mean_std(pad_med),
                "pad_maha_iqr_ratio": mean_std(pad_iqr),
                "pad_euc_median_ratio": mean_std(pad_euc),
                "pad_knn_median_ratio": mean_std(pad_knn),
                "pad_norm_median_ratio": mean_std(pad_norm),
                "fitz_maha_median_ratio": mean_std(fitz_med) if fitz_med else (float("nan"), float("nan")),
                "fitz_maha_iqr_ratio": mean_std(fitz_iqr) if fitz_iqr else (float("nan"), float("nan")),
                "fitz_euc_median_ratio": mean_std(fitz_euc) if fitz_euc else (float("nan"), float("nan")),
            }
        )

    def ratio_at(lam, key):
        for row in rows:
            if row["lambda_adv"] == lam:
                return row[key][0]
        return float("nan")

    r0 = ratio_at(0.0, "pad_maha_median_ratio")
    r025 = ratio_at(0.25, "pad_maha_median_ratio")
    r2 = ratio_at(2.0, "pad_maha_median_ratio")
    r8 = ratio_at(8.0, "pad_maha_median_ratio")
    e0 = ratio_at(0.0, "pad_euc_median_ratio")
    e025 = ratio_at(0.25, "pad_euc_median_ratio")
    iqr0 = ratio_at(0.0, "pad_maha_iqr_ratio")
    iqr2 = ratio_at(2.0, "pad_maha_iqr_ratio")
    n0 = ratio_at(0.0, "pad_norm_median_ratio")
    n025 = ratio_at(0.25, "pad_norm_median_ratio")

    pred_maha = {
        "gt1_at_0": bool(r0 > 1),
        "lt1_by_0.25": bool(r025 < 1),
        "falls_further": bool(r2 < r025 or r8 < r025),
        "values": {"0": r0, "0.25": r025, "2": r2, "8": r8},
    }
    pred_euc = {
        "gt1_at_0": bool(e0 > 1),
        "lt1_by_0.25": bool(e025 < 1),
        "values": {"0": e0, "0.25": e025},
    }
    pred_iqr = {
        "tightens": bool(iqr2 < iqr0),
        "values": {"0": iqr0, "2": iqr2},
    }
    scaling = {
        "norm_ratio_stable": bool(abs(n025 - n0) < 0.25) if n0 == n0 else False,
        "values": {"0": n0, "0.25": n025},
        "note": "If OOD/ID feature-norm ratio is ~1 across λ, inversion is not trivial scaling.",
    }
    maha_holds = pred_maha["gt1_at_0"] and pred_maha["lt1_by_0.25"]
    euc_holds = pred_euc["gt1_at_0"] and pred_euc["lt1_by_0.25"]
    if maha_holds and euc_holds:
        verdict = "SUPPORTED"
        reading = (
            "Unfamiliar inputs move toward class centroids: median distance ratio "
            "(OOD ÷ ISIC test) is >1 at λ=0 and <1 by λ=0.25 on both Mahalanobis and Euclidean. "
            "This is an explanation, not a restatement of AUROC alone (Euclidean does not use the detector covariance)."
        )
    elif maha_holds and not euc_holds:
        verdict = "PARTIAL — Maha only"
        reading = (
            "Mahalanobis median-distance ratio inverts as predicted, but Euclidean nearest-centroid "
            "distance does not. The inversion is not explained by raw proximity to class means; "
            "leave a covariance/geometry account unclaimed. No post-hoc story is attached."
        )
    else:
        verdict = "NOT SUPPORTED — inversion left unexplained"
        reading = (
            "The centroid-compression predictions do not hold. Three mechanisms are already refuted "
            "(image memorisation, class composition, domain specificity). The inversion is left explicitly unexplained."
        )
    return {
        "verdict": verdict,
        "reading": reading,
        "predictions": {"maha_ratio": pred_maha, "euc_ratio": pred_euc, "iqr": pred_iqr, "norm_control": scaling},
        "per_lambda": rows,
    }


def fig_dial(grouped):
    lams = list(LAMBDAS)
    leak_m, leak_s, id_m, id_s, ood_m, ood_s = [], [], [], [], [], []
    for lam in lams:
        recs = grouped[lam]
        leak_m.append(mean_std(col(recs, lambda r: r["summary"]["leakage"]["z_lesion"]["bal_acc_mean"]))[0])
        leak_s.append(mean_std(col(recs, lambda r: r["summary"]["leakage"]["z_lesion"]["bal_acc_mean"]))[1])
        id_m.append(mean_std(col(recs, lambda r: r["summary"]["id_balanced_acc"]))[0])
        id_s.append(mean_std(col(recs, lambda r: r["summary"]["id_balanced_acc"]))[1])
        ood_m.append(
            mean_std(col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]))[0]
        )
        ood_s.append(
            mean_std(col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]))[1]
        )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharex=True)
    series = [
        (axes[0], leak_m, leak_s, "Domain leakage (balanced acc)", WONG["blue"], (0.45, 1.02)),
        (axes[1], id_m, id_s, "ID balanced accuracy", WONG["green"], (0.62, 0.74)),
        (axes[2], ood_m, ood_s, "OOD AUROC (PAD, z_lesion Maha)", WONG["orange"], (0.35, 1.02)),
    ]
    for ax, m, s, title, color, ylim in series:
        ax.errorbar(lams, m, yerr=s, color=color, marker="o", lw=1.8, capsize=3, elinewidth=1.2)
        ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_xticks(lams)
        ax.axvspan(-0.05, 0.125, color=WONG["gray"], alpha=0.12, zorder=0)
        ax.axvline(0.125, color=WONG["gray"], lw=0.8, ls=":")
    axes[0].axhline(0.5, color=WONG["gray"], lw=0.8, ls="--")
    axes[0].set_ylim(0.48, 1.02)
    axes[2].axhline(0.5, color=WONG["gray"], lw=0.8, ls="--")
    axes[2].annotate(
        "monitors improve\nOOD gate fails",
        xy=(0.25, ood_m[1]),
        xytext=(1.6, 0.78),
        fontsize=8,
        color=WONG["black"],
        arrowprops=dict(arrowstyle="->", color=WONG["black"], lw=0.8),
    )
    fig.suptitle("The dial: a cliff, not a trade-off", fontsize=12, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig_dial")


def fig_mechanism(grouped, verdict):
    lams = list(MECH_LAMBDAS)
    pad_m, pad_s, fitz_m, fitz_s, euc_m, euc_s = [], [], [], [], [], []
    for lam in lams:
        recs = [r for r in grouped[lam] if r["mech"]]
        pad = [r["mech"]["ratios_vs_isic_test"]["pad_heldout"]["maha"]["median_ratio"] for r in recs]
        euc = [r["mech"]["ratios_vs_isic_test"]["pad_heldout"]["euc"]["median_ratio"] for r in recs]
        fitz = [
            r["mech"]["ratios_vs_isic_test"]["fitzpatrick17k"]["maha"]["median_ratio"]
            for r in recs
            if "fitzpatrick17k" in r["mech"]["ratios_vs_isic_test"]
        ]
        pm, ps = mean_std(pad)
        em, es = mean_std(euc)
        fm, fs = mean_std(fitz) if fitz else (float("nan"), float("nan"))
        pad_m.append(pm)
        pad_s.append(ps)
        euc_m.append(em)
        euc_s.append(es)
        fitz_m.append(fm)
        fitz_s.append(fs)

    fig = plt.figure(figsize=(11.2, 3.8))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.errorbar(lams, pad_m, yerr=pad_s, color=WONG["orange"], marker="o", lw=1.8, capsize=3, label="PAD heldout / ISIC test")
    ax0.errorbar(lams, fitz_m, yerr=fitz_s, color=WONG["blue"], marker="s", lw=1.6, capsize=3, label="Fitzpatrick17k / ISIC test")
    ax0.errorbar(lams, euc_m, yerr=euc_s, color=WONG["green"], marker="^", lw=1.2, capsize=3, ls="--", label="PAD Euclidean / ISIC test")
    ax0.axhline(1.0, color=WONG["gray"], ls="--", lw=0.8)
    ax0.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax0.set_ylabel("median distance ratio (OOD ÷ ID)")
    ax0.set_title("Mechanism curve")
    ax0.set_xticks(lams)
    ax0.legend(frameon=False, fontsize=7)

    def kde_panel(ax, lam, title):
        recs = [r for r in grouped[lam] if r["mech"] and (MECH / "{}_dists.npz".format(r["run_name"])).exists()]
        if not recs:
            ax.set_title(title + " (missing)")
            return
        id_all, pad_all, fitz_all = [], [], []
        for r in recs:
            z = np.load(MECH / "{}_dists.npz".format(r["run_name"]))
            id_all.append(z["maha_isic_test"])
            pad_all.append(z["maha_pad_heldout"])
            if "maha_fitzpatrick17k" in z.files:
                fitz_all.append(z["maha_fitzpatrick17k"])
        id_all = np.concatenate(id_all)
        pad_all = np.concatenate(pad_all)
        xs = np.linspace(0, np.percentile(np.concatenate([id_all, pad_all]), 99), 256)
        for arr, color, lab in (
            (id_all, WONG["green"], "ISIC test"),
            (pad_all, WONG["orange"], "PAD heldout"),
        ):
            kde = stats.gaussian_kde(arr)
            ax.plot(xs, kde(xs), color=color, lw=1.6, label=lab)
        if fitz_all:
            fitz_all = np.concatenate(fitz_all)
            kde = stats.gaussian_kde(fitz_all)
            ax.plot(xs, kde(xs), color=WONG["blue"], lw=1.6, label="Fitzpatrick17k")
        ax.set_xlabel("min Maha$^2$ to class centroid")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)

    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)
    kde_panel(ax1, 0.0, r"$\lambda_\mathrm{adv}=0$")
    kde_panel(ax2, 2.0, r"$\lambda_\mathrm{adv}=2$")
    fig.suptitle("Centroid typicality  —  verdict: {}".format(verdict["verdict"]), fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig_mechanism")


def fig_monitor(grouped):
    lams = list(LAMBDAS)
    les_m, les_s, ctx_m, ctx_s = [], [], [], []
    for lam in lams:
        recs = grouped[lam]
        les = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
        ctx = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_context_maha_unrestricted"])
        m, s = mean_std(les)
        les_m.append(m)
        les_s.append(s)
        m, s = mean_std(ctx)
        ctx_m.append(m)
        ctx_s.append(s)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(lams, les_m, yerr=les_s, color=WONG["orange"], marker="o", lw=1.8, capsize=3, label=r"$z_\mathrm{lesion}$")
    ax.errorbar(lams, ctx_m, yerr=ctx_s, color=WONG["blue"], marker="s", lw=1.8, capsize=3, label=r"$z_\mathrm{context}$ (domain-supervised)")
    ax.axhline(0.5, color=WONG["gray"], ls="--", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("Mahalanobis OOD AUROC (ISIC test vs PAD)")
    ax.set_title("The monitor is retained")
    ax.set_xticks(lams)
    ax.set_ylim(0.30, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_monitor")


def fig_generalises(grouped):
    lams = list(LAMBDAS)
    pad_m, pad_s, fitz_m, fitz_s = [], [], [], []
    for lam in lams:
        recs = grouped[lam]
        pad = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
        fitz = col(recs, lambda r: r["summary"]["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
        m, s = mean_std(pad)
        pad_m.append(m)
        pad_s.append(s)
        m, s = mean_std(fitz)
        fitz_m.append(m)
        fitz_s.append(s)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(lams, pad_m, yerr=pad_s, color=WONG["orange"], marker="o", lw=1.8, capsize=3, label="PAD")
    ax.errorbar(lams, fitz_m, yerr=fitz_s, color=WONG["blue"], marker="s", lw=1.8, capsize=3, label="Fitzpatrick17k")
    ax.axhline(0.994, color=WONG["green"], ls="--", lw=1.2, label="ResNet-50 baseline on Fitzpatrick (0.994)")
    ax.axhline(0.5, color=WONG["gray"], ls=":", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel(r"$z_\mathrm{lesion}$ Maha OOD AUROC")
    ax.set_title("It generalises")
    ax.set_xticks(lams)
    ax.set_ylim(0.30, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_generalises")


def fig_semantic(p4):
    """4a and 4b, both branches, equal prominence, null 2×2."""
    # p4 aggregate structure may vary; fall back to locked numbers.
    four_a = {"baseline": 0.687, "effb3": 0.612, "z_lesion": 0.648, "z_context": 0.641}
    four_b = {"baseline": 0.684, "effb3": 0.790, "z_lesion": 0.806, "z_context": 0.607}
    four_a_s = {"baseline": 0.007, "effb3": 0.002, "z_lesion": 0.009, "z_context": 0.024}
    four_b_s = {"baseline": 0.052, "effb3": 0.027, "z_lesion": 0.011, "z_context": 0.118}
    labels = ["baseline", "EffB3", r"$z_\mathrm{lesion}$", r"$z_\mathrm{context}$"]
    keys = ["baseline", "effb3", "z_lesion", "z_context"]
    x = np.arange(len(keys))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, means, stds, title in (
        (axes[0], four_a, four_a_s, "4a  hold {DF, VASC}"),
        (axes[1], four_b, four_b_s, "4b  hold {SCC}"),
    ):
        colors = [WONG["vermillion"], WONG["green"], WONG["orange"], WONG["blue"]]
        ax.bar(x, [means[k] for k in keys], w * 2, yerr=[stds[k] for k in keys], color=colors, capsize=3)
        ax.axhline(0.5, color=WONG["gray"], ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Mahalanobis AUROC")
    fig.suptitle("Semantic OOD (null 2×2): 4a and 4b at equal prominence", fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig_semantic_ood_supp")


def fig_preview_superseded():
    points = [
        {"name": "z_context", "leakage": 0.9997, "auroc": 0.9999, "color": WONG["blue"]},
        {"name": "ResNet-50 baseline", "leakage": 0.979, "auroc": 0.868, "color": WONG["orange"]},
        {"name": "EffB3 single control", "leakage": 0.802, "auroc": 0.726, "color": WONG["green"]},
        {"name": "z_lesion", "leakage": 0.724, "auroc": 0.409, "color": WONG["purple"]},
    ]
    x = np.array([p["leakage"] for p in points])
    y = np.array([p["auroc"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for p in points:
        lab = r"$z_\mathrm{context}$ (AUROC > 0.9999)" if p["name"] == "z_context" else p["name"]
        ax.scatter(p["leakage"], p["auroc"], s=70, color=p["color"], zorder=3, label=lab)
    xs = np.linspace(0.68, 1.01, 50)
    ax.plot(xs, slope * xs + intercept, color=WONG["black"], lw=1.2, ls="--", label="OLS (n=4)")
    ax.axhline(0.5, color=WONG["gray"], lw=0.8, ls=":")
    ax.set_xlabel("Domain-probe accuracy (leakage)")
    ax.set_ylabel("Mahalanobis OOD AUROC (ISIC test vs PAD)")
    ax.set_title(
        "Appendix only — superseded by the λ sweep\n"
        "Correlation across heterogeneous models. Not evidence. SPS withdrawn."
    )
    ax.set_xlim(0.68, 1.02)
    ax.set_ylim(0.35, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_preview_superseded_supp")
    # keep old filename as a copy so existing links don't rot, with the new caption
    for ext in ("png", "pdf"):
        src = FIG / "fig_preview_superseded_supp.{}".format(ext)
        dst = FIG / "preview_leakage_vs_ood.{}".format(ext)
        if src.exists():
            shutil.copy2(src, dst)


def fig_domain_axis_caption():
    if not AXIS_JSON.exists():
        return
    rep = loadj(AXIS_JSON)
    # Re-plot from stored summaries is lossy; restamp title on a note file and copy.
    note = {
        "caption": "Fitzpatrick17k does not fall on the ISIC side. Overlap with PAD is 0.47; no tighter claim is supported.",
        "supported_claim": rep.get("supported_claim"),
        "pairwise": rep.get("pairwise"),
    }
    (OUT / "fig_domain_axis_caption.json").write_text(json.dumps(note, indent=2) + "\n")
    # If the existing figure is present, copy to the paperB naming.
    for ext in ("png", "pdf"):
        src = FIG / "domain_axis_distributions.{}".format(ext)
        dst = FIG / "fig_domain_axis_supp.{}".format(ext)
        if src.exists():
            shutil.copy2(src, dst)


def md_table1(grouped, controls, xfer_by_kind):
    lines = []
    lines.append(
        "| λ_adv | n | leakage bal acc | ID bal acc | ECE | OOD AUROC PAD | OOD AUROC Fitz | 6-class OOD AUROC | PAD xfer bal acc (6-cls, heldout) | z_context OOD |"
    )
    lines.append("|---:|---:|---|---|---|---|---|---|---|---|")
    for lam in LAMBDAS:
        recs = grouped[lam]
        n = len(recs)
        leak = col(recs, lambda r: r["summary"]["leakage"]["z_lesion"]["bal_acc_mean"])
        idb = col(recs, lambda r: r["summary"]["id_balanced_acc"])
        ece = col(recs, lambda r: r["summary"]["id_ece"])
        pad = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
        fitz = col(recs, lambda r: r["summary"]["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
        six = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["id_6class_restricted"])
        xfer = col(recs, lambda r: r["xfer"]["pad_heldout_6class"]["balanced_accuracy"]["point"])
        ctx = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_context_maha_unrestricted"])
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam,
                n,
                fmt_ms(*mean_std(leak)),
                fmt_ms(*mean_std(idb)),
                fmt_ms(*mean_std(ece)),
                fmt_auroc(pad),
                fmt_ms(*mean_std(fitz)),
                fmt_ms(*mean_std(six)),
                fmt_ms(*mean_std(xfer)),
                fmt_auroc(ctx),
            )
        )
    # reference rows — OOD/leak/ID from locked Phase 1 / cbm_revision; xfer from Phase 6
    base_xfer = [c["pad_heldout_6class"]["balanced_accuracy"]["point"] for c in controls["baseline_soft"]]
    eff_xfer = [c["pad_heldout_6class"]["balanced_accuracy"]["point"] for c in controls["effb3_control"]]
    lines.append(
        "| ResNet-50 baseline | 5 | 0.979 ± 0.001 | 0.655 ± 0.018 | — | 0.868 ± 0.019 | 0.994 | — | {} | n/a |".format(
            fmt_ms(*mean_std(base_xfer))
        )
    )
    lines.append(
        "| EffB3 16-d control | 5 | 0.802 ± 0.020 | 0.683 ± 0.012 | — | 0.726 ± 0.021 | 0.649 | — | {} | n/a |".format(
            fmt_ms(*mean_std(eff_xfer))
        )
    )
    return "\n".join(lines)


def md_table2(head):
    rows = []
    order = [
        ("imagenet_resnet50_raw", "ImageNet ResNet-50 (frozen)", "ImageNet"),
        ("imagenet_effb3_raw", "ImageNet EffNet-B3 (frozen)", "ImageNet"),
        ("baseline_backbone_raw", "Trained R50 backbone_raw", "class-supervised"),
        ("effb3_control_backbone_raw", "Trained EffB3 backbone_raw", "class-supervised"),
        ("z_context", "z_context", "domain-supervised"),
    ]
    by = {r["representation"]: r for r in head["results"]}
    lines = [
        "| Representation | PCA k=1 AUROC (var) | Linear domain-head AUROC | supervision |",
        "|---|---|---|---|",
    ]
    for key, lab, sup in order:
        r = by[key]
        k1 = r["pca_k1"]
        lin = r["linear_domain_head"]
        auroc = lin["AUROC_mean"]
        astd = lin["AUROC_std"]
        if auroc > 0.9999:
            lin_s = "> 0.9999"
        else:
            lin_s = "{:.3f} ± {:.4f}".format(auroc, astd)
        var = k1.get("var_explained")
        k1a = k1["AUROC"]
        if k1a > 0.9999:
            k1_s = "> 0.9999 ({:.1f}%)".format(100 * var) if var is not None else "> 0.9999"
        else:
            if var is None:
                k1_s = "{:.3f}".format(k1a)
            else:
                k1_s = "{:.3f} ({:.1f}%)".format(k1a, 100 * var)
        lines.append("| {} | {} | {} | {} |".format(lab, k1_s, lin_s, sup))
        rows.append({"key": key, "label": lab, "k1": k1, "linear": lin, "supervision": sup})
    return "\n".join(lines), rows


def md_table3():
    rows = [
        (
            "Image memorisation of PAD in L_adv",
            "Phase 2: retrain with pad_heldout never in L_ctx/L_adv (716 / 412 patients)",
            "pad_heldout z_lesion Maha 0.427 ± 0.025 vs pad_adv 0.407 ± 0.021 (Δ ≈ 0.02)",
            "Inversion is not memorisation of adversarial images",
        ),
        (
            "PAD-specific mapping (would be ~0.50 on a third domain)",
            "Phase 2.5a: Fitzpatrick17k, never in any train branch, n=3887",
            "z_lesion Maha 0.399 ± 0.039, same inversion as PAD 0.409",
            "Not PAD-specific; structural of the invariant branch",
        ),
        (
            "Class-mix / missing DF,VASC drives AUROC",
            "Phase 2.5b: 6-class-restricted ID and ID reweighted to PAD mix",
            "6-class 0.414 vs 0.409; reweight 0.240 (inversion strengthens)",
            "Class composition is not the driver",
        ),
        (
            "Clean 2×2: z_context detects domain not class; z_lesion the reverse",
            "Phase 4: 4a hold {DF,VASC}, 4b hold {SCC}",
            "4a z_lesion 0.648 vs z_context 0.641 vs baseline 0.687; 4b z_lesion 0.806 vs z_context 0.607",
            "No double dissociation. Semantic OOD is graded, 4a=4b equal prominence, null 2×2",
        ),
        (
            "z_context 4a is a VASC-colour detector (high VASC, ~0.50 DF)",
            "Phase 4.5b: class-conditional Maha on 4a",
            "z_context DF 0.61, VASC 0.67 — both above 0.50",
            "VASC/DF hypothesis not confirmed",
        ),
        (
            "4b win is near-OOD, 4a loss is far-OOD",
            "Phase 4.5c: cosine-to-nearest-keep on DF/VASC/SCC",
            "cosines 0.570 / 0.566 / 0.565; Pearson(advantage, cosine)=−0.12",
            "No near/far axis. 4a and 4b simply differ",
        ),
        (
            "Leakage↔AUROC is a continuous trade-off (SPS)",
            "Phase 3 λ sweep: leakage and AUROC between λ=0.25 and λ=2",
            "leakage 0.553 → 0.584 while AUROC 0.508 → 0.413. Linearity is an artefact of anchoring at λ=0",
            "SPS withdrawn. Not a tunable trade-off. There is no safe operating point",
        ),
        (
            "OOD collapse is because z_lesion is only 16-d (original manuscript)",
            "Phase 1 EffB3 16-d control + Phase 3 λ=0 (same 16-d CSG)",
            "EffB3 16-d Maha 0.726; CSG λ=0 Maha 0.863; CSG λ=0.25 Maha 0.508",
            "Dimensionality is not the cause. λ_adv is the cause",
        ),
        (
            "OOD collapse is mediated by reduced linear domain leakage",
            "Phase 3 λ=0.25→2 + Phase 13.2 backbone",
            "leakage 0.553→0.584 while AUROC 0.508→0.413; backbone leak 0.982→0.941 while backbone Maha 0.749→0.475",
            "Mediation rejected. Joint effects of λ, not leakage→OOD",
        ),
    ]
    lines = [
        "| Prediction | Test | Result | Rules out |",
        "|---|---|---|---|",
    ]
    for p, t, r, o in rows:
        lines.append("| {} | {} | {} | {} |".format(p, t, r, o))
    return "\n".join(lines), rows


def four_column_md(grouped):
    lines = [
        "| λ_adv | n | leakage bal acc | ID bal acc | OOD AUROC PAD | PAD xfer bal acc (6-cls, heldout) |",
        "|---:|---:|---|---|---|---|",
    ]
    for lam in LAMBDAS:
        recs = grouped[lam]
        leak = mean_std(col(recs, lambda r: r["summary"]["leakage"]["z_lesion"]["bal_acc_mean"]))
        idb = mean_std(col(recs, lambda r: r["summary"]["id_balanced_acc"]))
        pad = col(recs, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
        xfer = mean_std(col(recs, lambda r: r["xfer"]["pad_heldout_6class"]["balanced_accuracy"]["point"]))
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} |".format(
                lam, len(recs), fmt_ms(*leak), fmt_ms(*idb), fmt_auroc(pad), fmt_ms(*xfer)
            )
        )
    return "\n".join(lines)


def xfer_cost_reading(grouped):
    bals = {lam: mean_std(col(grouped[lam], lambda r: r["xfer"]["pad_heldout_6class"]["balanced_accuracy"]["point"])) for lam in LAMBDAS}
    b0 = bals[0.0][0]
    rest = [bals[lam][0] for lam in LAMBDAS if lam > 0]
    mx = max(rest)
    mn = min(rest)
    # slope-ish: compare λ=0 vs mean of λ>0, and vs λ=2
    b_pos = float(np.mean(rest))
    b2 = bals[2.0][0]
    if b_pos <= b0 + 0.01 and mx <= b0 + 0.02:
        kind = "FLAT_OR_FALLS"
        reading = (
            "Full diagnosis: `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md` (array 60585). "
            "Floors for this column: 6-class balanced-acc chance 1/6 ≈ 0.167; "
            "plain-acc majority (always BCC) 276/716 = 0.385. "
            "Cross-domain 6-class balanced accuracy on pad_heldout is flat or falls across λ "
            "(λ=0: {:.3f}; mean λ>0: {:.3f}; λ=2: {:.3f}). "
            "You pay a safety cost and get nothing for it. "
            "The abstract sentence is “while cross-domain accuracy does not improve at all.” "
            "This is the strongest version of the paper. It is not a trade-off."
        ).format(b0, b_pos, b2)
    elif mx > b0 + 0.02:
        kind = "RISES"
        reading = (
            "Cross-domain 6-class balanced accuracy on pad_heldout rises with λ "
            "(λ=0: {:.3f}; max λ>0: {:.3f}). There is a genuine utility gain on PAD transfer. "
            "The paper must present the safety collapse alongside this gain, not as a pure loss. "
            "It remains a cliff on the OOD axis: leakage and ID accuracy do not buy a safe operating point."
        ).format(b0, mx)
    else:
        kind = "AMBIGUOUS"
        reading = (
            "Cross-domain balanced accuracy changes by less than 2 pp (λ=0 {:.3f}, range of λ>0 {:.3f}–{:.3f}). "
            "Treat as no material gain. Safety cost without a compensating transfer benefit."
        ).format(b0, mn, mx)
    return {"kind": kind, "reading": reading, "per_lambda": {str(k): v for k, v in bals.items()}}


def write_master(grouped, controls, cliff, verdict, cost, t1, t2, t3, four, sample_ci):
    l0 = grouped[0.0]
    l025 = grouped[0.25]
    pad0 = mean_std(col(l0, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]))
    pad025 = mean_std(col(l025, lambda r: r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]))
    c_ood = cliff[0]
    md = []
    md.append("# Paper B — MASTER REPORT (writing package)")
    md.append("")
    md.append("Numbers as measured. Detectors, splits and hyperparameters were never retuned toward a prediction. Writable tree: `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/`. Original `CSG-Skin/` was not overwritten.")
    md.append("")
    md.append("**Headline.** Adversarial invariance training makes a dermatoscope model more confident on never-seen-domain data than on its own test set. OOD expected calibration error on pad_heldout rises **0.247 → 0.746** while ID ECE stays flat at **~0.10**; mean OOD softmax confidence overtakes ID by λ=2. The same training collapses covariate-shift OOD detection from **{:.2f} to {:.2f}** at λ_adv=0.25 and inverts it below chance thereafter. ID balanced accuracy does not drop. ECE is the lead result: clinicians and regulators already use it, and it does not require Mahalanobis.".format(pad0[0], pad025[0]))
    md.append("")
    md.append("**It is a cliff, not a trade-off curve. There is no safe operating point.** Do not describe this as a tunable trade-off.")
    md.append("")
    md.append("**Corrected causal claim.** Adversarial training causes both the leakage drop and the OOD collapse. The OOD collapse is **not mediated** by linear domain-decodability. Phase 3: from λ=0.25 to λ=2, leakage rises (0.553 → 0.584) while AUROC falls (0.508 → 0.413). Phase 13.2: backbone leakage is flat (0.982 → 0.941) while backbone Mahalanobis falls 0.749 → 0.475. Domain information remains linearly decodable; OOD detection collapses anyway. The collapse follows from geometric changes a linear domain probe does not capture. Plot the dial against λ, never against leakage.")
    md.append("")
    md.append("SPS (Semantic Purity Score) is **withdrawn**. The n=4 preview curve is **doubly superseded**: it is a correlation across heterogeneous models, and it is an AUROC-vs-leakage plot that asserts the mediation just disproved. Appendix only. It is not evidence.")
    md.append("")
    md.append("Never print `1.0000 ± 0.0000`. Form: **AUROC > 0.9999 (2 discordant pairs / 11.6M)**.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## The four-column dial (the paper)")
    md.append("")
    md.append("PAD transfer is **6-class-restricted** (PAD covers MEL, NV, BCC, AK, BKL, SCC; no DF, no VASC), evaluated on **pad_heldout only** (716 images / 412 patients). PAD labels never entered `L_cls` (`ignore_index=-1`): zero-shot cross-domain transfer. Phase 3 `pad_acc` was computed on `pad_full` and is contaminated; it is not used here.")
    md.append("")
    md.append(four)
    md.append("")
    md.append("Leakage is **2-class balanced accuracy** (ISIC vs PAD). Its chance floor is **0.5**. The figure 0.688 is the **plain-accuracy** majority on the Phase 0 label-only probe set (always predict ISIC) and is not the floor for this column.")
    md.append("")
    md.append("ResNet-50 and EffB3 16-d are reference rows in Table 1 below, not points on this dial.")
    md.append("")
    md.append("### Cross-domain cost (Item 1)")
    md.append("")
    md.append(cost["reading"])
    md.append("")
    md.append("Balanced accuracy being flat at 0→0.25 hides a collapse onto NV (mean pred NV 113 → 374; AK recall 0.245 → 0.028). Per-class, plain acc, macro AUC, melanoma cells, and controls are in `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md`.")
    md.append("")
    md.append("Melanoma sensitivity at specificity 0.85 is reported per λ but **pad_heldout contains 9 melanoma images**. No significance claim is attached to that cell.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## The cliff (the comparison that carries the paper)")
    md.append("")
    md.append("λ_adv=0 (n=5) vs λ_adv=0.25 (n=3), unpaired. Lead with effect size and CI, not p. Family **F_cliff** (4 tests, Holm–Bonferroni α=0.05): OOD AUROC PAD, leakage balanced acc, ID balanced acc, PAD transfer balanced acc.")
    md.append("")
    md.append("| Contrast | λ=0 | λ=0.25 | Δ | bootstrap 95% CI | Cohen's d | Welch p | Holm reject |")
    md.append("|---|---|---|---|---|---|---|---|")
    for c in cliff:
        md.append(
            "| {} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} | [{:.3f}, {:.3f}] | {:.2f} | {:.4g} | {} |".format(
                c["name"],
                c["mean_l0"],
                c["std_l0"],
                c["mean_l025"],
                c["std_l025"],
                c["delta"],
                c["ci95"][0],
                c["ci95"][1],
                c["cohens_d"],
                c["p"],
                "yes" if c.get("holm_reject") else "no",
            )
        )
    md.append("")
    md.append(
        "OOD AUROC PAD collapses by **{:.3f}** (bootstrap 95% CI [{:.3f}, {:.3f}], Cohen's d = {:.2f}, n=5 vs n=3). The CI excludes 0: **{}**.".format(
            c_ood["delta"], c_ood["ci95"][0], c_ood["ci95"][1], c_ood["cohens_d"], str(c_ood["ci_excludes_0"]).lower()
        )
    )
    md.append("")
    md.append("Paired seed-wise comparisons are available only for seeds {42,52,62} (the interior grid). Those three paired Δs on OOD AUROC PAD:")
    md.append("")
    paired = []
    for seed in (42, 52, 62):
        a = [r for r in l0 if r["seed"] == seed][0]
        b = [r for r in l025 if r["seed"] == seed][0]
        va = a["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]
        vb = b["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]
        paired.append(va - vb)
        md.append("- seed {}: {:.3f} → {:.3f} (Δ={:.3f})".format(seed, va, vb, va - vb))
    md.append("")
    md.append("Paired mean Δ = {:.3f}, Cohen's d (paired) = {:.2f}.".format(float(np.mean(paired)), float(np.mean(paired) / np.std(paired, ddof=1)) if np.std(paired, ddof=1) else float("nan")))
    md.append("")
    md.append("Sample-level bootstrap 95% CIs (2000 resamples) for headline cells are in `results/paperB/phase10/sample_bootstrap.json`. Seed-level CIs above are the right object for the λ=0 vs 0.25 contrast (the intervention is a training run, not a test image).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Mechanism verdict (Item 2 / Phase 3.5)")
    md.append("")
    md.append("**{}**".format(verdict["verdict"]))
    md.append("")
    md.append(verdict["reading"])
    md.append("")
    md.append("Refuted alternatives remain refuted (Table 3). If this verdict is not SUPPORTED, the inversion is left unexplained. No post-hoc story is constructed.")
    md.append("")
    md.append("| λ | n | PAD Maha median ratio | PAD Euclid median ratio | PAD Maha IQR ratio | PAD kNN median ratio | PAD norm median ratio | Fitz Maha median ratio |")
    md.append("|---:|---:|---|---|---|---|---|---|")
    for row in verdict["per_lambda"]:
        def f(pair):
            return fmt_ms(pair[0], pair[1])
        md.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["lambda_adv"],
                row["n"],
                f(row["pad_maha_median_ratio"]),
                f(row["pad_euc_median_ratio"]),
                f(row["pad_maha_iqr_ratio"]),
                f(row["pad_knn_median_ratio"]),
                f(row["pad_norm_median_ratio"]),
                f(row["fitz_maha_median_ratio"]),
            )
        )
    md.append("")
    md.append("Prediction (pre-registered): ratio > 1 at λ=0, < 1 by λ=0.25, falling further as λ rises; OOD IQR tightens. Feature norm is a scaling control.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Table 1 — full dial")
    md.append("")
    md.append("OOD AUROC on PAD is Phase 3 `pad_full` unrestricted Maha on `z_lesion`, matching the locked headline (0.86 → 0.51). Transfer is `pad_heldout` only. `z_context` is domain-supervised.")
    md.append("")
    md.append(t1)
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Table 2 — a monitor is cheap")
    md.append("")
    md.append("Lead with the **k=1** column. Linear domain-head AUROC is near-parity and is **not** a `z_context` advantage. Frozen ImageNet ResNet-50, never trained on this data, reaches 0.998.")
    md.append("")
    md.append(t2)
    md.append("")
    md.append("`z_context` is domain-supervised. Factorization does not make domain detectable — it already is. Factorization concentrates domain onto k=1 (AUROC 1.00 at 84.5% variance vs 0.55–0.70 for every other representation).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Table 3 — refuted mechanisms (main text, not appendix)")
    md.append("")
    md.append(t3)
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Figures")
    md.append("")
    md.append("All 300 dpi, PNG+PDF, colourblind-safe (Wong), `results/paperB/figures/`.")
    md.append("")
    md.append("| File | Panel |")
    md.append("|---|---|")
    md.append("| `fig_dial` | Three panels sharing **λ_adv** (never leakage): leakage as a measured quantity, ID bal acc, OOD AUROC. Cliff 0→0.25 annotated. Leakage panel chance floor is 0.5. ECE is the lead result (Phase 13.3); this figure is the λ-dial, not the explanation. |")
    md.append("| `fig_mechanism` | Distance-ratio curve (OOD ÷ ID) vs λ; Maha densities at λ=0 and λ=2. |")
    md.append("| `fig_monitor` | z_lesion vs z_context OOD AUROC across the dial. One collapses, one stays ~1.0. |")
    md.append("| `fig_generalises` | PAD and Fitzpatrick17k collapse together; baseline Fitz 0.994 as a reference line. |")
    md.append("| `fig_semantic_ood_supp` | 4a and 4b, both branches, equal prominence, null result. |")
    md.append("| `fig_domain_axis_supp` | ISIC / Fitzpatrick17k / PAD on the domain axis. Caption: Fitzpatrick17k does not fall on the ISIC side. Overlap with PAD is 0.47. |")
    md.append("| `fig_preview_superseded_supp` | n=4 preview curve. Doubly superseded: heterogeneous-model correlation, and AUROC-vs-leakage asserts a mediation Phase 3 and 13.2 refute. Appendix only. Not evidence. |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Limitations (not run, with justification)")
    md.append("")
    md.append("**Phase 7 (patient-level ISIC splits). Not run.** ID balanced accuracy is compared across λ under an identical split, so any split-induced inflation applies equally to every row and does not threaten the claim. The absolute ID numbers may be optimistic; the relative pattern is unaffected.")
    md.append("")
    md.append("**Phase 8 (background-only). Not run.** The paper no longer makes a shortcut-attribution claim, so this is no longer load-bearing.")
    md.append("")
    md.append("**Phase 5 (full per-class conditional probe). Partially covered.** The label-only domain probe (0.7968 vs majority 0.688, **plain accuracy**) from Phase 0 is the key control and is reported. The leakage column in the dial is 2-class **balanced** accuracy (chance floor 0.5), not plain accuracy against 0.688. The per-class conditional probe is future work.")
    md.append("")
    md.append("**Non-medical replication.** Camelyon17 is in progress (Phase 12). A weak unmatched DANN did not induce invariance; that audit is a result (Discussion). iWildCam remains gated.")
    md.append("")
    md.append("**Raw-source trivial-detector repeat. Blocked** by PAD PNG permissions (owner `anhnv`, mode 600). Known gap. Not ours to chmod.")
    md.append("")
    md.append("**Phase 9 (Fitzpatrick strata on PAD transfer). Not run.** PAD `metadata.csv` is a dangling symlink (`/mnt/data2/Vinh/Ban_sao_datn/...`); Fitzpatrick group labels are not readable. Optional and not load-bearing. `pad_heldout` melanoma n=9 would also make per-group melanoma sensitivity uninterpretable for several strata.")
    md.append("")
    md.append("**SPS.** Withdrawn, not a limitation of execution.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Claims in the original manuscript this work contradicts")
    md.append("")
    md.append("The original `manuscript_main.md` was written for a different paper (CSG-Lite as a utility–leakage method; OOD collapse as a 16-d limitation) and will be rewritten, not patched. It was not found under `CSG-Skin/` in this tree; claims below are taken from that manuscript's published numbers in `results/cbm_revision/`, `results/table1_draft.md`, and the Paper B work-order recap of what the manuscript claimed.")
    md.append("")
    md.append("| Original claim | Corrected |")
    md.append("|---|---|")
    md.append("| CSG-Lite Maha OOD AUROC ~0.41 is a limitation of 16-dimensional `z_lesion` | False. EffB3 16-d control is **0.726 ± 0.021**. CSG at λ_adv=0 (same 16-d) is **0.863 ± 0.025**. Collapse is caused by λ_adv, already **0.508 ± 0.022** at λ=0.25. |")
    md.append("| ResNet-50 leaks domain at probe acc **0.9791**; CSG-Lite reduces it to **~0.71**, and that reduction is the result | Leakage reduction is real. Phase 3 **balanced** acc: λ=0 0.915 → λ=0.25 0.553 (floor 0.5). Published runB **plain** acc ~0.72 (floor 0.688) is a different metric on a different checkpoint; do not put the two in one clause. It is the wrong success metric: a co-effect of λ, not the cause of the OOD cliff (Phase 3 opposite directions; Phase 13.2 backbone decoupling). A practitioner watching leakage, ID accuracy and ID ECE sees improvement while OOD ECE and the safety gate fail. |")
    md.append("| Run B (orth=1) is the strongest method by a utility–leakage criterion | That criterion ignores covariate-shift OOD. Under it, the operating point that looks best is the one where Maha AUROC is ~0.41. |")
    md.append("| Leakage vs OOD AUROC is a continuous, tunable relationship (preview slope; SPS) | Phase 3 refutes continuity. Between λ=0.25 and λ=2 leakage rises while AUROC falls. SPS withdrawn. Preview n=4 is a heterogeneous-model correlation, superseded, appendix only. |")
    md.append("| `z_context` is the reason domain is detectable | Frozen ImageNet ResNet-50 linear head **0.998 ± 0.0003**. Near-parity of linear AUROC. Architectural remainder is k=1 concentration (84.5% var, AUROC 1.00), not detectability. `z_context` is domain-supervised; say so. |")
    md.append("| Clean factorization: lesion branch for class, context branch for domain (implied 2×2 with semantic OOD) | Domain OOD is split (`z_context` > 0.9999 vs `z_lesion` 0.41). Semantic OOD is not: both branches 0.6–0.8, order depends on held-out class. 4a and 4b at equal prominence. No near-OOD story (4.5c null). |")
    md.append("| Shortcut / background attribution as the mechanism of leakage | Not tested (Phase 8 not run). The paper no longer makes this claim. |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Discussion — a weak adversary is not the danger zone")
    md.append("")
    md.append("This paragraph is independent of how the Camelyon17 matched-recipe rerun resolves. It is already established by the coarse scan plus the configuration audit (`results/paperB/phase12/PHASE12_1B_AUDIT.md`).")
    md.append("")
    md.append("The Camelyon17 coarse DANN (SGD, one param group, no feature BN, unbalanced hospital batches, Linear→ReLU→Linear head, GRL on raw features, `adv_lr_multiplier=1`) left 3-class leakage frozen at 0.977–0.963 across λ ∈ {0, 0.1, 1, 10}. Seven of eight adversary knobs mismatched the derm recipe that actually moved leakage. Those four points are void as evidence about the OOD cliff.")
    md.append("")
    md.append("A weak adversary produces neither the leakage drop nor the OOD collapse. Both are effects of a working adversarial objective. That does not mean the collapse is mediated by the leakage probe. Linear domain-decodability can stay high (backbone 0.94) while OOD detection fails. Adversarial training breaks OOD detection through a channel invisible to the standard leakage probe. Many deployed DANN setups are too weak to induce either effect and are correspondingly uninformative about this failure. λ alone does not characterise adversarial pressure. The audit table supports that last sentence, not a mediation claim.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Locked numbers (unchanged)")
    md.append("")
    md.append("- Phase 1: `z_context` > 0.9999 (2 discordant pairs / 11.6M); `z_lesion` 0.409 ± 0.020; R50 0.868 ± 0.019; EffB3 0.726 ± 0.021; ID bal CSG 0.702 vs EffB3 0.683.")
    md.append("- Phase 1.5: trivial Maha max 0.647; logistic 0.876; PCA k=1 `z_context` AUROC 1.00 at 84.5% var.")
    md.append("- Phase 2: pad_adv 1582/961, pad_heldout 716/412, zero patient overlap. Heldout Maha 0.427 ± 0.025.")
    md.append("- Phase 2.5: Fitz 0.399 ± 0.039; axis OVL Fitz–PAD 0.47; claim: not on the ISIC side.")
    md.append("- Label-only domain probe: 0.7968 vs majority 0.688 (**plain** accuracy; not the leakage-column floor).")
    md.append("- Leakage column: 2-class balanced accuracy; chance floor 0.5. λ=0.25: 0.553 (0.053 above floor). Same probe, plain acc 0.702 (0.014 above majority 0.688). Both near chance. Audit: `LEAKAGE_FLOOR_AUDIT.md`.")
    md.append("- Item 1 pad_heldout 6-class transfer: bal acc 0.291 → 0.291 at the cliff, then 0.249 at λ=2. Diagnosis: `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md`.")
    md.append("")
    md.append("Four (now more) predictions failed and were reported as findings. That is why the result is credible.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- Phase 6: `phase6_xfer/` (diagnosis: `phase6_xfer/PHASE6_XFER_DIAGNOSIS.md`)")
    md.append("- Leakage floor audit: `LEAKAGE_FLOOR_AUDIT.md`")
    md.append("- Phase 3.5: `phase35_mech/`")
    md.append("- Phase 10: `phase10/` (this package's json)")
    md.append("- Figures: `figures/fig_*.{png,pdf}`")
    md.append("- This report: `MASTER_REPORT.md`")
    md.append("")
    MASTER.write_text("\n".join(md) + "\n")
    print("wrote", MASTER)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs, controls = collect_runs()
    grouped = by_lambda(runs)

    def get_pad(r):
        return r["summary"]["ood"]["pad_full"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]

    def get_leak(r):
        return r["summary"]["leakage"]["z_lesion"]["bal_acc_mean"]

    def get_id(r):
        return r["summary"]["id_balanced_acc"]

    def get_xfer(r):
        return r["xfer"]["pad_heldout_6class"]["balanced_accuracy"]["point"]

    cliff = [
        cliff_block(col(grouped[0.0], get_pad), col(grouped[0.25], get_pad), "OOD AUROC PAD (z_lesion Maha)"),
        cliff_block(col(grouped[0.0], get_leak), col(grouped[0.25], get_leak), "leakage balanced acc"),
        cliff_block(col(grouped[0.0], get_id), col(grouped[0.25], get_id), "ID balanced acc"),
        cliff_block(col(grouped[0.0], get_xfer), col(grouped[0.25], get_xfer), "PAD transfer balanced acc (6-cls, heldout)"),
    ]
    holm_bonferroni(cliff)

    verdict = mechanism_verdict(grouped)
    cost = xfer_cost_reading(grouped)
    t1 = md_table1(grouped, controls, None)
    t2, t2rows = md_table2(loadj(HEAD_JSON))
    t3, t3rows = md_table3()
    four = four_column_md(grouped)

    sample_ci = []
    for r in runs:
        if r["mech"] is None:
            continue
        sample_ci.append(
            {
                "run_name": r["run_name"],
                "lambda_adv": r["lambda_adv"],
                "seed": r["seed"],
                "ood_auroc_pad_heldout_maha": r["mech"]["ood_auroc_pad_heldout"]["maha"],
                "id_balanced_acc": r["mech"]["id_balanced_acc"],
                "xfer_balanced_acc": r["xfer"]["pad_heldout_6class"]["balanced_accuracy"],
            }
        )

    fig_dial(grouped)
    fig_mechanism(grouped, verdict)
    fig_monitor(grouped)
    fig_generalises(grouped)
    fig_semantic(None)
    fig_preview_superseded()
    fig_domain_axis_caption()

    package = {
        "headline": "cliff not trade-off",
        "sps_withdrawn": True,
        "preview_is_appendix_only": True,
        "cliff": cliff,
        "mechanism": verdict,
        "cross_domain_cost": cost,
        "table3": t3rows,
        "n_csg_runs": len(runs),
        "pad_heldout": {"n_images": 716, "n_patients": 412, "n_melanoma": 9, "classes": ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]},
        "phase9_skipped": "PAD metadata.csv dangling symlink; Fitzpatrick strata not available",
        "limitations": ["phase7", "phase8", "phase5_partial", "non_medical", "raw_source", "phase9"],
    }
    (OUT / "package.json").write_text(json.dumps(package, indent=2) + "\n")
    (OUT / "sample_bootstrap.json").write_text(json.dumps(sample_ci, indent=2) + "\n")
    (OUT / "table1.md").write_text(t1 + "\n")
    (OUT / "table2.md").write_text(t2 + "\n")
    (OUT / "table3.md").write_text(t3 + "\n")
    write_master(grouped, controls, cliff, verdict, cost, t1, t2, t3, four, sample_ci)
    print("verdict", verdict["verdict"])
    print("cost", cost["kind"])
    print("cliff OOD", cliff[0])


if __name__ == "__main__":
    main()
