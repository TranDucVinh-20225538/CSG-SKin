#!/usr/bin/env python3
"""Manuscript figures 1–6 and S1–S3 (plotting only; data under results/paperB/)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import paper_figure_style as pfs

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB")
P3 = PAPERB / "phase3_sweep"
P13_RUN = PAPERB / "phase13" / "per_run"
REPORT = PAPERB / "figures" / "FIGURES_REPORT.md"
MISSING: list[str] = []


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def load_phase3_summaries():
    rows = []
    for p in sorted(P3.glob("runB_orth1_ladv*/summary.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def by_lambda_summaries(rows):
    d = defaultdict(list)
    for r in rows:
        d[float(r["lambda_adv"])].append(r)
    return d


def load_phase13_runs():
    return [json.loads(p.read_text()) for p in sorted(P13_RUN.glob("*.json"))]


def by_lambda_runs(runs):
    d = defaultdict(list)
    for r in runs:
        d[float(r["lambda_adv"])].append(r)
    return d


def agg_lam(by, lam, fn):
    recs = by.get(float(lam), [])
    return mean_std([fn(r) for r in recs])


def intervention_band_x(lams=pfs.LAMS):
    i0 = pfs.lam_index(0.25, lams)
    i1 = pfs.lam_index(2.0, lams)
    return i0 - 0.5, i1 + 0.5


def fig1_intervention(shared_ylim=(0.4, 1.0)):
    rows = load_phase3_summaries()
    by = by_lambda_summaries(rows)
    dual = json.loads((PAPERB / "phase16" / "table1_ood_pad_dual_column.json").read_text())["by_lambda"]
    lams = pfs.LAMS
    x = pfs.lam_xpos(lams)

    panels = [
        ("a", "Leakage", lambda r: r["leakage"]["z_lesion"]["bal_acc_mean"], True, False, True),
        ("b", "ID bal. acc.", lambda r: r["id_balanced_acc"], False, False, True),
        ("c", "OOD AUROC", None, False, True, True),
        ("d", "OOD ECE", None, False, False, False),
    ]

    by13 = by_lambda_runs(load_phase13_runs())

    fig = plt.figure(figsize=(9.6, 2.85), constrained_layout=True)
    gs = GridSpec(1, 4, figure=fig, wspace=0.32)
    x0, x1 = intervention_band_x(lams)

    for i, (tag, title, fn, show_floor, show_chance, shared_scale) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        pfs.style_axis(ax)
        pfs.set_lambda_axis(ax, lams)
        ax.axvspan(x0, x1, color="#CCCCCC", alpha=0.25, zorder=0)
        means, stds = [], []
        for lam in lams:
            if tag == "d":
                m, s = agg_lam(by13, lam, lambda r: r["confidence"]["pad_heldout"]["ece"])
            elif tag == "c":
                block = dual[_dual_pad_key(lam)]
                m, s = block["pad_heldout"]["mean"], block["pad_heldout"]["std"]
            else:
                m, s = agg_lam(by, lam, fn)
            means.append(m)
            stds.append(s)
        pfs.plot_series(ax, x, means, stds, 0, "runB orth=1")
        if show_floor:
            pfs.floor_line(ax, 0.5)
        if show_chance:
            pfs.chance_line(ax)
        ax.set_title(f"({tag}) {title}", fontsize=8, pad=6)
        if shared_scale:
            ax.set_ylim(*shared_ylim)
        if tag == "a":
            ax.set_ylabel("Accuracy / AUROC")
        elif tag == "d":
            ax.set_ylabel("ECE")
        if tag == "b":
            lo, hi = float(np.nanmin(means)), float(np.nanmax(means))
            ax.text(
                0.97,
                0.06,
                f"obs. {lo:.3f}–{hi:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color=pfs.INK,
            )

    fig.text(
        0.5,
        0.01,
        r"shaded: practitioner-favourable $\lambda$ (0.25–2)",
        ha="center",
        va="bottom",
        fontsize=7,
        color=pfs.INK,
    )
    pfs.save_figure(fig, "fig1_intervention")


def fig2_confidence():
    by = by_lambda_runs(load_phase13_runs())
    lams = pfs.LAMS
    x = pfs.lam_xpos(lams)
    series = (
        ("isic_test", "ISIC test"),
        ("pad_heldout", "pad held-out"),
        ("fitzpatrick17k", "Fitzpatrick17k"),
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    pfs.style_axis(ax)
    pfs.set_lambda_axis(ax, lams)
    curves = []
    for slot, (key, lab) in enumerate(series):
        means, stds = [], []
        for lam in lams:
            m, s = agg_lam(by, lam, lambda r, k=key: r["confidence"][k]["msp_mean"])
            means.append(m)
            stds.append(s)
        pfs.plot_series(ax, x, means, stds, slot, lab)
        curves.append((lab, np.array(means)))
    # Crossing: first λ where max(OOD) > ID
    id_y = curves[0][1]
    ood_y = np.maximum(curves[1][1], curves[2][1])
    cross = None
    for i in range(1, len(lams)):
        if ood_y[i] > id_y[i] and ood_y[i - 1] <= id_y[i - 1]:
            cross = x[i - 1] + (x[i] - x[i - 1]) * (id_y[i - 1] - ood_y[i - 1]) / (
                (ood_y[i] - ood_y[i - 1]) - (id_y[i] - id_y[i - 1]) + 1e-12
            )
            break
    if cross is None:
        for i, lam in enumerate(lams):
            if ood_y[i] > id_y[i]:
                cross = float(x[i])
                break
    if cross is not None:
        ax.axvline(cross, color=pfs.CHANCE, ls=":", lw=1.0, zorder=1)
        ax.text(cross + 0.05, 0.92, "OOD conf.\nexceeds ID", fontsize=7, color=pfs.INK, va="top")
    ax.set_ylabel("Mean max-softmax confidence")
    ax.set_title("Figure 2: confidence on unseen domains")
    ax.legend(frameon=False, loc="lower right")
    pfs.save_figure(fig, "fig2_confidence")


def fig3_detectors():
    rows = load_phase3_summaries()
    by = by_lambda_summaries(rows)
    lams = pfs.LAMS
    x = pfs.lam_xpos(lams)
    dets = (
        ("MSP", lambda r: r["ood"]["pad_heldout"]["z_lesion"]["MSP"]),
        ("Energy", lambda r: r["ood"]["pad_heldout"]["z_lesion"]["Energy_T1"]),
        ("cosine", lambda r: r["ood"]["pad_heldout"]["z_lesion"]["cosine_max"]),
        (
            "Mahalanobis",
            lambda r: r["ood"]["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"],
        ),
        ("kNN (50)", lambda r: r["ood"]["pad_heldout"]["z_lesion"]["knn_k50"]),
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    pfs.style_axis(ax)
    pfs.set_lambda_axis(ax, lams)
    pfs.chance_line(ax)
    for slot, (lab, fn) in enumerate(dets):
        means, stds = [], []
        for lam in lams:
            m, s = agg_lam(by, lam, fn)
            means.append(m)
            stds.append(s)
        pfs.plot_series(ax, x, means, stds, slot, lab, ms=5.5, marker_edgewidth=0.9)
    ax.set_ylabel("AUROC (PAD held-out)")
    ax.set_title("Figure 3: detectors on shared 16-d latent")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    pfs.save_figure(fig, "fig3_detectors")


def fig4_generalises():
    rows = load_phase3_summaries()
    by = by_lambda_summaries(rows)
    lams = pfs.LAMS
    x = pfs.lam_xpos(lams)
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    pfs.style_axis(ax)
    pfs.set_lambda_axis(ax, lams)
    pfs.chance_line(ax)
    for slot, (lab, fn) in enumerate(
        (
            ("PAD held-out", lambda r: r["ood"]["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]),
            (
                "Fitzpatrick17k",
                lambda r: r["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"],
            ),
        )
    ):
        means, stds = [], []
        for lam in lams:
            m, s = agg_lam(by, lam, fn)
            means.append(m)
            stds.append(s)
        pfs.plot_series(ax, x, means, stds, slot, lab)
    # Fitz reference lines (phase 2.5 aggregates)
    agg25 = json.loads((PAPERB / "phase2_5" / "phase25_aggregate.json").read_text())["methods"]
    ax.axhline(agg25["baseline_soft"]["fitz_maha"]["mean"], color=pfs.SERIES[2]["color"], ls=":", lw=1.0, zorder=1)
    ax.text(x[-1] + 0.15, agg25["baseline_soft"]["fitz_maha"]["mean"], "ResNet-50\n0.994", fontsize=6, va="center", color=pfs.INK)
    ax.axhline(agg25["effb3_control"]["fitz_maha"]["mean"], color=pfs.SERIES[3]["color"], ls=":", lw=1.0, zorder=1)
    ax.text(x[-1] + 0.15, agg25["effb3_control"]["fitz_maha"]["mean"], "EffB3\n0.649", fontsize=6, va="center", color=pfs.INK)
    m0, _ = agg_lam(by, 0.0, lambda r: r["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
    ax.plot(x[0], m0, marker="*", ms=12, color=pfs.INK, zorder=4, linestyle="None")
    ax.annotate(r"$\lambda{=}0$: 0.764", xy=(x[0], m0), xytext=(0.6, m0 + 0.06), fontsize=7, color=pfs.INK, arrowprops=dict(arrowstyle="-", color=pfs.INK, lw=0.6))
    ax.set_ylabel("Mahalanobis AUROC")
    ax.set_title("Figure 4: beyond the adversarial domain")
    ax.legend(frameon=False, loc="upper right")
    pfs.save_figure(fig, "fig4_generalises")


def fig5_no_mediation():
    runs = load_phase13_runs()
    by = by_lambda_runs(runs)
    lams = pfs.LAMS
    x = pfs.lam_xpos(lams)
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.6), sharex=True)
    panels = (
        ("Backbone domain leakage (bal. acc.)", "backbone_raw_lesion", "leakage_bal_acc_mean", False),
        ("Backbone Mahalanobis AUROC", "backbone_raw_lesion", "mahalanobis_eps1e-3", True),
    )
    ymins, ymaxs = [], []
    for ax, (title, depth, key, chance) in zip(axes, panels):
        pfs.style_axis(ax)
        means, stds = [], []
        for lam in lams:
            m, s = agg_lam(by, lam, lambda r, d=depth, k=key: r["depths_pad_full"][d][k])
            means.append(m)
            stds.append(s)
        pfs.plot_series(ax, x, means, stds, 0, "backbone")
        ymins.append(min(np.array(means) - np.array(stds)))
        ymaxs.append(max(np.array(means) + np.array(stds)))
        if chance:
            pfs.chance_line(ax)
        ax.set_title(title)
        ax.set_ylabel("Score")
    pfs.set_lambda_axis(axes[1], lams)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pfs.LAM_LABELS)
    axes[0].set_xlabel(r"$\lambda_{\mathrm{adv}}$")
    lo, hi = min(ymins), max(ymaxs)
    pad = 0.05 * (hi - lo + 1e-6)
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Figure 5: leakage flat, OOD score falls (shared y-range)", fontsize=8)
    fig.tight_layout()
    pfs.save_figure(fig, "fig5_leakage_no_mediation")


def _dual_pad_key(lam: float) -> str:
    for k in ("0", "0.25", "0.5", "1", "2", "4", "8"):
        if abs(float(k) - float(lam)) < 1e-9:
            return k
    raise KeyError(lam)


def fig6_site_shift():
    dense = json.loads((PAPERB / "phase16" / "b1" / "dense3seed_aggregate.json").read_text())["rows"]
    dense_rev = json.loads((PAPERB / "phase16" / "b1_rev" / "b1_rev_aggregate.json").read_text())["dense3seed"]
    dual = json.loads((PAPERB / "phase16" / "table1_ood_pad_dual_column.json").read_text())["by_lambda"]

    lam_a = [r["lambda"] for r in dense]
    x_a = np.arange(len(lam_a), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.8))

    ax = axes[0]
    pfs.style_axis(ax)
    ax.set_xticks(x_a)
    ax.set_xticklabels([str(v).replace(".0", "") if v == int(v) else str(v) for v in lam_a])
    ax.set_xlabel(r"$\lambda_{\mathrm{adv}}$")
    pfs.chance_line(ax)
    for slot, key, lab in (
        (0, "leakage_mean", "Leakage"),
        (1, "knn50_mean", "kNN (50)"),
        (2, "cosine_mean", "cosine"),
    ):
        means = [r[key] for r in dense]
        stds = [r[key.replace("_mean", "_std")] for r in dense]
        pfs.plot_series(ax, x_a, means, stds, slot, lab)
    ax.set_title("(a) BCN → HAM (collapse reproduces)")
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    pfs.style_axis(ax)
    lams_b = pfs.LAMS
    x_b = pfs.lam_xpos(lams_b)
    pfs.set_lambda_axis(ax, lams_b)
    pfs.chance_line(ax)
    # HAM → BCN (3 seeds at dense λ)
    lam_rev = [r["lambda"] for r in dense_rev]
    x_rev = np.array([pfs.lam_index(l, lams_b) for l in lam_rev], dtype=float)
    means_r = [r["maha_mean"] for r in dense_rev]
    stds_r = [r["maha_std"] for r in dense_rev]
    pfs.plot_series(ax, x_rev, means_r, stds_r, 0, "HAM → BCN Maha")
    # Cross-modality PAD held-out
    means_c, stds_c = [], []
    for lam in lams_b:
        block = dual[_dual_pad_key(lam)]
        means_c.append(block["pad_heldout"]["mean"])
        stds_c.append(block["pad_heldout"]["std"])
    pfs.plot_series(ax, x_b, means_c, stds_c, 1, "Cross-modality (PAD held-out)")
    ax.annotate(f"{means_r[0]:.2f}", xy=(x_rev[0], means_r[0]), xytext=(-12, 8), textcoords="offset points", fontsize=7, color=pfs.INK)
    ax.annotate(f"{means_r[-1]:.2f}", xy=(x_rev[-1], means_r[-1]), xytext=(4, -10), textcoords="offset points", fontsize=7, color=pfs.INK)
    ax.annotate(f"{means_c[0]:.2f}", xy=(x_b[0], means_c[0]), xytext=(-12, -12), textcoords="offset points", fontsize=7, color=pfs.INK)
    ax.annotate(f"{means_c[pfs.lam_index(2.0)]:.2f}", xy=(x_b[pfs.lam_index(2.0)], means_c[pfs.lam_index(2.0)]), xytext=(4, 6), textcoords="offset points", fontsize=7, color=pfs.INK)
    ax.set_title("(b) HAM → BCN vs cross-modality")
    ax.set_ylabel("Mahalanobis AUROC")
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Figure 6: same-modality site shift", fontsize=8)
    fig.tight_layout()
    pfs.save_figure(fig, "fig6_site_shift")


def figS1_semantic():
    agg = json.loads((PAPERB / "phase4_semantic_ood" / "phase4_aggregate.json").read_text())["aggregate"]
    configs = ("4a", "4b")
    methods = ("baseline", "effb3", "runB_orth1")
    labels = {"baseline": "ResNet-50", "effb3": "EffB3", "runB_orth1": "CSG runB"}
    branches = (
        ("z_lesion", "z_lesion_auroc_mean", "z_lesion"),
        ("z_context", "z_context_auroc_mean", "z_context"),
        ("backbone", "backbone_raw_auroc_mean", "backbone"),
    )
    x = np.arange(len(methods), dtype=float)
    w = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.8), sharey=True)
    for ax, cfg in zip(axes, configs):
        pfs.style_axis(ax)
        pfs.chance_line(ax)
        for j, (_, key, blab) in enumerate(branches):
            heights, errs = [], []
            for method in methods:
                row = next(r for r in agg if r["config"] == cfg and r["method"] == method)
                m = row.get(key)
                if m is None or (isinstance(m, float) and np.isnan(m)):
                    heights.append(np.nan)
                    errs.append(0.0)
                else:
                    heights.append(float(m))
                    errs.append(float(row.get(key.replace("_mean", "_std"), 0.0) or 0.0))
            st = pfs.SERIES[j]
            ax.bar(
                x + (j - 1) * w,
                heights,
                width=w,
                color=st["color"],
                edgecolor="none",
                label=blab,
                yerr=errs,
                capsize=2,
                error_kw=dict(ecolor=pfs.INK, lw=0.8),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in methods])
        ax.set_title(f"({cfg}) DF+VASC withheld" if cfg == "4a" else f"({cfg}) SCC withheld")
        ax.set_ylabel("Semantic OOD AUROC")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.02, 1.22))
    fig.suptitle("Figure S1: semantic OOD (null; equal panel weight)", fontsize=8)
    fig.tight_layout()
    pfs.save_figure(fig, "figS1_semantic_ood")


def figS2_domain_axis():
    d = np.load(PAPERB / "phase1_6" / "domain_axis_projections.npz")
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    pfs.style_axis(ax)
    for slot, (key, lab) in enumerate((("isic", "ISIC test"), ("fitz", "Fitzpatrick17k"), ("pad", "PAD"))):
        st = pfs.SERIES[slot]
        ax.hist(d[key], bins=40, density=True, histtype="step", lw=2, color=st["color"], ls=st["ls"], label=lab)
    ax.set_xlabel(r"Domain axis ($k=1$ context PCA, PAD-positive)")
    ax.set_ylabel("Density")
    ax.set_title("Figure S2: domain axis distributions")
    ax.legend(frameon=False)
    pfs.save_figure(fig, "figS2_domain_axis")


def figS3_preview():
    pts = [
        (0.9997, 0.9999, "z_context"),
        (0.979, 0.868, "ResNet-50"),
        (0.802, 0.726, "EffB3"),
        (0.724, 0.409, r"$z_{\mathrm{lesion}}$"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    pfs.style_axis(ax)
    pfs.chance_line(ax)
    for i, (lx, ay, lab) in enumerate(pts):
        st = pfs.SERIES[i]
        ax.scatter(lx, ay, s=64, color=st["color"], marker=st["marker"], label=lab, zorder=3)
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(0.68, 1.01, 50)
    ax.plot(xs, slope * xs + intercept, color=pfs.INK, ls="--", lw=1.2, label="OLS (n=4)")
    ax.set_xlabel("Domain-probe accuracy (leakage)")
    ax.set_ylabel("Mahalanobis OOD AUROC")
    ax.set_title("Figure S3: superseded correlational preview")
    ax.legend(frameon=False, fontsize=7)
    pfs.save_figure(fig, "figS3_preview_superseded")


def grayscale_check():
    from PIL import Image

    out = Path(pfs.ROOT_FIG)
    stems = (
        "fig1_intervention",
        "fig2_confidence",
        "fig3_detectors",
        "fig4_generalises",
        "fig5_leakage_no_mediation",
        "fig6_site_shift",
        "figS1_semantic_ood",
        "figS2_domain_axis",
        "figS3_preview_superseded",
    )
    lines = ["# Grayscale distinguishability (line/marker)\n"]
    for stem in stems:
        png = out / f"{stem}.png"
        if not png.exists():
            continue
        im = Image.open(png).convert("L")
        path = out / f"{stem}_gray.png"
        im.save(path)
        lines.append(f"- {png.name}: saved {path.name}\n")
    return "".join(lines)


def write_report():
    lines = [
        "# Manuscript figures report\n",
        "\nOutputs: `results/paperB/figures/` at 300 dpi PDF+PNG.\n",
        "\n## Data sources\n",
        "- Fig 1a–c: `phase3_sweep/*/summary.json`\n",
        "- Fig 1d, 2, 5: `phase13/per_run/*.json` (`confidence`, `depths_pad_full`)\n",
        "- Fig 1c, 3, 4: PAD held-out / Fitz from Phase 3 summaries\n",
        "- Fig 4 refs: `phase2_5/phase25_aggregate.json` (Fitz baselines)\n",
        "- Fig 6a: `phase16/b1/dense3seed_aggregate.json` (λ ∈ {0, 0.25, 1} only)\n",
        "- Fig 6b: `phase16/b1_rev` dense3seed + `table1_ood_pad_dual_column.json`\n",
        "- S1: `phase4_semantic_ood/phase4_aggregate.json`\n",
        "- S2: `phase1_6/domain_axis_projections.npz`\n",
        "- S3: Phase 1.5 preview points (`plot_preview_curve.py`)\n",
        "\n## Notes\n",
        "- λ axis is **categorical** (even spacing); see figure captions.\n",
        "- Fig 6a uses three λ points (dense 3-seed protocol); not a full seven-λ grid.\n",
        "- Fig S3 plots leakage on x-axis (supplementary, explicitly superseded); main figures do not.\n",
    ]
    if MISSING:
        lines.append("\n## Missing / not plotted\n")
        for m in MISSING:
            lines.append(f"- {m}\n")
    lines.append("\n## Grayscale exports\n")
    lines.append(grayscale_check())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("".join(lines))


def main():
    pfs.setup_matplotlib()
    fig1_intervention()
    fig2_confidence()
    fig3_detectors()
    fig4_generalises()
    fig5_no_mediation()
    fig6_site_shift()
    figS1_semantic()
    figS2_domain_axis()
    figS3_preview()
    write_report()
    print("Wrote figures to", pfs.ROOT_FIG)
    print("Report:", REPORT)


if __name__ == "__main__":
    main()
