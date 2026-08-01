#!/usr/bin/env python3
"""Generate draft-ready figures for benchmark analysis."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule
from src.models.csg_lightning import CSGLiteLightning
from src.utils.paths import PROJECT_ROOT


def configure_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.2,
        }
    )


def parse_args():
    p = argparse.ArgumentParser(description="Generate UMAP/TSNE, trade-off, and stability plots.")
    p.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "master_metadata_lesion_only_soft.csv",
    )
    p.add_argument(
        "--ckpt_for_embedding",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "csg_lite" / "runB_s42" / "best-31.ckpt",
    )
    p.add_argument("--batch_size", type=int, default=128)
    # Keep default 0 to avoid shared-memory issues on constrained environments.
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--fig_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    p.add_argument("--max_points", type=int, default=3000, help="Max points for TSNE scatter.")
    p.add_argument("--dpi", type=int, default=300, help="Raster DPI for PNG export (journal-friendly >=300).")
    p.add_argument("--tsne_perplexity", type=float, default=50.0, help="TSNE perplexity.")
    return p.parse_args()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def collect_latents_and_domains(ckpt_path, metadata, batch_size, num_workers):
    lit = CSGLiteLightning.load_from_checkpoint(str(ckpt_path), strict=False)
    model = lit.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dm = SkinDataModule(metadata_csv=metadata, batch_size=batch_size, num_workers=num_workers)
    dm.setup()
    loader = dm.test_dataloader()  # ISIC test + PAD, shuffle=False
    df = dm.ood_test_dataset.data.reset_index(drop=True)
    domains = (df["domain"].astype(str) == "pad_ufes").astype(np.int64).to_numpy()

    z_les, z_ctx = [], []
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        _logits, _dctx, _dadv, zles_b, zctx_b = model(images, x_lesion=None, return_latents=True)
        z_les.append(zles_b.cpu())
        z_ctx.append(zctx_b.cpu())
    z_les = torch.cat(z_les, dim=0).numpy()
    z_ctx = torch.cat(z_ctx, dim=0).numpy()
    if len(domains) != z_les.shape[0]:
        raise RuntimeError("Domain length mismatch with latent rows.")
    return z_les, z_ctx, domains


def _save_png_and_pdf(fig, png_path, dpi):
    png_path = Path(png_path)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight")


def plot_tsne_disentanglement(z_les, z_ctx, domains, out_path, max_points=3000, perplexity=50.0, dpi=320):
    rng = np.random.default_rng(42)
    n = z_les.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        z_les = z_les[idx]
        z_ctx = z_ctx[idx]
        domains = domains[idx]

    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
    emb_les = tsne.fit_transform(z_les)
    emb_ctx = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity).fit_transform(z_ctx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    names = {0: "ISIC", 1: "PAD-UFES"}
    colors = {0: "#1f77b4", 1: "#d62728"}
    for ax, emb, title in [
        (axes[0], emb_les, "(A) Lesion latent (mixed domains desired)"),
        (axes[1], emb_ctx, "(B) Context latent (domain separation expected)"),
    ]:
        for d in [0, 1]:
            m = domains == d
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.7, label=names[d], c=colors[d])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("Latent Space Visualization of Lesion and Context Representations")
    fig.text(0.5, 0.02, "t-SNE shown for visualization only.", ha="center", va="center", fontsize=11)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=dpi)
    plt.close(fig)


def _method_row(label, summary_agg, leakage_agg):
    return {
        "Method": label,
        "Acc": float(summary_agg["id_acc"]["mean"]),
        "AccStd": float(summary_agg["id_acc"]["std"]),
        "Leakage": float(leakage_agg["z_lesion_acc_mean"]["mean"]),
        "LeakageStd": float(leakage_agg["z_lesion_acc_mean"]["std"]),
    }


def _is_pareto_best(rows):
    """Pareto best for maximize Acc and minimize Leakage."""
    out = {}
    for i, r in enumerate(rows):
        dominated = False
        for j, q in enumerate(rows):
            if i == j:
                continue
            if q["Acc"] >= r["Acc"] and q["Leakage"] <= r["Leakage"] and (
                q["Acc"] > r["Acc"] or q["Leakage"] < r["Leakage"]
            ):
                dominated = True
                break
        out[r["Method"]] = not dominated
    return out


def plot_tradeoff(out_path, dpi=320):
    rows = []
    # Baseline Soft aggregate
    p_base = PROJECT_ROOT / "results" / "table1_baseline_soft_aggregate.json"
    if p_base.is_file():
        base = _load_json(p_base)
        rows.append(_method_row("Baseline Soft", base["aggregate"], base["aggregate"]))

    # Run A / GRL aggregate
    p_runa_agg = PROJECT_ROOT / "results" / "table1_runA_grl_aggregate.json"
    if p_runa_agg.is_file():
        runa_agg = _load_json(p_runa_agg)
        rows.append(_method_row("Run A / GRL", runa_agg["aggregate"], runa_agg["aggregate"]))

    # Run B (orth=5) aggregate
    p_runb = PROJECT_ROOT / "results" / "table1_runB_aggregate.json"
    if p_runb.is_file():
        runb = _load_json(p_runb)
        rows.append(_method_row("Run B (orth=5.0)", runb["aggregate"], runb["aggregate"]))

    # Run B (orth=1) aggregate
    p_runb_orth1 = PROJECT_ROOT / "results" / "table1_runB_orth1_aggregate.json"
    if p_runb_orth1.is_file():
        runb_orth1 = _load_json(p_runb_orth1)
        rows.append(_method_row("Run B (orth=1.0)", runb_orth1["aggregate"], runb_orth1["aggregate"]))

    if not rows:
        return

    marker_map = {
        "Baseline Soft": "o",
        "Run A / GRL": "s",
        "Run B (orth=5.0)": "^",
        "Run B (orth=1.0)": "D",
    }
    color_map = {
        "Baseline Soft": "#6c757d",
        "Run A / GRL": "#1f77b4",
        "Run B (orth=5.0)": "#ff7f0e",
        "Run B (orth=1.0)": "#2ca02c",
    }
    row_map = {r["Method"]: r for r in rows}
    needed = ["Baseline Soft", "Run A / GRL", "Run B (orth=5.0)", "Run B (orth=1.0)"]
    if not all(k in row_map for k in needed):
        return

    offsets = {
        "Baseline Soft": (0.006, -0.0002),  # right side
        "Run A / GRL": (-0.007, -0.0015),  # slightly lower
        "Run B (orth=5.0)": (0.003, 0.0010),  # above right
        "Run B (orth=1.0)": (0.0045, -0.0010),  # move right a bit
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160, sharey=False)
    panel_defs = [
        {
            "title": "(A) Overall Comparison",
            "methods": needed,
            "xlim": (0.68, 1.00),
            "ylim": (0.77, 0.82),
            "elinewidth": 1.8,
            "capsize": 4,
        },
        {
            "title": "(B) Zoomed Comparison",
            "methods": ["Run A / GRL", "Run B (orth=5.0)", "Run B (orth=1.0)"],
            "xlim": (0.695, 0.740),
            "ylim": (0.807, 0.813),
            "elinewidth": 1.4,
            "capsize": 3,
        },
    ]

    for ax, panel in zip(axes, panel_defs):
        for method in panel["methods"]:
            r = row_map[method]
            is_baseline = method == "Baseline Soft"
            ax.errorbar(
                r["Leakage"],
                r["Acc"],
                xerr=r["LeakageStd"],
                yerr=r["AccStd"],
                fmt=marker_map.get(method, "o"),
                markersize=10,
                color=color_map.get(method, "#333333"),
                markeredgecolor="#7a7a7a" if is_baseline else "#111111",
                markeredgewidth=0.8 if is_baseline else 1.0,
                capsize=panel["capsize"],
                elinewidth=panel["elinewidth"],
                alpha=0.6 if is_baseline else 1.0,
                zorder=4,
            )
            dx, dy = offsets.get(method, (0.003, 0.0006))
            ax.text(r["Leakage"] + dx, r["Acc"] + dy, method, fontsize=11)

        ax.set_xlim(*panel["xlim"])
        ax.set_ylim(*panel["ylim"])
        ax.set_title(panel["title"])
        ax.set_xlabel("Leakage Probe Accuracy (lower is better)")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("ID Test Accuracy")
    axes[1].set_ylabel("ID Test Accuracy")
    fig.suptitle("Utility-Leakage Trade-off Across Methods", fontsize=15.5)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_leakage_barplot(out_path, dpi=320):
    files = [
        PROJECT_ROOT / "results" / "table1_baseline_soft_aggregate.json",
        PROJECT_ROOT / "results" / "table1_runA_grl_aggregate.json",
        PROJECT_ROOT / "results" / "table1_runB_aggregate.json",
        PROJECT_ROOT / "results" / "table1_runB_orth1_aggregate.json",
    ]
    methods = []
    means = []
    stds = []
    for fp in files:
        if not fp.is_file():
            continue
        obj = _load_json(fp)
        label = obj.get("label", fp.stem)
        agg = obj["aggregate"]["z_lesion_acc_mean"]
        methods.append(label)
        means.append(float(agg["mean"]))
        stds.append(float(agg["std"]))
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    x = np.arange(len(methods))
    colors = ["#6c757d", "#1f77b4", "#ff7f0e", "#2ca02c"][: len(methods)]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="#222222", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Leakage Probe Accuracy")
    ax.set_title("Domain Leakage Comparison Across Methods")
    ax.set_ylim(0.65, 1.02)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2.0, m + 0.01, "{:.3f}".format(m), ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=dpi)
    plt.close(fig)


def _load_runb_metrics():
    files = sorted((PROJECT_ROOT / "results" / "csg_lite").glob("runB_s*/csv_logs/metrics.csv"))
    if not files:
        return []
    return [pd.read_csv(f) for f in files]


def _epoch_series(df, key):
    x = df[["epoch", key]].dropna()
    if x.empty:
        return None
    return x.groupby("epoch", as_index=True)[key].last()


def _stack_by_epoch(series_list):
    epochs = sorted(set().union(*[set(s.index.tolist()) for s in series_list if s is not None]))
    if not epochs:
        return None, None, None
    arr = []
    for s in series_list:
        if s is None:
            continue
        arr.append([float(s.get(ep, np.nan)) for ep in epochs])
    mat = np.asarray(arr, dtype=np.float64)
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return np.asarray(epochs), mean, std


def plot_training_stability(out_path, dpi=320):
    dfs = _load_runb_metrics()
    if not dfs:
        return

    metrics = {
        "val/acc": "Validation Accuracy",
        "train/loss_epoch": "Total Loss",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, metrics.items()):
        ss = [_epoch_series(df, key) for df in dfs]
        ep, m, s = _stack_by_epoch(ss)
        if ep is None:
            ax.set_title(title + " (missing)")
            continue
        ax.plot(ep, m, lw=2)
        ax.fill_between(ep, m - s, m + s, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Run B Training Stability Across Seeds (mean ± std)")
    fig.tight_layout()
    _save_png_and_pdf(fig, out_path, dpi=dpi)
    plt.close(fig)

    # Supplementary-only internals.
    fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
    for ax, key, title in zip(axes_s, ["train/loss_adv", "train/loss_orth"], ["GRL Loss", "Orthogonality Loss"]):
        ss = [_epoch_series(df, key) for df in dfs]
        ep, m, s = _stack_by_epoch(ss)
        if ep is None:
            ax.set_title(title + " (missing)")
            continue
        ax.plot(ep, m, lw=2)
        ax.fill_between(ep, m - s, m + s, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    fig_s.suptitle("Supplementary: Optimization Internals")
    fig_s.tight_layout()
    _save_png_and_pdf(fig_s, Path(out_path).with_name("figure_training_stability_runB_supplementary.png"), dpi=dpi)
    plt.close(fig_s)


def main():
    args = parse_args()
    configure_plot_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42, workers=True)

    z_les, z_ctx, domains = collect_latents_and_domains(
        args.ckpt_for_embedding, args.metadata, args.batch_size, args.num_workers
    )
    plot_tsne_disentanglement(
        z_les,
        z_ctx,
        domains,
        args.fig_dir / "figure_umap_tsne_disentanglement.png",
        max_points=args.max_points,
        perplexity=args.tsne_perplexity,
        dpi=args.dpi,
    )
    plot_tradeoff(args.fig_dir / "figure_tradeoff_utility_vs_leakage.png", dpi=args.dpi)
    plot_leakage_barplot(args.fig_dir / "figure_leakage_barplot.png", dpi=args.dpi)
    plot_training_stability(args.fig_dir / "figure_training_stability_runB.png", dpi=args.dpi)
    print("Saved figures to {}".format(args.fig_dir))


if __name__ == "__main__":
    main()

