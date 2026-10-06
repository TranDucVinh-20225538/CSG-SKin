"""Shared matplotlib style for Paper B manuscript figures (work order palette)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT_FIG = "/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/figures"

LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LAM_LABELS = ("0", "0.25", "0.5", "1", "2", "4", "8")

# Okabe–Ito subset, fixed order (max 5 series).
SERIES = (
    {"color": "#0072B2", "ls": "-", "marker": "o"},
    {"color": "#D55E00", "ls": "--", "marker": "s"},
    {"color": "#009E73", "ls": "-.", "marker": "^"},
    {"color": "#000000", "ls": ":", "marker": "D"},
    {"color": "#E69F00", "ls": "-", "marker": "v"},
)

INK = "#333333"
GRID = "#DDDDDD"
CHANCE = "#888888"


def setup_matplotlib():
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def lam_xpos(lam_values=None):
    lams = lam_values if lam_values is not None else LAMS
    return np.arange(len(lams), dtype=float)


def lam_index(lam: float, lam_values=None) -> int:
    lams = lam_values if lam_values is not None else LAMS
    for i, x in enumerate(lams):
        if abs(float(x) - float(lam)) < 1e-9:
            return i
    raise KeyError(lam)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def set_lambda_axis(ax, lam_values=None, xlabel=r"$\lambda_{\mathrm{adv}}$"):
    lams = lam_values if lam_values is not None else LAMS
    xp = lam_xpos(lams)
    ax.set_xticks(xp)
    ax.set_xticklabels([str(x).replace(".0", "") if x == int(x) else str(x) for x in lams])
    ax.set_xlabel(xlabel)
    ax.set_xlim(xp[0] - 0.35, xp[-1] + 0.35)


def chance_line(ax, label_once=True):
    ax.axhline(0.5, color=CHANCE, ls="--", lw=0.8, zorder=1)
    if label_once:
        ax.text(0.02, 0.502, "chance", transform=ax.get_yaxis_transform(), fontsize=7, color=CHANCE, va="bottom")


def floor_line(ax, y=0.5):
    ax.axhline(y, color=CHANCE, ls="--", lw=0.8, zorder=1)


def plot_series(ax, xpos, means, stds, slot: int, label: str, direct_label=False):
    st = SERIES[slot]
    y = np.asarray(means, dtype=float)
    e = np.asarray(stds, dtype=float)
    line, = ax.plot(
        xpos,
        y,
        color=st["color"],
        ls=st["ls"],
        marker=st["marker"],
        lw=2,
        ms=8,
        label=label,
        zorder=3,
    )
    ax.fill_between(xpos, y - e, y + e, color=st["color"], alpha=0.2, linewidth=0, zorder=2)
    if direct_label and len(xpos):
        ax.annotate(
            label,
            xy=(xpos[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color=INK,
            va="center",
        )
    return line


def save_figure(fig, stem: str):
    from pathlib import Path

    out = Path(ROOT_FIG)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
