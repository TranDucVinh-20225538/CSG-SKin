#!/usr/bin/env python3
"""Generate Figure 1 workflow diagram (top-down 3-stage layout)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 1 framework diagram.")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/mnt/data2/Vinh/CSG-Skin/results/figures"),
        help="Directory to save figure1_framework.png/.pdf",
    )
    p.add_argument("--dpi", type=int, default=300, help="PNG export dpi")
    return p.parse_args()


def add_box(ax, x, y, w, h, text, fc, ec="#333333", fs=11, fw="normal", shadow=True):
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.004, y - 0.004),
                w,
                h,
                boxstyle="round,pad=0.010,rounding_size=0.015",
                linewidth=0,
                facecolor="#000000",
                alpha=0.07,
                zorder=1,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=fw, family="DejaVu Sans", zorder=3)


def add_arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="#4a4a4a",
            zorder=4,
        )
    )


def draw_stage1(ax):
    ax.text(0.08, 0.86, "1. Multi-domain Inputs", fontsize=13, weight="semibold", family="DejaVu Sans")
    add_box(ax, 0.10, 0.77, 0.30, 0.08, "ISIC\n(source domain)", fc="#f1f1f1", fs=10.5)
    add_box(ax, 0.60, 0.77, 0.30, 0.08, "PAD-UFES\n(external domain)", fc="#f1f1f1", fs=10.5)

    add_box(ax, 0.23, 0.625, 0.54, 0.12, "Soft Lesion-centered Preprocessing", fc="#f5f5f5", fs=11, fw="semibold")
    add_box(ax, 0.30, 0.655, 0.12, 0.05, "Original", fc="#ececec", fs=9.8, shadow=False)
    add_box(ax, 0.44, 0.655, 0.12, 0.05, "Mask", fc="#ececec", fs=9.8, shadow=False)
    add_box(ax, 0.58, 0.655, 0.12, 0.05, "Soft Crop", fc="#ececec", fs=9.8, shadow=False)
    add_arrow(ax, 0.42, 0.68, 0.44, 0.68)
    add_arrow(ax, 0.56, 0.68, 0.58, 0.68)

    add_arrow(ax, 0.25, 0.77, 0.44, 0.74)
    add_arrow(ax, 0.75, 0.77, 0.56, 0.74)
    add_arrow(ax, 0.50, 0.63, 0.50, 0.56)


def draw_stage2(ax):
    ax.text(0.08, 0.53, "2. Dual-Encoder Disentanglement", fontsize=13, weight="semibold", family="DejaVu Sans")
    add_box(ax, 0.05, 0.20, 0.90, 0.33, "", fc="#f8fafc", ec="#d8dee6", shadow=False)

    # Left (lesion) branch
    add_box(ax, 0.13, 0.43, 0.29, 0.07, "Lesion Encoder\n(EfficientNet-B3)", fc="#dbeafe", ec="#2563eb", fs=10.8)
    add_box(ax, 0.19, 0.35, 0.17, 0.05, "z_lesion", fc="#eaf2ff", ec="#2563eb", fs=11)
    add_box(ax, 0.15, 0.26, 0.25, 0.06, "Diagnosis Head\n8-class prediction", fc="#eaf2ff", ec="#2563eb", fs=10.2)

    # Right (context) branch
    add_box(ax, 0.58, 0.43, 0.29, 0.07, "Context Encoder\n(EfficientNet-B3)", fc="#ffedd5", ec="#ea580c", fs=10.8)
    add_box(ax, 0.64, 0.35, 0.17, 0.05, "z_context", fc="#fff4e8", ec="#ea580c", fs=11)
    add_box(ax, 0.60, 0.26, 0.25, 0.06, "Domain Head\nISIC vs PAD", fc="#fff4e8", ec="#ea580c", fs=10.2)

    # Intra-branch arrows
    add_arrow(ax, 0.275, 0.43, 0.275, 0.40)
    add_arrow(ax, 0.275, 0.35, 0.275, 0.32)
    add_arrow(ax, 0.725, 0.43, 0.725, 0.40)
    add_arrow(ax, 0.725, 0.35, 0.725, 0.32)

    # Orthogonality center
    add_box(ax, 0.44, 0.34, 0.12, 0.08, "Orthogonality\nz_lesion || z_context", fc="#f3f4f6", ec="#6b7280", fs=9.8)
    add_arrow(ax, 0.36, 0.375, 0.44, 0.375)
    add_arrow(ax, 0.64, 0.375, 0.56, 0.375)

    # Loss callout
    add_box(ax, 0.83, 0.33, 0.13, 0.14, "Losses\n\n• CE_cls\n• CE_ctx\n• GRL_adv\n• L_orth", fc="#f8fafc", ec="#64748b", fs=9.6)

    add_arrow(ax, 0.50, 0.22, 0.50, 0.17)


def draw_stage3(ax):
    ax.text(0.08, 0.14, "3. Outputs and Evaluation", fontsize=13, weight="semibold", family="DejaVu Sans")
    add_box(ax, 0.10, 0.05, 0.24, 0.08, "Robust Diagnosis", fc="#dcfce7", ec="#16a34a", fs=10.8, fw="semibold")
    add_box(ax, 0.38, 0.05, 0.24, 0.08, "Reduced Leakage", fc="#dcfce7", ec="#16a34a", fs=10.8, fw="semibold")
    add_box(ax, 0.66, 0.05, 0.24, 0.08, "Improved Trade-off", fc="#dcfce7", ec="#16a34a", fs=10.8, fw="semibold")
    ax.text(
        0.50,
        0.015,
        "Evaluated by: Accuracy, Balanced Accuracy, ECE, Leakage Probe, t-SNE",
        ha="center",
        va="center",
        fontsize=10.0,
        family="DejaVu Sans",
        color="#2f2f2f",
    )


def build_figure():
    fig, ax = plt.subplots(figsize=(10.5, 12), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Framework Overview of the Proposed Shortcut-Guided Cross-domain Diagnosis Model",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        family="DejaVu Sans",
    )

    draw_stage1(ax)
    draw_stage2(ax)
    draw_stage3(ax)
    return fig


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    png_path = args.output_dir / "figure1_framework.png"
    pdf_path = args.output_dir / "figure1_framework.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved: {}".format(png_path))
    print("Saved: {}".format(pdf_path))


if __name__ == "__main__":
    main()

