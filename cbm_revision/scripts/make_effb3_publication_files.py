#!/usr/bin/env python3
"""Create publication-ready aggregate files for the EffNet-B3 backbone control."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "effb3_control"
CBM = ROOT / "results" / "cbm_revision"


def _mean_std(xs):
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt(mean, std):
    return f"{mean:.4f} +- {std:.4f}"


def _cohens_d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / max(nx + ny - 2, 1))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def _wilcoxon_p(x, y):
    try:
        return float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        return float("nan")


def _load_cbm_json(name):
    return json.loads((CBM / name).read_text(encoding="utf-8"))


def _load_effb3_rows():
    metrics = {}
    with (OUT / "metrics.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            metrics[seed] = {
                "id_acc": float(row["acc"]),
                "id_balanced_acc": float(row["balanced_acc"]),
                "id_ece": float(row["ece"]),
            }
    with (OUT / "leakage_probe.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            metrics[seed]["z_lesion_acc_mean"] = float(row["leakage_probe_mean"])
    maha = {}
    with (OUT / "ood_scores.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["score_type"] == "Mahalanobis":
                maha[int(row["seed"])] = float(row["auroc"])
    for seed, val in maha.items():
        metrics[seed]["ood_maha_auroc"] = val
    return [
        {"seed": seed, **metrics[seed]}
        for seed in sorted(metrics)
        if "ood_maha_auroc" in metrics[seed]
    ]


def _extract_method(label, obj):
    rows = obj["runs"]
    return {
        "label": label,
        "seeds": [int(r["seed"]) for r in rows],
        "id_acc": [float(r["id_acc"]) for r in rows],
        "id_balanced_acc": [float(r["id_balanced_acc"]) for r in rows],
        "id_ece": [float(r["id_ece"]) for r in rows],
        "z_lesion_acc_mean": [float(r["z_lesion_acc_mean"]) for r in rows],
        "ood_maha_auroc": [float(r["ood_maha_auroc"]) for r in rows],
    }


def _extract_effb3(rows):
    return {
        "label": "EffNet-B3 Single Encoder",
        "seeds": [int(r["seed"]) for r in rows],
        "id_acc": [float(r["id_acc"]) for r in rows],
        "id_balanced_acc": [float(r["id_balanced_acc"]) for r in rows],
        "id_ece": [float(r["id_ece"]) for r in rows],
        "z_lesion_acc_mean": [float(r["z_lesion_acc_mean"]) for r in rows],
        "ood_maha_auroc": [float(r["ood_maha_auroc"]) for r in rows],
    }


def write_aggregate_stats(methods):
    fields = [
        "Method",
        "n",
        "Accuracy",
        "Balanced Accuracy",
        "ECE",
        "Leakage Probe Accuracy",
        "Mahalanobis OOD AUROC",
    ]
    with (OUT / "aggregate_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in methods:
            acc = _mean_std(m["id_acc"])
            bacc = _mean_std(m["id_balanced_acc"])
            ece = _mean_std(m["id_ece"])
            leak = _mean_std(m["z_lesion_acc_mean"])
            ood = _mean_std(m["ood_maha_auroc"])
            writer.writerow(
                {
                    "Method": m["label"],
                    "n": len(m["seeds"]),
                    "Accuracy": _fmt(*acc),
                    "Balanced Accuracy": _fmt(*bacc),
                    "ECE": _fmt(*ece),
                    "Leakage Probe Accuracy": _fmt(*leak),
                    "Mahalanobis OOD AUROC": _fmt(*ood),
                }
            )


def write_significance(methods):
    eff = next(m for m in methods if m["label"] == "EffNet-B3 Single Encoder")
    fields = [
        "Comparison",
        "Metric",
        "EffB3_mean",
        "Other_mean",
        "Mean_difference_EffB3_minus_Other",
        "Welch_t_p",
        "Wilcoxon_p",
        "Cohens_d",
        "Interpretation",
    ]
    metrics = [
        ("Balanced Accuracy", "id_balanced_acc", "higher_is_better"),
        ("Leakage Probe Accuracy", "z_lesion_acc_mean", "lower_is_better"),
        ("ECE", "id_ece", "lower_is_better"),
        ("Mahalanobis OOD AUROC", "ood_maha_auroc", "higher_is_better"),
    ]
    rows = []
    for other in methods:
        if other["label"] == eff["label"]:
            continue
        for metric_name, key, direction in metrics:
            x = np.asarray(eff[key], dtype=np.float64)
            y = np.asarray(other[key], dtype=np.float64)
            diff = float(x.mean() - y.mean())
            d = _cohens_d(x, y)
            p_t = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
            p_w = _wilcoxon_p(x, y) if len(x) == len(y) else float("nan")
            if direction == "lower_is_better":
                interp = "EffB3 lower" if diff < 0 else "EffB3 higher"
            else:
                interp = "EffB3 higher" if diff > 0 else "EffB3 lower"
            rows.append(
                {
                    "Comparison": f"EffNet-B3 Single Encoder vs {other['label']}",
                    "Metric": metric_name,
                    "EffB3_mean": f"{x.mean():.4f}",
                    "Other_mean": f"{y.mean():.4f}",
                    "Mean_difference_EffB3_minus_Other": f"{diff:.4f}",
                    "Welch_t_p": f"{p_t:.6f}",
                    "Wilcoxon_p": "" if np.isnan(p_w) else f"{p_w:.6f}",
                    "Cohens_d": f"{d:.4f}",
                    "Interpretation": interp,
                }
            )
    with (OUT / "significance_effb3.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(methods):
    style = {
        "ResNet50 Baseline": ("#8c8c8c", "o", 0.6),
        "EffNet-B3 Single Encoder": ("#9467bd", "P", 0.95),
        "Run A / GRL": ("#1f77b4", "s", 0.95),
        "Run B (orth=1.0)": ("#2ca02c", "D", 0.95),
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    for m in methods:
        leak = _mean_std(m["z_lesion_acc_mean"])
        bacc = _mean_std(m["id_balanced_acc"])
        color, marker, alpha = style[m["label"]]
        ax.errorbar(
            leak[0],
            bacc[0],
            xerr=leak[1],
            yerr=bacc[1],
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1.3,
            capsize=3,
            alpha=alpha,
            label=m["label"],
        )
    offsets = {
        "ResNet50 Baseline": (0.004, -0.001),
        "EffNet-B3 Single Encoder": (0.004, 0.001),
        "Run A / GRL": (-0.018, -0.004),
        "Run B (orth=1.0)": (0.004, 0.002),
    }
    for m in methods:
        leak = _mean_std(m["z_lesion_acc_mean"])[0]
        bacc = _mean_std(m["id_balanced_acc"])[0]
        dx, dy = offsets[m["label"]]
        ax.text(leak + dx, bacc + dy, m["label"], fontsize=9.5, color="#222222")
    ax.set_xlabel("Domain Leakage Probe Accuracy (lower is better)", fontsize=11)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title("Backbone Control: Utility-Leakage Trade-off", fontsize=13, pad=10)
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.68, 1.00)
    ax.set_ylim(0.63, 0.715)
    ax.tick_params(labelsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "backbone_control_tradeoff.png", dpi=300)
    plt.close(fig)


def write_markdown(methods):
    by_label = {m["label"]: m for m in methods}
    eff = by_label["EffNet-B3 Single Encoder"]
    base = by_label["ResNet50 Baseline"]
    runb = by_label["Run B (orth=1.0)"]
    eff_bacc = _mean_std(eff["id_balanced_acc"])
    eff_leak = _mean_std(eff["z_lesion_acc_mean"])
    base_bacc = _mean_std(base["id_balanced_acc"])
    base_leak = _mean_std(base["z_lesion_acc_mean"])
    runb_bacc = _mean_std(runb["id_balanced_acc"])
    runb_leak = _mean_std(runb["z_lesion_acc_mean"])
    lines = [
        "# EffNet-B3 Backbone Control Results",
        "",
        "## Key Aggregate Results",
        "",
        "| Method | Balanced Accuracy | ECE | Leakage | Mahalanobis OOD AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in methods:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                m["label"],
                _fmt(*_mean_std(m["id_balanced_acc"])),
                _fmt(*_mean_std(m["id_ece"])),
                _fmt(*_mean_std(m["z_lesion_acc_mean"])),
                _fmt(*_mean_std(m["ood_maha_auroc"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The EffNet-B3 single-encoder control improves balanced accuracy over the ResNet50 baseline "
                f"({_fmt(*eff_bacc)} vs {_fmt(*base_bacc)}), confirming that the stronger backbone contributes to utility."
            ),
            "",
            (
                f"However, its domain leakage remains high ({_fmt(*eff_leak)}), far above Run B "
                f"({_fmt(*runb_leak)}). This supports the manuscript claim that disentanglement, rather than backbone strength alone, "
                "is responsible for reducing shortcut/domain information in the lesion representation."
            ),
            "",
            (
                f"Run B keeps the best utility-leakage balance: balanced accuracy {_fmt(*runb_bacc)} with substantially lower leakage "
                "than both single-encoder baselines."
            ),
            "",
            "## Generated Files",
            "",
            "- `aggregate_stats.csv`",
            "- `significance_effb3.csv`",
            "- `backbone_control_tradeoff.png`",
            "- `concise_results.md`",
        ]
    )
    (OUT / "concise_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    methods = [
        _extract_method("ResNet50 Baseline", _load_cbm_json("baseline_soft_n5.json")),
        _extract_effb3(_load_effb3_rows()),
        _extract_method("Run A / GRL", _load_cbm_json("runA_grl_n5.json")),
        _extract_method("Run B (orth=1.0)", _load_cbm_json("runB_orth1_n5.json")),
    ]
    write_aggregate_stats(methods)
    write_significance(methods)
    plot_tradeoff(methods)
    write_markdown(methods)
    for name in [
        "aggregate_stats.csv",
        "significance_effb3.csv",
        "backbone_control_tradeoff.png",
        "concise_results.md",
    ]:
        print(f"Saved: {OUT / name}")


if __name__ == "__main__":
    main()

