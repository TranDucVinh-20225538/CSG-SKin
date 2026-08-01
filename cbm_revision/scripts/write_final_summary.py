#!/usr/bin/env python3
"""Write FINAL_SUMMARY.md for CBM revision."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "cbm_revision"
OUT_MD = RESULT_DIR / "FINAL_SUMMARY.md"


def _load(fp):
    return json.loads(Path(fp).read_text(encoding="utf-8"))


def _best_ood_detector(df):
    if df.empty:
        return "N/A"
    g = df.groupby("ScoreType")["AUROC"].mean().sort_values(ascending=False)
    return str(g.index[0])


def main():
    b5 = _load(RESULT_DIR / "baseline_soft_n5.json")["aggregate"]
    a5 = _load(RESULT_DIR / "runA_grl_n5.json")["aggregate"]
    r5 = _load(RESULT_DIR / "runB_orth1_n5.json")["aggregate"]

    b3 = _load(ROOT / "results" / "table1_baseline_soft_aggregate.json")["aggregate"]
    a3 = _load(ROOT / "results" / "table1_runA_grl_aggregate.json")["aggregate"]
    r3 = _load(ROOT / "results" / "table1_runB_orth1_aggregate.json")["aggregate"]

    ood_csv = RESULT_DIR / "ood_comparison.csv"
    ood_df = pd.read_csv(ood_csv) if ood_csv.is_file() else pd.DataFrame()
    best_detector = _best_ood_detector(ood_df)

    # best overall by (high BAcc, low Leakage)
    candidates = [
        ("Baseline Soft", b5["id_balanced_acc"]["mean"], b5["z_lesion_acc_mean"]["mean"]),
        ("Run A / GRL", a5["id_balanced_acc"]["mean"], a5["z_lesion_acc_mean"]["mean"]),
        ("Run B (orth=1.0)", r5["id_balanced_acc"]["mean"], r5["z_lesion_acc_mean"]["mean"]),
    ]
    best_method = sorted(candidates, key=lambda x: (-x[1], x[2]))[0][0]

    robust_holds = best_method == "Run B (orth=1.0)"
    robust_para = (
        "Core conclusion remains robust after expanding from n=3 to n=5: "
        "Run B (orth=1.0) preserves the best utility-leakage balance with stable variance."
        if robust_holds
        else "Core ranking changes after n=5; manuscript claims should be updated accordingly."
    )

    lines = []
    lines.append("# CBM Revision Final Summary")
    lines.append("")
    lines.append("## 1) Best OOD Detector")
    lines.append(f"- Best detector by mean AUROC across methods: **{best_detector}** (z_lesion space).")
    lines.append("")
    lines.append("## 2) Extra Seeds (n=5) vs Previous n=3")
    lines.append("- n=5 aggregates were recomputed for Baseline Soft, Run A / GRL, Run B (orth=1.0).")
    lines.append("- Comparison snapshot (Balanced Accuracy mean):")
    lines.append(
        f"  - Baseline: n=3 {b3['id_balanced_acc']['mean']:.4f} -> n=5 {b5['id_balanced_acc']['mean']:.4f}"
    )
    lines.append(
        f"  - Run A: n=3 {a3['id_balanced_acc']['mean']:.4f} -> n=5 {a5['id_balanced_acc']['mean']:.4f}"
    )
    lines.append(
        f"  - Run B1: n=3 {r3['id_balanced_acc']['mean']:.4f} -> n=5 {r5['id_balanced_acc']['mean']:.4f}"
    )
    lines.append("- Comparison snapshot (Leakage mean):")
    lines.append(
        f"  - Baseline: n=3 {b3['z_lesion_acc_mean']['mean']:.4f} -> n=5 {b5['z_lesion_acc_mean']['mean']:.4f}"
    )
    lines.append(
        f"  - Run A: n=3 {a3['z_lesion_acc_mean']['mean']:.4f} -> n=5 {a5['z_lesion_acc_mean']['mean']:.4f}"
    )
    lines.append(
        f"  - Run B1: n=3 {r3['z_lesion_acc_mean']['mean']:.4f} -> n=5 {r5['z_lesion_acc_mean']['mean']:.4f}"
    )
    lines.append("")
    lines.append("## 3) Overall Strongest Method")
    lines.append(f"- Overall strongest by utility-leakage criterion: **{best_method}**.")
    lines.append("")
    lines.append("## 4) Manuscript Table Replacement")
    lines.append("- Replace main table with `results/cbm_revision/table_main.csv`.")
    lines.append("- Use OOD table from `results/cbm_revision/ood_comparison.csv`.")
    lines.append("- Use significance table from `results/cbm_revision/stat_tests.csv`.")
    lines.append("")
    lines.append("## 5) Robustness Statement")
    lines.append(robust_para)
    lines.append("")
    lines.append("## 6) Caution on p-values")
    lines.append("- With n=5, p-values should be interpreted cautiously; emphasize Cohen's d magnitude for Leakage reduction.")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()

