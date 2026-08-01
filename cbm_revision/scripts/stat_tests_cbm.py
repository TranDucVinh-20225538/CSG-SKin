#!/usr/bin/env python3
"""Statistical tests for CBM revision (n=5), emphasizing Cohen's d."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "cbm_revision"
OUT_CSV = RESULT_DIR / "stat_tests.csv"
OUT_JSON = RESULT_DIR / "stat_tests.json"


def _load(fp):
    return json.loads(Path(fp).read_text(encoding="utf-8"))


def _cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / max(na + nb - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _welch(a, b):
    try:
        from scipy.stats import ttest_ind

        return float(ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        return float("nan")


def _wilcoxon_if_valid(a, b):
    try:
        from scipy.stats import wilcoxon

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if len(a) != len(b):
            return float("nan")
        d = a - b
        if np.allclose(d, 0.0):
            return 1.0
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _interpret_sig(p):
    if np.isnan(p):
        return "unknown"
    return "yes" if p < 0.05 else "no"


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load(RESULT_DIR / "baseline_soft_n5.json")
    runa = _load(RESULT_DIR / "runA_grl_n5.json")
    runb1 = _load(RESULT_DIR / "runB_orth1_n5.json")

    # pick best by balanced accuracy
    cand = [("Run A / GRL", runa), ("Run B (orth=1.0)", runb1)]
    best_name, best_obj = max(cand, key=lambda x: float(x[1]["aggregate"]["id_balanced_acc"]["mean"]))

    def series(obj, key):
        return [float(r[key]) for r in obj["runs"] if key in r]

    tests = []
    comparisons = [
        ("Run A / GRL vs Baseline", runa),
        ("Run B (orth=1.0) vs Baseline", runb1),
        (f"{best_name} vs Baseline", best_obj),
    ]
    # Remove duplicated comparison rows when best method is already listed above.
    dedup = {}
    for name, obj in comparisons:
        dedup[name] = obj
    comparisons = list(dedup.items())
    metrics = [
        ("Balanced Accuracy", "id_balanced_acc"),
        ("Leakage", "z_lesion_acc_mean"),
    ]
    base_rows = baseline["runs"]

    for metric_name, key in metrics:
        base = [float(r[key]) for r in base_rows if key in r]
        for comp_name, comp_obj in comparisons:
            comp = [float(r[key]) for r in comp_obj["runs"] if key in r]
            p_t = _welch(comp, base)
            d = _cohen_d(comp, base)
            tests.append(
                {
                    "Metric": metric_name,
                    "Comparison": comp_name,
                    "Test": "Welch t-test",
                    "p_value": p_t,
                    "effect_size": d,
                    "significant": _interpret_sig(p_t),
                }
            )
            p_w = _wilcoxon_if_valid(comp, base)
            tests.append(
                {
                    "Metric": metric_name,
                    "Comparison": comp_name,
                    "Test": "Wilcoxon signed-rank",
                    "p_value": p_w,
                    "effect_size": d,
                    "significant": _interpret_sig(p_w),
                }
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Metric", "Comparison", "Test", "p_value", "effect_size", "significant"])
        w.writeheader()
        for r in tests:
            w.writerow(r)
    OUT_JSON.write_text(json.dumps({"tests": tests, "note": "Given n=5, prioritize Cohen's d magnitude."}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()

