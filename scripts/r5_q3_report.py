#!/usr/bin/env python3
"""Paper B R5 Q3 report: backbone generality of the λ pattern (lesion-level, λ ∈ {0, 2} × 5 seeds).

Arms: dual encoder with DINOv2-S/14 (or B/14 fallback; ConvNeXt-T dropped by the user 2026-10-15) (results/paperB/r5/q3/runs), single-encoder
ResNet-50 (results/paperB/r5/p9/stage2, shared with P9 stage 2), and the published dual EfficientNet-B3 rows (R2 item 4)
as the reference. CPU only. Writes results/paperB/r5/q3/{summary.csv, BOOTSTRAP.json, REPORT.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

import pandas as pd  # noqa: E402

Q3 = R.R5 / "q3"
P9S2 = R.R5 / "p9" / "stage2"
LAMS = (0.0, 2.0)
OODS = ("pad_heldout", "fitzpatrick17k") + R.EXTRA_SETS


def keep(tag, suffix):
    return tag.endswith(suffix) if suffix else "_rec" not in tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="current")
    a = ap.parse_args()
    suffix = "" if a.recipe == "current" else ("_recp8" if a.recipe == "p8" else "_rec" + Path(a.recipe).stem)
    out = Q3 if not suffix else Q3 / ("recipe" + suffix)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted((Q3 / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if keep(s["tag"], suffix) and s["penalty"] == "none":
            runs.append(("dual " + s["backbone"], s, f.parent))
    for f in sorted(P9S2.glob("single_resnet50_*/summary.json")):
        s = json.loads(f.read_text())
        if keep(s["tag"], suffix) and float(s["lambda_adv"]) in LAMS and s["penalty"] == "none":
            runs.append(("single resnet50 ({})".format(s["config"]), s, f.parent))
    if not runs:
        print("no Q3 runs yet")
        return
    rows = [{"arm": arm, "tag": s["tag"], "lambda_adv": float(s["lambda_adv"]), "seed": s["seed"],
             **R.table1_values(s), "knn_pad_heldout": s["ood"]["pad_heldout"]["knn_k50"],
             "nv_nearest_pad_heldout": s["ood"]["pad_heldout"]["majority_nearest_frac"],
             "leakage_pca16": s["leakage"].get("test_vs_pad_full_pca16", {}).get("bal_acc_mean"),
             **{"maha_" + e: s["ood"][e]["mahalanobis_classcond"] for e in R.EXTRA_SETS if e in s["ood"]},
             "gate": s["gate"]["status"], "gpu_hours": s.get("gpu_hours")} for arm, s, _d in runs]
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    cells = defaultdict(dict)
    for arm, s, d in runs:
        cells["{}|lam{:g}".format(arm, float(s["lambda_adv"]))][int(s["seed"])] = str(d)
    B = R.isic_bootstrap(cells, OODS)
    (out / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")
    ref = R.item4_reference()
    ref = ref[ref.lambda_adv.isin(LAMS)]

    L = ["# Paper B R5 Q3: backbone generality (lesion-level, λ ∈ {0, 2})", "",
         "Recipe `{}`. Table 1 columns; mean ± SD over seeds. Bootstrap: PREREGISTER_BOOTSTRAP draws, ID by lesion, "
         "pad_heldout by patient, Fitzpatrick / MED-NODE / SD-198 by image, SCIN by case.".format(a.recipe), "",
         "| arm | λ | n | " + " | ".join(R.TABLE1_HEAD) + " | pad_heldout rule | Fitz rule |",
         "|---|---:|---:|" + "---:|" * len(R.TABLE1_HEAD) + "---|---|"]
    tex = []
    for lam in LAMS:
        g = ref[ref.lambda_adv == lam]
        if len(g):
            vals = [R.ms(g[c]) for c in ("leakage", "id_bal_acc", "id_ece")] + ["(R4)", R.ms(g.ood_auroc),
                                                                                  R.ms(g.fitz_auroc), "(R4)"]
            L.append("| dual effb3 (R2 item 4, reference) | {:g} | {} | {} | (R3 A2) | (R3 A2) |".format(
                lam, len(g), " | ".join(vals)))
    for k in sorted(cells):
        arm, lam = k.split("|lam")
        g = df[(df.arm == arm) & (df.lambda_adv == float(lam))]
        b = B.get(k, {})
        ru = [b.get(o, {}).get("mahalanobis_classcond", {}).get("rule", "—") for o in ("pad_heldout", "fitzpatrick17k")]
        L.append("| {} | {} | {} | {} | {} | {} |".format(arm, lam, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS),
                                                         *ru))
        tex.append("{} $\\lambda={}$ & {} \\\\".format(arm, lam, " & ".join(R.tex_ms(g[c]) for c in R.TABLE1_COLS)))
    L += ["", "Extra OOD sets (Mahalanobis) and diagnostics:", "",
          "| arm | λ | kNN pad_heldout | NV-nearest pad_heldout | leakage PCA-16 | " +
          " | ".join("Maha " + e for e in R.EXTRA_SETS) + " |", "|---|---:|" + "---:|" * (3 + len(R.EXTRA_SETS))]
    for k in sorted(cells):
        arm, lam = k.split("|lam")
        g = df[(df.arm == arm) & (df.lambda_adv == float(lam))]
        L.append("| {} | {} | {} |".format(arm, lam, " | ".join(
            R.ms(g[c]) if c in g else "—" for c in ("knn_pad_heldout", "nv_nearest_pad_heldout", "leakage_pca16",
                                                     *["maha_" + e for e in R.EXTRA_SETS]))))
    L += ["", "```latex"] + tex + ["```", "", "## Results in plain words", ""]
    for arm in sorted(df.arm.unique()):
        parts = []
        for lam in LAMS:
            g = df[(df.arm == arm) & (df.lambda_adv == lam)]
            if not len(g):
                continue
            b = B.get("{}|lam{:g}".format(arm, lam), {}).get("pad_heldout", {}).get("mahalanobis_classcond")
            parts.append("at λ = {:g} ({} seeds) leakage {}, ID balanced accuracy {}, pad_heldout Mahalanobis AUROC {}{}, "
                         "Fitzpatrick {}".format(lam, len(g), R.ms(g.leakage), R.ms(g.id_bal_acc), R.ms(g.ood_auroc),
                                                 "" if b is None else " (pooled CI {:.3f}-{:.3f}, {}/{} seeds below 0.5, "
                                                 "rule {})".format(*b["pooled_ci"], b["k_below"], b["n_seeds"], b["rule"]),
                                                 R.ms(g.fitz_auroc)))
        L.append("{}: {}.".format(arm, "; ".join(parts)))
        L.append("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
