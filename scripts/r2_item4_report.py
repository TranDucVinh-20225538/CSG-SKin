#!/usr/bin/env python3
"""R2 Item 4 report: lesion-level vs image-level ISIC splits, λ ∈ {0, 2}, matched seeds {42, 52, 62}."""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C
from analyze_phase13 import conf_pack

OUT = C.R2 / "item4"
IMG_RUNS = C.PAPERB / "results" / "paperB" / "phase3_sweep"
IMG_FEAT = C.FEAT13
LES_RUNS = OUT / "runs"
LES_FEAT = OUT / "features"
LAMS = (0.0, 2.0)
SEEDS = (42, 52, 62)
COLS = (
    ("leakage", "Leakage bal acc"),
    ("id_bal", "ID bal acc"),
    ("id_ece", "ID ECE"),
    ("ood_ece", "OOD ECE (pad_heldout, 6-cls)"),
    ("maha", "Maha pad_heldout"),
    ("cosine", "Cosine pad_heldout"),
    ("knn", "kNN pad_heldout"),
    ("msp", "MSP pad_heldout"),
    ("energy", "Energy pad_heldout"),
    ("fitz", "Fitz AUROC (Maha)"),
)


def row(run_dir, feat_dir, tag):
    s = json.loads((run_dir / tag / "summary.json").read_text())
    h = s["ood"]["pad_heldout"]["z_lesion"]
    hold = np.load(feat_dir / tag / "pad_heldout.npz")
    ood_ece = conf_pack(hold["logits"], hold["labels"], restrict=C.PAD_CLASSES)[0]["ece"]
    return {
        "leakage": s["leakage"]["z_lesion"]["bal_acc_mean"],
        "id_bal": s["id_balanced_acc"],
        "id_ece": s["id_ece"],
        "ood_ece": ood_ece,
        "maha": h["mahalanobis_classcond"]["unrestricted"],
        "cosine": h["cosine_max"],
        "knn": h["knn_k50"],
        "msp": h["MSP"],
        "energy": h["Energy_T1"],
        "fitz": s["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"],
    }


def pattern(t):
    """Directional pattern of the main table, fixed in PRECOMMIT_ITEM4.json."""
    m = {k: {l: np.mean([r[k] for r in t[l]]) for l in LAMS} for k, _ in COLS}
    return {
        "leakage_falls": m["leakage"][2.0] < m["leakage"][0.0],
        "id_bal_not_lower_by_0.02": m["id_bal"][2.0] - m["id_bal"][0.0] > -0.02,
        "id_ece_within_0.03": abs(m["id_ece"][2.0] - m["id_ece"][0.0]) <= 0.03,
        "ood_ece_rises": m["ood_ece"][2.0] > m["ood_ece"][0.0],
        "maha_above_0.5_at_0_below_at_2": m["maha"][0.0] > 0.5 > m["maha"][2.0],
        "all_five_detectors_fall": all(m[k][2.0] < m[k][0.0] for k in ("maha", "cosine", "knn", "msp", "energy")),
        "fitz_above_0.5_at_0_below_at_2": m["fitz"][0.0] > 0.5 > m["fitz"][2.0],
    }


def main():
    tabs = {}
    for name, rd, fd in (("image", IMG_RUNS, IMG_FEAT), ("lesion", LES_RUNS, LES_FEAT)):
        tabs[name] = {l: [row(rd, fd, "runB_orth1_ladv{}_s{}".format(C.lam_tag(l), s)) for s in SEEDS] for l in LAMS}
    pats = {k: pattern(v) for k, v in tabs.items()}
    unchanged = pats["image"] == pats["lesion"]
    verdict = ("Pattern unchanged -> the limitation defence becomes evidence; drop the limitation."
               if unchanged else
               "Pattern changes -> a finding about the main table; takes priority over everything else in R2.")
    split = json.loads((OUT / "split_record.json").read_text())
    head = C.git_head()
    (OUT / "item4_results.json").write_text(json.dumps(
        {"commit": head, "tables": {k: {str(l): v for l, v in d.items()} for k, d in tabs.items()},
         "pattern": pats, "unchanged": unchanged, "verdict": verdict, "split_record": split}, indent=2) + "\n")

    L = ["# R2 Item 4 — lesion-level ISIC splits\n",
         "Commit: `{}`. Pre-commit: `results/paperB/r2/PRECOMMIT_ITEM4.json`.\n".format(head),
         "**Verdict:** {}\n".format(verdict),
         "Seeds {42, 52, 62} at λ ∈ {0, 2} for both splits (matched; not the five-seed mean). Everything except the ISIC split is fixed. "
         "Detectors on z_lesion^norm, fit on ISIC train.\n",
         "Null `lesion_id` policy: each such image is its own group (kept). Affected: {} of {} ISIC images ({:.2%}); "
         "train {:.2%}, val {:.2%}, test {:.2%}. Lesion overlap train/test, val/test, train/val: 0 / 0 / 0.\n".format(
             split["n_null_lesion_all_isic"], split["train"]["n"] + split["val"]["n"] + split["test"]["n"],
             split["null_lesion_frac_all_isic"], split["train"]["null_lesion_frac"], split["val"]["null_lesion_frac"],
             split["test"]["null_lesion_frac"]),
         "**Image-level split (the paper's main table): 3041 of 5067 ISIC test images (60.0%) share a `lesion_id` with ISIC train "
         "or val; the lesion-disjoint test set is 2026 images (436 of them with null `lesion_id`, each its own group).** "
         "This belongs in the manuscript's Methods, not only in Limitations.\n",
         "## Table 1 columns, mean ± s.d. over 3 seeds\n",
         "| Split | λ | " + " | ".join(lab for _, lab in COLS) + " |",
         "|---|---:|" + "---|" * len(COLS)]
    for name in ("image", "lesion"):
        for l in LAMS:
            L.append("| {} | {:g} | ".format(name, l) + " | ".join(
                C.fmt(*C.mean_sd([r[k] for r in tabs[name][l]])[:2]) for k, _ in COLS) + " |")
    L.append("")
    L.append("## Per seed\n")
    L.append("| Split | λ | seed | " + " | ".join(lab for _, lab in COLS) + " |")
    L.append("|---|---:|---:|" + "---|" * len(COLS))
    for name in ("image", "lesion"):
        for l in LAMS:
            for s, r in zip(SEEDS, tabs[name][l]):
                L.append("| {} | {:g} | {} | ".format(name, l, s) + " | ".join("{:.3f}".format(r[k]) for k, _ in COLS) + " |")
    L.append("")
    L.append("## Pre-committed pattern checks\n")
    L.append("| Check | image-level | lesion-level |")
    L.append("|---|---|---|")
    for k in pats["image"]:
        L.append("| {} | {} | {} |".format(k, pats["image"][k], pats["lesion"][k]))
    L.append("")
    L.append("## Caveats\n")
    L.append("- Image-level λ=2 seeds reuse the Phase 2 checkpoints (same protocol); lesion-level λ=2 runs train fresh.")
    L.append("- The lesion-level ISIC test set is a different image set from the image-level one; ID columns are not paired by image.")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "verdict:", verdict, flush=True)


if __name__ == "__main__":
    main()
