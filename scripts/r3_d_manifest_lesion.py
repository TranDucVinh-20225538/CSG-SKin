#!/usr/bin/env python3
"""Lesion-level manifest entries at n = 5 (R3 B1).

1. Appends to paper/number_manifest.json one entry per lesion-level number the text prints now
   (`written` = the printed value, source = the n = 5 result), so stale text fails the gate.
2. Writes paper/number_manifest_lesion_table1.json: entries for a lesion-level Table 1, with `written`
   formatted from the n = 5 result, for use once that table exists in the .tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "paper" / "number_manifest.json"
B1 = "r3/b1/b1_results.json"
SW = "phase3_sweep/sweep_aggregate.json"
RES = json.loads((ROOT / "results" / "paperB" / B1).read_text())


def s(lam, col, stat="mean"):
    return ["summary", lam, col, stat]


TEXT = [
    ("les_leak_l0", "0.925", None, B1, s("0", "leakage")),
    ("les_leak_l2", "0.565", None, B1, s("2", "leakage")),
    ("les_idece_l0", "0.182", None, B1, s("0", "id_ece")),
    ("les_idece_l2", "0.175", None, B1, s("2", "id_ece")),
    ("les_oodece_l0", "0.198", None, B1, s("0", "ood_ece")),
    ("les_oodece_l2", "0.714", None, B1, s("2", "ood_ece")),
    ("les_maha_l0", "0.815", "pad_heldout", B1, s("0", "maha")),
    ("les_maha_l2", "0.400", "pad_heldout", B1, s("2", "maha")),
    ("les_fitz_l0", "0.727", "fitzpatrick", B1, s("0", "fitz")),
    ("les_fitz_l2", "0.407", "fitzpatrick", B1, s("2", "fitz")),
    ("les_idbal_l0", "0.504", None, B1, s("0", "id_bal")),
    ("les_idbal_l2", "0.464", None, B1, s("2", "id_bal")),
    ("les_idbal_sd_l2", "0.028", None, B1, s("2", "id_bal", "sd")),
    ("img_idbal_l0_vs_lesion", "0.693", None, SW, ["by_group", "runB_orth1_ladv0", "id_balanced_acc", "mean"]),
]

BS = "reviewer_r1/bootstrap_results.json"
DX = "phase6_xfer/diagnosis_aggregate.json"
I2 = ["by_block", "z_lesion_norm|pad_heldout"]
P13 = "phase13/per_run_aggregate.json"
ABSTRACT = [
    ("abs_maha_l025", "0.513", "pad_heldout", BS,
     ["results", "pad_heldout", "0.25", "mahalanobis_classcond", "point_mean_over_seeds"]),
    ("abs_ci_lo_l2", "0.414", "pad_heldout", BS,
     ["results", "pad_heldout", "2.0", "mahalanobis_classcond", "mean_across_seeds_bootstrap_ci", "lo"]),
    ("abs_ci_hi_l2", "0.439", "pad_heldout", BS,
     ["results", "pad_heldout", "2.0", "mahalanobis_classcond", "mean_across_seeds_bootstrap_ci", "hi"]),
    ("abs_leak_l0", "0.915", None, SW, ["by_group", "runB_orth1_ladv0", "leakage.z_lesion.bal_acc_mean", "mean"]),
    ("abs_leak_l025", "0.553", None, SW, ["by_group", "runB_orth1_ladv0p25", "leakage.z_lesion.bal_acc_mean", "mean"]),
    ("abs_idbal_l0", "0.692", "isic", SW, ["by_group", "runB_orth1_ladv0", "id_balanced_acc", "mean"]),
    ("abs_conf_pad_l0", "0.489", None, P13,
     ["by_group", "runB_orth1_ladv0", "confidence.pad_heldout.msp_mean", "mean"]),
    ("abs_conf_pad_l2", "0.930", None, P13,
     ["by_group", "runB_orth1_ladv2", "confidence.pad_heldout.msp_mean", "mean"]),
    ("abs_ak_recall_l0", "0.244", None, DX, ["csg", 0, "recall", "AK", "mean"]),
    ("abs_ak_recall_l2", "0.016", None, DX, ["csg", 4, "recall", "AK", "mean"]),
    ("abs_hosp_knn_l0", "0.658", None, "phase16/b1/dense3seed_aggregate.json", ["rows", 0, "knn50_mean"]),
    ("abs_hosp_knn_l025", "0.486", None, "phase16/b1/dense3seed_aggregate.json", ["rows", 1, "knn50_mean"]),
    ("abs_hosp_maha_l1", "0.683", None, "reviewer_r1/bootstrap_phase16_ham_bcn_ladv1.json", ["point_mean_over_seeds"]),
    ("abs_nv_frac_ood_l0", "0.332", None, "r2/item2/item2_aggregate.json", I2 + ["0", "frac_nv_nearest_ood", "mean"]),
    ("abs_nv_frac_ood_l2", "0.601", None, "r2/item2/item2_aggregate.json", I2 + ["2", "frac_nv_nearest_ood", "mean"]),
    ("abs_auroc_nv_l0", "0.807", None, "r2/item2/item2_aggregate.json", I2 + ["0", "auroc_nv_nearest", "mean"]),
    ("abs_auroc_nv_l2", "0.282", None, "r2/item2/item2_aggregate.json", I2 + ["2", "auroc_nv_nearest", "mean"]),
    ("abs_auroc_nonnv_l0", "0.861", None, "r2/item2/item2_aggregate.json", I2 + ["0", "auroc_non_nv_nearest", "mean"]),
    ("abs_auroc_nonnv_l2", "0.644", None, "r2/item2/item2_aggregate.json", I2 + ["2", "auroc_non_nv_nearest", "mean"]),
    ("abs_med_nv_l0", "49.8", None, "r2/item2/item2_aggregate.json", I2 + ["0", "med_ood_to_NV", "mean"]),
    ("abs_med_nv_l2", "17.5", None, "r2/item2/item2_aggregate.json", I2 + ["2", "med_ood_to_NV", "mean"]),
    ("abs_n_isic", "25{,}331", "isic", "phase1_5/preprocess_audit.json",
     ["domains", "raw_source", "isic", "width", "n"]),
]
TEXT += [(i, w, l, f, k) for i, w, l, f, k in ABSTRACT]

COLS = ["leakage", "id_bal", "id_ece", "ood_ece", "maha", "fitz", "xdom", "cosine", "knn", "msp", "energy"]


def main():
    entries = json.loads(MAN.read_text())
    have = {e["id"] for e in entries}
    for eid, written, label, f, key in TEXT:
        if eid not in have:
            note = ("R3 D: abstract" if eid.startswith("abs_") else
                    "R3 D: lesion-level text vs n = 5 (B1); fails while the text prints the n = 3 value")
            entries.append({"id": eid, "written": written, "label": label, "file": f, "key": key, "note": note})
    MAN.write_text(json.dumps(entries, indent=2) + "\n")

    t1 = []
    for lam in ("0", "2"):
        for col in COLS:
            for stat in ("mean", "sd"):
                v = RES["summary"][lam][col][stat]
                t1.append({"id": "t1les_{}_{}_l{}".format(col, stat, lam), "written": "{:.3f}".format(v),
                           "label": {"maha": "pad_heldout", "fitz": "fitzpatrick"}.get(col),
                           "file": B1, "key": s(lam, col, stat),
                           "note": "lesion-level Table 1, n = 5"})
    (ROOT / "paper" / "number_manifest_lesion_table1.json").write_text(json.dumps(t1, indent=2) + "\n")
    print(len(entries), "entries;", len(t1), "lesion Table 1 entries")


if __name__ == "__main__":
    main()
