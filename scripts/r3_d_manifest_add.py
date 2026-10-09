#!/usr/bin/env python3
"""Append the R3-D triage entries (categories c, d, e) to paper/number_manifest.json; idempotent by id."""
import json
from pathlib import Path

MAN = Path(__file__).resolve().parents[1] / "paper" / "number_manifest.json"
SW = "phase3_sweep/sweep_aggregate.json"
DX = "phase6_xfer/diagnosis_aggregate.json"
EX = "r2/item5/exploratory_results.json"
I2 = "r2/item2/item2_aggregate.json"
NV = ["by_block", "z_lesion_norm|pad_heldout"]

NEW = [
    # (d) markdown-only, now in the A3 JSONs
    ("img_overlap_pct", "60.0", "isic", "r2/item4/image_level_overlap.json", ["pct_sharing"]),
    ("pr_l0", "4.879", None, "phase13/participation_ratio.json", ["by_lambda", "0", "pr_z_lesion_norm", "mean"]),
    ("pr_l025", "4.500", None, "phase13/participation_ratio.json", ["by_lambda", "0.25", "pr_z_lesion_norm", "mean"]),
    # (c) source not indexed before R3-D
    ("item5_cos_l1", "0.589", "camelyon", EX, ["aggregate", "1.0", "Cosine", "mean"]),
    ("item5_knn_l1", "0.648", "camelyon", EX, ["aggregate", "1.0", "kNN", "mean"]),
    ("item5_msp_l1", "0.589", "camelyon", EX, ["aggregate", "1.0", "MSP", "mean"]),
    ("nv_nearest_id_l0", "0.528", "pad_heldout", I2, NV + ["0", "frac_nv_nearest_id", "mean"]),
    ("nv_nearest_id_l2", "0.531", "pad_heldout", I2, NV + ["2", "frac_nv_nearest_id", "mean"]),
    ("ratio_nv_l0", "6.63", "pad_heldout", I2, NV + ["0", "ratio_NV", "mean"]),
    ("ratio_nv_l2", "2.37", "pad_heldout", I2, NV + ["2", "ratio_NV", "mean"]),
    # (e) indexed, but the free scan picked a different candidate
    ("img_idbal_l2", "0.707", "isic", SW, ["by_group", "runB_orth1_ladv2", "id_balanced_acc", "mean"]),
    ("img_fitz_maha_sd_l2", "0.068", "fitzpatrick", SW,
     ["by_group", "runB_orth1_ladv2", "ood.fitzpatrick17k.z_lesion.mahalanobis_classcond.unrestricted", "sd"]),
    ("fitz_ci_lo_l2", "0.386", "fitzpatrick", "reviewer_r1/bootstrap_results.json",
     ["results", "fitzpatrick17k", "2.0", "mahalanobis_classcond", "mean_across_seeds_bootstrap_ci", "lo"]),
    ("resnet50_fitz", "0.994", "fitzpatrick", "phase2_5/phase25_aggregate.json",
     ["methods", "baseline_soft", "fitz_maha", "mean"]),
    ("xfer_effb3_bal", "0.298", None, DX, ["controls", "effb3_control", "bal", "mean"]),
    ("xfer_acc_l2", "0.184", None, DX, ["csg", 4, "acc", "mean"]),
    ("xfer_majority", "0.385", None, DX, ["floors", "plain_majority_always_BCC"]),
    ("xfer_auc_l0", "0.605", None, DX, ["csg", 0, "auc", "mean"]),
    ("xfer_auc_l2", "0.581", None, DX, ["csg", 4, "auc", "mean"]),
    ("mmd_maha", "0.665", "pad_full", "phase15b/mmd/mmd_aggregate.json",
     ["by_group", "mmd_w10000", "ood_pad_full.mahalanobis_classcond", "mean"]),
    ("mmd_knn", "0.738", "pad_full", "phase15b/mmd/mmd_aggregate.json",
     ["by_group", "mmd_w10000", "ood_pad_full.knn_k50", "mean"]),
    ("mmd_energy", "0.694", "pad_full", "phase15b/mmd/mmd_aggregate.json",
     ["by_group", "mmd_w10000", "ood_pad_full.Energy_T1", "mean"]),
    ("hambcn_laplacian", "0.532", "isic", "phase16/b0/summary.json",
     ["handcrafted_site_unmasked", "laplacian_stats", "bal_acc_mean"]),
    ("hambcn_leak_l0", "0.963", "isic", "phase16/b1/dense3seed_aggregate.json", ["rows", 0, "leakage_mean"]),
    ("hambcn_cos_l0", "0.576", "isic", "phase16/b1/dense3seed_aggregate.json", ["rows", 0, "cosine_mean"]),
    ("single_leak_l0", "0.970", None, "phase15/single_dann/single_dann_aggregate.json",
     ["by_group", "single_dann_ladv0", "leakage.pad_full.bal_acc_mean", "mean"]),
    ("single_leak_l025", "0.763", "pad_full", "phase15/single_dann/single_dann_aggregate.json",
     ["by_group", "single_dann_ladv0p25", "leakage.pad_full.bal_acc_mean", "mean"]),
    ("hambcn_ci_lo", "0.666", "isic", "reviewer_r1/bootstrap_phase16_ham_bcn_ladv1.json",
     ["mean_across_seeds_bootstrap_ci", "lo"]),
    ("hambcn_ci_hi", "0.699", "isic", "reviewer_r1/bootstrap_phase16_ham_bcn_ladv1.json",
     ["mean_across_seeds_bootstrap_ci", "hi"]),
    ("camelyon_adv_ce_l0", "0.11", "camelyon", "r2/item5/coarse_decision_primary.json", ["coarse", "0", "adv_min_ce"]),
]

entries = json.loads(MAN.read_text())
have = {e["id"] for e in entries}
for eid, written, label, f, key in NEW:
    if eid not in have:
        entries.append({"id": eid, "written": written, "label": label, "file": f, "key": key,
                        "note": "R3-D triage (results/paperB/r3/d/triage.md)"})
MAN.write_text(json.dumps(entries, indent=2) + "\n")
print(len(entries), "entries")
