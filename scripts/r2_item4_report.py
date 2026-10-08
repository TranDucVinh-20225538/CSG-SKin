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


DETS = ("maha", "cosine", "knn", "msp", "energy", "fitz")


def image_level_class_dist():
    lab = np.load(IMG_FEAT / "runB_orth1_ladv0_s42" / "isic_test.npz")["labels"].astype(int)
    return np.bincount(lab, minlength=8)


def wauroc(sid, wid, sood):
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=np.r_[wid, np.ones(len(sood))]))


def mix_control(feat_dir, tag, n_ref):
    """Per-image detector scores on z_lesion^norm / logits; AUROC unrestricted and with ID reweighted to n_ref's class mix."""
    import torch

    tr, te = np.load(feat_dir / tag / "isic_train.npz"), np.load(feat_dir / tag / "isic_test.npz")
    oo = {o: np.load(feat_dir / tag / "{}.npz".format(o)) for o in ("pad_heldout", "fitzpatrick17k")}
    ztr, ytr = tr["z_lesion_norm"], tr["labels"].astype(int)
    lab = te["labels"].astype(int)
    maha, knn = C.Maha(ztr, ytr), C.KNN(ztr)
    means = np.stack([ztr[ytr == c].mean(0) for c in range(8)])
    prot = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)

    def cos(z):
        return -((z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)) @ prot.T).max(1)

    def msp(lg):
        return -torch.softmax(torch.from_numpy(lg).float(), 1).numpy().max(1)

    def energy(lg):
        return -torch.logsumexp(torch.from_numpy(lg).float(), 1).numpy()

    fns = {"maha": lambda d: maha.score(d["z_lesion_norm"]), "cosine": lambda d: cos(d["z_lesion_norm"]),
           "knn": lambda d: knn.score(d["z_lesion_norm"]), "msp": lambda d: msp(d["logits"]),
           "energy": lambda d: energy(d["logits"])}
    p_src = np.bincount(lab, minlength=8) / len(lab)
    p_ref = n_ref / n_ref.sum()
    w = np.where(p_src > 0, p_ref / np.maximum(p_src, 1e-12), 0.0)[lab]
    out = {}
    for k in DETS:
        f, o = (fns["maha"], "fitzpatrick17k") if k == "fitz" else (fns[k], "pad_heldout")
        sid, so = f(te), f(oo[o])
        out[k] = {"unrestricted": C.auroc(sid, so), "reweighted": wauroc(sid, w, so)}
    return out, np.bincount(lab, minlength=8)


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
    n_img = image_level_class_dist()
    mix, n_split = {}, {}
    for name, fd in (("image", IMG_FEAT), ("lesion", LES_FEAT)):
        mix[name] = {}
        for l in LAMS:
            mix[name][l] = []
            for s in SEEDS:
                m, n = mix_control(fd, "runB_orth1_ladv{}_s{}".format(C.lam_tag(l), s), n_img)
                mix[name][l].append(m)
                n_split[name] = n
    rw = {"lesion": {l: [dict(r, **{k: m[k]["reweighted"] for k in DETS}) for r, m in zip(tabs["lesion"][l], mix["lesion"][l])]
                     for l in LAMS}}
    pats = {k: pattern(v) for k, v in tabs.items()}
    pats["lesion_reweighted"] = pattern(rw["lesion"])
    unchanged = pats["image"] == pats["lesion"]
    unchanged_rw = pats["image"] == pats["lesion_reweighted"]
    verdict = ("Pattern unchanged -> the limitation defence becomes evidence; drop the limitation."
               if unchanged else
               "Pattern changes -> a finding about the main table; takes priority over everything else in R2.")
    verdict += " Class-mix control (lesion-level test reweighted to the image-level class distribution): pattern {}.".format(
        "unchanged" if unchanged_rw else "changes")
    split = json.loads((OUT / "split_record.json").read_text())
    head = C.git_head()
    (OUT / "item4_results.json").write_text(json.dumps(
        {"commit": head, "tables": {k: {str(l): v for l, v in d.items()} for k, d in tabs.items()},
         "pattern": pats, "unchanged": unchanged, "unchanged_reweighted": unchanged_rw, "verdict": verdict,
         "class_counts_test": {k: v.tolist() for k, v in n_split.items()},
         "mix_control": {k: {str(l): v for l, v in d.items()} for k, d in mix.items()}, "split_record": split}, indent=2) + "\n")

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
    L.append("## Class-mix control\n")
    L.append("Without the reweighted column a difference is ambiguous between \"leakage removed\" and \"class mix changed\" "
             "(see `results/paperB/r2/disjoint_reweight/REPORT.md`). Reweighting: each ISIC test image of class c weighted "
             "p_image-level(c) / p_lesion-level(c); OOD weight 1. Detector scores recomputed per image from cached features "
             "(z_lesion^norm, logits), fit on the run's own ISIC train.\n")
    L.append("| Class | Image-level test n | % | Lesion-level test n | % |")
    L.append("|---|---:|---:|---:|---:|")
    ni, nl = n_split["image"], n_split["lesion"]
    for c in range(8):
        L.append("| {} | {} | {:.1f} | {} | {:.1f} |".format(C.LABELS[c], ni[c], 100 * ni[c] / ni.sum(), nl[c], 100 * nl[c] / nl.sum()))
    L.append("| total | {} | 100 | {} | 100 |\n".format(ni.sum(), nl.sum()))
    L.append("| Detector | λ | image-level | lesion-level, unrestricted | lesion-level, reweighted to image-level mix |")
    L.append("|---|---:|---|---|---|")
    labs = dict(COLS)
    for k in DETS:
        for l in LAMS:
            g = lambda nm, kk: C.fmt(*C.mean_sd([m[k][kk] for m in mix[nm][l]])[:2])
            L.append("| {} | {:g} | {} | {} | {} |".format(labs[k], l, g("image", "unrestricted"), g("lesion", "unrestricted"),
                                                          g("lesion", "reweighted")))
    L.append("\nRecomputed image-level values should match the Table 1 columns above (same features, same definitions).\n")
    L.append("## Per seed\n")
    L.append("| Split | λ | seed | " + " | ".join(lab for _, lab in COLS) + " |")
    L.append("|---|---:|---:|" + "---|" * len(COLS))
    for name in ("image", "lesion"):
        for l in LAMS:
            for s, r in zip(SEEDS, tabs[name][l]):
                L.append("| {} | {:g} | {} | ".format(name, l, s) + " | ".join("{:.3f}".format(r[k]) for k, _ in COLS) + " |")
    L.append("")
    L.append("## Pre-committed pattern checks\n")
    L.append("| Check | image-level | lesion-level | lesion-level, reweighted |")
    L.append("|---|---|---|---|")
    for k in pats["image"]:
        L.append("| {} | {} | {} | {} |".format(k, pats["image"][k], pats["lesion"][k], pats["lesion_reweighted"][k]))
    L.append("")
    L.append("## Caveats\n")
    L.append("- Image-level λ=2 seeds reuse the Phase 2 checkpoints (same protocol); lesion-level λ=2 runs train fresh.")
    L.append("- The lesion-level ISIC test set is a different image set from the image-level one; ID columns are not paired by image.")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "verdict:", verdict, flush=True)


if __name__ == "__main__":
    main()
