#!/usr/bin/env python3
"""R4: (1) lesion-level Table 1, all seven λ at n = 5; (2) image-level Table 1 rows λ ∈ {0.25, 1} from five seeds
(λ ∈ {0, 2} recomputed as a reproduction check); (3) provenance of the ResNet-50 Fitzpatrick 0.994; (4) five
detectors on pad_heldout for the MMD arm next to the single-encoder GRL arm.

Writes results/paperB/r4/REPORT.md and r4_results.json."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402
from r2_item4_report import LES_FEAT, LES_RUNS, row  # noqa: E402

RES = C.PAPERB / "results" / "paperB"
OUT = RES / "r4"
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ENDPOINT_SEEDS = (42, 52, 62, 72, 82)
LES_MID_SEEDS = (42, 43, 44, 45, 46)
DET = (("mahalanobis_classcond", "Mahalanobis"), ("cosine_max", "Cosine"), ("knn_k50", "kNN k=50"),
       ("MSP", "MSP"), ("Energy_T1", "Energy"))


def ms(xs):
    xs = [float(x) for x in xs]
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "n": len(xs)}


def pm(d, sd=True):
    if d is None:
        return "—"
    return "{:.3f} ± {:.3f}".format(d["mean"], d["sd"]) if sd and d["sd"] is not None else "{:.3f}".format(d["mean"])


def tex_pm(d, sd=True):
    return "{:.3f}\\,$\\pm$\\,{:.3f}".format(d["mean"], d["sd"]) if sd else "{:.3f}".format(d["mean"])


def tag(lam, s):
    return "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), s)


def lesion_table():
    out, missing = {}, []
    for lam in LAMS:
        seeds = ENDPOINT_SEEDS if lam in (0.0, 2.0) else LES_MID_SEEDS
        rows = {}
        for s in seeds:
            t = tag(lam, s)
            if not (LES_RUNS / t / "summary.json").exists() or not (LES_FEAT / t / "pad_heldout.npz").exists():
                missing.append(t)
                continue
            rows[s] = row(LES_RUNS, LES_FEAT, t)
        out["{:g}".format(lam)] = {"seeds": sorted(rows), **{k: ms([r[k] for r in rows.values()])
                                                            for k in ("leakage", "id_bal", "maha", "fitz")}}
    return out, missing


def image_rows():
    out, missing = {}, []
    for lam in (0.0, 0.25, 1.0, 2.0):
        cols = defaultdict(list)
        for s in ENDPOINT_SEEDS:
            t = tag(lam, s)
            p13, xf = RES / "phase13" / "per_run" / "{}.json".format(t), RES / "phase6_xfer" / "{}.json".format(t)
            if not p13.exists() or not xf.exists():
                missing.append(t)
                continue
            sm = json.loads((RES / "phase3_sweep" / t / "summary.json").read_text())
            cols["leakage"].append(sm["leakage"]["z_lesion"]["bal_acc_mean"])
            cols["id_bal"].append(sm["id_balanced_acc"])
            cols["id_ece"].append(sm["id_ece"])
            cols["ood_ece"].append(json.loads(p13.read_text())["confidence"]["pad_heldout"]["ece"])
            cols["maha"].append(sm["ood"]["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
            cols["fitz"].append(sm["ood"]["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"])
            cols["xdom"].append(json.loads(xf.read_text())["pad_heldout_6class"]["balanced_accuracy"]["point"])
        out["{:g}".format(lam)] = {k: ms(v) for k, v in cols.items()}
    return out, missing


def resnet50():
    agg = json.loads((RES / "phase2_5" / "phase25_aggregate.json").read_text())
    return {"file": "phase2_5/phase25_aggregate.json", "key": ["methods", "baseline_soft", "fitz_maha"],
            "label_in_aggregate_phase25_py": "ResNet-50 backbone_raw", **agg["methods"]["baseline_soft"]["fitz_maha"]}


def mmd():
    groups, checks = defaultdict(list), []
    for p in sorted((OUT / "mmd").glob("*.json")):
        r = json.loads(p.read_text())
        groups[(r["objective"], float(r["weight"]))].append(r)
        checks.append(r["max_abs_pad_full_diff"])
    arm = {}
    for (obj, w), rs in sorted(groups.items(), key=lambda x: (x[0][0] != "erm", x[0][1])):
        arm["{}_w{:g}".format(obj, w)] = {
            "objective": obj, "weight": w, "seeds": sorted(r["seed"] for r in rs),
            "leakage": ms([r["leakage_bal_acc"] for r in rs]), "id_bal": ms([r["id_balanced_acc"] for r in rs]),
            "id_acc": ms([r["id_acc"] for r in rs]),
            "pad_heldout": {k: ms([r["ood"]["pad_heldout"][k] for r in rs]) for k, _ in DET},
            "pad_full": {k: ms([r["ood"]["pad_full"][k] for r in rs]) for k, _ in DET}}
    grl = defaultdict(list)
    for p in sorted((RES / "phase15" / "single_dann").glob("single_dann_ladv*_s*/summary.json")):
        s = json.loads(p.read_text())
        grl[float(s["lambda_adv"])].append(s)
    grl_out = {"{:g}".format(l): {
        "seeds": sorted(s["seed"] for s in ss),
        "leakage": ms([s["leakage"]["pad_full"]["bal_acc_mean"] for s in ss]),
        "id_bal": ms([s["id_balanced_acc"] for s in ss]),
        "id_acc": ms([s["id_acc"] for s in ss]),
        "pad_heldout": {k: ms([s["ood"]["pad_heldout"][k] for s in ss]) for k, _ in DET}} for l, ss in sorted(grl.items())}
    return arm, grl_out, (max(checks) if checks else None), sum(len(v) for v in groups.values())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    les, les_miss = lesion_table()
    img, img_miss = image_rows()
    rn = resnet50()
    arm, grl, chk, n_mmd = mmd()
    head = C.git_head()
    pending = les_miss + img_miss + ([] if n_mmd == 24 else ["MMD scoring: {} of 24 runs".format(n_mmd)])
    L = ["# R4 — Table 1 at n = 5, ResNet-50 Fitzpatrick source, MMD arm on pad_heldout", "",
         "Commit: `{}`. Code: `scripts/r4_report.py`; new runs `slurm/r2_item4.sbatch` (lesion λ ∈ {{0.25, 0.5, 1, 4, 8}}, "
         "seeds 45, 46), `scripts/r4_extra_phase3_runs.py` (Phase 13 features and Phase 6 cross-domain eval for the image-level "
         "λ ∈ {{0.25, 1}} seeds 72, 82, unchanged code), `scripts/r4_mmd_pad_heldout.py` (unchanged Phase 15 detector code).".format(head), ""]
    L += ["**Verdict:** " + ("all four items complete." if not pending else "incomplete; missing: " + ", ".join(pending)), ""]

    L += ["## (1) Lesion-level Table 1", "",
          "Seeds: λ ∈ {0, 2} 42/52/62/72/82; other λ 42/43/44/45/46. Leakage = z_lesion probe balanced accuracy; "
          "Mahalanobis class-conditional, fit on ISIC train only.", "",
          "| λ | n | Leakage | ID bal acc | Maha pad_heldout | Maha Fitzpatrick |", "|---:|---:|---|---|---|---|"]
    for l, r in les.items():
        L.append("| {} | {} | {} | {} | {} | {} |".format(l, len(r["seeds"]), pm(r["leakage"]), pm(r["id_bal"], False),
                                                       pm(r["maha"]), pm(r["fitz"])))
    L += ["", "LaTeX rows, image-level format:", "", "```"]
    for l, r in les.items():
        L.append("{:<4} & {} & {} & {} & {} & {} \\\\".format(l, len(r["seeds"]), tex_pm(r["leakage"]), tex_pm(r["id_bal"], False),
                                                             tex_pm(r["maha"]), tex_pm(r["fitz"])))
    L += ["```", ""]

    L += ["## (2) Image-level Table 1 rows at n = 5", "",
          "Sources per cell: Phase 3 `summary.json` (leakage, ID bal acc, ID ECE, OOD AUROC = Mahalanobis pad_heldout "
          "unrestricted, Fitzpatrick); Phase 13 `per_run` `confidence.pad_heldout.ece` (OOD ECE); Phase 6 "
          "`pad_heldout_6class.balanced_accuracy.point` (X-dom). λ ∈ {0, 2} reproduce the published rows.", "",
          "| λ | n | Leakage | ID bal acc | ID ECE | OOD ECE | OOD AUROC | Fitz AUROC | X-dom bal acc |",
          "|---:|---:|---|---|---|---|---|---|---|"]
    for l, r in img.items():
        if not r:
            continue
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            l, r["leakage"]["n"], pm(r["leakage"]), pm(r["id_bal"], False), pm(r["id_ece"], False), pm(r["ood_ece"], False),
            pm(r["maha"]), pm(r["fitz"]), pm(r["xdom"])))
    L += ["", "```"]
    for l in ("0.25", "1"):
        r = img[l]
        if r:
            L.append("{:<4} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                l, r["leakage"]["n"], tex_pm(r["leakage"]), tex_pm(r["id_bal"], False), tex_pm(r["id_ece"], False),
                tex_pm(r["ood_ece"], False), tex_pm(r["maha"]), tex_pm(r["fitz"]), tex_pm(r["xdom"])))
    L += ["```", ""]

    L += ["## (3) ResNet-50 Fitzpatrick 0.994", "",
          "Source: `{}` `{}` = {:.4f} ± {:.4f}, n = {} (labelled \"ResNet-50 backbone_raw\" in "
          "`scripts/aggregate_phase25.py`). The file is now tracked; it was gitignored with `phase2_5/`.".format(
              rn["file"], ".".join(rn["key"]), rn["mean"], rn["std"], rn["n"]), ""]

    L += ["## (4) MMD arm on pad_heldout, next to gradient reversal", "",
          "Pipeline check: pad_full recomputed from the same checkpoints; max |recomputed − stored| = {}.".format(
              "{:.1e}".format(chk) if chk is not None else "—"), "",
          "| arm | seeds | Leakage (pad_full probe) | ID acc | ID bal acc | " + " | ".join(lab for _, lab in DET) + " |",
          "|---|---|---|---|---|" + "---|" * len(DET)]
    for k, r in arm.items():
        L.append("| {} | {} | {} | {} | {} | ".format(k, "/".join(map(str, r["seeds"])), pm(r["leakage"]), pm(r["id_acc"]), pm(r["id_bal"]))
                 + " | ".join(pm(r["pad_heldout"][d]) for d, _ in DET) + " |")
    for l, r in grl.items():
        L.append("| GRL single-encoder λ={} | {} | {} | {} | {} | ".format(l, "/".join(map(str, r["seeds"])), pm(r["leakage"]),
                                                                          pm(r["id_acc"]), pm(r["id_bal"]))
                 + " | ".join(pm(r["pad_heldout"][d]) for d, _ in DET) + " |")
    L += ["", "Caveats: MMD/ERM arm has 3 seeds per weight; leakage for both arms is the pad_full probe stored at training time."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "r4_results.json").write_text(json.dumps({
        "commit": head, "pending": pending, "lesion_table1": les, "image_table1": img, "resnet50_fitz": rn,
        "mmd_arm": arm, "grl_arm": grl, "mmd_pad_full_max_abs_diff": chk}, indent=1) + "\n")
    print("pending", pending)


if __name__ == "__main__":
    main()
