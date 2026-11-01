#!/usr/bin/env python3
"""Paper B R8 D2 report (decisions/paperB_r8_d2.md). DINOv2 MK-MMD selected from D1."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_p9v2 as V  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R8 / "d2"
ERM_DIR = R.R7 / "t1" / "erm"
OODS = ("pad_heldout", "fitzpatrick17k") + R.EXTRA_SETS
UNSEEN = ("fitzpatrick17k", "mednode", "sd198", "scin")


def classify(inv, competent, boot):
    pad = boot.get("align|lam0", {}).get("pad_heldout", {}).get("mahalanobis_classcond", {})
    pad_a = pad.get("rule") == "A"
    none_unseen = all(
        boot.get("align|lam0", {}).get(o, {}).get("mahalanobis_classcond", {}).get("rule") != "A"
        for o in ("pad_heldout",) + UNSEEN
    )
    if inv and competent and pad_a:
        return ("answered",
                "invariance reached, competent, Rule A on pad_heldout: W1 answered for this single-encoder model")
    if inv and competent and none_unseen:
        return ("no_invert",
                "invariance reached, competent, Rule A not met on pad_heldout and every unseen set: "
                "scope stays the dual encoder")
    if not inv:
        return ("not_tested",
                "invariance not reached over 5 seeds (seed-42 pass did not hold): the test did not take place")
    return ("other",
            "gap>0.03 or inversion on some sets only: reported as found; settles neither question")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((R.R8 / "d1" / "SELECTED_dinov2_vits14.json").read_text())
    rows, cells = [], defaultdict(dict)
    for f in sorted((OUT / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if "v1" not in s:
            p = V.heldout_probe(f.parent)
            s["v1"] = p["V1"]["bal_acc_mean"]
            s["v1_sd"] = p["V1"]["bal_acc_sd"]
            f.write_text(json.dumps(s, indent=2) + "\n")
        rows.append({"arm": "align MK-MMD w30", "tag": s["tag"], "seed": s["seed"],
                     "lambda_adv": float(s["lambda_adv"]), "v1": s["v1"], "v1_sd": s.get("v1_sd"),
                     **R.table1_values(s), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["align|lam0"][int(s["seed"])] = str(f.parent)
    # ERM λ=0 reference from T1
    for f in sorted(ERM_DIR.glob("*/summary.json")):
        s = json.loads(f.read_text())
        rows.append({"arm": "ERM DINOv2", "tag": s["tag"], "seed": s["seed"],
                     "lambda_adv": 0.0, "v1": None, "v1_sd": None,
                     **R.table1_values(s), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["ERM|lam0"][int(s["seed"])] = str(f.parent)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    B = R.isic_bootstrap(cells, OODS)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")
    align = df[df.arm.str.startswith("align")]
    erm = df[df.arm.str.startswith("ERM")]
    v1_mean = float(align.v1.mean()) if len(align) else float("nan")
    id_a = float(align.id_bal_acc.mean()) if len(align) else float("nan")
    id_e = float(erm.id_bal_acc.mean()) if len(erm) else float("nan")
    gap = id_e - id_a
    inv = v1_mean <= 0.60
    competent = gap <= 0.03
    case, reason = classify(inv, competent, B)
    L = ["# Paper B R8 D2: single-encoder detection test (MK-MMD)", "",
         "Design: `decisions/paperB_r8_d2.md`. Commit `{}`.".format(R.git_head()), "",
         "Selected from D1: `{}` (family=align; **not DANN**).".format(sel.get("tag")), "",
         "## Reading-rule case (computed first)", "",
         "**{}.** {}.".format(case, reason), "",
         "5-seed mean V1 = {:.3f} (invariance ≤0.60: {}). Competence gap = {:.3f} (ERM {:.3f} − align {:.3f}; ≤0.03: {}).".format(
             v1_mean, inv, gap, id_e, id_a, competent), "",
         "| arm | n | V1 | " + " | ".join(R.TABLE1_HEAD) + " |",
         "|---|---:|---:|" + "|".join(["---:"] * len(R.TABLE1_HEAD)) + "|"]
    for arm in ("align MK-MMD w30", "ERM DINOv2"):
        g = df[df.arm == arm]
        if not len(g):
            continue
        v1s = "—" if g.v1.isna().all() else R.ms(g["v1"])
        L.append("| {} | {} | {} | {} |".format(
            arm, len(g), v1s, " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS)))
    pad = B.get("align|lam0", {}).get("pad_heldout", {}).get("mahalanobis_classcond", {})
    L += ["", "pad_heldout Rule A: {} (k_below={}/{}).".format(
        pad.get("rule"), pad.get("k_below"), pad.get("n_seeds")), "",
          "GPU hours (D2 runs): {:.2f}.".format(float(align.gpu_hours.fillna(0).sum())), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "OUTCOME.json").write_text(json.dumps({
        "case": case, "reason": reason, "v1_mean": v1_mean, "gap": gap,
        "invariance": inv, "competent": competent, "pad_rule": pad.get("rule"),
        "selected": sel}, indent=2) + "\n")
    print(case, reason)


if __name__ == "__main__":
    main()
