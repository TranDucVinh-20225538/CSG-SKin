#!/usr/bin/env python3
"""Paper B R7b T1c report (decisions/paperB_r7b_t1c.md). Same case rules as T1."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7B / "t1c"
OODS = ("pad_heldout", "fitzpatrick17k") + R.EXTRA_SETS
EFFB3_ERM_ID = (0.558, 0.022)
LABELLED_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")


def classify_case(gap, boot, leak_l2):
    near = gap <= 0.03
    pad_inv = boot.get("dual|lam2", {}).get("pad_heldout", {}).get("mahalanobis_classcond", {}).get("rule") == "A"
    n_unseen = sum(
        1 for o in LABELLED_UNSEEN
        if boot.get("dual|lam2", {}).get(o, {}).get("mahalanobis_classcond", {}).get("rule") == "A"
    )
    no_inv_any = (not pad_inv) and all(
        boot.get("dual|lam2", {}).get(o, {}).get("mahalanobis_classcond", {}).get("rule") != "A"
        for o in ("pad_heldout",) + LABELLED_UNSEEN
    )
    if near and pad_inv and n_unseen >= 2:
        return "A", "near ERM (gap={:.3f}) and Rule A on pad_heldout plus {}/3 labelled unseen".format(gap, n_unseen)
    if near and no_inv_any:
        return "B", "near ERM (gap={:.3f}) and Rule A fails on pad_heldout and every unseen set".format(gap)
    return "C", ("gap={:.3f} (near={}), pad Rule A={}, labelled-unseen Rule A count={}, "
                 "leakage λ=2={}".format(gap, near, pad_inv, n_unseen,
                                         "—" if leak_l2 is None else "{:.3f}".format(leak_l2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel_path = OUT / "SELECTION.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if sel.get("status") == "stop_after_selection":
        L = ["# Paper B R7b T1c: higher backbone LR", "",
             "Design: `decisions/paperB_r7b_t1c.md`. Commit `{}`.".format(R.git_head()), "",
             "## Reading-rule case (computed first)", "",
             "T1c stopped after selection: dual λ=0 chose backbone LR × 0.1. **T1 is the result.** "
             "No new Case A/B/C from T1c main runs.", "",
             "See `SELECTION.md` / `SELECTION.json`. T1 stays in the paper.", ""]
        if (OUT / "SELECTION.md").exists():
            L += ["", (OUT / "SELECTION.md").read_text()]
        (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
        print("stop after selection")
        return

    dual_m = float(sel.get("dual_chosen_mult", float("nan")))
    erm_m = float(sel.get("erm_chosen_mult", float("nan")))
    rows, cells = [], defaultdict(dict)

    def bb_ok(s, want):
        m = s.get("backbone_lr_mult")
        if m is None:
            return abs(want - 0.1) < 1e-9  # T1 reuse
        return abs(float(m) - want) < 1e-9

    for f in sorted((OUT / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if not bb_ok(s, dual_m):
            continue
        arm = "dual " + s["backbone"]
        rows.append({"arm": arm, "tag": s["tag"], "lambda_adv": float(s["lambda_adv"]), "seed": s["seed"],
                     "backbone_lr_mult": s.get("backbone_lr_mult", dual_m),
                     **R.table1_values(s), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["{}|lam{:g}".format(arm, float(s["lambda_adv"]))][int(s["seed"])] = str(f.parent)
    # reuse T1 dual seed42 if dual_m==0.1 (should have stopped); if dual_m matches a selection run already in runs/
    for f in sorted((OUT / "erm").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if not bb_ok(s, erm_m):
            continue
        arm = "ERM " + s["backbone"]
        rows.append({"arm": arm, "tag": s["tag"], "lambda_adv": float(s["lambda_adv"]), "seed": s["seed"],
                     "backbone_lr_mult": s.get("backbone_lr_mult", erm_m),
                     **R.table1_values(s), "gpu_hours": s.get("gpu_hours"),
                     "degraded": bool(s["id_balanced_acc"] < 0.25)})
        cells["{}|lam{:g}".format(arm, float(s["lambda_adv"]))][int(s["seed"])] = str(f.parent)

    if not rows:
        (OUT / "REPORT.md").write_text("# Paper B R7b T1c\n\nNo main runs yet.\n")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    B = R.isic_bootstrap(cells, OODS)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")
    dual0 = df[(df.arm.str.startswith("dual")) & (df.lambda_adv == 0.0)]
    erm0 = df[(df.arm.str.startswith("ERM")) & (df.lambda_adv == 0.0)]
    dual2 = df[(df.arm.str.startswith("dual")) & (df.lambda_adv == 2.0)]
    gap = float(erm0.id_bal_acc.mean()) - float(dual0.id_bal_acc.mean()) if len(erm0) and len(dual0) else float("nan")
    leak_l2 = float(dual2.leakage.mean()) if len(dual2) else None
    boot_map = {}
    for k, v in B.items():
        if k.startswith("dual ") and "|lam2" in k:
            boot_map["dual|lam2"] = v
    case, reason = classify_case(gap, boot_map, leak_l2)
    L = ["# Paper B R7b T1c: higher backbone LR", "",
         "Design: `decisions/paperB_r7b_t1c.md`. Commit `{}`.".format(R.git_head()), "",
         "## Reading-rule case (computed first)", "",
         "**Case {}.** {}.".format(case, reason), "",
         "Chosen backbone LR: dual × {:g}, ERM × {:g}.".format(dual_m, erm_m), "",
         "Competence gap = {:.3f}. EffB3-ERM reference {:.3f} ± {:.3f}.".format(gap, *EFFB3_ERM_ID), ""]
    if (OUT / "SELECTION.md").exists():
        L += [(OUT / "SELECTION.md").read_text(), ""]
    L += ["| arm | λ | n | " + " | ".join(R.TABLE1_HEAD) + " |",
          "|---|---:|---:|" + "|".join(["---:"] * len(R.TABLE1_HEAD)) + "|"]
    for arm in sorted(df.arm.unique()):
        for lam in sorted(df[df.arm == arm].lambda_adv.unique()):
            g = df[(df.arm == arm) & (df.lambda_adv == lam)]
            L.append("| {} | {:g} | {} | {} |".format(
                arm, lam, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS)))
    L += ["", "```latex"]
    for arm in sorted(df.arm.unique()):
        for lam in sorted(df[df.arm == arm].lambda_adv.unique()):
            g = df[(df.arm == arm) & (df.lambda_adv == lam)]
            L.append("{} $\\lambda={:g}$ & {} \\\\".format(
                arm, lam, " & ".join(R.ms(g[c]).replace(" ± ", "\\,$\\pm$\\,") for c in R.TABLE1_COLS)))
    L += ["```", "", "GPU hours (main rows): {:.2f}.".format(float(df.gpu_hours.fillna(0).sum())), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "OUTCOME.json").write_text(json.dumps({"case": case, "reason": reason, "gap": gap,
                                                   "dual_mult": dual_m, "erm_mult": erm_m}, indent=2) + "\n")
    print("Case", case, reason)


if __name__ == "__main__":
    main()
