#!/usr/bin/env python3
"""Paper B R9 E2 report (stage S always; C/M if present). decisions/paperB_r9_e2.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r9" / "e2"


def _pm(xs):
    a = np.asarray(xs, float)
    if len(a) == 0:
        return "—"
    if len(a) == 1:
        return "{:.3f}".format(a[0])
    return "{:.3f} ± {:.3f}".format(a.mean(), a.std(ddof=1))


def stage_s_table():
    sel = json.loads((OUT / "SELECTED_RECIPE.json").read_text()) if (OUT / "SELECTED_RECIPE.json").exists() else None
    gate = json.loads((OUT / "GATE_S.json").read_text()) if (OUT / "GATE_S.json").exists() else None
    rows = []
    for cid in ("A4", "A7"):
        for s in (42, 52):
            p = OUT / "S" / "runs" / "{}_e80_ladv0_s{}".format(cid, s) / "summary.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())["metrics"]
            rows.append({"recipe": cid, "epochs": 80, "seed": s,
                         "id_val": m["val"]["bal_acc"], "id_test": m["id"]["bal_acc"],
                         "degraded": bool(m.get("degraded"))})
    for s in (42, 52):
        p = OUT / "erm_ref" / "single_dann_ladv0_s{}".format(s) / "summary.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["metrics"]
        rows.append({"recipe": "ERM", "epochs": 80, "seed": s,
                     "id_val": m["val"]["bal_acc"], "id_test": m["id"]["bal_acc"],
                     "degraded": bool(m.get("degraded"))})
    return rows, sel, gate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sel, gate = stage_s_table()
    L = ["# Paper B R9 E2: stronger λ = 0 recipe (80 epochs)", "",
         "Design: `decisions/paperB_r9_e2.md`. Commit `{}`.".format(R.C.git_head()), "",
         "## Stage S", "",
         "| recipe | epochs | seed | ID val | ID test | degraded |",
         "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append("| {} | {} | {} | {:.3f} | {:.3f} | {} |".format(
            r["recipe"], r["epochs"], r["seed"], r["id_val"], r["id_test"], r["degraded"]))
    L.append("")
    if sel:
        L += ["Selection: **{}** (val means A4={:.3f}, A7={:.3f}).".format(
            sel["winner"], sel["per_recipe"]["A4"]["val_mean"], sel["per_recipe"]["A7"]["val_mean"]), ""]
        erm = sel["erm"]
        L.append("ERM reference = {:.3f} (max of 0.558 and 80ep two-seed mean {:.3f}).".format(
            erm["erm_reference"], erm["erm_80_seed_mean"]))
    if gate:
        L += ["", "### Gate S", "",
              "**{}** — selected ID test mean = {:.3f}, threshold = ERM−0.03 = {:.3f}.".format(
                  "met" if gate["met"] else "not met", gate["id_test_mean"], gate["threshold"]), ""]
        if not gate["met"]:
            L.append("**competence gap not closed by an 80-epoch schedule**")
            L.append("")
    if (OUT / "GATE_C.json").exists():
        gc = json.loads((OUT / "GATE_C.json").read_text())
        L += ["## Stage C", "",
              "Gate C **{}** — five-seed ID test mean = {:.3f}, threshold = {:.3f}.".format(
                  "met" if gc["met"] else "not met", gc["id_test_mean_5seed"], gc["threshold"]), ""]
        if not gc["met"]:
            L.append("**competence gap not closed by an 80-epoch schedule**")
            L.append("")
    if (OUT / "M" / "OUTCOME.json").exists():
        mo = json.loads((OUT / "M" / "OUTCOME.json").read_text())
        L += ["## Stage M", "", "**Case {}.** {}".format(mo.get("case"), mo.get("reason", "")), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
