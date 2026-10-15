#!/usr/bin/env python3
"""Paper B R5 P8: held-out leakage probe (P6 V1) for the P8 main runs next to the current-recipe dual encoder
(Table 1 / R2 item 4) at the same λ, one probe for both (decisions/paperB_r5_p8.md, deviation 2026-10-15 23:15).

V1 = ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit by lesion / patient, probe seeds 42/52/62, on z_lesion_norm,
computed here with r5_p9v2.v1_probe on each run's own isic_test.npz / pad_heldout.npz for every row (P8 last.ckpt and
best-val; Table 1 best-val from the item-4 features; Table 1 last.ckpt from results/paperB/r5/t1last when present).
The P6 per-run value (pad_heldout sliced from pad_full) is listed for reference.

Outputs: results/paperB/r5/p8/main/heldout_probe/{per_run/*.json, summary.csv, REPORT.md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r5_p9v2 as V  # noqa: E402

MAIN = R.P8_DIR / "main"
OUT = MAIN / "heldout_probe"
LAMS = (0.0, 0.5, 2.0)
T1_SEEDS = {0.0: (42, 52, 62, 72, 82), 0.5: (42, 43, 44, 45, 46), 2.0: (42, 52, 62, 72, 82)}
P6 = R.RES / "r5" / "p6" / "per_run"
T1LAST = R.RES / "r5" / "t1last"


def v1(feat_dir):
    feat_dir = Path(feat_dir)
    cache = OUT / "per_run" / "{}__{}.json".format(feat_dir.parent.name, feat_dir.name)
    if cache.exists():
        return json.loads(cache.read_text())["V1"]
    G = V.groups()
    te, ho = np.load(feat_dir / "isic_test.npz"), np.load(feat_dir / "pad_heldout.npz")
    assert np.array_equal(te["labels"].astype(int), G["yi"]), "isic_test row order"
    val = V.v1_probe(te["z_lesion_norm"], ho["z_lesion_norm"], G["gi"], G["gp"])["bal_acc_mean"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"features": str(feat_dir), "V1": val}, indent=1) + "\n")
    return val


def main():
    rows = []
    for lam in LAMS:
        for s in R.SEEDS:
            tag = "main_ladv{}_s{}".format(R.lam_tag(lam), s)
            p = MAIN / "runs" / tag / "summary.json"
            if not p.exists() or json.loads(p.read_text()).get("checkpoint_policy") != R.CKPT_POLICY:
                continue
            j = json.loads(p.read_text())
            rows.append({"model": "P8 (A4)", "lambda": lam, "seed": s, "checkpoint": "last", "epoch": j["primary_epoch"],
                         "V1": v1(MAIN / "features" / tag), "padfull_leakage": j["metrics"]["leakage"]["bal_acc_mean"]})
            b = j["secondary_bestval"]
            rows.append({"model": "P8 (A4)", "lambda": lam, "seed": s, "checkpoint": "best-val", "epoch": b["epoch"],
                         "V1": v1(b["features"]), "padfull_leakage": b["metrics"]["leakage"]["bal_acc_mean"]})
        for s in T1_SEEDS[lam]:
            run = "runB_orth1_ladv{}_s{}".format(R.lam_tag(lam), s)
            p6 = json.loads((P6 / "{}.json".format(run)).read_text())["z_lesion_norm"]["V1"]["bal_acc_mean"]
            (ck,) = list((R.CKPT / "r2_item4" / run).glob("best-*.ckpt"))
            rows.append({"model": "Table 1 (current recipe)", "lambda": lam, "seed": s, "checkpoint": "best-val",
                         "epoch": int(ck.stem.split("-")[1]), "V1": v1(R.ITEM4_FEAT / run), "V1_P6_reference": p6})
            tl = T1LAST / "per_run" / "{}.json".format(run)
            if tl.exists():
                t = json.loads(tl.read_text())
                rows.append({"model": "Table 1 (current recipe)", "lambda": lam, "seed": s, "checkpoint": "last",
                             "epoch": t["last_epoch"], "V1": t["last"]["heldout_leakage_V1"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    L = ["# Held-out leakage (P6 V1, z_lesion_norm): P8 vs current-recipe dual encoder at the same λ", "",
         "Same probe function and protocol for every row (V1: ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit by "
         "lesion / patient, probe seeds 42/52/62). Chance = 0.5; lower = less domain information.", "",
         "| λ | model | checkpoint | n | epochs | V1 held-out leakage | P6 reference (Table 1) |",
         "|---:|---|---|---:|---|---|---|"]
    tex = []
    for (lam, model, ck), d in df.groupby(["lambda", "model", "checkpoint"], sort=True):
        ref = R.pm(R.ms(d["V1_P6_reference"])) if "V1_P6_reference" in d and d["V1_P6_reference"].notna().any() else "—"
        L.append("| {:g} | {} | {} | {} | {}–{} | {} | {} |".format(lam, model, ck, len(d), d.epoch.min(), d.epoch.max(),
                                                                 R.pm(R.ms(d.V1)), ref))
        tex.append("{:g} & {} & {} & {} & {} \\\\".format(lam, model, ck, len(d), R.tex_pm(R.ms(d.V1))))
    L += ["", "LaTeX rows (λ & model & checkpoint & n & V1):", "", "```"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
