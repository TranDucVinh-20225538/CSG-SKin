#!/usr/bin/env python3
"""Paper B R8 D1 report + selection (decisions/paperB_r8_d1.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_p9v2 as V  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R8 / "d1"
ERM = {"effb3": 0.558, "dinov2_vits14": 0.565}
FLOOR = {b: v - 0.05 for b, v in ERM.items()}
TIE_V1 = 0.01


def family(row):
    if row["penalty"] != "none":
        return "align"  # MK-MMD / CORAL
    return "grl"


def is_pass(row):
    return (float(row["v1"]) <= 0.60
            and float(row["id_balanced_acc"]) >= FLOOR[row["backbone"]]
            and not bool(row["degraded"]))


def select_backbone(df, backbone):
    sub = df[(df.backbone == backbone) & (df.pass_flag)].copy()
    if not len(sub):
        return None
    grl = sub[sub.family == "grl"]
    pool = grl if len(grl) else sub
    # lowest V1; ties within 0.01 → smaller λ or weight, then smaller k
    vmin = float(pool.v1.min())
    tied = pool[pool.v1 <= vmin + TIE_V1].copy()

    def sort_key(r):
        lam_or_w = float(r["penalty_weight"]) if r["penalty"] != "none" else float(r["lambda_adv"])
        k = 0
        cfg = str(r["config"])
        if "_k" in cfg:
            k = int(cfg.rsplit("_k", 1)[1])
        return (float(r["v1"]), lam_or_w, k)

    best = min((tied.iloc[i] for i in range(len(tied))), key=sort_key)
    return {
        "backbone": backbone,
        "tag": best.tag,
        "config": best.config,
        "lambda_adv": float(best.lambda_adv),
        "penalty": best.penalty,
        "penalty_weight": float(best.penalty_weight),
        "v1": float(best.v1),
        "id_balanced_acc": float(best.id_balanced_acc),
        "family": best.family,
        "n_passing": int(len(sub)),
        "n_passing_grl": int(len(grl)),
        "note": ("D2 tests single-encoder alignment, not DANN"
                 if best.family == "align" else "standard DANN (GRL) selected"),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted((OUT / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if "v1" not in s and (f.parent / "features" / "isic_test.npz").exists():
            p = V.heldout_probe(f.parent)
            s["v1"] = p["V1"]["bal_acc_mean"]
            s["v1_sd"] = p["V1"]["bal_acc_sd"]
            s["degraded"] = bool(s["id_balanced_acc"] < 0.25)
            f.write_text(json.dumps(s, indent=2) + "\n")
        row = {
            "tag": s["tag"], "backbone": s["backbone"], "config": s.get("config"),
            "lambda_adv": float(s["lambda_adv"]), "penalty": s.get("penalty", "none"),
            "penalty_weight": float(s.get("penalty_weight") or 0),
            "v1": float(s.get("v1")) if s.get("v1") is not None else float("nan"),
            "v1_sd": float(s.get("v1_sd") or 0),
            "id_val": float(s.get("val_balanced_acc") or float("nan")),
            "id_balanced_acc": float(s["id_balanced_acc"]),
            "degraded": bool(s.get("degraded", s["id_balanced_acc"] < 0.25)),
            "gpu_hours": float(s.get("gpu_hours") or 0),
            "dir": str(f.parent),
        }
        row["family"] = family(row)
        row["pass_flag"] = bool(is_pass(row)) if row["v1"] == row["v1"] else False
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "summary.csv", index=False)
    gpu = float(df.gpu_hours.sum()) if len(df) else 0.0
    sels = {}
    for bb in ("effb3", "dinov2_vits14"):
        sel = select_backbone(df, bb) if len(df) else None
        path = OUT / "SELECTED_{}.json".format(bb)
        if sel:
            path.write_text(json.dumps(sel, indent=2) + "\n")
            sels[bb] = sel
        else:
            path.write_text(json.dumps({"backbone": bb, "status": "no_pass",
                                        "message": "W1 not addressed by R8 for {}".format(bb)}, indent=2) + "\n")
            sels[bb] = None

    L = ["# Paper B R8 D1: single-encoder invariance screen", "",
         "Design: `decisions/paperB_r8_d1.md`. Commit `{}`.".format(R.git_head()), "",
         "GPU hours used: {:.2f} / 30.".format(gpu),
         "ERM floors: EffB3 ≥ {:.3f}; DINOv2 ≥ {:.3f}.".format(FLOOR["effb3"], FLOOR["dinov2_vits14"]), "",
         "| backbone | config | λ/w | family | V1 ± sd | ID val | ID test | degraded | pass |",
         "|---|---|---|---|---|---:|---:|---|---|"]
    for _, r in df.sort_values(["backbone", "family", "lambda_adv", "penalty_weight"]).iterrows() if len(df) else []:
        lw = r.penalty_weight if r.penalty != "none" else r.lambda_adv
        L.append("| {} | {} | {:g} | {} | {:.3f} ± {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            r.backbone, r.config, lw, r.family, r.v1, r.v1_sd, r.id_val, r.id_balanced_acc,
            r.degraded, r.pass_flag))
    L += ["", "## Selection", ""]
    for bb, sel in sels.items():
        if sel is None:
            L.append("- **{}**: no run passed. W1 not addressed by R8 for {}.".format(bb, bb))
        else:
            L.append("- **{}**: selected `{}` (family={}, V1={:.3f}). {}.".format(
                bb, sel["tag"], sel["family"], sel["v1"], sel["note"]))
    if not len(df):
        L.append("No D1 runs finished yet.")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "n=", len(df), "gpu_h=", round(gpu, 2))


if __name__ == "__main__":
    main()
