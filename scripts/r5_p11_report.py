#!/usr/bin/env python3
"""Paper B R5 P11 report: MK-MMD and scaled CORAL on z_lesion_norm (dual encoder, λ_adv = 0). CPU only.

--select   seed-42 scan → SELECTION.json (val vs pad_adv → 0.60, classifier intact; validation-side only)
default    summary.csv, BOOTSTRAP.json, REPORT.md with the comparable-invariance outcome
           (decisions/paperB_r5_p11.md amendment 2026-10-16): held-out V1 (P6/A3) within 0.03 of GRL λ = 2 and
           ID bal acc within 0.05 of GRL λ = 0; else “equivalent invariance not reached”.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_p9v2 as V  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

P11 = R.R5 / "p11"
P6 = R.R5 / "p6" / "per_run"
ITEM4 = R.PAPERB / "results" / "paperB" / "r2" / "item4" / "runs"
TARGET, INTACT, TIE = 0.60, 0.95, 0.01
V1_TOL, ID_TOL = 0.03, 0.05
PENALTIES = ("mkmmd", "coral")
DETS = R.DETS


def keep(tag, suffix):
    return tag.endswith(suffix) if suffix else "_rec" not in tag


def load(suffix):
    out = []
    for f in sorted((P11 / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        if keep(s["tag"], suffix):
            s["_dir"] = str(f.parent)
            log = f.parent / "adversary_log.jsonl"
            pen = [json.loads(x).get("penalty") for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
            pen = [p for p in pen if p is not None]
            s["_pen_flag"] = ("penalty not optimised" if s["penalty"] != "none" and len(pen) >= 2 and pen[-1] >= pen[0]
                              else "")
            out.append(s)
    return out


def heldout_v1(run_dir):
    """P6 V1 on z_lesion_norm (ISIC test ∪ pad_heldout). Cached beside the run."""
    run_dir = Path(run_dir)
    cache = run_dir / "heldout_probe_V1.json"
    if cache.exists():
        return float(json.loads(cache.read_text())["V1"])
    G = V.groups()
    te = np.load(run_dir / "features" / "isic_test.npz")
    ho = np.load(run_dir / "features" / "pad_heldout.npz")
    zkey = "z_lesion_norm" if "z_lesion_norm" in te else "z"
    assert np.array_equal(te["labels"].astype(int), G["yi"]), "isic_test row order"
    val = float(V.v1_probe(te[zkey], ho[zkey], G["gi"], G["gp"])["bal_acc_mean"])
    cache.write_text(json.dumps({"V1": val, "protocol": "P6 V1 / A3"}, indent=1) + "\n")
    return val


def grl_refs():
    v1 = [json.loads(p.read_text())["z_lesion_norm"]["V1"]["bal_acc_mean"]
          for p in sorted(P6.glob("runB_orth1_ladv2_s*.json"))]
    id0 = [json.loads(p.read_text())["id_balanced_acc"]
           for p in sorted(ITEM4.glob("runB_orth1_ladv0_s*/summary.json"))]
    return {"V1_lambda2": float(np.mean(v1)), "V1_lambda2_per_seed": v1, "n_V1": len(v1),
            "ID_lambda0": float(np.mean(id0)), "ID_lambda0_per_seed": id0, "n_ID": len(id0)}


def flat(s):
    m = re.search(r"_ep(\d+)", s["tag"])
    v1 = heldout_v1(s["_dir"])
    return {"tag": s["tag"], "penalty": s["penalty"], "weight": float(s["penalty_weight"]), "epochs": int(s["max_epochs"]),
            "epochs_flag": int(m.group(1)) if m else 0,
            "seed": int(s["seed"]), "lambda_adv": float(s["lambda_adv"]), **R.table1_values(s),
            "val_bal_acc": s["val_balanced_acc"],
            "selection_probe": s["leakage"]["val_vs_pad_adv_selection"]["bal_acc_mean"],
            "heldout_V1": v1,
            "knn_pad_heldout": s["ood"]["pad_heldout"]["knn_k50"],
            "nv_nearest_pad_heldout": s["ood"]["pad_heldout"]["majority_nearest_frac"],
            "reweighted_pad_heldout": (s["ood"]["pad_heldout"].get("decomposition") or {}).get("reweighted_to_id_mix"),
            "flag": s.get("_pen_flag", ""), "gpu_hours": s.get("gpu_hours"),
            **{"{}_{}".format(o, d): s["ood"][o][d] for o in ("pad_heldout", "fitzpatrick17k") for d in DETS}}


def select(df):
    ctl = df[(df.penalty == "none") & (df.seed == 42)]
    if ctl.empty:
        raise SystemExit("P11 control run missing; selection not made")
    floor = INTACT * float(ctl.val_bal_acc.iloc[0])
    sel = {"rule": "closest selection probe to {} among val bal acc >= {} x control; ties < {} -> smaller weight, "
                   "then 40 epochs".format(TARGET, INTACT, TIE), "control_val_bal_acc": float(ctl.val_bal_acc.iloc[0]),
           "intact_floor": floor}
    for pen in PENALTIES:
        g = df[(df.penalty == pen) & (df.seed == 42)].copy()
        if g.empty:
            sel[pen] = None
            continue
        g["dist"] = (g.selection_probe - TARGET).abs()
        ok = g[g.val_bal_acc >= floor]
        flag = None
        if ok.empty:
            ok = g[g.selection_probe <= 0.75].sort_values("val_bal_acc", ascending=False).head(1)
            flag = "no config kept the classifier intact"
        else:
            best = ok.dist.min()
            ok = ok[ok.dist < best + TIE].sort_values(["weight", "epochs"])
        if ok.empty:
            sel[pen] = {"selected": None, "flag": "no config reached selection probe <= 0.75"}
            continue
        r = ok.iloc[0]
        sel[pen] = {"weight": float(r.weight), "epochs": int(r.epochs), "epochs_flag": int(r.epochs_flag),
                    "selection_probe": float(r.selection_probe),
                    "val_bal_acc": float(r.val_bal_acc), "flag": flag,
                    "scan": g[["weight", "epochs", "selection_probe", "val_bal_acc"]].to_dict("records")}
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="current")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    suffix = "" if a.recipe == "current" else ("_recp8" if a.recipe == "p8" else "_rec" + Path(a.recipe).stem)
    out = P11 if not suffix else P11 / ("recipe" + suffix)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(suffix)
    if not runs:
        print("no P11 runs yet")
        return
    df = pd.DataFrame([flat(s) for s in runs])
    if a.select:
        sel = select(df)
        (out / "SELECTION.json").write_text(json.dumps(sel, indent=2) + "\n")
        print(json.dumps({k: v for k, v in sel.items() if k in PENALTIES}, indent=1))
        return
    df.to_csv(out / "summary.csv", index=False)
    sel = json.loads((out / "SELECTION.json").read_text()) if (out / "SELECTION.json").exists() else {}
    refs = grl_refs()
    cells = defaultdict(dict)
    for s in runs:
        if s["penalty"] != "none":
            cells["{}|w{:g}|ep{}".format(s["penalty"], float(s["penalty_weight"]), s["max_epochs"])][int(s["seed"])] = s["_dir"]
    multi = {k: v for k, v in cells.items() if len(v) >= 3}
    B = R.isic_bootstrap(multi, dets=DETS) if multi else {}
    (out / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")

    # Comparable invariance (amendment 2026-10-16): any MK-MMD/CORAL setting, prefer 5-seed cells then seed-42 scan.
    comparable = []
    for k, seeds in sorted(cells.items()):
        g = df[df.tag.isin([Path(d).name for d in seeds.values()])]
        v1, idb = float(g.heldout_V1.mean()), float(g.id_bal_acc.mean())
        ok = abs(v1 - refs["V1_lambda2"]) <= V1_TOL and abs(idb - refs["ID_lambda0"]) <= ID_TOL
        comparable.append({"arm": k, "n": len(g), "heldout_V1": v1, "id_bal_acc": idb,
                           "d_V1": v1 - refs["V1_lambda2"], "d_ID": idb - refs["ID_lambda0"], "meets": ok})
    any_ok = any(c["meets"] for c in comparable)
    outcome = {"refs": refs, "V1_tol": V1_TOL, "ID_tol": ID_TOL, "settings": comparable,
               "outcome": ("see per-setting collapse/inversion below" if any_ok else "equivalent invariance not reached")}
    (out / "COMPARABLE_INVARIANCE.json").write_text(json.dumps(outcome, indent=2) + "\n")

    L = ["# Paper B R5 P11: MK-MMD (DAN) and scaled CORAL — comparable invariance (held-out V1)", "",
         "Recipe `{}`. Dual encoder EfficientNet-B3, λ_adv = 0, last.ckpt primary. MK-MMD as Long et al. 2015 / "
         "Wu et al. 2023 (arXiv:2309.11705; DAN on rotated MNIST: Maha 32.9, ODIN 30.8). "
         "Comparable invariance: |held-out V1 − GRL λ=2 V1| ≤ {:.2f} and |ID − GRL λ=0 ID| ≤ {:.2f} "
         "(decisions/paperB_r5_p11.md amendment 2026-10-16).".format(a.recipe, V1_TOL, ID_TOL), "",
         "GRL references: V1(λ=2) = {:.3f} (n={}); ID(λ=0) = {:.3f} (n={}).".format(
             refs["V1_lambda2"], refs["n_V1"], refs["ID_lambda0"], refs["n_ID"]), "",
         "## Scan (seed 42) and selection", "",
         "| penalty | weight | epochs | selection probe | val bal acc | held-out V1 | ID bal acc | "
         "pad_heldout Maha | comparable? | flag |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for _, r in df[df.seed == 42].sort_values(["penalty", "epochs", "weight"]).iterrows():
        meet = abs(r.heldout_V1 - refs["V1_lambda2"]) <= V1_TOL and abs(r.id_bal_acc - refs["ID_lambda0"]) <= ID_TOL
        L.append("| {} | {:g} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} |".format(
            r.penalty, r.weight, r.epochs, r.selection_probe, r.val_bal_acc, r.heldout_V1, r.id_bal_acc, r.ood_auroc,
            "yes" if meet else "no", r.flag))
    for pen in PENALTIES:
        v = sel.get(pen)
        if v:
            L.append("")
            L.append("Selected {}: {}.".format(pen, "none ({})".format(v.get("flag")) if v.get("weight") is None else
                                              "weight {:g}, {} epochs (selection probe {:.3f}, val bal acc {:.3f}){}".format(
                                                  v["weight"], v["epochs"], v["selection_probe"], v["val_bal_acc"],
                                                  "; " + v["flag"] if v.get("flag") else "")))

    L += ["", "## Comparable invariance (all MK-MMD / CORAL settings)", "",
          "| arm | n | held-out V1 | ΔV1 vs GRL λ2 | ID bal acc | ΔID vs GRL λ0 | meets |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for c in comparable:
        L.append("| {} | {} | {:.3f} | {:+.3f} | {:.3f} | {:+.3f} | {} |".format(
            c["arm"], c["n"], c["heldout_V1"], c["d_V1"], c["id_bal_acc"], c["d_ID"], "yes" if c["meets"] else "no"))
    L += ["", "**Outcome: {}.**".format(outcome["outcome"]), ""]

    tex, para = [], []
    if not any_ok:
        para.append("Equivalent invariance not reached: no MK-MMD or CORAL setting has held-out V1 within {:.2f} of "
                    "GRL λ = 2 ({:.3f}) and ID bal acc within {:.2f} of GRL λ = 0 ({:.3f}). No claim is made that "
                    "“MMD does not collapse”.".format(V1_TOL, refs["V1_lambda2"], ID_TOL, refs["ID_lambda0"]))
    else:
        L += ["## Settings that meet comparable invariance (5 detectors, rule A)", "",
              "| arm | n | " + " | ".join(R.DET_LABEL[d] for d in DETS) + " | Maha rule A pad_heldout | "
              "Maha rule A Fitz | collapse | inversion |",
              "|---|---:|" + "---:|" * len(DETS) + "---|---|---|---|"]
        for c in comparable:
            if not c["meets"]:
                continue
            k = c["arm"]
            g = df[df.tag.isin([Path(d).name for d in cells[k].values()])]
            b = B.get(k, {}) if k in multi else {}
            pts = [R.ms(g["pad_heldout_{}".format(d)]) for d in DETS]
            ru_h = b.get("pad_heldout", {}).get("mahalanobis_classcond", {}).get("rule", "—")
            ru_f = b.get("fitzpatrick17k", {}).get("mahalanobis_classcond", {}).get("rule", "—")
            maha = float(g.ood_auroc.mean())
            collapse = 0.45 <= maha <= 0.55
            invert = ru_h == "A" if ru_h != "—" else maha < 0.5
            L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                k, c["n"], " | ".join(pts), ru_h, ru_f, "yes" if collapse else "no", "yes" if invert else "no"))
            # Fitzpatrick row
            L.append("| {} (Fitz) | {} | {} | — | {} | — | — |".format(
                k, c["n"], " | ".join(R.ms(g["fitzpatrick17k_{}".format(d)]) for d in DETS), ru_f))
            tex.append("{} & {} & {} \\\\".format(k.replace("|", " "), c["n"], " & ".join(
                R.tex_ms(g["pad_heldout_{}".format(d)]) for d in DETS)))
            bp = b.get("pad_heldout", {}).get("mahalanobis_classcond")
            para.append(
                "{} (n={}): meets comparable invariance (V1={:.3f}, ID={:.3f}). Pad_heldout Mahalanobis {}; "
                "rule A {}; collapse {}; inversion {}. Fitzpatrick Mahalanobis {}.".format(
                    k, c["n"], c["heldout_V1"], c["id_bal_acc"], R.ms(g.ood_auroc),
                    ru_h if bp is None else "{} (pooled CI {:.3f}-{:.3f}, {}/{})".format(
                        ru_h, *bp["pooled_ci"], bp["k_below"], bp["n_seeds"]),
                    "yes" if collapse else "no", "yes" if invert else "no", R.ms(g.fitz_auroc)))

    L += ["", "```latex"] + tex + ["```", "", "## Results in plain words", ""]
    for p in para:
        L += [p, ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md", "outcome=", outcome["outcome"])


if __name__ == "__main__":
    main()
