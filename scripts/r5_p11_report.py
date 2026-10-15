#!/usr/bin/env python3
"""Paper B R5 P11 report: MK-MMD and scaled CORAL on z_lesion_norm (dual encoder, λ_adv = 0) vs GRL at matched
invariance on pad_heldout. CPU only.

--select   reads the seed-42 scan + control and writes results/paperB/r5/p11/SELECTION.json (rule below; validation-side
           numbers only, never an OOD or test number)
default    writes results/paperB/r5/p11/{summary.csv, BOOTSTRAP.json, REPORT.md}

Selection rule (fixed in decisions/paperB_r5_p11.md): per penalty, among scan configs with "classifier intact"
(ISIC val balanced accuracy ≥ 0.95 × the control's), the one whose selection probe (ISIC val vs pad_adv) is closest to
0.60; ties (|Δ| < 0.01) -> smaller weight, then 40 epochs. If no config is intact: the config with the highest val
balanced accuracy among those with selection probe ≤ 0.75, flagged; else none.
Matched GRL comparison: for each P11 cell, the R2 item-4 GRL runs (λ > 0) whose Table 1 leakage is within ±0.05 of the
cell's mean leakage.
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

import pandas as pd  # noqa: E402

P11 = R.R5 / "p11"
TARGET, INTACT, TIE, MATCH = 0.60, 0.95, 0.01, 0.05
PENALTIES = ("mkmmd", "coral")


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


def flat(s):
    m = re.search(r"_ep(\d+)", s["tag"])
    return {"tag": s["tag"], "penalty": s["penalty"], "weight": float(s["penalty_weight"]), "epochs": int(s["max_epochs"]),
            "epochs_flag": int(m.group(1)) if m else 0,
            "seed": int(s["seed"]), "lambda_adv": float(s["lambda_adv"]), **R.table1_values(s),
            "val_bal_acc": s["val_balanced_acc"],
            "selection_probe": s["leakage"]["val_vs_pad_adv_selection"]["bal_acc_mean"],
            "knn_pad_heldout": s["ood"]["pad_heldout"]["knn_k50"],
            "nv_nearest_pad_heldout": s["ood"]["pad_heldout"]["majority_nearest_frac"],
            "reweighted_pad_heldout": (s["ood"]["pad_heldout"].get("decomposition") or {}).get("reweighted_to_id_mix"),
            "flag": s.get("_pen_flag", ""), "gpu_hours": s.get("gpu_hours")}


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
    cells = defaultdict(dict)
    for s in runs:
        if s["penalty"] != "none":
            cells["{}|w{:g}|ep{}".format(s["penalty"], float(s["penalty_weight"]), s["max_epochs"])][int(s["seed"])] = s["_dir"]
    multi = {k: v for k, v in cells.items() if len(v) >= 3}
    B = R.isic_bootstrap(multi) if multi else {}
    (out / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")
    grl = R.item4_reference()
    grl_pos = grl[grl.lambda_adv > 0]

    L = ["# Paper B R5 P11: moment matching (MK-MMD, scaled CORAL) vs GRL at matched invariance", "",
         "Recipe `{}`. Dual encoder EfficientNet-B3, λ_adv = 0, penalty on z_lesion_norm between the ISIC and PAD rows "
         "of each batch. Leakage = Table 1 probe (ISIC test vs PAD full).".format(a.recipe), "",
         "## Scan (seed 42) and selection", "",
         "| penalty | weight | epochs | selection probe (val vs pad_adv) | val bal acc | leakage | ID bal acc | "
         "pad_heldout Maha | flag |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in df[df.seed == 42].sort_values(["penalty", "epochs", "weight"]).iterrows():
        L.append("| {} | {:g} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            r.penalty, r.weight, r.epochs, r.selection_probe, r.val_bal_acc, r.leakage, r.id_bal_acc, r.ood_auroc,
            r.flag))
    for pen in PENALTIES:
        v = sel.get(pen)
        if v:
            L.append("")
            L.append("Selected {}: {}.".format(pen, "none ({})".format(v.get("flag")) if v.get("weight") is None else
                                              "weight {:g}, {} epochs (selection probe {:.3f}, val bal acc {:.3f}){}".format(
                                                  v["weight"], v["epochs"], v["selection_probe"], v["val_bal_acc"],
                                                  "; " + v["flag"] if v.get("flag") else "")))
    L += ["", "## Selected configs (5 seeds) vs GRL at matched leakage", "",
          "| arm | n | " + " | ".join(R.TABLE1_HEAD) + " | pad_heldout rule | Fitz rule | GRL runs within ±0.05 "
          "leakage (n, mean λ) | GRL Maha pad_heldout | GRL Fitz |", "|---|---:|" + "---:|" * len(R.TABLE1_HEAD) +
          "---|---|---|---:|---:|"]
    tex, para = [], []
    for k in sorted(multi):
        tags = [Path(d).name for d in multi[k].values()]
        g = df[df.tag.isin(tags)]
        b = B.get(k, {})
        ru = [b.get(o, {}).get("mahalanobis_classcond", {}).get("rule", "—") for o in ("pad_heldout", "fitzpatrick17k")]
        lk = float(g.leakage.mean())
        m = grl_pos[(grl_pos.leakage - lk).abs() <= MATCH]
        L.append("| {} | {} | {} | {} | {} | n = {}, mean λ {} | {} | {} |".format(
            k, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS), *ru, len(m),
            "—" if m.empty else "{:.2f}".format(m.lambda_adv.mean()), R.ms(m.ood_auroc), R.ms(m.fitz_auroc)))
        tex.append("{} & {} \\\\".format(k.replace("|", " ").replace("_", "\\_"),
                                         " & ".join(R.tex_ms(g[c]) for c in R.TABLE1_COLS)))
        bp = b.get("pad_heldout", {}).get("mahalanobis_classcond")
        para.append("{} ({} seeds): leakage {}, ID balanced accuracy {}, pad_heldout Mahalanobis AUROC {}{}; the {} GRL "
                    "runs with leakage within ±{:.2f} of {:.3f} have pad_heldout Mahalanobis {} and Fitzpatrick {}.".format(
                        k, len(g), R.ms(g.leakage), R.ms(g.id_bal_acc), R.ms(g.ood_auroc),
                        "" if bp is None else " (pooled CI {:.3f}-{:.3f}, {}/{} seeds below 0.5, rule {})".format(
                            *bp["pooled_ci"], bp["k_below"], bp["n_seeds"], bp["rule"]),
                        len(m), MATCH, lk, R.ms(m.ood_auroc), R.ms(m.fitz_auroc)))
    L += ["", "Diagnostics (selected cells): kNN pad_heldout, NV-nearest fraction, pad_heldout reweighted to the ISIC mix.",
          "", "| arm | kNN | NV-nearest | reweighted |", "|---|---:|---:|---:|"]
    for k in sorted(multi):
        g = df[df.tag.isin([Path(d).name for d in multi[k].values()])]
        L.append("| {} | {} | {} | {} |".format(k, R.ms(g.knn_pad_heldout), R.ms(g.nv_nearest_pad_heldout),
                                               R.ms(g.reweighted_pad_heldout)))
    L += ["", "```latex"] + tex + ["```", "", "## Results in plain words", ""]
    for p in para:
        L += [p, ""]
    if not para:
        L += ["Seed stage not run yet.", ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
