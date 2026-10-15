#!/usr/bin/env python3
"""Paper B R5 P9 report: single-encoder invariance toolbox ladder (ResNet-50, EfficientNet-B3) and the conditional
stage 2 (selected rung × λ ∈ {0, 0.5, 2} × 5 seeds). CPU only: reads summary.json / scores.npz.

Writes results/paperB/r5/p9/{summary.csv, SELECTED_<backbone>[_recp8].json, BOOTSTRAP_stage2.json, REPORT.md}
(non-current recipes: report files under p9/recipe_<suffix>/).
Selection (per backbone, written before any run): the first ladder rung, in ladder order, whose Table 1 leakage probe
(ISIC test vs PAD full) is ≤ 0.60. None reached -> no SELECTED file (stage 2 and Q3 use 'baseline').
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_recipe  # noqa: E402

import pandas as pd  # noqa: E402

P9 = R.R5 / "p9"
LADDER = ("baseline", "T1_ganin_warmup", "T2_bank_reinit", "T3_strong", "T4_long")
STOP = 0.60
DUAL_REF = "dual-encoder lesion-level λ ∈ {0.5, 1, 2}: leakage 0.555-0.66 (R2 item 4 / Table 1)"


def row(s):
    g = s.get("gate", {})
    best = g.get("best_epoch") or {}
    last = g.get("last_epoch") or {}
    return {"tag": s["tag"], "backbone": s["backbone"], "config": s["config"], "lambda_adv": s["lambda_adv"],
            "seed": s["seed"], "max_epochs": s["max_epochs"], **R.table1_values(s),
            "leakage_pca16": s["leakage"].get("test_vs_pad_full_pca16", {}).get("bal_acc_mean"),
            "leakage_selection": s["leakage"]["val_vs_pad_adv_selection"]["bal_acc_mean"],
            "z_dim": s["leakage"].get("z_dim"), "gate": g.get("status"), "adv_ce_best": best.get("loss_adv"),
            "adv_ce_last": last.get("loss_adv"), "inner_ce_last": last.get("inner_ce"), "gpu_hours": s.get("gpu_hours")}


def load(stage, suffix):
    out = []
    for f in sorted((P9 / stage).glob("*/summary.json")):
        s = json.loads(f.read_text())
        if (suffix and not s["tag"].endswith(suffix)) or (not suffix and "_rec" in s["tag"]):
            continue
        s["_dir"] = str(f.parent)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="current")
    a = ap.parse_args()
    suffix = r5_recipe.tag(a.recipe)
    lad, st2 = load("ladder", suffix), load("stage2", suffix)
    skipped = [json.loads(f.read_text()) for f in sorted((P9 / "ladder").glob("*/skipped.json"))]
    if not lad and not st2:
        print("no P9 runs yet")
        return
    out = P9 if not suffix else P9 / ("recipe" + suffix)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row(s) for s in lad + st2])
    df.to_csv(out / "summary.csv", index=False)

    L = ["# Paper B R5 P9: single-encoder invariance toolbox", "",
         "Recipe `{}`. Target: Table 1 leakage probe ≤ {:.2f} ({}). Ladder at λ = 2, seed 42; cap 20 GPU-h over both "
         "backbones. Leakage is the probe on the full scoring space (z_dim shown); the PCA-16 column is the same probe "
         "on the 16 leading ISIC-train components (dimension-matched to the dual encoder's z_lesion_norm).".format(
             a.recipe, STOP, DUAL_REF), ""]
    used = sum(float(s.get("gpu_hours") or 0) for s in lad)
    L += ["Ladder GPU-h used: {:.2f} of 20.".format(used), ""]
    sel = {}
    for bb in ("resnet50", "effb3"):
        rows = sorted([s for s in lad if s["backbone"] == bb], key=lambda s: LADDER.index(s["config"]))
        L += ["## Ladder: {}".format(bb), "",
              "| rung | leakage | PCA-16 | selection probe | ID bal acc | OOD AUROC pad_heldout | Fitz AUROC | adversary "
              "gate | best adv CE | last adv CE | last inner CE | GPU-h | flag |", "|---|" + "---:|" * 11 + "---|"]
        base = next((s["id_balanced_acc"] for s in rows if s["config"] == "baseline"), None)
        for s in rows:
            r = row(s)
            flag = "classifier degraded" if base is not None and r["id_bal_acc"] < base - 0.05 else ""
            L.append("| {} | {:.3f} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {} | {:.2f} | {} |".format(
                r["config"], r["leakage"], "—" if r["leakage_pca16"] is None else "{:.3f}".format(r["leakage_pca16"]),
                r["leakage_selection"], r["id_bal_acc"], r["ood_auroc"], r["fitz_auroc"], r["gate"],
                *["—" if r[k] is None else "{:.3f}".format(r[k]) for k in ("adv_ce_best", "adv_ce_last", "inner_ce_last")],
                r["gpu_hours"] or 0.0, flag))
        sk = [k for k in skipped if "_{}_".format(bb) in k["tag"]]
        for k in sk:
            L.append("")
            L.append("Skipped `{}`: reached by {}, ladder GPU-h used {:.2f}, this run's estimate {:.2f}.".format(
                k["tag"], k["reached_by"] or "none", k["ladder_gpu_h_used"], k["this_run_est"]))
        hit = next((s for s in rows if s["leakage"]["test_vs_pad_full"]["bal_acc_mean"] <= STOP), None)
        L.append("")
        if hit is not None:
            sel[bb] = {"config": hit["config"], "tag": hit["tag"],
                       "leakage": hit["leakage"]["test_vs_pad_full"]["bal_acc_mean"],
                       "id_bal_acc": hit["id_balanced_acc"], "rule": "first rung with leakage <= {}".format(STOP)}
            (P9 / "SELECTED_{}{}.json".format(bb, suffix)).write_text(json.dumps(sel[bb], indent=2) + "\n")
            L.append("{}: rung `{}` reached leakage {:.3f} (ID balanced accuracy {:.3f}); stage 2 can use it.".format(
                bb, hit["config"], sel[bb]["leakage"], sel[bb]["id_bal_acc"]))
        else:
            best = min(rows, key=lambda s: s["leakage"]["test_vs_pad_full"]["bal_acc_mean"]) if rows else None
            L.append("{}: no rung reached {:.2f}{}.".format(bb, STOP, "" if best is None else
                     "; lowest leakage {:.3f} at `{}`".format(best["leakage"]["test_vs_pad_full"]["bal_acc_mean"],
                                                           best["config"])))
        L.append("")

    if st2:
        cells = defaultdict(dict)
        for s in st2:
            cells["{}|{}|lam{:g}".format(s["backbone"], s["config"], s["lambda_adv"])][s["seed"]] = s["_dir"]
        B = R.isic_bootstrap(cells)
        (out / "BOOTSTRAP_stage2.json").write_text(json.dumps(B, indent=2) + "\n")
        L += ["## Stage 2 (Table 1 columns, mean ± SD over seeds)", "",
              "| cell | n | " + " | ".join(R.TABLE1_HEAD) + " | pad_heldout rule | Fitz rule |",
              "|---|---:|" + "---:|" * len(R.TABLE1_HEAD) + "---|---|"]
        tex = []
        for k in sorted(cells):
            g = df[df.tag.isin([Path(d).name for d in cells[k].values()])]
            b = B.get(k, {})
            ru = [b.get(o, {}).get("mahalanobis_classcond", {}).get("rule", "—") for o in ("pad_heldout", "fitzpatrick17k")]
            L.append("| {} | {} | {} | {} | {} |".format(k, len(g), " | ".join(R.ms(g[c]) for c in R.TABLE1_COLS), *ru))
            tex.append("{} & {} \\\\".format(k.replace("|", " ").replace("_", "\\_"),
                                             " & ".join(R.tex_ms(g[c]) for c in R.TABLE1_COLS)))
        L += ["", "```latex"] + tex + ["```", ""]
        L += ["## Stage 2 in plain words", ""]
        for k in sorted(cells):
            g = df[df.tag.isin([Path(d).name for d in cells[k].values()])]
            b = B.get(k, {}).get("pad_heldout", {}).get("mahalanobis_classcond")
            L.append("{} ({} seeds): leakage {}, ID balanced accuracy {}, pad_heldout Mahalanobis AUROC {}{}.".format(
                k, len(g), R.ms(g.leakage), R.ms(g.id_bal_acc), R.ms(g.ood_auroc),
                "" if b is None else " (pooled CI {:.3f}-{:.3f}, {}/{} seeds below 0.5, rule {})".format(
                    *b["pooled_ci"], b["k_below"], b["n_seeds"], b["rule"])))
            L.append("")
    L += ["## Ladder in plain words", ""]
    for bb in ("resnet50", "effb3"):
        rows = sorted([s for s in lad if s["backbone"] == bb], key=lambda s: LADDER.index(s["config"]))
        if rows:
            L.append("{}: leakage by rung {}; ID balanced accuracy by rung {}.".format(
                bb, ", ".join("{} {:.3f}".format(s["config"], s["leakage"]["test_vs_pad_full"]["bal_acc_mean"]) for s in rows),
                ", ".join("{:.3f}".format(s["id_balanced_acc"]) for s in rows)))
            L.append("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
