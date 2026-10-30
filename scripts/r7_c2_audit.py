#!/usr/bin/env python3
"""Paper B R7 C2: which run set each figure and clinical number uses (decisions/paperB_r7_c2.md).

Audits scripts/plot_manuscript_figures.py (FIGURES_REPORT.md is stale). No .tex edits.
If Fig 2 still uses phase13, regenerate MSP confidence from t1last last.ckpt and report old vs new.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = R.R7 / "c2"
FEAT = R.R5 / "t1last" / "features_last"
SUM = R.R5 / "t1last" / "summary.csv"
P13 = R.PAPERB / "results" / "paperB" / "phase13" / "per_run"
CLIN = R.R5 / "posthoc" / "clinical_t1last" / "results.json"
PLOT = R.PAPERB / "scripts" / "plot_manuscript_figures.py"


def msp_mean(pack):
    lg = torch.from_numpy(np.asarray(pack["logits"])).float()
    return float(torch.softmax(lg, 1).max(1).values.mean())


def t1last_confidence_by_lam():
    """Mean MSP confidence on isic_test / pad_heldout / fitz from t1last features_last."""
    by = defaultdict(lambda: defaultdict(list))
    with open(SUM, newline="") as f:
        runs = [r for r in csv.DictReader(f) if r["checkpoint"] == "last"]
    for r in runs:
        tag, lam = r["run"], float(r["lambda"])
        for name in ("isic_test", "pad_heldout", "fitzpatrick17k"):
            p = FEAT / tag / "{}.npz".format(name)
            if not p.exists():
                continue
            pack = dict(np.load(p, allow_pickle=True))
            by[lam][name].append(msp_mean(pack))
    out = {}
    for lam, d in sorted(by.items()):
        out[lam] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
                    for k, v in d.items()}
    return out


def phase13_confidence_by_lam():
    by = defaultdict(lambda: defaultdict(list))
    for p in sorted(P13.glob("*.json")):
        r = json.loads(p.read_text())
        lam = float(r["lambda_adv"])
        for k in ("isic_test", "pad_heldout", "fitzpatrick17k"):
            if k in r.get("confidence", {}):
                by[lam][k].append(float(r["confidence"][k]["msp_mean"]))
    out = {}
    for lam, d in sorted(by.items()):
        out[lam] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
                    for k, v in d.items()}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = PLOT.read_text()
    inventory = [
        {"panel": "Fig 1a–d", "script_fn": "fig1_intervention", "run_set": "Table 1 last.ckpt (r5/t1last/summary.csv)",
         "checkpoint": "last.ckpt", "note": "load_t1last_by_lambda; OOD ECE from UNSEEN_ECE.json"},
        {"panel": "Fig 2", "script_fn": "fig2_confidence", "run_set": "phase13/per_run (STALE vs Table 1)",
         "checkpoint": "phase13 (best-era)", "note": "still load_phase13_runs; regenerate from t1last below"},
        {"panel": "Fig 3", "script_fn": "fig3_detectors", "run_set": "phase3_sweep summaries",
         "checkpoint": "phase3", "note": "load_phase3_summaries"},
        {"panel": "Fig 4–6 / S*", "script_fn": "(various)", "run_set": "see plot_manuscript_figures.py",
         "checkpoint": "mixed / legacy", "note": "FIGURES_REPORT.md header is stale; trust the script"},
        {"panel": "§4.4 / App clinical %", "script_fn": "clinical_t1last", "run_set": "Table 1 last.ckpt (clinical_t1last)",
         "checkpoint": "last.ckpt", "note": str(CLIN) if CLIN.exists() else "missing clinical_t1last"},
    ]
    old = phase13_confidence_by_lam()
    new = t1last_confidence_by_lam()
    cmp = {}
    for lam in sorted(set(old) | set(new)):
        cmp[str(lam)] = {}
        for k in ("isic_test", "pad_heldout", "fitzpatrick17k"):
            o = old.get(lam, {}).get(k)
            n = new.get(lam, {}).get(k)
            if o and n:
                cmp[str(lam)][k] = {"old_phase13": o, "new_t1last": n, "delta_mean": n["mean"] - o["mean"]}
            else:
                cmp[str(lam)][k] = {"old_phase13": o, "new_t1last": n}
    (OUT / "fig2_old_vs_new.json").write_text(json.dumps(cmp, indent=2) + "\n")
    (OUT / "t1last_confidence.json").write_text(json.dumps({str(k): v for k, v in new.items()}, indent=2) + "\n")
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    L = ["# Paper B R7 C2: figure and clinical run-set audit", "",
         "Design: `decisions/paperB_r7_c2.md`. Commit `{}`.".format(R.git_head()), "",
         "`FIGURES_REPORT.md` is stale. Audit is against `scripts/plot_manuscript_figures.py`.", "",
         "| Panel | Script | Run set | Checkpoint | Note |",
         "|---|---|---|---|---|"]
    for r in inventory:
        L.append("| {} | `{}` | {} | {} | {} |".format(
            r["panel"], r["script_fn"], r["run_set"], r["checkpoint"], r["note"]))
    L += ["", "## Fig 2: phase13 (old) vs t1last last.ckpt (new)", "",
          "Fig 2 still calls `load_phase13_runs()`. Regenerated MSP confidence from "
          "`results/paperB/r5/t1last/features_last` (Table 1 last.ckpt).", "",
          "| λ | set | old mean | new mean | Δ |",
          "|---:|---|---:|---:|---:|"]
    for lam, blk in sorted(cmp.items(), key=lambda x: float(x[0])):
        for k, v in blk.items():
            if v.get("old_phase13") and v.get("new_t1last"):
                L.append("| {} | {} | {:.3f} | {:.3f} | {:+.3f} |".format(
                    lam, k, v["old_phase13"]["mean"], v["new_t1last"]["mean"], v["delta_mean"]))
    L += ["", "Clinical percentages in §4.4 / App. H: `clinical_t1last` (Table 1 last.ckpt), already aligned.",
          "No `.tex` edits in this item.", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
