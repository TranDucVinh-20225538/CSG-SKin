#!/usr/bin/env python3
"""R10b B1: bootstrap CIs for R1 shift 50% cells (λ ∈ {0.5, 2}, all R1 OOD sets incl. SCIN).

Keeps other cells' CIs from the last full run if present; only recomputes shift 50%.
Writes results/paperB/r10b/B1_shift50_boots.json and patches r6/r1/cur/REPORT.md sections.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_report as RR  # noqa: E402

OUT = R.RES / "r10b"
DET = "mahalanobis_classcond"
OODS = list(RR.R1_OODS)
SHIFTS = (50,)
LAMS = (0.5, 2.0)


def latex_row(sh, lam, ood, b):
    x = b[DET]
    ra = "met" if b["rule_A"] else "not met"
    return (
        f"{sh}\\% & {lam:g} & {RR.R1_OOD_LAB[ood]} & "
        f"${x['mean']:.3f}\\pm{x['sd']:.3f}$ & $[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]$ & "
        f"{x['k_below']}/{x['n_seeds']} & {ra} \\\\"
    )


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    root = RR.item_dir("r1", "cur")
    fmt = "r1_cur_shift{sh}_ladv{{lam}}_s{{s}}"
    src = []
    seed_lists = {}
    for lam in LAMS:
        for sh in SHIFTS:
            pairs = RR.own_runs(root, fmt.format(sh=sh), lam)
            lab = RR.SHIFT_LABEL[sh]
            src.append((lab, lam, pairs))
            seed_lists[f"{lab}|{lam:g}"] = sorted(pairs.keys())
            print(f"cell {lab} λ={lam:g} seeds={sorted(pairs.keys())} n={len(pairs)}", flush=True)
            if set(pairs.keys()) != set(R.SEEDS):
                print("WARNING seed list != {42,52,62,72,82}", flush=True)

    data = R.load_data()
    idg = R.idx_groups(R.T.lesion_groups(data["test"])[0])
    cells = RR.metrics(src)
    excl = RR.degraded_by_cell(cells)
    print("bootstrapping…", flush=True)
    boots = RR.boots_for(src, OODS, data, idg, exclude_by_cell=excl)
    rows = []
    latex = []
    for lab, lam, _ in src:
        for ood in OODS:
            b = boots.get((lab, lam, ood))
            if b is None or DET not in b:
                continue
            x = b[DET]
            rows.append({
                "shift": 50, "lambda": lam, "set": ood, "label": lab,
                "mean": x["mean"], "sd": x["sd"], "ci": [x["ci_lo"], x["ci_hi"]],
                "k": f"{x['k_below']}/{x['n_seeds']}", "rule_A": b["rule_A"],
                "unit_ood": b.get("unit_ood"), "excluded": b.get("excluded_seeds"),
                "latex": latex_row(50, lam, ood, b),
            })
            latex.append(rows[-1]["latex"])
            print(f"  {lab} λ={lam:g} {ood}: {x['mean']:.3f}±{x['sd']:.3f} "
                  f"[{x['ci_lo']:.3f},{x['ci_hi']:.3f}] k={x['k_below']}/{x['n_seeds']} "
                  f"RuleA={'met' if b['rule_A'] else 'not met'}", flush=True)

    # E1 does not use shift 50% refs (only 0% / 100%); record that.
    e1_note = ("R9 E1 references are shift 0% and shift 100% only; no shift-50% boots required. "
               "E1 --no_boot from R10 left unchanged.")
    payload = {
        "commit": R.C.git_head(), "seed_lists": seed_lists, "rows": rows, "latex": latex,
        "e1": e1_note, "cpu_hours": (time.time() - t0) / 3600.0,
        "n_boot": 2000, "rng": 20261005,
    }
    (OUT / "B1_shift50_boots.json").write_text(json.dumps(payload, indent=1) + "\n")
    md = ["# R10b B1 — R1 shift 50% bootstrap", "",
          f"Commit `{payload['commit']}`. PREREGISTER 2000 draws, rng 20261005.",
          "", e1_note, "",
          "| shift | λ | set | mean±sd | pooled CI | k | Rule A |",
          "|---:|---:|---|---:|---|---:|---|"]
    for r in rows:
        md.append(
            f"| 50% | {r['lambda']:g} | {r['set']} | {r['mean']:.3f}±{r['sd']:.3f} | "
            f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['k']} | "
            f"{'met' if r['rule_A'] else 'not met'} |"
        )
    md += ["", "LaTeX:", "```"] + latex + ["```", ""]
    (OUT / "B1_shift50_boots.md").write_text("\n".join(md))
    print("wrote", OUT / "B1_shift50_boots.json", "cpu_h", round(payload["cpu_hours"], 3), flush=True)


if __name__ == "__main__":
    main()
