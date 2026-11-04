#!/usr/bin/env python3
"""R10b B3: pooled CI + k + Rule A for A4 rows of tab:unseen (sets in C3_unseen.json)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r10b"
C3 = R.RES / "r10" / "C3_unseen.json"
DET = "mahalanobis_classcond"
FEAT = R.RES / "r5" / "p8" / "main" / "features"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    c3 = json.loads(C3.read_text())
    a4 = c3["a4"]
    data = R.load_data()
    idg = R.idx_groups(R.T.lesion_groups(data["test"])[0])
    rows = []
    latex = []
    for lam_s, block in sorted(a4.items(), key=lambda kv: float(kv[0])):
        lam = float(lam_s)
        sets = [k for k in block if isinstance(block[k], dict) and "mean" in block[k]]
        feat_dirs = {}
        for s in R.SEEDS:
            d = FEAT / "main_ladv{}_s{}".format(R.lam_tag(lam), s)
            if (d / "isic_train.npz").exists() and (d / "isic_test.npz").exists():
                # skip degraded via summary if present
                sp = R.RES / "r5" / "p8" / "main" / "runs" / d.name / "summary.json"
                if sp.exists():
                    m = json.loads(sp.read_text())["metrics"]
                    if m.get("degraded") or m["id"]["bal_acc"] < R.DEGRADED_ID_BAL:
                        print("exclude degraded", d.name, flush=True)
                        continue
                feat_dirs[s] = d
        print(f"A4 λ={lam:g} n_seeds={len(feat_dirs)} sets={sets}", flush=True)
        for ood in sets:
            if not all((feat_dirs[s] / f"{ood}.npz").exists() for s in feat_dirs):
                print("missing features for", ood, flush=True)
                continue
            og, unit = R.ood_groups(ood, data)
            b = R.bootstrap_cell(feat_dirs, ood, idg, og, dets=(DET,))
            if b is None or DET not in b:
                continue
            x = b[DET]
            old = block[ood]
            row = {
                "backbone": "a4", "lambda": lam, "set": ood,
                "old_mean": old.get("mean"), "old_sd": old.get("sd"),
                "mean": x["mean"], "sd": x["sd"], "ci": [x["ci_lo"], x["ci_hi"]],
                "k": f"{x['k_below']}/{x['n_seeds']}", "rule_A": b["rule_A"],
                "unit_ood": unit,
            }
            row["latex"] = (
                f"A4 & {lam:g} & {ood} & ${x['mean']:.3f}\\pm{x['sd']:.3f}$ & "
                f"$[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]$ & {x['k_below']}/{x['n_seeds']} & "
                f"{'met' if b['rule_A'] else 'not met'} \\\\"
            )
            rows.append(row)
            latex.append(row["latex"])
            print(f"  {ood}: {x['mean']:.3f}±{x['sd']:.3f} [{x['ci_lo']:.3f},{x['ci_hi']:.3f}] "
                  f"k={x['k_below']}/{x['n_seeds']} RuleA={b['rule_A']}", flush=True)

    payload = {"commit": R.C.git_head(), "rows": rows, "latex": latex,
               "cpu_hours": (time.time() - t0) / 3600.0}
    (OUT / "B3_a4_ci.json").write_text(json.dumps(payload, indent=1) + "\n")
    # merge into C3 store
    enriched = json.loads(C3.read_text())
    for r in rows:
        blk = enriched["a4"][str(float(r["lambda"]))][r["set"]]
        blk["ci"] = r["ci"]
        blk["k"] = r["k"]
        blk["rule_A"] = r["rule_A"]
        blk["unit_ood"] = r["unit_ood"]
    (OUT / "C3_unseen_with_ci.json").write_text(json.dumps(enriched, indent=1) + "\n")
    md = ["# R10b B3 — A4 tab:unseen CIs", "", f"Commit `{payload['commit']}`.", "",
          "| λ | set | mean±sd | pooled CI | k | Rule A |",
          "|---:|---|---:|---|---:|---|"]
    for r in rows:
        md.append(
            f"| {r['lambda']:g} | {r['set']} | {r['mean']:.3f}±{r['sd']:.3f} | "
            f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['k']} | "
            f"{'met' if r['rule_A'] else 'not met'} |"
        )
    md += ["", "LaTeX:", "```"] + latex + ["```", ""]
    (OUT / "B3_a4_ci.md").write_text("\n".join(md))
    print("wrote B3", "cpu_h", round(payload["cpu_hours"], 3), flush=True)


if __name__ == "__main__":
    main()
