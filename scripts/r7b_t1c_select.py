#!/usr/bin/env python3
"""Paper B R7b T1c selection: pick backbone LR mult per arm (decisions/paperB_r7b_t1c.md).

Candidates: ×0.1 from T1 seed 42; ×0.3 and ×1.0 from T1c selection runs.
Choose max ISIC val balanced accuracy (last.ckpt); tie <0.005 → smaller mult.
If dual choice is ×0.1, stop (T1 is the result).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

OUT = R.R7B / "t1c"
T1 = R.R7 / "t1"
TIE = 0.005


def load_cand(mult, path: Path):
    s = json.loads(path.read_text())
    return {
        "mult": mult,
        "val_balanced_acc": float(s["val_balanced_acc"]),
        "id_balanced_acc": float(s.get("id_balanced_acc")) if s.get("id_balanced_acc") is not None else None,
        "tag": s.get("tag"),
        "path": str(path),
    }


def pick(cands):
    vmax = max(c["val_balanced_acc"] for c in cands)
    tied = [c for c in cands if (vmax - c["val_balanced_acc"]) < TIE]
    return min(tied, key=lambda c: c["mult"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dual, erm = [], []
    d0 = T1 / "runs" / "r7_t1_dual_dinov2_vits14_ladv0_s42" / "summary.json"
    e0 = T1 / "erm" / "r7_t1_single_dinov2_vits14_baseline_ladv0_s42" / "summary.json"
    if d0.exists():
        dual.append(load_cand(0.1, d0))
    if e0.exists():
        erm.append(load_cand(0.1, e0))
    for m in (0.3, 1.0):
        bb = R.lam_tag(m)
        dp = OUT / "runs" / "r7b_t1c_dual_dinov2_vits14_ladv0_s42_bb{}".format(bb) / "summary.json"
        ep = OUT / "erm" / "r7b_t1c_single_dinov2_vits14_baseline_ladv0_s42_bb{}".format(bb) / "summary.json"
        if dp.exists():
            dual.append(load_cand(m, dp))
        if ep.exists():
            erm.append(load_cand(m, ep))
    if len(dual) < 3 or len(erm) < 3:
        out = {"status": "incomplete", "dual_candidates": dual, "erm_candidates": erm}
        (OUT / "SELECTION.json").write_text(json.dumps(out, indent=2) + "\n")
        print("selection incomplete: dual", len(dual), "erm", len(erm))
        return 1
    d_pick, e_pick = pick(dual), pick(erm)
    stop = abs(d_pick["mult"] - 0.1) < 1e-12
    out = {
        "status": "stop_after_selection" if stop else "proceed_main",
        "tie_eps": TIE,
        "dual_chosen_mult": d_pick["mult"],
        "dual_chosen_val_bal": d_pick["val_balanced_acc"],
        "erm_chosen_mult": e_pick["mult"],
        "erm_chosen_val_bal": e_pick["val_balanced_acc"],
        "dual_candidates": dual,
        "erm_candidates": erm,
        "note": ("If dual chooses ×0.1, T1c stops after selection (T1 is the result). "
                 "Test-set numbers reported but not used for choice."),
    }
    (OUT / "SELECTION.json").write_text(json.dumps(out, indent=2) + "\n")
    L = ["# Paper B R7b T1c selection", "",
         "Dual chosen × {:g} (val bal {:.4f}). ERM chosen × {:g} (val bal {:.4f}).".format(
             d_pick["mult"], d_pick["val_balanced_acc"], e_pick["mult"], e_pick["val_balanced_acc"]), "",
         "Status: **{}**.".format(out["status"]), "",
         "| arm | × | val bal | ID bal (test, not used) |",
         "|---|---:|---:|---:|"]
    for arm, cands in (("dual λ=0", dual), ("ERM", erm)):
        for c in sorted(cands, key=lambda x: x["mult"]):
            L.append("| {} | {:g} | {:.4f} | {:.4f} |".format(
                arm, c["mult"], c["val_balanced_acc"], c["id_balanced_acc"] or float("nan")))
    (OUT / "SELECTION.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "dual_chosen_mult", "erm_chosen_mult",
                                          "dual_chosen_val_bal", "erm_chosen_val_bal")}, indent=2))
    return 2 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
