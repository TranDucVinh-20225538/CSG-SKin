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
MULTS = (0.1, 0.3, 1.0)


def val_bal(summary_path: Path):
    s = json.loads(summary_path.read_text())
    return float(s["val_balanced_acc"]), s


def pick(rows):
    """rows: list of (mult, val_bal, summary, path). Higher val wins; ties (<0.005) → smaller mult."""
    vmax = max(v for _, v, _, _ in rows)
    tied = [(m, v, s, p) for m, v, s, p in rows if (vmax - v) < TIE]
    return min(tied, key=lambda x: x[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dual_cands, erm_cands = [], []
    # ×0.1 from T1
    d0 = T1 / "runs" / "r7_t1_dual_dinov2_vits14_ladv0_s42" / "summary.json"
    e0 = T1 / "erm" / "r7_t1_single_dinov2_vits14_baseline_ladv0_s42" / "summary.json"
    if d0.exists():
        v, s = val_bal(d0)
        dual_cands.append((0.1, v, s, str(d0)))
    if e0.exists():
        v, s = val_bal(e0)
        erm_cands.append((0.1, v, s, str(e0)))
    # ×0.3, ×1.0 from T1c
    for m in (0.3, 1.0):
        bb = R.lam_tag(m)
        dp = OUT / "runs" / "r7b_t1c_dual_dinov2_vits14_ladv0_s42_bb{}".format(bb) / "summary.json"
        ep = OUT / "erm" / "r7b_t1c_single_dinov2_vits14_baseline_ladv0_s42_bb{}".format(bb) / "summary.json"
        if dp.exists():
            v, s = val_bal(dp)
            dual_cands.append((m, v, s, str(dp)))
        if ep.exists():
            v, s = val_bal(ep)
            erm_cands.append((m, v, s, str(ep)))
    if len(dual_cands) < 3 or len(erm_cands) < 3:
        print("selection incomplete: dual", len(dual_cands), "erm", len(erm_cands))
        (OUT / "SELECTION.json").write_text(json.dumps({
            "status": "incomplete", "dual_candidates": dual_cands, "erm_candidates": erm_cands}, indent=2) + "\n")
        return 1
    d_pick = pick(dual_cands)
    e_pick = pick(erm_cands)
    stop = abs(d_pick[0] - 0.1) < 1e-12
    out = {
        "status": "stop_after_selection" if stop else "proceed_main",
        "tie_eps": TIE,
        "dual_chosen_mult": d_pick[0],
        "dual_chosen_val_bal": d_pick[1],
        "erm_chosen_mult": e_pick[0],
        "erm_chosen_val_bal": e_pick[1],
        "dual_candidates": [
            {"mult": m, "val_balanced_acc": v, "id_balanced_acc": s.get("id_balanced_acc"),
             "path": p, "tag": s.get("tag")} for m, v, s, p in dual_cands],
        "erm_candidates": [
            {"mult": m, "val_balanced_acc": v, "id_balanced_acc": s.get("id_balanced_acc"),
             "path": p, "tag": s.get("tag")} for m, v, s, p in erm_cands],
        "note": ("If dual chooses ×0.1, T1c stops after selection (T1 is the result). "
                 "Test-set numbers reported but not used for choice."),
    }
    (OUT / "SELECTION.json").write_text(json.dumps(out, indent=2) + "\n")
    # short markdown
    L = ["# Paper B R7b T1c selection", "",
         "Dual chosen × {:.1f} (val bal {:.4f}). ERM chosen × {:.1f} (val bal {:.4f}).".format(
             d_pick[0], d_pick[1], e_pick[0], e_pick[1]), "",
         "Status: **{}**.".format(out["status"]), "",
         "| arm | × | val bal | ID bal (test, not used) |",
         "|---|---:|---:|---:|"]
    for arm, cands in (("dual λ=0", dual_cands), ("ERM", erm_cands)):
        for m, v, s, _p in sorted(cands, key=lambda x: x[0]):
            L.append("| {} | {:g} | {:.4f} | {:.4f} |".format(arm, m, v, float(s.get("id_balanced_acc", float("nan")))))
    (OUT / "SELECTION.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: out[k] for k in ("status", "dual_chosen_mult", "erm_chosen_mult")}, indent=2))
    return 0 if not stop else 2


if __name__ == "__main__":
    sys.exit(main() or 0)
