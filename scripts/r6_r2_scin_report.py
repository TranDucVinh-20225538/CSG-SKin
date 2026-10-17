#!/usr/bin/env python3
"""CPU SCIN score + case-level cluster-CI report for Paper B R6 R2 (separate from primary fix reading).

Decisions: decisions/paperB_r6_r2_scin.md
- Same five detectors as R1; ID lesion / SCIN case (P7a group) bootstrap (PREREGISTER).
- First wave: completed λ=2 cells; re-run after held CDAN / λ=0.5 finish.
- Optional --extract_missing: fill scin.npz from last.ckpt when absent.

Usage:
  python scripts/r6_r2_scin_report.py [--recipe current] [--extract_missing] [--num_workers 4] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_report as RR  # noqa: E402

OOD = "scin"
METHODS = ("cdan", "iwdan", "iwcdan", "iwdan_oracle")
CKPT_ROOT = R.CKPT / "r6_r2"


def _mapping_note():
    meta = pd.read_csv(R.P7A_SETS / "scin.csv", keep_default_na=False)
    mapped = meta["mapped"].astype(str).str.strip()
    n_map = int((mapped != "").sum())
    n = int(len(meta))
    return {
        "n_images": n,
        "n_mapped_isic": n_map,
        "frac_mapped_isic": float(n_map / max(n, 1)),
        "n_groups_case": int(meta["group"].astype(str).nunique()),
        "mapped_counts": meta.loc[mapped != "", "mapped"].value_counts().to_dict(),
    }


def extract_missing_scin(rt, num_workers, device):
    """For each R2 run with last.ckpt but no scin.npz, extract only missing feature packs."""
    data = R.load_data()
    out_root = R.RES / "r6" / "r2" / rt
    ckpt_root = CKPT_ROOT / rt
    n_done = 0
    for ckpt_dir in sorted(ckpt_root.glob("r2_{}_*_s*".format(rt))):
        tag = ckpt_dir.name
        last = ckpt_dir / "last.ckpt"
        feat_dir = out_root / "features" / tag
        if not last.exists():
            continue
        if (feat_dir / "scin.npz").exists() and (feat_dir / "isic_train.npz").exists():
            continue
        print("extract_missing", tag, flush=True)
        net = R.load_net(last, device)
        R.extract(net, data, feat_dir, "full", num_workers, device)
        # refresh summary ood.scin if summary exists
        run_dir = out_root / "runs" / tag
        summary_p = run_dir / "summary.json"
        if summary_p.exists() and (feat_dir / "scin.npz").exists():
            Fs = R.load_feats(feat_dir)
            m = R.run_metrics(Fs)
            summary = json.loads(summary_p.read_text())
            summary.setdefault("metrics", {}).setdefault("ood", {})["scin"] = m["ood"]["scin"]
            summary["scin_rescore"] = {
                "from_checkpoint": str(last),
                "note": "CPU extract/score for separate REPORT_SCIN.md",
            }
            summary_p.write_text(json.dumps(summary, indent=2) + "\n")
        n_done += 1
    print("extract_missing done", n_done, flush=True)


def build_src(root, rt):
    src = [("λ=0 (R1 shift100 last.ckpt)", 0.0, RR.r1_shift100_runs(rt, 0.0))]
    for lam in (0.5, 2.0):
        src.append(("DANN (R1 shift100 last.ckpt)", lam, RR.r1_shift100_runs(rt, lam)))
        for m in METHODS:
            lab = "IWDAN_ORACLE" if m == "iwdan_oracle" else m.upper()
            src.append((lab, lam, RR.own_runs(root, "r2_" + rt + "_" + m + "_ladv{lam}_s{s}", lam)))
    # keep cells that have SCIN features (or at least isic_test + will skip if no scin)
    out = []
    for label, lam, pairs in src:
        keep = {}
        for s, (f, r) in pairs.items():
            if (Path(f) / "scin.npz").exists() and (Path(f) / "isic_train.npz").exists():
                keep[s] = (f, r)
        if keep:
            out.append((label, lam, keep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="current")
    ap.add_argument("--extract_missing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_boot", action="store_true")
    a = ap.parse_args()
    rt = R.recipe_tag(R.resolve_recipe(a.recipe))
    root = R.RES / "r6" / "r2" / rt
    device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")

    if a.extract_missing:
        extract_missing_scin(rt, a.num_workers, device)

    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    idg = R.idx_groups(lesion_g)
    og, unit = R.ood_groups(OOD, data)
    map_note = _mapping_note()

    src = build_src(root, rt)
    if not src:
        raise SystemExit("no R2/R1 cells with scin.npz under {}".format(root / "features"))

    print("cells", [(lab, lam, sorted(pairs)) for lab, lam, pairs in src], flush=True)
    print("SCIN unit", unit, "mapping", map_note, flush=True)

    boots = {} if a.no_boot else RR.boots_for(src, [OOD], data, idg, dets=R.BOOT_DETS)
    cells = RR.metrics(src)

    L = [
        "# Paper B R6 R2 — SCIN (secondary, separate report)",
        "",
        "Design: `decisions/paperB_r6_r2_scin.md`. Commit `{}`.".format(R.C.git_head()),
        "",
        "## Scope",
        "",
        "- **Not** in the R2 estimated-vs-oracle fix reading (primary sets remain Fitzpatrick17k / MED-NODE / SD-198).",
        "- Only **{:.3f}%** of SCIN images map to ISIC classes ({}/{} images; {} cases). "
        "Mapped counts: {}.".format(
            100.0 * map_note["frac_mapped_isic"], map_note["n_mapped_isic"], map_note["n_images"],
            map_note["n_groups_case"],
            ", ".join("{}={}".format(k, v) for k, v in sorted(map_note["mapped_counts"].items()))),
        "- Detectors: same five as R1. Cluster CIs: PREREGISTER 2,000 draws, rng 20261005; "
        "**ID = lesion**, **OOD = case** (P7a `group`, unit=`{}`).".format(unit),
        "- Baselines: R1 shift 100% RECIPE=`current` `last.ckpt`. R2 methods: CDAN / IWDAN / IWCDAN / IWDAN_ORACLE "
        "as present on disk; re-run this script when held λ = 0.5 / remaining CDAN seeds finish.",
        "- No new dermatology datasets.",
        "",
        "### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)",
        "",
    ]
    L += R.detector_lines(cells, boots, OOD) + [""]

    # Per-seed Mahalanobis table
    L += ["### Mahalanobis per seed (point + case-level CI)", "",
          "| cell | λ | seed | point | ci_lo | ci_hi | CI entirely < 0.5 |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for label, lam, runs in cells:
        b = boots.get((label, lam, OOD)) or {}
        per = (b.get("mahalanobis_classcond") or {}).get("per_seed") or {}
        for s in sorted(runs):
            pt = runs[s]["ood"][OOD]["mahalanobis_classcond"]
            ps = per.get(str(s))
            if ps:
                L.append("| {} | {:g} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                    label, lam, s, ps["point"], ps["ci_lo"], ps["ci_hi"],
                    "yes" if ps["below_chance_entire_ci"] else "no"))
            else:
                L.append("| {} | {:g} | {} | {:.3f} | — | — | — |".format(label, lam, s, pt))
    L.append("")

    # Paired Δ / gap on SCIN only (descriptive)
    L += ["## Paired-by-seed Δ vs DANN and gap recovered (SCIN only; descriptive)", "",
          "Same formulas as main R2 report; **does not** enter the fix reading.", "",
          "| method | λ | Δ Maha vs DANN [case-level cluster CI] | gap recovered (method−DANN)/(λ0−DANN) [seed boot CI] |",
          "|---|---:|---|---|"]
    gap_json, delta_json = {}, {}
    for lam in (0.5, 2.0):
        dann_pairs = RR.r1_shift100_runs(rt, lam)
        l0_pairs = RR.r1_shift100_runs(rt, 0.0)
        if not dann_pairs or not l0_pairs:
            continue
        dann_feat = {s: f for s, (f, _r) in dann_pairs.items() if (Path(f) / "scin.npz").exists()}
        l0_feat = {s: f for s, (f, _r) in l0_pairs.items() if (Path(f) / "scin.npz").exists()}
        for m in METHODS:
            lab = "IWDAN_ORACLE" if m == "iwdan_oracle" else m.upper()
            pairs = RR.own_runs(root, "r2_" + rt + "_" + m + "_ladv{lam}_s{s}", lam)
            feat = {s: f for s, (f, _r) in pairs.items() if (Path(f) / "scin.npz").exists()}
            if not feat or not dann_feat:
                continue
            if a.no_boot:
                dlt = gap = None
            else:
                dlt = RR.paired_delta_cluster_ci(feat, dann_feat, OOD, idg, og)
                gap = RR.gap_recovered_seed_boot(feat, dann_feat, l0_feat, OOD)
            key = "{}|{:g}|{}".format(lab, lam, OOD)
            if dlt:
                delta_json[key] = dlt
            if gap:
                gap_json[key] = gap
            L.append("| {} | {:g} | {} | {} |".format(lab, lam, RR._fmt_delta(dlt), RR._fmt_gap(gap)))
    L += ["",
          "Prediction (decisions): descriptive only — SCIN under DANN λ = 2 near chance (~0.52); "
          "IWDAN may sit slightly higher; not a gate.",
          ""]

    report_p = root / "REPORT_SCIN.md"
    report_p.write_text("\n".join(L) + "\n")
    payload = {
        "commit": R.C.git_head(),
        "design": "decisions/paperB_r6_r2_scin.md",
        "ood": OOD,
        "unit_ood": unit,
        "mapping": map_note,
        "bootstrap": {("{}|{:g}".format(lab, lam)): boots.get((lab, lam, OOD))
                      for lab, lam, _ in src},
        "paired_delta_vs_dann": delta_json,
        "gap_recovered": gap_json,
        "cells_present": [(lab, lam, sorted(runs)) for lab, lam, runs in cells],
        "note": "Secondary; 0.615% ISIC-mapped; excluded from R2 fix reading",
    }
    # json-safe boots (numpy types already float)
    (root / "scin_report.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print("wrote", report_p, flush=True)
    print("wrote", root / "scin_report.json", flush=True)
    # brief stdout table
    print("\nSCIN Maha (mean ± sd [CI])", flush=True)
    for label, lam, _runs in cells:
        b = boots.get((label, lam, OOD))
        print(" ", label, "λ={:g}".format(lam), RR.fmt_ci(b) if b else "—", flush=True)


if __name__ == "__main__":
    main()
