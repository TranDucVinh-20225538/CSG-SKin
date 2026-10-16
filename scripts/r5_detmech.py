#!/usr/bin/env python3
"""Paper B R5 per-detector mechanism (decisions/paperB_r5_detmech.md). CPU.

Per-class AUROC and decomposition for each of 5 detectors at λ ∈ {0, 2}, old recipe and A4 (last.ckpt),
on pad_heldout and fitzpatrick17k — to explain Maha/cosine inversion vs Energy/kNN/MSP Fitz-only collapse.

Usage: r5_detmech.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r5" / "detmech"
SEEDS = R.SEEDS
LAMS = (0.0, 2.0)
OODS = ("pad_heldout", "fitzpatrick17k")
DETS = R.DETS


def feat_dirs():
    old = R.ITEM4_FEAT
    a4 = R.P8_DIR / "main" / "features"
    out = []
    for lam in LAMS:
        for s in SEEDS:
            out.append(("old_recipe", lam, s, old / "runB_orth1_ladv{}_s{}".format(R.lam_tag(lam), s)))
            out.append(("a4", lam, s, a4 / "main_ladv{}_s{}".format(R.lam_tag(lam), s)))
    return out


def run_one(recipe, lam, seed, feat_dir):
    feat_dir = Path(feat_dir)
    need = ("isic_train", "isic_test") + OODS
    if not all((feat_dir / "{}.npz".format(n)).exists() for n in need):
        return None
    Fs = R.load_feats(feat_dir, need)
    sc = R.Scorer(Fs["isic_train"])
    te = Fs["isic_test"]
    sid = sc.scores(te)
    yid = te["labels"].astype(int)
    rec = {"recipe": recipe, "lambda": lam, "seed": seed, "feat_dir": str(feat_dir), "ood": {}}
    for ood in OODS:
        so = sc.scores(Fs[ood])
        yood = Fs[ood]["labels"].astype(int)
        has_labels = bool((yood >= 0).all()) and ood.startswith("pad_")
        blk = {}
        for d in DETS:
            overall = R.auroc(sid[d], so[d])
            entry = {"auroc": overall}
            # ID-side per-class
            entry["id_class_auroc"] = {
                R.LABELS[c]: R.auroc(sid[d][yid == c], so[d]) for c in range(R.NCLS) if (yid == c).any()
            }
            if has_labels:
                entry["ood_class_auroc"] = {
                    R.LABELS[c]: R.auroc(sid[d], so[d][yood == c])
                    for c in R.PAD_CLASSES if (yood == c).sum() >= 1
                }
                entry["decomposition"] = R.decomposition(sid[d], yid, so[d], yood)
            blk[d] = entry
        rec["ood"][ood] = blk
    return rec


def ms(xs):
    a = np.asarray(xs, float)
    if a.size == 0:
        return None
    if a.size == 1:
        return {"mean": float(a[0]), "sd": None, "n": 1}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size)}


def pm(m):
    if m is None:
        return "—"
    if m.get("sd") is None:
        return "{:.3f}".format(m["mean"])
    return "{:.3f} ± {:.3f}".format(m["mean"], m["sd"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild REPORT.md / summary.csv from existing per_run JSON (no recompute)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir(exist_ok=True)
    rows = []
    if a.report_only:
        for p in sorted((OUT / "per_run").glob("*.json")):
            rec = json.loads(p.read_text())
            for ood in OODS:
                for det in DETS:
                    rows.append({"recipe": rec["recipe"], "lambda": rec["lambda"], "seed": rec["seed"],
                                 "ood": ood, "detector": det, "auroc": rec["ood"][ood][det]["auroc"]})
        print("report-only: loaded", len(rows), "summary rows from per_run", flush=True)
    else:
        for recipe, lam, seed, d in feat_dirs():
            rec = run_one(recipe, lam, seed, d)
            if rec is None:
                print("missing", recipe, lam, seed, d, flush=True)
                continue
            tag = "{}_ladv{}_s{}".format(recipe, R.lam_tag(lam), seed)
            (OUT / "per_run" / "{}.json".format(tag)).write_text(json.dumps(rec, indent=1, default=float) + "\n")
            for ood in OODS:
                for det in DETS:
                    rows.append({"recipe": recipe, "lambda": lam, "seed": seed, "ood": ood, "detector": det,
                                 "auroc": rec["ood"][ood][det]["auroc"]})
            print("ok", tag, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)

    L = [
        "# Paper B R5 — per-detector mechanism",
        "",
        "Design: `decisions/paperB_r5_detmech.md`. last.ckpt features for A4; item4 for old recipe.",
        "",
    ]
    for recipe in ("old_recipe", "a4"):
        L += ["## {}".format(recipe), ""]
        for ood in OODS:
            L += ["### {}".format(ood), "",
                  "| detector | λ = 0 AUROC | λ = 2 AUROC | Δ (2−0) |",
                  "|---|---|---|---|"]
            for det in DETS:
                m0 = ms(df[(df.recipe == recipe) & (df.ood == ood) & (df.detector == det)
                           & np.isclose(df["lambda"], 0.0)]["auroc"])
                m2 = ms(df[(df.recipe == recipe) & (df.ood == ood) & (df.detector == det)
                           & np.isclose(df["lambda"], 2.0)]["auroc"])
                dlt = "—" if m0 is None or m2 is None else "{:+.3f}".format(m2["mean"] - m0["mean"])
                L.append("| {} | {} | {} | {} |".format(det, pm(m0), pm(m2), dlt))
            L.append("")
    L += [
        "## Decomposition (Mahalanobis, pad_heldout; mean over seeds)",
        "",
        "|Δ| = |reweighted − identity|; **≈ 0 is the identity check** (reweighting algebra), not a failed "
        "detector. Compare those two columns to **unweighted pooled AUROC** (ordinary ID-vs-OOD AUROC).",
        "",
        "| recipe | λ | unweighted pooled AUROC | measured AUROC, ID reweighted to PAD mix | "
        "Σ w_c·AUROC_c (identity) | |Δ| | NV OOD AUROC | non-NV OOD AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for recipe in ("old_recipe", "a4"):
        for lam in LAMS:
            blocks, pooled = [], []
            for s in SEEDS:
                p = OUT / "per_run" / "{}_ladv{}_s{}.json".format(recipe, R.lam_tag(lam), s)
                if not p.exists():
                    continue
                blk = json.loads(p.read_text())["ood"]["pad_heldout"]["mahalanobis_classcond"]
                dec = blk.get("decomposition")
                if dec:
                    blocks.append(dec)
                if blk.get("auroc") is not None:
                    pooled.append(blk["auroc"])
            if not blocks:
                continue
            L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} |".format(
                recipe, lam,
                pm(ms(pooled)),
                pm(ms([b["reweighted_to_ood_mix"] for b in blocks])),
                pm(ms([b["identity_sum"] for b in blocks])),
                pm(ms([b["identity_abs_diff"] for b in blocks])),
                pm(ms([(b.get("ood_nv_vs_rest") or {}).get("nv") for b in blocks])),
                pm(ms([(b.get("ood_nv_vs_rest") or {}).get("non_nv") for b in blocks])),
            ))
    L += ["", "## Plain words", "",
          "Compare λ = 0 vs λ = 2 for Mahalanobis/cosine (inversion on pad_heldout / Fitz) against "
          "Energy/kNN/MSP (collapse on Fitz, milder on pad_heldout). Per-run JSON holds per-class AUROC "
          "and decomposition for every detector. In the decomposition table, |Δ| ≈ 0 confirms the "
          "reweight↔identity algebra; the unweighted pooled AUROC is the ordinary detection number.", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-40:]))
    print("wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
