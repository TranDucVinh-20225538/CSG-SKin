#!/usr/bin/env python3
"""Paper B R5 Q5: the separate frozen-encoder monitor next to the DANN-feature monitor on every Q1–Q3 run
(decisions/paperB_r5_q5.md). CPU, cached features.

Inputs
  manifest  results/paperB/r5/q4/manifest.csv (r5_q4_manifest.py --extra <Q1/Q2/Q3 run lists>). Rows are selected by
            --families (default: every family that is not one of the built-in R3/R4 families).
  frozen    per task, a directory of r5_frozen_extract.py outputs (<split>.npz with z, labels, groups) built on frame CSVs
            whose row order equals the run feature files: optional manifest column `frozen_dir`, else --frozen task=DIR,
            else (task isic_skin) results/paperB/r5/frozen/features/imagenet_r50.
For Q1 (CXR) the frames are written by the Q1 pipeline (path, label_idx, group = patient) and extracted with
  r5_frozen_extract.py --encoder imagenet_r50 --frames_dir <Q1 frames> --out_root results/paperB/r5/frozen/features_cxr
The frozen monitor does not depend on the run, so it is scored once per (task, frozen encoder) with a cluster bootstrap
CI (ID and OOD resampled by the `groups` stored in the npz); the run monitor is Mahalanobis / kNN on the run's own
representation, mean ± s.d. over seeds per (family, λ).
Usage: r5_q5_frozen_runs.py [--families f1 f2] [--frozen cxr=/path ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r5_q4_collapse as Q4  # noqa: E402

OUT = R.R5 / "q5"
BUILTIN = ("csg_dual_lesion", "csg_dual_image", "single_dann_image", "csg_dual_lesion_advlr")
DEFAULT_FROZEN = {"isic_skin": R.R5 / "frozen" / "features" / "imagenet_r50"}


def monitor_scores(ztr, ytr, zid, zoods, nc):
    m, k = Q4.PooledMaha(ztr, ytr, nc), R.C.KNN(ztr)
    return {"maha": (m.score(zid), {o: m.score(z) for o, z in zoods.items()}),
            "knn": (k.score(zid), {o: k.score(z) for o, z in zoods.items()})}


def frozen_block(fdir, id_train, id_test, oods, nc):
    ld = lambda s: dict(np.load(Path(fdir) / "{}.npz".format(s), allow_pickle=True))  # noqa: E731
    tr, te = ld(id_train), ld(id_test)
    oo = {o: ld(o) for o in oods}
    sc = monitor_scores(tr["z"], tr["labels"].astype(int), te["z"], {o: f["z"] for o, f in oo.items()}, nc)
    res = {}
    for det, (sid, so) in sc.items():
        for o in oods:
            lo, hi = R.cluster_ci(sid, te["groups"].astype(str), so[o], oo[o]["groups"].astype(str))
            res["{}|{}".format(det, o)] = {"auroc": R.auroc(sid, so[o]), "ci_lo": lo, "ci_hi": hi}
    return res, len(te["z"]), {o: len(f["z"]) for o, f in oo.items()}


def run_block(r):
    d = Path(r["feat_dir"])
    ld = lambda s: dict(np.load(d / "{}.npz".format(s)))  # noqa: E731
    tr, te = ld(r["id_train"]), ld(r["id_test"])
    oods = str(r["oods"]).split(";")
    oo = {o: ld(o) for o in oods}
    rep = r["rep"]
    sc = monitor_scores(tr[rep], tr["labels"].astype(int), te[rep], {o: f[rep] for o, f in oo.items()}, int(r["n_classes"]))
    rec = {k: r[k] for k in ("run_id", "task", "family", "lambda", "seed")}
    rec["n_id"] = len(te[rep])
    for det, (sid, so) in sc.items():
        for o in oods:
            rec["run_{}|{}".format(det, o)] = R.auroc(sid, so[o])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Q4.OUT / "manifest.csv")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--frozen", nargs="*", default=[], help="task=DIR")
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    sel = man[man.family.isin(a.families)] if a.families else man[~man.family.isin(BUILTIN)]
    ready = sel[sel.status == "ready"]
    fro = {**{k: Path(v) for k, v in DEFAULT_FROZEN.items()}, **{k: Path(v) for k, v in (x.split("=", 1) for x in a.frozen)}}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_block(r) for r in ready.to_dict("records")]
    df = pd.DataFrame(rows)
    frozen = {}
    for task, g in ready.groupby("task"):
        r0 = g.iloc[0].to_dict()
        fdir = Path(r0["frozen_dir"]) if "frozen_dir" in g and pd.notna(r0.get("frozen_dir")) else fro.get(task)
        if fdir is None or not (fdir / "{}.npz".format(r0["id_test"])).exists():
            frozen[task] = {"status": "frozen features missing", "dir": str(fdir)}
            continue
        cells, n_id, n_ood = frozen_block(fdir, r0["id_train"], r0["id_test"], str(r0["oods"]).split(";"), int(r0["n_classes"]))
        n_run = set(df[df.task == task].n_id)
        frozen[task] = {"status": "ok", "dir": str(fdir), "cells": cells, "n_id": n_id, "n_ood": n_ood,
                        "row_count_matches_runs": n_run == {n_id}}
    df.to_csv(OUT / "summary.csv", index=False)
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "frozen": frozen, "runs": rows})
    L = ["# Paper B R5 Q5 — frozen-encoder monitor vs DANN-feature monitor on the Q1–Q3 runs", "",
         "Design: `decisions/paperB_r5_q5.md`. Commit `{}`. Runs ready: {} of {} selected (families: {}).".format(
             R.git_head(), len(ready), len(sel), ", ".join(sorted(sel.family.unique())) or "none"), ""]
    tex = []
    for task, fr in frozen.items():
        d = df[df.task == task]
        oods = str(ready[ready.task == task].oods.iloc[0]).split(";")
        L += ["## Task " + task, "", "Frozen: `{}` ({}).".format(fr["dir"], fr["status"]), "",
              "| monitor | family | λ | n | " + " | ".join("Maha {}".format(o) for o in oods) + " | " + " | ".join("kNN {}".format(o) for o in oods) + " |",
              "|---|---|---:|---:|" + "---|" * (2 * len(oods))]
        if fr["status"] == "ok":
            c = fr["cells"]
            L.append("| frozen | — | — | — | " + " | ".join("{:.3f} [{:.3f}, {:.3f}]".format(c["{}|{}".format(det, o)]["auroc"], c["{}|{}".format(det, o)]["ci_lo"],
                                                                                         c["{}|{}".format(det, o)]["ci_hi"]) for det in ("maha", "knn") for o in oods) + " |")
            tex.append("{} frozen & -- & ".format(task) + " & ".join("{:.3f}".format(c["maha|{}".format(o)]["auroc"]) for o in oods) + " \\\\")
        for (fam, lam), g in d.groupby(["family", "lambda"]):
            L.append("| run | {} | {:g} | {} | ".format(fam, lam, len(g)) + " | ".join(
                R.fmt(g["run_{}|{}".format(det, o)].mean(), g["run_{}|{}".format(det, o)].std(ddof=1)) for det in ("maha", "knn") for o in oods) + " |")
            tex.append("{} {} & {:g} & ".format(task, fam.replace("_", "\\_"), lam) + " & ".join(
                R.tex_pm(g["run_maha|{}".format(o)].mean(), g["run_maha|{}".format(o)].std(ddof=1)) for o in oods) + " \\\\")
        if fr["status"] == "ok" and len(d):
            o = oods[0]
            lo = d.groupby("lambda")["run_maha|{}".format(o)].mean()
            L += ["", "Plain reading: on {} the frozen monitor gives Mahalanobis {:.3f} on {}; the run monitor ranges {:.3f}–{:.3f} across λ "
                  "(lowest at λ = {:g}).".format(task, fr["cells"]["maha|{}".format(o)]["auroc"], o, lo.min(), lo.max(), lo.idxmin())]
        L.append("")
    L += ["## LaTeX rows (task/monitor & λ & Mahalanobis per OOD set)", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
