#!/usr/bin/env python3
"""Paper B R11 X5: dose–response AUROC vs t; inversion reading rule.

Reading: reproduces inversion if Rule A on ≥2 of 3 labelled unseen at t=1 and none at t=0 (λ=2).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from r11_x3_train import git_head  # noqa: E402
from r11_x5_train import jobs, tag_for, OUT, OODS, TS, SEEDS  # noqa: E402

DEG = 0.25
LABELLED_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def ms(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "n": len(xs)}


def cell_key(lam, t):
    return "lam{:g}|t{:g}".format(lam, t)


def bootstrap_cell_runs(runs_feat):
    if not runs_feat:
        return None
    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    id_groups = R.idx_groups(lesion_g)
    feat_dirs = {s: Path(fd) for s, fd in runs_feat}
    out = {}
    for ood in OODS:
        # synth_isic_test: image-level groups
        if ood == "synth_isic_test":
            n = len(np.load(next(iter(feat_dirs.values())) / "synth_isic_test.npz")["labels"])
            og = [np.array([i]) for i in range(n)]
            unit = "image (synth test)"
        else:
            og, unit = R.ood_groups(ood, data)
        b = R.bootstrap_cell(feat_dirs, ood, id_groups, og)
        if b:
            b["unit"] = unit
        out[ood] = b
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = defaultdict(list)
    for j in jobs():
        tag = tag_for(j["lam"], j["seed"], j["t"])
        p = OUT / "runs" / tag / "summary.json"
        if not p.exists():
            print("missing", tag, flush=True)
            continue
        raw[cell_key(j["lam"], j["t"])].append(json.loads(p.read_text()))

    cells = {}
    for key, runs in sorted(raw.items()):
        id_vals, v1_vals = [], []
        feat_for_boot, degraded, per = [], [], []
        for s in sorted(runs, key=lambda r: r["seed"]):
            seed, idb, v1 = int(s["seed"]), float(s["id_balanced_acc"]), float(s["heldout_leakage_V1"])
            maha = {o: s["metrics"]["ood"].get(o, {}).get("mahalanobis_classcond") for o in OODS}
            per.append({"seed": seed, "tag": s["tag"], "id_bal": idb, "v1": v1, "ood_maha": maha})
            if idb < DEG:
                degraded.append(seed)
                continue
            id_vals.append(idb)
            v1_vals.append(v1)
            feat_for_boot.append((seed, OUT / "features" / s["tag"]))
        print("bootstrap", key, "n", len(feat_for_boot), flush=True)
        boots = bootstrap_cell_runs(feat_for_boot)
        cells[key] = {
            "id_bal": ms(id_vals), "v1": ms(v1_vals), "n_seeds": len(id_vals),
            "degraded_seeds": degraded, "per_seed": per,
            "point_maha": {o: ms([p["ood_maha"].get(o) for p in per if p["seed"] not in degraded])
                           for o in OODS},
            "bootstrap": {
                o: None if not boots or not boots.get(o) else {
                    "rule_A": boots[o]["rule_A"],
                    "mahalanobis_classcond": {
                        k: boots[o]["mahalanobis_classcond"][k]
                        for k in ("mean", "sd", "ci_lo", "ci_hi", "k_below", "n_seeds")
                        if k in boots[o]["mahalanobis_classcond"]},
                } for o in OODS
            },
        }

    # Reading rule at λ=2
    c1 = cells.get(cell_key(2.0, 1.0))
    c0 = cells.get(cell_key(2.0, 0.0))
    hits1 = hits0 = []
    if c1 and c1.get("bootstrap"):
        hits1 = [o for o in LABELLED_UNSEEN if c1["bootstrap"].get(o, {}) and c1["bootstrap"][o]["rule_A"]]
    if c0 and c0.get("bootstrap"):
        hits0 = [o for o in LABELLED_UNSEEN if c0["bootstrap"].get(o, {}) and c0["bootstrap"][o]["rule_A"]]
    reproduces = (len(hits1) >= 2) and (len(hits0) == 0)

    verdict = {
        "reproduces_inversion": reproduces,
        "rule_A_hits_t1": hits1,
        "rule_A_hits_t0": hits0,
        "note": "Rule A ≥2/3 labelled unseen at t=1 and none at t=0 (λ=2)",
    }

    payload = {
        "design": "decisions/paperB_r11_x5.md", "commit": git_head(),
        "cells": cells, "verdict": verdict, "seeds": list(SEEDS), "t_levels": list(TS),
    }
    (OUT / "RESULTS.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    L = [
        "# Paper B R11 X5 — semi-synthetic controlled label shift",
        "",
        "Design: `decisions/paperB_r11_x5.md`. Commit `{}`.".format(git_head()),
        "",
        "Frozen ERM + smartphone synth target (blur σ=1.5, +10%R/−10%B, JPEG q=40). "
        "Head as X3. t = (1−t)·ISIC + t·PAD mix on synth stream. Seeds {}.".format(list(SEEDS)),
        "",
        "## Dose–response (Mahalanobis AUROC mean, λ = 2)",
        "",
        "| t | pad | Fitz | mednode | SD-198 | synth_test | V1 | ID bal | RuleA unseen |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]

    def pm(d):
        if not d:
            return "—"
        return "{:.3f}".format(d["mean"]) + (" ± {:.3f}".format(d["sd"]) if d.get("sd") is not None else "")

    for t in TS:
        c = cells.get(cell_key(2.0, t))
        if not c:
            L.append("| {:g} | — | — | — | — | — | — | — | — |".format(t))
            continue
        b = c["bootstrap"]
        hits = [o for o in LABELLED_UNSEEN if b.get(o) and b[o]["rule_A"]]
        pt = c["point_maha"]
        # prefer bootstrap mean when present
        def auroc(o):
            bb = b.get(o) if b else None
            if bb and bb.get("mahalanobis_classcond"):
                return pm(bb["mahalanobis_classcond"])
            return pm(pt.get(o))
        L.append("| {:g} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            t, auroc("pad_heldout"), auroc("fitzpatrick17k"), auroc("mednode"),
            auroc("sd198"), auroc("synth_isic_test"), pm(c["v1"]), pm(c["id_bal"]),
            hits or "none"))

    L += ["", "## λ = 0 reference (Mahalanobis)", "",
          "| t | pad | Fitz | mednode | SD-198 | synth_test | V1 | ID bal |",
          "|---:|---|---|---|---|---|---|---|"]
    for t in TS:
        c = cells.get(cell_key(0.0, t))
        if not c:
            L.append("| {:g} | — | — | — | — | — | — | — |".format(t))
            continue
        b, pt = c["bootstrap"], c["point_maha"]

        def auroc(o):
            bb = b.get(o) if b else None
            if bb and bb.get("mahalanobis_classcond"):
                return pm(bb["mahalanobis_classcond"])
            return pm(pt.get(o))
        L.append("| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
            t, auroc("pad_heldout"), auroc("fitzpatrick17k"), auroc("mednode"),
            auroc("sd198"), auroc("synth_isic_test"), pm(c["v1"]), pm(c["id_bal"])))

    L += ["", "## Reading rule", "",
          "- Rule A hits at t=1 (λ=2): **{}**.".format(hits1 or "none"),
          "- Rule A hits at t=0 (λ=2): **{}**.".format(hits0 or "none"),
          "- **reproduces_inversion = {}**.".format(reproduces),
          ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "reproduces", reproduces, flush=True)
    print("DONE_X5_SCORE", flush=True)


if __name__ == "__main__":
    main()
