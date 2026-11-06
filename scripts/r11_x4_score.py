#!/usr/bin/env python3
"""Paper B R11 X4: aggregate BCN→HAM — V1, Rule A, reading rules."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from r11_x4_train import jobs, tag_for, OUT, OODS, LABELLED_UNSEEN, git_head, SPLIT_PATH  # noqa: E402

DEG = 0.25
INVARIANT_V1 = 0.60


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


def cell_key(lam, shift):
    return "lam{:g}|shift{}".format(lam, int(round(100 * shift)))


def ham_groups(feat_dir):
    """Lesion groups for ham_heldout aligned to feature row order (path order in extract)."""
    # reconstruct from split
    import phase16_bcn_ham_data as bdata
    df = bdata.load_bcn_ham_frame()
    splits, _ = bdata.prepare_b1_splits(df, 42, ham_heldout_frac=0.30)
    return R.idx_groups(splits["ham_heldout"]["lesion_id"].astype(str).to_numpy()), "lesion"


def bootstrap_runs(runs_feat):
    if not runs_feat:
        return None
    data = R.load_data()
    # ID groups: BCN test lesions — need from first feat dir's bcn_test length vs split
    import phase16_bcn_ham_data as bdata
    df = bdata.load_bcn_ham_frame()
    splits, _ = bdata.prepare_b1_splits(df, 42, ham_heldout_frac=0.30)
    id_groups = R.idx_groups(splits["bcn_test"]["lesion_id"].astype(str).to_numpy())
    feat_dirs = {s: Path(fd) for s, fd in runs_feat}
    out = {}
    for ood in OODS:
        if ood == "ham_heldout":
            og, unit = ham_groups(next(iter(feat_dirs.values())))
        else:
            og, unit = R.ood_groups(ood, data)
        b = R.bootstrap_cell(feat_dirs, ood, id_groups, og)
        if b:
            b["unit"] = unit
        out[ood] = b
    return out


def reading(cells):
    c100 = cells.get(cell_key(2.0, 1.0))
    c0 = cells.get(cell_key(2.0, 0.0))
    if not c100:
        return {"case": None, "note": "λ=2 shift100 missing"}
    v1 = c100["v1"]["mean"] if c100.get("v1") else None
    if v1 is None or v1 > INVARIANT_V1:
        return {"case": "C", "invariant": False, "v1": v1,
                "note": "not invariant (V1 > 0.60); settles nothing"}
    b100 = c100.get("bootstrap") or {}
    hits100 = [o for o in LABELLED_UNSEEN if b100.get(o) and b100[o].get("rule_A")]
    hits0 = []
    if c0 and c0.get("bootstrap"):
        hits0 = [o for o in LABELLED_UNSEEN if c0["bootstrap"].get(o) and c0["bootstrap"][o].get("rule_A")]
    if len(hits100) >= 2 and len(hits0) == 0:
        return {"case": "reproduced_label_shift_necessary", "invariant": True, "v1": v1,
                "hits_shift100": hits100, "hits_shift0": hits0,
                "note": "inversion reproduced and label shift necessary in this pair"}
    if len(hits100) >= 2 and len(hits0) >= 1:
        return {"case": "reproduced_label_shift_not_necessary", "invariant": True, "v1": v1,
                "hits_shift100": hits100, "hits_shift0": hits0,
                "note": "inversion reproduced, label shift not necessary here"}
    return {"case": "not_reproduced", "invariant": True, "v1": v1,
            "hits_shift100": hits100, "hits_shift0": hits0,
            "note": "met on fewer than 2 at shift 100%; not reproduced in this pair"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = defaultdict(list)
    for j in jobs():
        tag = tag_for(j["lam"], j["seed"], j["shift"])
        p = OUT / "runs" / tag / "summary.json"
        if not p.exists():
            print("missing", tag, flush=True)
            continue
        raw[cell_key(j["lam"], j["shift"])].append(json.loads(p.read_text()))

    cells = {}
    for key, runs in sorted(raw.items()):
        id_vals, v1_vals, feat_for, degraded, per = [], [], [], [], []
        for s in sorted(runs, key=lambda r: r["seed"]):
            seed = int(s["seed"])
            idb, v1 = float(s["id_balanced_acc"]), float(s["heldout_leakage_V1"])
            per.append({"seed": seed, "tag": s["tag"], "id_bal": idb, "v1": v1,
                        "ood_maha": {o: s["metrics"]["ood"].get(o, {}).get("mahalanobis_classcond") for o in OODS}})
            if idb < DEG:
                degraded.append(seed)
                continue
            id_vals.append(idb)
            v1_vals.append(v1)
            feat_for.append((seed, OUT / "features" / s["tag"]))
        print("bootstrap", key, "n", len(feat_for), flush=True)
        boots = bootstrap_runs(feat_for)
        cells[key] = {
            "id_bal": ms(id_vals), "v1": ms(v1_vals), "n_seeds": len(id_vals),
            "degraded_seeds": degraded, "per_seed": per,
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

    verdict = reading(cells)
    tv = json.loads(SPLIT_PATH.read_text()) if SPLIT_PATH.exists() else {}
    payload = {"design": "decisions/paperB_r11_x4.md", "commit": git_head(),
               "tv_priors": tv, "cells": cells, "verdict": verdict}
    (OUT / "RESULTS.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    def pm(d):
        if not d:
            return "—"
        return "{:.3f}".format(d["mean"]) + (" ± {:.3f}".format(d["sd"]) if d.get("sd") is not None else "")

    L = [
        "# Paper B R11 X4 — BCN → HAM",
        "",
        "Design: `decisions/paperB_r11_x4.md`. Commit `{}`.".format(git_head()),
        "",
        "## TV class priors (report first)",
        "",
        "- TV(BCN train, HAM adv) = {}.".format(tv.get("tv_bcn_train_vs_ham_adv")),
        "- TV(BCN train, HAM heldout) = {}.".format(tv.get("tv_bcn_train_vs_ham_heldout")),
        "- ham_heldout lesions frac target ≥ 20% (split uses 30%).",
        "",
        "## Cells",
        "",
        "| λ | shift | ID bal | V1 | RuleA Fitz | med | SD | ham_heldout | n |",
        "|---:|---:|---|---|---|---|---|---|---:|",
    ]
    for key in sorted(cells.keys()):
        c = cells[key]
        lam, sh = key.split("|")[0].replace("lam", ""), key.split("|")[1].replace("shift", "")
        b = c["bootstrap"]

        def ra(o):
            x = b.get(o) if b else None
            return "—" if not x else str(x["rule_A"])
        L.append("| {} | {}% | {} | {} | {} | {} | {} | {} | {} |".format(
            lam, sh, pm(c["id_bal"]), pm(c["v1"]), ra("fitzpatrick17k"), ra("mednode"),
            ra("sd198"), ra("ham_heldout"), c["n_seeds"]))
    L += ["", "## Verdict (λ = 2)", "",
          "- **{}**: {}.".format(verdict.get("case"), verdict.get("note")),
          "- V1 = {}; hits@100 = {}; hits@0 = {}.".format(
              verdict.get("v1"), verdict.get("hits_shift100"), verdict.get("hits_shift0")),
          "",
          "ham_heldout and SCIN reported, not counted.",
          ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "case", verdict.get("case"), flush=True)
    print("DONE_X4_SCORE", flush=True)


if __name__ == "__main__":
    main()
