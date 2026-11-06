#!/usr/bin/env python3
"""Paper B R11 X3: aggregate runs — ID bal, V1, detectors, Rule A, reading cases A/B/C.

Design: decisions/paperB_r11_x3.md. CPU after train array.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from r11_x3_train import jobs, tag_for, OODS, OUT  # noqa: E402

ERM_REF = 0.558  # decisions/paperB_r11_x3.md; r5/p8/erm_ref/REFERENCE.json mean
COMPETENT_FLOOR = ERM_REF - 0.03
INVARIANT_V1 = 0.60
DEG = 0.25
LABELLED_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")  # + pad_heldout separately


def git_head():
    return subprocess.run(
        ["git", "-C", str(R.PAPERB), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


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


def load_run(tag):
    p = OUT / "runs" / tag / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def bootstrap_cells(cell_runs):
    """cell_runs: list of (seed, feat_dir Path) excluding degraded."""
    if not cell_runs:
        return None
    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    id_groups = R.idx_groups(lesion_g)
    feat_dirs = {s: Path(fd) for s, fd in cell_runs}
    out = {}
    for ood in OODS:
        og, unit = R.ood_groups(ood, data)
        b = R.bootstrap_cell(feat_dirs, ood, id_groups, og)
        if b:
            b["unit"] = unit
        out[ood] = b
    return out


def ms(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "n": len(xs)}


def cell_key(lam, shift):
    return "lam{:g}|shift{}".format(lam, int(round(100 * shift)))


def reading_case(grid_cells, shift0_cell):
    """Apply decisions/paperB_r11_x3.md reading rules. λ=2 shift100 decides."""
    c2 = grid_cells.get(cell_key(2.0, 1.0))
    if c2 is None:
        return {"case": None, "note": "λ=2 shift100 missing"}

    # Not invariant at any λ?
    any_inv = False
    for k, c in grid_cells.items():
        if c and c.get("v1") and c["v1"]["mean"] <= INVARIANT_V1:
            any_inv = True
            break
    if not any_inv:
        return {
            "case": "C",
            "label": "Not invariant at any λ",
            "competent": None,
            "invariant_l2": False,
            "rule_A_pad": None,
            "rule_A_unseen_hits": None,
            "note": "invariance not reached; settles nothing. Stop after λ grid.",
        }

    v1 = c2["v1"]["mean"]
    idb = c2["id_bal"]["mean"]
    invariant = v1 <= INVARIANT_V1
    competent = idb >= COMPETENT_FLOOR
    boots = c2.get("bootstrap") or {}
    rule_pad = bool(boots.get("pad_heldout", {}).get("rule_A"))
    unseen_hits = [o for o in LABELLED_UNSEEN if boots.get(o, {}).get("rule_A")]
    n_unseen = len(unseen_hits)

    all_labelled = ("pad_heldout",) + LABELLED_UNSEEN
    none_rule = all(not boots.get(o, {}).get("rule_A") for o in all_labelled)

    if not competent:
        case, note = "not_competent", "reported, settles neither"
    elif invariant and rule_pad and n_unseen >= 2:
        case, note = "A", "inversion at ERM competence"
    elif invariant and none_rule:
        case, note = "B", "no inversion at ERM competence"
    else:
        case, note = "intermediate", (
            "competent+invariant but Rule A pattern matches neither A nor B; report as found")

    out = {
        "case": case, "label": note,
        "competent": competent, "invariant_l2": invariant,
        "id_bal": idb, "v1": v1,
        "competent_floor": COMPETENT_FLOOR, "erm_ref": ERM_REF,
        "rule_A_pad": rule_pad, "rule_A_unseen_hits": unseen_hits,
        "note": note,
    }
    if case == "A" and shift0_cell:
        b0 = shift0_cell.get("bootstrap") or {}
        hits0 = [o for o in all_labelled if b0.get(o, {}).get("rule_A")]
        out["shift0_control"] = {
            "rule_A_hits": hits0,
            "label_shift_necessary": len(hits0) == 0,
            "note": "label shift necessary if Rule A on none of labelled unseen at shift 0%",
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # group runs into cells
    cells_raw = defaultdict(list)
    for j in jobs():
        tag = tag_for(j["lam"], j["seed"], j["shift"])
        s = load_run(tag)
        if s is None:
            print("missing", tag, flush=True)
            continue
        cells_raw[cell_key(j["lam"], j["shift"])].append(s)

    cells = {}
    for key, runs in sorted(cells_raw.items()):
        id_vals, v1_vals = [], []
        feat_for_boot = []
        degraded = []
        per = []
        for s in sorted(runs, key=lambda r: r["seed"]):
            seed = int(s["seed"])
            idb = float(s["id_balanced_acc"])
            v1 = float(s["heldout_leakage_V1"])
            rec = {
                "seed": seed, "tag": s["tag"], "id_bal": idb, "v1": v1,
                "ood_maha": {o: s["metrics"]["ood"].get(o, {}).get("mahalanobis_classcond")
                             for o in OODS},
            }
            per.append(rec)
            if idb < DEG:
                degraded.append(seed)
                continue
            id_vals.append(idb)
            v1_vals.append(v1)
            feat_for_boot.append((seed, OUT / "features" / s["tag"]))
        print("bootstrap", key, "n", len(feat_for_boot), flush=True)
        boots = bootstrap_cells(feat_for_boot)
        cells[key] = {
            "id_bal": ms(id_vals),
            "v1": ms(v1_vals),
            "n_seeds": len(id_vals),
            "degraded_seeds": degraded,
            "per_seed": per,
            "bootstrap": {
                o: None if boots is None or boots.get(o) is None else {
                    "rule_A": boots[o]["rule_A"],
                    "mahalanobis_classcond": {k: boots[o]["mahalanobis_classcond"][k]
                                             for k in ("mean", "sd", "ci_lo", "ci_hi", "k_below", "n_seeds")
                                             if k in boots[o]["mahalanobis_classcond"]},
                } for o in OODS
            },
        }

    grid = {k: v for k, v in cells.items() if k.endswith("|shift100") or "|shift100" in k}
    # keys are lamX|shift100
    grid_cells = {k: v for k, v in cells.items() if k.endswith("shift100")}
    shift0 = cells.get(cell_key(2.0, 0.0))
    verdict = reading_case(grid_cells, shift0)

    payload = {
        "design": "decisions/paperB_r11_x3.md",
        "commit": git_head(),
        "erm_ref": ERM_REF,
        "competent_floor": COMPETENT_FLOOR,
        "invariant_v1": INVARIANT_V1,
        "cells": cells,
        "verdict": verdict,
    }
    (OUT / "RESULTS.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    # REPORT
    L = [
        "# Paper B R11 X3 — adversary on head over frozen ERM",
        "",
        "Design: `decisions/paperB_r11_x3.md`. Commit `{}`.".format(git_head()),
        "",
        "Frozen ERM EffB3 features (no aug). Head: 2-layer MLP → 16-d + classifier + derm adversary/GRL. "
        "40 epochs, last.ckpt. ERM ref ID bal = {:.3f}; competent ≥ {:.3f}; invariant V1 ≤ {:.2f}.".format(
            ERM_REF, COMPETENT_FLOOR, INVARIANT_V1),
        "",
        "## Cells",
        "",
        "| λ | shift | ID bal | V1 | Rule A pad | Rule A Fitz | Rule A med | Rule A SD | n |",
        "|---:|---:|---|---|---|---|---|---|---:|",
    ]
    for key in sorted(cells.keys(), key=lambda k: (0 if "shift100" in k else 1, k)):
        c = cells[key]
        lam = key.split("|")[0].replace("lam", "")
        sh = key.split("|")[1].replace("shift", "")
        idb = c["id_bal"]
        v1 = c["v1"]
        def pm(d):
            if not d:
                return "—"
            return "{:.3f}".format(d["mean"]) + (" ± {:.3f}".format(d["sd"]) if d.get("sd") is not None else "")
        b = c["bootstrap"]

        def ra(o):
            x = b.get(o) if b else None
            if not x:
                return "—"
            return str(x["rule_A"])
        L.append("| {} | {}% | {} | {} | {} | {} | {} | {} | {} |".format(
            lam, sh, pm(idb), pm(v1), ra("pad_heldout"), ra("fitzpatrick17k"),
            ra("mednode"), ra("sd198"), c["n_seeds"]))

    L += ["", "## Verdict (λ = 2, shift 100 %)", ""]
    v = verdict
    L.append("- **Case {}**: {}.".format(v.get("case"), v.get("note")))
    L.append("- competent = {}; invariant = {}; ID bal = {}; V1 = {}.".format(
        v.get("competent"), v.get("invariant_l2"),
        None if v.get("id_bal") is None else "{:.3f}".format(v["id_bal"]),
        None if v.get("v1") is None else "{:.3f}".format(v["v1"])))
    L.append("- Rule A pad = {}; unseen hits = {}.".format(v.get("rule_A_pad"), v.get("rule_A_unseen_hits")))
    if v.get("shift0_control"):
        L.append("- shift 0% control: {}.".format(v["shift0_control"]))
    L += ["", "## Notes", "",
          "- Cache: `results/paperB/r11/x3/cache/` (symlinks to erm_ref; no augmentation).",
          "- λ=0.5 written as ladv0p5 in tags.",
          ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "case", v.get("case"), flush=True)
    print("DONE_X3_SCORE", flush=True)


if __name__ == "__main__":
    main()
