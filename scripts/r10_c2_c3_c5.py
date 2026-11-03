#!/usr/bin/env python3
"""R10 C2/C3/C5: fill MED-NODE / SD-198 / SCIN (and A4 / DINOv2) table rows from cached features.

C2 tab:dial — Mahalanobis on mednode/sd198/scin for every λ in r5/t1last (last.ckpt features).
C3 tab:unseen — DINOv2 dual from r7/t1; missing A4 sets from r5/p8/main.
C5 tab:single-dino — MED-NODE / SD-198 / SCIN for r7/t1b.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r10"
OODS = ("mednode", "sd198", "scin")
DET = "mahalanobis_classcond"


def pm(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if len(a) == 0:
        return None
    if len(a) == 1:
        return {"mean": float(a[0]), "sd": 0.0, "n": 1, "vals": a.tolist()}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(len(a)), "vals": a.tolist()}


def latex_pm(st):
    if st is None:
        return "--"
    return "${:.3f}\\pm{:.3f}$".format(st["mean"], st["sd"])


def c2_dial(data, idg):
    """Score mednode/sd198/scin on t1last features_last for every λ."""
    root = R.RES / "r5" / "t1last" / "features_last"
    # parse λ from run name runB_orth1_ladv{}_s{}
    cells = {}  # lam -> {ood: {seed: auroc}}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"runB_orth1_ladv(.+)_s(\d+)", d.name)
        if not m:
            continue
        lam = float(m.group(1).replace("p", "."))
        seed = int(m.group(2))
        need = ("isic_train", "isic_test") + OODS
        if not all((d / "{}.npz".format(n)).exists() for n in need):
            continue
        Fs = R.load_feats(d, need)
        met = R.run_metrics(Fs, ood_names=list(OODS))
        cells.setdefault(lam, {}).setdefault("_id", {})[seed] = float(met["id"]["bal_acc"])
        for o in OODS:
            cells[lam].setdefault(o, {})[seed] = float(met["ood"][o][DET])
    # bootstrap Rule A
    boots = {}
    for lam, blk in cells.items():
        for o in OODS:
            if o not in blk:
                continue
            feat = {}
            for seed, _au in blk[o].items():
                tag = "runB_orth1_ladv{}_s{}".format(R.lam_tag(lam), seed)
                feat[seed] = root / tag
            try:
                og, unit = R.ood_groups(o, data)
            except FileNotFoundError:
                continue
            # exclude degraded
            excl = {s for s, idb in blk.get("_id", {}).items() if idb < R.DEGRADED_ID_BAL}
            b = R.bootstrap_cell(feat, o, idg, og, dets=(DET,), exclude_seeds=excl or None)
            if b is not None:
                b["unit"] = unit
                boots[(lam, o)] = b
    # aggregate
    rows = []
    for lam in sorted(cells):
        for o in OODS:
            st = pm(list(cells[lam].get(o, {}).values()))
            b = boots.get((lam, o))
            k = None if b is None or DET not in b else "{}/{}".format(b[DET]["k_below"], b[DET]["n_seeds"])
            ra = None if b is None else b.get("rule_A")
            ci = None if b is None or DET not in b else (b[DET]["ci_lo"], b[DET]["ci_hi"])
            rows.append({"lambda": lam, "set": o, "stat": st, "k": k, "rule_A": ra, "ci": ci,
                         "latex": "{} & {} & {} & {} & {} & {} \\\\".format(
                             lam, o, latex_pm(st),
                             "[{:.3f}, {:.3f}]".format(*ci) if ci else "--",
                             k or "--",
                             ("met" if ra else "not met") if ra is not None else "--")})
    return {"rows": rows, "per_lam": {str(k): {o: pm(list(v.get(o, {}).values())) for o in OODS}
                                      for k, v in cells.items()}}


def c3_unseen():
    """DINOv2 dual (r7/t1) + A4 missing sets (mednode/sd198/scin) from p8 main."""
    out = {"dinov2": {}, "a4": {}}
    # DINOv2
    import csv
    path = R.RES / "r7" / "t1" / "summary.csv"
    by = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if not r["arm"].startswith("dual"):
                continue
            if r.get("degraded") in ("True", "true", "1"):
                continue
            lam = float(r["lambda_adv"])
            by.setdefault(lam, []).append(r)
    for lam, rows in by.items():
        out["dinov2"][str(lam)] = {
            "pad_heldout": pm([float(r["ood_auroc"]) for r in rows]),
            "fitzpatrick17k": pm([float(r["fitz_auroc"]) for r in rows]),
            "mednode": pm([float(r["maha_mednode"]) for r in rows]),
            "sd198": pm([float(r["maha_sd198"]) for r in rows]),
            "scin": pm([float(r["maha_scin"]) for r in rows]),
            "n": len(rows),
        }
    # A4 from p8 main summaries + bootstrap-ish seed means only (CIs from existing if needed)
    for lam in (0.0, 2.0):
        vals = {o: [] for o in ("pad_heldout", "fitzpatrick17k") + OODS}
        for s in R.SEEDS:
            sp = R.RES / "r5" / "p8" / "main" / "runs" / "main_ladv{}_s{}".format(R.lam_tag(lam), s) / "summary.json"
            if not sp.exists():
                continue
            m = json.loads(sp.read_text())["metrics"]
            if m.get("degraded") or m["id"]["bal_acc"] < R.DEGRADED_ID_BAL:
                continue
            for o in vals:
                vals[o].append(float(m["ood"][o][DET]))
        out["a4"][str(lam)] = {o: pm(v) for o, v in vals.items()}
    return out


def c5_single_dino():
    """Score MED-NODE/SD-198/SCIN on r7/t1b features."""
    root = R.RES / "r7" / "t1b" / "runs"
    per = {o: [] for o in OODS}
    leakage, idb, pad, fitz = [], [], [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "ladv2" not in d.name:
            continue
        feat = d / "features"
        need = ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k") + OODS
        if not all((feat / "{}.npz".format(n)).exists() for n in need):
            continue
        Fs = R.load_feats(feat, need + ("pad_full",) if (feat / "pad_full.npz").exists() else need)
        # ensure pad_full for leakage if present
        names = list(need)
        if (feat / "pad_full.npz").exists():
            names.append("pad_full")
        Fs = R.load_feats(feat, names)
        met = R.run_metrics(Fs, ood_names=list(OODS) + ["pad_heldout", "fitzpatrick17k"])
        if met.get("degraded") or met["id"]["bal_acc"] < R.DEGRADED_ID_BAL:
            continue
        idb.append(met["id"]["bal_acc"])
        if "leakage" in met:
            leakage.append(met["leakage"]["bal_acc_mean"])
        pad.append(met["ood"]["pad_heldout"][DET])
        fitz.append(met["ood"]["fitzpatrick17k"][DET])
        for o in OODS:
            per[o].append(met["ood"][o][DET])
    return {
        "leakage": pm(leakage), "id_bal": pm(idb),
        "pad_heldout": pm(pad), "fitzpatrick17k": pm(fitz),
        **{o: pm(v) for o, v in per.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="c2,c3,c5")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = [x.strip() for x in a.items.split(",") if x.strip()]
    data = R.load_data()
    idg = R.idx_groups(R.T.lesion_groups(data["test"])[0])
    res = {"commit": R.C.git_head()}
    if "c2" in items:
        print("C2 …", flush=True)
        res["c2"] = c2_dial(data, idg)
        (OUT / "C2_dial_extra.json").write_text(json.dumps(res["c2"], indent=1, default=float) + "\n")
    if "c3" in items:
        print("C3 …", flush=True)
        res["c3"] = c3_unseen()
        (OUT / "C3_unseen.json").write_text(json.dumps(res["c3"], indent=1, default=float) + "\n")
    if "c5" in items:
        print("C5 …", flush=True)
        res["c5"] = c5_single_dino()
        (OUT / "C5_single_dino.json").write_text(json.dumps(res["c5"], indent=1, default=float) + "\n")
    (OUT / "C2_C3_C5.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("wrote", OUT / "C2_C3_C5.json")


if __name__ == "__main__":
    main()
