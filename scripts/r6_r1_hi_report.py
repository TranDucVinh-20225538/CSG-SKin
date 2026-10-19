#!/usr/bin/env python3
"""Paper B R6 R1-hi report (decisions/paperB_r6_r1_hi.md).

Usage:
  r6_r1_hi_report.py                 # full report from completed runs
  r6_r1_hi_report.py --gate_check    # print JSON: need_advlr3 bool (λ=8 gate empty)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as C5  # noqa: E402
import r6_common as R6  # noqa: E402

OUT = R6.RES / "r6" / "r1_hi" / "cur"
R1_CUR = R6.RES / "r6" / "r1" / "cur"
REP = "z_lesion_norm"
PSEEDS = (42, 52, 62)
ID_BAL_MIN = 0.25
GATE_MLP = 0.55
OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "pad_full", "scin")
UNSEEN3 = ("fitzpatrick17k", "mednode", "sd198")
DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max", "MSP", "Energy_T1")


def _bal(y, pred):
    return float(balanced_accuracy_score(y, pred))


def heldout_probes(feat_dir: Path) -> dict:
    te = dict(np.load(feat_dir / "isic_test.npz"))
    hold = dict(np.load(feat_dir / "pad_heldout.npz"))
    full = dict(np.load(feat_dir / "pad_full.npz"))
    rk = REP if REP in te else "z"
    gi, gp = C5.id_test_groups(), C5.pad_heldout_patients()
    Xh = np.r_[te[rk], hold[rk]]
    dh = np.r_[np.zeros(len(te["labels"])), np.ones(len(hold["labels"]))]
    gh = np.r_[np.char.add("I", gi.astype(str)), np.char.add("P", gp.astype(str))]
    Xp = np.r_[te[rk], full[rk]]
    dp = np.r_[np.zeros(len(te["labels"])), np.ones(len(full["labels"]))]
    lin, mlp, knn, pf = [], [], [], []
    for s in PSEEDS:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(Xh, dh, gh))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(Xh[a], dh[a])
        lin.append(_bal(dh[b], clf.predict(Xh[b])))
        m = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=200, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=10, random_state=s))
        m.fit(Xh[a], dh[a])
        mlp.append(_bal(dh[b], m.predict(Xh[b])))
        k = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=20))
        k.fit(Xh[a], dh[a])
        knn.append(_bal(dh[b], k.predict(Xh[b])))
        atr, ate, ytr, yte = train_test_split(Xp, dp, test_size=0.3, random_state=s, stratify=dp)
        clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf2.fit(atr, ytr)
        pf.append(_bal(yte, clf2.predict(ate)))
    def sm(v):
        a = np.asarray(v, float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}
    return {"heldout_linear": sm(lin), "heldout_mlp": sm(mlp), "heldout_knn20": sm(knn),
            "pad_full_linear": sm(pf), "heldout_mlp_mean": float(np.mean(mlp))}


_PAD_PAT = re.compile(r"PAT_(\d+)_(\d+)_(\d+)")


def _pad_patient_ids_from_master(n: int) -> np.ndarray:
    """Master pad_ufes rows have no patient_id column; parse PAT_<id>_… from path."""
    df = C5.T.p2.load_master()
    pad = df[df.domain == "pad_ufes"].reset_index(drop=True)
    g = []
    for i, p in enumerate(pad.path.astype(str)):
        m = _PAD_PAT.search(p)
        g.append(m.group(1) if m else "row{}".format(i))
    g = np.asarray(g, dtype=str)
    if len(g) != n:
        return np.arange(n).astype(str)
    return g


def ood_groups(name: str, n: int):
    if name == "pad_heldout":
        g = C5.pad_heldout_patients()
        return g if len(g) == n else np.arange(n).astype(str)
    if name == "pad_full":
        return _pad_patient_ids_from_master(n)
    if name == "fitzpatrick17k":
        try:
            return C5.fitz_groups("cluster_union")
        except Exception:  # noqa: BLE001
            return C5.fitz_groups("image")
    csv = C5.R5 / "p7a" / "sets" / f"{name}.csv"
    if csv.is_file():
        g = pd.read_csv(csv, keep_default_na=False).group.astype(str).to_numpy()
        if len(g) == n:
            return g
    return np.arange(n).astype(str)


def score_run(feat_dir: Path) -> dict:
    tr = dict(np.load(feat_dir / "isic_train.npz"))
    te = dict(np.load(feat_dir / "isic_test.npz"))
    oods = {}
    for o in OODS:
        p = feat_dir / f"{o}.npz"
        if p.is_file():
            oods[o] = dict(np.load(p))
    scores, _ = C5.detector_scores(tr, te, oods, rep=REP if REP in tr else "z")
    gid = C5.id_test_groups()
    pre = C5.prereg()
    n_boot, rng0 = int(pre["n_resamples"]), int(pre["rng_seed"])
    out = {}
    for det in DETS:
        sid, so = scores[det]
        out[det] = {}
        for o, s_ood in so.items():
            point = C5.auroc(sid, s_ood)
            lo, hi = C5.cluster_ci(sid, gid, s_ood, ood_groups(o, len(s_ood)), n=n_boot, seed=rng0)
            out[det][o] = {"point": point, "ci_lo": lo, "ci_hi": hi,
                           "ci_lo_gt_0.5": bool(lo > 0.5), "ci_hi_lt_0.5": bool(hi < 0.5)}
    return out


def r1_shift0_l2_id_bal():
    """{seed: id_bal} from R1 cur shift 0% λ=2."""
    out = {}
    csvp = R1_CUR / "summary.csv"
    if csvp.is_file():
        for r in pd.read_csv(csvp).to_dict("records"):
            if str(r.get("cell")) == "shift 0%" and float(r.get("lambda")) == 2.0:
                out[int(r["seed"])] = float(r["id_bal"])
    if out:
        return out
    for p in sorted((R1_CUR / "runs").glob("r1_cur_shift0_ladv2_s*/summary.json")):
        s = json.loads(p.read_text())
        out[int(s["seed"])] = float(s["metrics"]["id"]["bal_acc"])
    return out


def load_runs():
    rows = []
    for p in sorted((OUT / "runs").glob("*/summary.json")):
        s = json.loads(p.read_text())
        tag = s["tag"]
        feat = OUT / "features" / tag
        if not (feat / "isic_test.npz").is_file():
            continue
        id_bal = float(s["metrics"]["id"]["bal_acc"])
        degraded = id_bal < ID_BAL_MIN
        probes = heldout_probes(feat)
        dets = score_run(feat)
        gate_pass = (not degraded) and probes["heldout_mlp_mean"] <= GATE_MLP
        rows.append({
            "tag": tag, "lambda": float(s["lambda_adv"]), "seed": int(s["seed"]),
            "wave": s.get("wave") or s.get("extra", {}).get("wave") or (
                "advlr3" if "advlr3" in tag else "main"),
            "adv_lr_multiplier": float(s.get("adv_lr_multiplier") or
                                       (s.get("extra") or {}).get("adv_lr_multiplier", 30.0)),
            "id_bal": id_bal, "degraded": degraded,
            "heldout_mlp": probes["heldout_mlp_mean"],
            "gate_pass": gate_pass,
            "probes": probes, "detectors": dets,
            "summary_path": str(p),
        })
    return rows


def cell_key(r):
    if r["wave"] == "advlr3":
        return ("advlr3", 4.0)
    return ("main", r["lambda"])


def analyse(rows):
    base = r1_shift0_l2_id_bal()
    for r in rows:
        b = base.get(r["seed"])
        r["id_bal_r1_shift0_l2"] = b
        r["id_bal_drop_vs_r1_l2"] = None if b is None else float(r["id_bal"] - b)

    cells = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)

    cell_out = []
    for (wave, lam), rs in sorted(cells.items()):
        nd = [x for x in rs if not x["degraded"]]
        gated = [x for x in nd if x["gate_pass"]]
        # pooled Maha CIs over gated seeds for UNSEEN3
        inv_sets, noninv_sets = [], []
        for ood in UNSEEN3:
            pts, los, his = [], [], []
            for x in gated:
                d = x["detectors"]["mahalanobis_classcond"].get(ood)
                if not d:
                    continue
                pts.append(d["point"]); los.append(d["ci_lo"]); his.append(d["ci_hi"])
            if not pts:
                continue
            # cell: mean point; CI = percentile of seed-mean bootstrap of points is heavy —
            # use: count how many gated seeds have ci_lo>0.5 / ci_hi<0.5; also mean of ci_lo
            mean_pt = float(np.mean(pts))
            mean_lo, mean_hi = float(np.mean(los)), float(np.mean(his))
            # primary: CI lo > 0.5 — use mean of per-seed ci_lo as cell summary + require ≥2 sets
            noninv = mean_lo > 0.5
            invert = mean_hi < 0.5
            if noninv:
                noninv_sets.append(ood)
            if invert:
                inv_sets.append(ood)
            # also store per-seed detail
        primary_ok = len(noninv_sets) >= 2
        opposite = len(inv_sets) >= 2
        id_vals = [x["id_bal"] for x in nd] or [x["id_bal"] for x in rs]
        drops = [x["id_bal_drop_vs_r1_l2"] for x in nd if x["id_bal_drop_vs_r1_l2"] is not None]
        cell_out.append({
            "wave": wave, "lambda": lam, "n_seeds": len(rs),
            "n_degraded": sum(x["degraded"] for x in rs),
            "n_non_degraded": len(nd),
            "n_gate_pass": len(gated),
            "gate_seeds": [x["seed"] for x in gated],
            "id_bal_mean": float(np.mean(id_vals)) if id_vals else float("nan"),
            "id_bal_sd": float(np.std(id_vals, ddof=1)) if len(id_vals) > 1 else 0.0,
            "id_bal_drop_vs_r1_l2_mean": float(np.mean(drops)) if drops else float("nan"),
            "primary_no_inversion": primary_ok if gated else None,
            "opposite_inversion_sufficient": opposite if gated else None,
            "noninv_sets": noninv_sets, "invert_sets": inv_sets,
            "seeds": [{
                "seed": x["seed"], "id_bal": x["id_bal"], "degraded": x["degraded"],
                "heldout_mlp": x["heldout_mlp"], "gate_pass": x["gate_pass"],
                "id_bal_drop_vs_r1_l2": x["id_bal_drop_vs_r1_l2"],
                "probes": x["probes"],
                "maha_unseen": {o: x["detectors"]["mahalanobis_classcond"].get(o) for o in UNSEEN3},
            } for x in sorted(rs, key=lambda z: z["seed"])],
        })
    return cell_out


def need_advlr3(cells):
    c8 = next((c for c in cells if c["wave"] == "main" and c["lambda"] == 8.0), None)
    if c8 is None:
        return True  # λ=8 not scored yet → caller should not launch; treat as unknown
    return c8["n_gate_pass"] == 0


def write_report(rows, cells):
    OUT.mkdir(parents=True, exist_ok=True)
    C5.write_json(OUT / "per_run.json", rows)
    C5.write_json(OUT / "cells.json", cells)
    flat = []
    for r in rows:
        flat.append({
            "tag": r["tag"], "wave": r["wave"], "lambda": r["lambda"], "seed": r["seed"],
            "id_bal": r["id_bal"], "degraded": r["degraded"], "heldout_mlp": r["heldout_mlp"],
            "gate_pass": r["gate_pass"], "id_bal_drop_vs_r1_l2": r["id_bal_drop_vs_r1_l2"],
            "heldout_linear": r["probes"]["heldout_linear"]["mean"],
            "heldout_knn20": r["probes"]["heldout_knn20"]["mean"],
            "pad_full_linear": r["probes"]["pad_full_linear"]["mean"],
            **{f"maha_{o}": (r["detectors"]["mahalanobis_classcond"].get(o) or {}).get("point")
               for o in UNSEEN3},
        })
    pd.DataFrame(flat).to_csv(OUT / "summary.csv", index=False)

    c8 = next((c for c in cells if c["wave"] == "main" and c["lambda"] == 8.0), None)
    c4 = next((c for c in cells if c["wave"] == "main" and c["lambda"] == 4.0), None)
    c3 = next((c for c in cells if c["wave"] == "advlr3"), None)
    not_reached = False
    if c8 and c8["n_gate_pass"] == 0:
        if c3 is None:
            fallback_note = "λ=8 gate empty → adversary LR ×3 at λ=4 required (not yet run)."
        elif c3["n_gate_pass"] == 0:
            not_reached = True
            fallback_note = "held-out invariance without label shift not reached"
        else:
            fallback_note = "λ=8 gate empty; adv-LR×3 λ=4 produced gated seeds."
    else:
        fallback_note = "λ=8 gate met; adv-LR×3 not run."

    lines = [
        "# Paper B R6 R1-hi — high-λ shift 0% (RECIPE=current, last.ckpt)",
        "",
        f"Design: `decisions/paperB_r6_r1_hi.md`. Commit `{C5.git_head()}`.",
        "**On origin/paperb/reviewer-r1:** _(fill after push)._",
        "",
        "Degraded: ID bal < 0.25 (flagged, excluded). Gate: held-out MLP ≤ 0.55 (P6 V1).",
        "Predictions apply to gated, non-degraded seeds only.",
        "",
        f"Fallback status: {fallback_note}",
        "",
    ]
    if not_reached:
        lines += ["> **held-out invariance without label shift not reached**", ""]

    lines += [
        "## Cells",
        "",
        "| wave | λ | n | degraded | gate pass | ID bal | ΔID vs R1 shift0 λ2 | primary no-inv | opposite inv |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for c in cells:
        def fmt(x):
            return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (
                f"{x:.3f}" if isinstance(x, float) else str(x))
        lines.append(
            f"| {c['wave']} | {c['lambda']} | {c['n_seeds']} | {c['n_degraded']} | "
            f"{c['n_gate_pass']}/{c['n_non_degraded']} | "
            f"{fmt(c['id_bal_mean'])} ± {fmt(c['id_bal_sd'])} | "
            f"{fmt(c['id_bal_drop_vs_r1_l2_mean'])} | "
            f"{c['primary_no_inversion']} | {c['opposite_inversion_sufficient']} |"
        )

    for c in cells:
        lines += [f"", f"### wave={c['wave']} λ={c['lambda']}", ""]
        lines += [
            "| seed | ID bal | Δ vs R1 λ2 | deg | heldout lin/MLP/kNN | pad_full lin | gate | "
            "Maha Fitz [CI] | Maha MED [CI] | Maha SD [CI] |",
            "|---:|---|---|---|---|---|---|---|---|---|",
        ]
        for s in c["seeds"]:
            pr = s["probes"]
            def maha(o):
                d = s["maha_unseen"].get(o) or {}
                if not d:
                    return "—"
                return f"{d['point']:.3f} [{d['ci_lo']:.3f},{d['ci_hi']:.3f}]"
            drop = s["id_bal_drop_vs_r1_l2"]
            lines.append(
                f"| {s['seed']} | {s['id_bal']:.3f} | "
                f"{'—' if drop is None else f'{drop:+.3f}'} | {s['degraded']} | "
                f"{pr['heldout_linear']['mean']:.3f}/{pr['heldout_mlp']['mean']:.3f}/"
                f"{pr['heldout_knn20']['mean']:.3f} | {pr['pad_full_linear']['mean']:.3f} | "
                f"{s['gate_pass']} | {maha('fitzpatrick17k')} | {maha('mednode')} | {maha('sd198')} |"
            )
        if c["gate_seeds"]:
            lines.append(
                f"Gated seeds: {c['gate_seeds']}. "
                f"Non-inverting sets (≥ CI lo>0.5): {c['noninv_sets']}; "
                f"inverting sets (CI hi<0.5): {c['invert_sets']}."
            )

    lines += [
        "",
        "## Detectors",
        "",
        "All five detectors with cluster CIs are in `per_run.json` → `detectors`.",
        "Primary prediction / opposite reading use Mahalanobis on Fitz / MED-NODE / SD-198 only.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "REPORT.md", flush=True)
    return {"need_advlr3": bool(c8 and need_advlr3(cells) and c3 is None),
            "not_reached": not_reached, "n_runs": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_check", action="store_true")
    a = ap.parse_args()
    rows = load_runs()
    cells = analyse(rows) if rows else []
    if a.gate_check:
        c8 = next((c for c in cells if c["wave"] == "main" and c["lambda"] == 8.0), None)
        main_done = c8 is not None and c8["n_seeds"] >= 5
        need = bool(main_done and c8["n_gate_pass"] == 0)
        print(json.dumps({"main_done": main_done, "need_advlr3": need,
                          "n_gate_pass_l8": None if c8 is None else c8["n_gate_pass"]}))
        return
    if not rows:
        print("no completed R1-hi runs under", OUT, flush=True)
        return
    write_report(rows, cells)


if __name__ == "__main__":
    main()
