#!/usr/bin/env python3
"""Class-aware held-out domain probes (a)/(b)/(c) — decisions/paperB_r6_class_aware_heldout.md.

(a) unweighted P6 V1; (b) P6 V3 class-reweighted; (c) within-class (≥20/side) size-weighted mean.
CPU on cached features. Resume skip_done; --report_only; --require_oracle_complete for headline.
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R6 / "class_aware_heldout"
REP = "z_lesion_norm"
PSEEDS = (42, 52, 62)
BOOT_N, BOOT_SEED = 2000, 20261017
MIN_PER_SIDE = 20
NEED = ("isic_test.npz", "pad_heldout.npz")

# (cell_key, family_label, feat_root, tag_regex)
CELLS = (
    ("r2_iwdan_l2", "R2 IWDAN λ=2", R.R6 / "r2" / "cur" / "features",
     r"r2_cur_iwdan_ladv2_s(\d+)$"),
    ("r2_iwcdan_l2", "R2 IWCDAN λ=2", R.R6 / "r2" / "cur" / "features",
     r"r2_cur_iwcdan_ladv2_s(\d+)$"),
    ("r2_oracle_l2", "R2 IWDAN_ORACLE λ=2", R.R6 / "r2" / "cur" / "features",
     r"r2_cur_iwdan_oracle_ladv2_s(\d+)$"),
    ("r1_shift0_l2", "R1 shift0% λ=2", R.R6 / "r1" / "cur" / "features",
     r"r1_cur_shift0_ladv2_s(\d+)$"),
    ("r1_shift100_l2", "R1 shift100% λ=2", R.R6 / "r1" / "cur" / "features",
     r"r1_cur_shift100_ladv2_s(\d+)$"),
)


def inventory():
    rows = []
    for cell, fam, root, pat in CELLS:
        if not root.is_dir():
            continue
        rx = re.compile(pat)
        for d in sorted(root.iterdir()):
            m = rx.fullmatch(d.name)
            if not m or not all((d / n).is_file() for n in NEED):
                continue
            rows.append({"cell": cell, "family": fam, "tag": d.name, "seed": int(m.group(1)),
                         "feat_root": str(root)})
    return rows


def fit_eval(Xtr, ytr, Xte, yte, s, wtr=None, wte=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
    clf.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    pred = clf.predict(Xte)
    return float(balanced_accuracy_score(yte, pred, sample_weight=wte))


def class_weights(y_isic, y_pad):
    p_pad = np.bincount(y_pad, minlength=8) / max(len(y_pad), 1)
    p_isic = np.bincount(y_isic, minlength=8) / max(len(y_isic), 1)
    return np.array([p_pad[c] / p_isic[c] if p_isic[c] > 0 else 0.0 for c in y_isic], float)


def _summarize(vals):
    a = np.asarray(vals, float)
    return {"bal_acc_mean": float(a.mean()), "bal_acc_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "bal_acc_per_probe_seed": [float(x) for x in a]}


def analyse_run(feat_root: Path, tag: str) -> dict:
    te = R.load(tag, "isic_test", feat_root=feat_root)
    hold = R.load(tag, "pad_heldout", feat_root=feat_root)
    gi, gp = R.id_test_groups(), R.pad_heldout_patients()
    assert len(gi) == len(te["labels"]) and len(gp) == len(hold["labels"])
    yi = te["labels"].astype(int)
    yh = hold["labels"].astype(int)
    Xi, Xh = te[REP], hold[REP]
    X = np.r_[Xi, Xh]
    d = np.r_[np.zeros(len(yi)), np.ones(len(yh))]
    g = np.r_[np.char.add("I", gi.astype(str)), np.char.add("P", gp.astype(str))]
    w_isic = class_weights(yi, yh)
    w = np.r_[w_isic, np.ones(len(yh))]

    a_vals, b_vals = [], []
    for s in PSEEDS:
        tr, te_ix = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
        a_vals.append(fit_eval(X[tr], d[tr], X[te_ix], d[te_ix], s))
        b_vals.append(fit_eval(X[tr], d[tr], X[te_ix], d[te_ix], s, wtr=w[tr], wte=w[te_ix]))

    # (c) within-class
    per_class = {}
    eligible = []
    for c in range(8):
        ni = int((yi == c).sum())
        nh = int((yh == c).sum())
        ok = ni >= MIN_PER_SIDE and nh >= MIN_PER_SIDE
        entry = {"n_isic": ni, "n_pad": nh, "eligible": ok}
        if not ok:
            per_class[str(c)] = entry
            continue
        mi = np.flatnonzero(yi == c)
        mh = np.flatnonzero(yh == c) + len(yi)
        idx = np.r_[mi, mh]
        Xc, dc, gc = X[idx], d[idx], g[idx]
        # need ≥2 groups on each domain for a split; else skip
        n_gi = len(set(gc[dc == 0].tolist()))
        n_gp = len(set(gc[dc == 1].tolist()))
        if n_gi < 2 or n_gp < 2:
            entry["eligible"] = False
            entry["skip_reason"] = "too_few_groups"
            per_class[str(c)] = entry
            continue
        c_vals = []
        for s in PSEEDS:
            tr, te_ix = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(Xc, dc, gc))
            c_vals.append(fit_eval(Xc[tr], dc[tr], Xc[te_ix], dc[te_ix], s))
        sm = _summarize(c_vals)
        entry.update(sm)
        entry["weight"] = float(ni + nh)
        per_class[str(c)] = entry
        eligible.append(entry)

    if eligible:
        wts = np.array([e["weight"] for e in eligible], float)
        means = np.array([e["bal_acc_mean"] for e in eligible], float)
        c_mean = float(np.average(means, weights=wts))
        # probe-seed weighted means for sd
        per_ps = []
        for j in range(len(PSEEDS)):
            mjs = np.array([e["bal_acc_per_probe_seed"][j] for e in eligible], float)
            per_ps.append(float(np.average(mjs, weights=wts)))
        c_sum = _summarize(per_ps)
        c_sum["bal_acc_mean"] = c_mean  # size-weighted over classes of probe-seed means' mean
        # prefer size-weighted mean of class means (matches commit text)
        c_sum = {"bal_acc_mean": c_mean,
                 "bal_acc_sd": float(np.sqrt(np.average((means - c_mean) ** 2, weights=wts))) if len(means) > 1 else 0.0,
                 "bal_acc_per_probe_seed": per_ps,
                 "n_eligible_classes": len(eligible),
                 "eligible_classes": [int(k) for k, v in per_class.items() if v.get("eligible")]}
    else:
        c_sum = {"bal_acc_mean": float("nan"), "bal_acc_sd": 0.0, "bal_acc_per_probe_seed": [],
                 "n_eligible_classes": 0, "eligible_classes": []}

    labels = list(getattr(R, "LABELS", None) or ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"])
    return {
        "rep": REP,
        "unweighted_a": _summarize(a_vals),
        "class_balanced_b": _summarize(b_vals),
        "within_class_c": c_sum,
        "per_class": per_class,
        "class_names": labels,
        "min_per_side": MIN_PER_SIDE,
        "n": {"isic_test": int(len(yi)), "pad_heldout": int(len(yh))},
    }


def seed_mean_ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 1:
        return {"mean": float("nan"), "sd": 0.0, "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if len(v) < 2:
        return {"mean": float(v[0]), "sd": 0.0, "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 1}
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT_N)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(v))}


def run_available():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir(exist_ok=True)
    inv = inventory()
    R.write_json(OUT / "inventory.json", {"n": len(inv), "runs": inv, "commit": R.git_head()})
    for meta in inv:
        key = f"{meta['cell']}__{meta['tag']}"
        outp = OUT / "per_run" / f"{key}.json"
        if outp.is_file():
            continue
        print(f"[class_aware] {key}", flush=True)
        rec = {**meta, **analyse_run(Path(meta["feat_root"]), meta["tag"])}
        R.write_json(outp, rec)
    return inv


def _load_rows():
    return [json.loads(p.read_text()) for p in sorted((OUT / "per_run").glob("*.json"))]


def _fmt(st):
    if st["n"] < 1 or not np.isfinite(st["mean"]):
        return "—"
    if st["n"] < 3 or not np.isfinite(st["ci_lo"]):
        return "{:.3f} ± {:.3f} (n={})".format(st["mean"], st["sd"], st["n"])
    return "{:.3f} ± {:.3f} [{:.3f}, {:.3f}] (n={})".format(
        st["mean"], st["sd"], st["ci_lo"], st["ci_hi"], st["n"])


def report(require_oracle_complete=False):
    rows = _load_rows()
    if not rows:
        print("no per_run yet", flush=True)
        return False
    flat = []
    for r in rows:
        flat.append({
            "cell": r["cell"], "family": r["family"], "tag": r["tag"], "seed": r["seed"],
            "a": r["unweighted_a"]["bal_acc_mean"],
            "b": r["class_balanced_b"]["bal_acc_mean"],
            "c": r["within_class_c"]["bal_acc_mean"],
            "n_elig": r["within_class_c"].get("n_eligible_classes"),
        })
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "summary.csv", index=False)

    oracle_seeds = sorted(df[df.cell == "r2_oracle_l2"].seed.unique())
    oracle_complete = set(oracle_seeds) >= {42, 52, 62, 72, 82}
    if require_oracle_complete and not oracle_complete:
        print("oracle cell incomplete; seeds=", oracle_seeds, flush=True)
        return False

    lines = [
        "# Paper B — class-aware held-out domain probes",
        "",
        f"Design: `decisions/paperB_r6_class_aware_heldout.md`. Commit `{R.git_head()}`.",
        "Pool: ISIC-test ∪ pad_heldout, GroupShuffleSplit 70/30 (lesion/patient). Rep: `z_lesion_norm`.",
        "**(a)** unweighted (P6 V1); **(b)** class-balanced (P6 V3); **(c)** within-class "
        f"(≥{MIN_PER_SIDE}/side), class-size-weighted mean.",
        "",
        f"Oracle λ=2 seeds on disk: {oracle_seeds}; complete={oracle_complete}.",
        "",
        "| cell | n | (a) unweighted | (b) class-balanced V3 | (c) within-class |",
        "|---|---:|---|---|---|",
    ]
    cells_out = []
    for cell, fam, _root, _pat in CELLS:
        g = df[df.cell == cell]
        if g.empty:
            lines.append(f"| {fam} | 0 | — | — | — |")
            continue
        sa, sb, sc = seed_mean_ci(g["a"]), seed_mean_ci(g["b"]), seed_mean_ci(g["c"])
        note = ""
        if cell == "r2_oracle_l2" and not oracle_complete:
            note = " *(partial — waiting s82)*"
        lines.append("| {}{} | {} | {} | {} | {} |".format(
            fam, note, len(g), _fmt(sa), _fmt(sb), _fmt(sc)))
        cells_out.append({"cell": cell, "family": fam, "n": int(len(g)),
                          "a": sa, "b": sb, "c": sc, "oracle_complete": oracle_complete})

    # per-seed oracle table when any exist
    go = df[df.cell == "r2_oracle_l2"].sort_values("seed")
    if len(go):
        lines += ["", "## Oracle λ=2 per seed", "",
                  "| seed | (a) | (b) | (c) |", "|---:|---:|---:|---:|"]
        for _, r in go.iterrows():
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(int(r.seed), r.a, r.b, r.c))

    lines += ["", f"**On origin/paperb/reviewer-r1:** _(fill after push)._", ""]
    R.write_json(OUT / "cells.json", cells_out)
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "REPORT.md", "oracle_complete=", oracle_complete, flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--require_oracle_complete", action="store_true",
                    help="exit 1 if oracle λ=2 does not have all 5 seeds")
    a = ap.parse_args()
    if not a.report_only:
        run_available()
    ok = report(require_oracle_complete=a.require_oracle_complete)
    if a.require_oracle_complete and not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
