#!/usr/bin/env python3
"""Paper B R5 P6: which images the leakage probe is trained / tested on, and the probe recomputed on images the
adversary never saw (decisions/paperB_r5_p6.md). CPU, cached features of the 35 lesion-level runs.

Table 1 leakage (train_phase3_sweep.leakage_probe): logistic regression (StandardScaler, max_iter 4000), 70/30
stratified split of ISIC-test ∪ PAD-full, 3 seeds (42/52/62), balanced accuracy. PAD-full = pad_adv (1,582 images the
adversary was trained on) ∪ pad_heldout (716). The adversary's ISIC side is ISIC train; ISIC test is never seen by it.

Variants (same classifier, same 3 probe seeds):
  V0  replication: ISIC-test ∪ PAD-full, 70/30 image-level stratified split (checks against summary.json)
  V1  held-out only: ISIC-test ∪ pad_heldout, 70/30 split grouped by lesion (ISIC) / patient (PAD)
  V2  train on adversary images, test on held-out images: fit on ISIC-train (subsample, n = 5,067) ∪ pad_adv,
      evaluate on ISIC-test ∪ pad_heldout
  V3  V1 with the ISIC side reweighted to the pad_heldout class mix over the six shared classes (sample weights in fit and
      in the balanced accuracy), so class composition cannot carry the domain signal
Representations: z_lesion_norm (Table 1), z_lesion, backbone_raw_lesion, z_context.
Usage: r5_p6_leakage_probe.py [--report_only]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "p6"
REPS = ("z_lesion_norm", "z_lesion", "backbone_raw_lesion", "z_context")
PSEEDS = (42, 52, 62)


def pad_heldout_mask():
    """Boolean over pad_full rows (load_master order) marking pad_heldout images."""
    spec = json.loads(R.T.p3.P2_SPLIT.read_text())
    df = R.T.p2.load_master()
    pad = df[df.domain == "pad_ufes"].reset_index(drop=True)
    m = pad.path.isin(set(spec["pad_heldout_paths"])).to_numpy()
    assert m.sum() == 716 and (~m).sum() == 1582
    return m


def fit_eval(Xtr, ytr, Xte, yte, s, wtr=None, wte=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
    clf.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    pred, prob = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
    return {"bal_acc": float(balanced_accuracy_score(yte, pred, sample_weight=wte)),
            "auroc": float(roc_auc_score(yte, prob, sample_weight=wte))}


def class_weights(y_isic, y_pad):
    p_pad = np.bincount(y_pad, minlength=8) / len(y_pad)
    p_isic = np.bincount(y_isic, minlength=8) / len(y_isic)
    return np.array([p_pad[c] / p_isic[c] if p_isic[c] > 0 else 0.0 for c in y_isic])


def analyse_run(lam, seed, tag, hmask):
    tr, te, full, hold = (R.load(tag, s) for s in ("isic_train", "isic_test", "pad_full", "pad_heldout"))
    dev = {rep: float(np.abs(full[rep][hmask] - hold[rep]).max()) for rep in REPS}
    for rep in REPS:
        assert np.allclose(full[rep][hmask], hold[rep], atol=1e-2), ("pad_full / pad_heldout row mapping", rep, dev[rep])
    assert np.array_equal(full["labels"][hmask].astype(int), hold["labels"].astype(int))
    gi, gp = R.id_test_groups(), R.pad_heldout_patients()
    yi, yh, yadv = te["labels"].astype(int), hold["labels"].astype(int), full["labels"][~hmask].astype(int)
    rec = {"lambda": lam, "seed": seed, "run": tag, "n": {"isic_test": len(yi), "pad_full": len(full["labels"]),
                                                          "pad_heldout": len(yh), "pad_adv": int((~hmask).sum())},
           "pad_full_vs_heldout_max_abs_dev": dev}
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(tr["labels"]), size=len(yi), replace=False)
    for rep in REPS:
        r = {}
        X0 = np.r_[te[rep], full[rep]]
        d0 = np.r_[np.zeros(len(yi)), np.ones(len(full[rep]))]
        r["V0"] = [fit_eval(*_split(X0, d0, s), s) for s in PSEEDS]
        X1 = np.r_[te[rep], hold[rep]]
        d1 = np.r_[np.zeros(len(yi)), np.ones(len(yh))]
        g1 = np.r_[np.char.add("I", gi.astype(str)), np.char.add("P", gp.astype(str))]
        v1, v3 = [], []
        w_isic = class_weights(yi, yh)
        w1 = np.r_[w_isic, np.ones(len(yh))]
        for s in PSEEDS:
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X1, d1, g1))
            v1.append(fit_eval(X1[a], d1[a], X1[b], d1[b], s))
            v3.append(fit_eval(X1[a], d1[a], X1[b], d1[b], s, wtr=w1[a], wte=w1[b]))
        r["V1"], r["V3"] = v1, v3
        X2tr = np.r_[tr[rep][sub], full[rep][~hmask]]
        d2tr = np.r_[np.zeros(len(sub)), np.ones(int((~hmask).sum()))]
        r["V2"] = [fit_eval(X2tr, d2tr, X1, d1, s) for s in PSEEDS]
        rec[rep] = {k: {"bal_acc_mean": float(np.mean([x["bal_acc"] for x in v])), "bal_acc_sd": float(np.std([x["bal_acc"] for x in v], ddof=1)),
                        "auroc_mean": float(np.mean([x["auroc"] for x in v]))} for k, v in r.items()}
    summ = json.loads((R.C.R2 / "item4" / "runs" / tag / "summary.json").read_text())
    rec["stored_table1_leakage_z_lesion"] = summ["leakage"]["z_lesion"]["bal_acc_mean"]
    rec["stored_z_context"] = summ["leakage"]["z_context"]["bal_acc_mean"]
    rec["ood_counts_pad_heldout"] = np.bincount(yh, minlength=8).tolist()
    rec["ood_counts_pad_adv"] = np.bincount(yadv, minlength=8).tolist()
    return rec


def _split(X, d, s):
    a, b, c, e = train_test_split(X, d, test_size=0.3, random_state=s, stratify=d)
    return a, c, b, e


def report():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "per_run").glob("*.json"))]
    flat = []
    for r in rows:
        for rep in REPS:
            for v, b in r[rep].items():
                flat.append({"lambda": r["lambda"], "seed": r["seed"], "rep": rep, "variant": v, **b,
                             "stored": r["stored_table1_leakage_z_lesion"] if rep == "z_lesion_norm" else None})
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "summary.csv", index=False)
    lams = sorted(df["lambda"].unique())
    L = ["# Paper B R5 P6 — leakage probe: training/test images and held-out recomputation", "",
         "Design: `decisions/paperB_r5_p6.md`. Commit `{}`. Balanced accuracy of a logistic domain probe, mean over 3 probe "
         "seeds, then mean ± s.d. over training seeds. Chance = 0.5.".format(R.git_head()), "",
         "## Which images (Table 1 protocol)", "",
         "| set | n | seen by adversary | seen by classifier | in Table 1 probe |", "|---|---:|---|---|---|",
         "| ISIC train (lesion-level) | 16,211 | yes (ISIC side) | yes | no |",
         "| ISIC test | 5,067 | no | no | yes (70/30 split) |",
         "| pad_adv | 1,582 | yes (PAD side) | no (labels unused) | yes (69 % of PAD-full) |",
         "| pad_heldout | 716 | no | no | yes (31 % of PAD-full) |", ""]
    tex = []
    for rep in REPS:
        L += ["## " + rep, "", "| λ | n | stored Table 1 | V0 replication | V1 held-out only | V2 adv-train → held-out | V3 V1 class-reweighted |",
              "|---:|---:|---|---|---|---|---|"]
        for lam in lams:
            d = df[(df.rep == rep) & (df["lambda"] == lam)]
            g = lambda v: R.fmt(d[d.variant == v].bal_acc_mean.mean(), d[d.variant == v].bal_acc_mean.std(ddof=1))  # noqa: E731
            st = d[d.variant == "V0"].stored.dropna()
            L.append("| {:g} | {} | {} | {} | {} | {} | {} |".format(lam, d.seed.nunique(), R.fmt(st.mean(), st.std(ddof=1)) if len(st) else "—",
                                                                g("V0"), g("V1"), g("V2"), g("V3")))
            if rep == "z_lesion_norm":
                tex.append("{:g} & {} & {} & {} & {} \\\\".format(lam, *(R.tex_pm(d[d.variant == v].bal_acc_mean.mean(), d[d.variant == v].bal_acc_mean.std(ddof=1))
                                                                    for v in ("V0", "V1", "V2", "V3"))))
        L.append("")
    d = df[df.rep == "z_lesion_norm"]
    m = lambda v, l: d[(d.variant == v) & (d["lambda"] == l)].bal_acc_mean.mean()  # noqa: E731
    L += ["## Plain reading", "", "On z_lesion^norm the Table 1 protocol (V0) gives {:.3f} at λ = 0 and {:.3f} at λ = 2. On images the "
          "adversary never saw (V1) it gives {:.3f} and {:.3f}; a probe fitted on the adversary's own images and tested on held-out "
          "images (V2) gives {:.3f} and {:.3f}; removing class composition (V3) gives {:.3f} and {:.3f}.".format(
              m("V0", 0.0), m("V0", 2.0), m("V1", 0.0), m("V1", 2.0), m("V2", 0.0), m("V2", 2.0), m("V3", 0.0), m("V3", 2.0)), "",
          "## LaTeX rows (λ & V0 & V1 & V2 & V3), z_lesion^norm", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    (OUT / "per_run").mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        hmask = pad_heldout_mask()
        rr = R.runs(need=("isic_train", "isic_test", "pad_full", "pad_heldout"))
        assert len(rr) == 35, len(rr)
        for lam, seed, tag in rr:
            dest = OUT / "per_run" / "{}.json".format(tag)
            if not dest.exists():
                R.write_json(dest, analyse_run(lam, seed, tag, hmask))
                print("done", tag, flush=True)
    report()


if __name__ == "__main__":
    main()
