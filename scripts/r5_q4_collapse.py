#!/usr/bin/env python3
"""Paper B R5 Q4: a label-free collapse statistic computed from ID data only, and its rank correlation with OOD AUROC
across every run in the manifest (decisions/paperB_r5_q4.md). CPU, cached features.

Statistics (per run, scoring representation of the manifest; fit on ID train, evaluated on ID test; no OOD data and no
OOD labels; ID test labels are not used either):
  S1  maj_share_excess   share of ID-test points whose nearest class mean (pooled-covariance Mahalanobis, the detector's
                         own geometry) is the majority class, minus the majority prior in ID train      [PRIMARY]
  S1r maj_share          the share itself
  S2  fisher_ratio       tr(S_between) / tr(S_within) of ID train in the representation (class separation)
  S3  part_ratio         participation ratio of the ID-test covariance eigenvalues / dimension (effective rank)
  S4  pred_maj_excess    share of ID-test argmax predictions = majority, minus the train prior
Outcome: Mahalanobis OOD AUROC on the run's primary OOD set (and every listed OOD set), same features.
Analysis: Spearman ρ across ready runs with a 95% run-bootstrap CI (2,000, seed 20261005); within family; partial ρ given λ
(rank residuals); leave-one-group-out threshold transfer for "AUROC < 0.5" (threshold = Youden-optimal on the other groups).
Groups: task when ≥ 2 tasks are ready, otherwise family (until Q1 exists this is leave-one-protocol/family-out).
Usage: r5_q4_collapse.py [--manifest results/paperB/r5/q4/manifest.csv] [--holdout_by task|family|protocol]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "q4"
STATS = ("maj_share_excess", "maj_share", "fisher_ratio", "part_ratio", "pred_maj_excess")
PRIMARY = "maj_share_excess"


class PooledMaha:
    def __init__(self, ztr, ytr, n_classes):
        z, y = np.asarray(ztr, np.float64), np.asarray(ytr, int)
        self.present = np.array([c for c in range(n_classes) if (y == c).any()])
        self.means = np.stack([z[y == c].mean(0) for c in self.present])
        cen = z - self.means[np.searchsorted(self.present, y)]
        cov = cen.T @ cen / max(len(z) - len(self.present), 1) + 1e-3 * np.eye(z.shape[1])
        self.P = np.linalg.inv(cov)

    def d2(self, z):
        z = np.asarray(z, np.float64)
        zp, mp = z @ self.P, self.means @ self.P
        return np.maximum(np.einsum("nd,nd->n", zp, z)[:, None] + np.einsum("cd,cd->c", mp, self.means)[None] - 2.0 * zp @ self.means.T, 0)

    def nearest(self, z):
        return self.present[self.d2(z).argmin(1)]

    def score(self, z):
        return self.d2(z).min(1)


def collapse_stats(ztr, ytr, z_eval, logits_eval, majority, n_classes, maha=None):
    """Label-free (on z_eval) collapse statistics; ID train labels define the class means and the prior."""
    ytr = np.asarray(ytr, int)
    m = maha or PooledMaha(ztr, ytr, n_classes)
    prior = float((ytr == majority).mean())
    share = float((m.nearest(z_eval) == majority).mean())
    z = np.asarray(ztr, np.float64)
    mu = z.mean(0)
    sb = sum(float((ytr == c).sum()) * np.sum((m.means[i] - mu) ** 2) for i, c in enumerate(m.present))
    sw = float(sum(np.sum((z[ytr == c] - m.means[i]) ** 2) for i, c in enumerate(m.present)))
    ev = np.clip(np.linalg.eigvalsh(np.cov(np.asarray(z_eval, np.float64), rowvar=False)), 0, None)
    pr = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30) / len(ev))
    out = {"maj_share": share, "maj_share_excess": share - prior, "fisher_ratio": sb / max(sw, 1e-30), "part_ratio": pr,
           "train_prior_majority": prior}
    if logits_eval is not None:
        out["pred_maj_excess"] = float((np.asarray(logits_eval).argmax(1) == majority).mean()) - prior
    return out


def analyse_run(r):
    d = Path(r["feat_dir"])
    ld = lambda s: dict(np.load(d / "{}.npz".format(s)))  # noqa: E731
    tr, te = ld(r["id_train"]), ld(r["id_test"])
    rep, nc, maj = r["rep"], int(r["n_classes"]), int(r["majority"])
    m = PooledMaha(tr[rep], tr["labels"], nc)
    rec = {k: r[k] for k in ("run_id", "task", "protocol", "family", "lambda", "seed")}
    rec.update(collapse_stats(tr[rep], tr["labels"], te[rep], te.get("logits"), maj, nc, m))
    sid = m.score(te[rep])
    for o in str(r["oods"]).split(";"):
        rec["maha_auroc|" + o] = R.auroc(sid, m.score(ld(o)[rep]))
    rec["ood_auroc_primary"] = rec["maha_auroc|" + r["primary_ood"]]
    return rec


def boot_rho(x, y, n=2000, seed=20261005):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2:
            v.append(spearmanr(x[i], y[i])[0])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"), float("nan"))


def partial_rho(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def youden(stat, inv):
    best, thr = -1.0, np.nan
    for t in np.unique(stat):
        p = stat >= t
        j = (p & inv).sum() / max(inv.sum(), 1) + (~p & ~inv).sum() / max((~inv).sum(), 1) - 1
        if j > best:
            best, thr = j, t
    return thr


def transfer(df, stat, by):
    rows = []
    for g in sorted(df[by].unique()):
        trn, tst = df[df[by] != g], df[df[by] == g]
        inv_tr, inv_te = (trn.ood_auroc_primary < 0.5).to_numpy(), (tst.ood_auroc_primary < 0.5).to_numpy()
        if inv_tr.all() or not inv_tr.any():
            rows.append({"heldout": g, "n": len(tst), "threshold": np.nan, "note": "training groups have one outcome class"})
            continue
        t = youden(trn[stat].to_numpy(), inv_tr)
        p = tst[stat].to_numpy() >= t
        rows.append({"heldout": g, "n": len(tst), "n_inverted": int(inv_te.sum()), "threshold": float(t),
                     "sensitivity": float((p & inv_te).sum() / inv_te.sum()) if inv_te.any() else np.nan,
                     "specificity": float((~p & ~inv_te).sum() / (~inv_te).sum()) if (~inv_te).any() else np.nan})
    return rows


def correlations(df, stats, target="ood_auroc_primary"):
    rows = []
    for fam, d in [("all", df)] + list(df.groupby("family")):
        for s in stats:
            d2 = d[[s, target, "lambda"]].dropna()
            if len(d2) < 5:
                continue
            rho = float(spearmanr(d2[s], d2[target])[0])
            lo, hi = boot_rho(d2[s].to_numpy(), d2[target].to_numpy())
            rows.append({"subset": fam, "stat": s, "n_runs": len(d2), "rho": rho, "ci_lo": lo, "ci_hi": hi,
                         "partial_rho_given_lambda": partial_rho(d2[s], d2[target], d2["lambda"]) if d2["lambda"].nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=OUT / "manifest.csv")
    ap.add_argument("--holdout_by", default=None)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    ready = man[man.status == "ready"]
    (OUT / "per_run").mkdir(parents=True, exist_ok=True)
    recs = []
    for r in ready.to_dict("records"):
        dest = OUT / "per_run" / "{}.json".format(r["run_id"].replace("/", "__"))
        if dest.exists():
            recs.append(pd.read_json(dest, typ="series").to_dict())
            continue
        rec = analyse_run(r)
        R.write_json(dest, rec)
        recs.append(rec)
        print("done", r["run_id"], flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "summary.csv", index=False)
    cor = correlations(df, STATS)
    cor.to_csv(OUT / "correlations.csv", index=False)
    by = a.holdout_by or ("task" if df.task.nunique() >= 2 else "family")
    tr = pd.DataFrame(transfer(df, PRIMARY, by))
    tr.to_csv(OUT / "transfer.csv", index=False)
    L = ["# Paper B R5 Q4 — label-free collapse statistic vs OOD AUROC", "",
         "Design: `decisions/paperB_r5_q4.md`. Commit `{}`. Ready runs: {} of {} in the manifest ({}). Outcome: Mahalanobis AUROC on each run's "
         "primary OOD set. ρ = Spearman across runs, 95% CI from 2,000 run resamples.".format(
             R.git_head(), len(ready), len(man), ", ".join("{} {}".format(k, v) for k, v in ready.family.value_counts().items())), "",
         "| subset | statistic | n runs | ρ [95% CI] | partial ρ given λ |", "|---|---|---:|---|---|"]
    tex = []
    for x in cor.itertuples():
        L.append("| {} | {} | {} | {:.3f} [{:.3f}, {:.3f}] | {} |".format(x.subset, x.stat, x.n_runs, x.rho, x.ci_lo, x.ci_hi, R.fmt(x.partial_rho_given_lambda)))
        if x.stat == PRIMARY:
            tex.append("{} & {} & {:.2f} [{:.2f}, {:.2f}] & {} \\\\".format(x.subset.replace("_", "\\_"), x.n_runs, x.rho, x.ci_lo, x.ci_hi,
                                                                         R.fmt(x.partial_rho_given_lambda, nd=2)))
    L += ["", "## Threshold transfer (primary statistic; outcome = AUROC < 0.5; held out by {})".format(by), "",
          "| held out | n | n inverted | threshold | sensitivity | specificity |", "|---|---:|---:|---|---|---|"]
    for x in tr.to_dict("records"):
        L.append("| {} | {} | {} | {} | {} | {} |".format(x["heldout"], x["n"], x.get("n_inverted", "—"), R.fmt(x.get("threshold")),
                                                         R.fmt(x.get("sensitivity")), R.fmt(x.get("specificity"))))
    p = cor[(cor.subset == "all") & (cor.stat == PRIMARY)]
    if len(p):
        p = p.iloc[0]
        L += ["", "## Plain reading", "", "Across {} runs the excess majority-centroid share on ID test has Spearman ρ = {:.2f} [{:.2f}, {:.2f}] "
              "with Mahalanobis OOD AUROC on the primary OOD set; holding λ fixed (partial ρ) it is {}.".format(
                  p.n_runs, p.rho, p.ci_lo, p.ci_hi, R.fmt(p.partial_rho_given_lambda, nd=2))]
    L += ["", "## LaTeX rows (subset & n & ρ [CI] & partial ρ), primary statistic", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
