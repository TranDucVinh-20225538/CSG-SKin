#!/usr/bin/env python3
"""Paper B R6 R4: label-free warning signals computed on unlabeled deployment data (decisions/paperB_r6_r4.md). CPU.

Deployment set = the run's primary OOD split (pad_heldout for ISIC runs), labels never used by any signal.
Signals per run (manifest rows with status ready, results/paperB/r5/q4/manifest.csv):
  W1 pred_maj_frac_excess   share of deployment images predicted as the majority class minus the ID-train prior
  W2 bbse_tv                total variation between the BBSE estimate of the deployment class proportions and the
                            source prior (confusion matrix from labelled ID test; Lipton et al. 2018; clipped, renormalised)
  W2b bbse_maj_ratio        BBSE majority proportion / source majority prior
  W3 q4_maj_share_excess    the Q4 statistic (ID test only)
  W3d deploy_maj_share_excess  the Q4 geometry on the deployment features (majority-centroid share minus prior)
Validation only (uses OOD labels where they exist): true deployment TV and BBSE absolute error.
Outcome: Mahalanobis AUROC on the deployment set (same features). Spearman ρ with run-bootstrap CI; threshold for
"AUROC < 0.5" chosen on one group and tested on the others (Youden; groups = task, else family).
Usage: r6_r4_warning_signals.py [--manifest ...] [--holdout_by task|family|protocol]
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

OUT = R.R6 / "r4"
SIGNALS = ("pred_maj_frac_excess", "bbse_tv", "bbse_maj_ratio", "q4_maj_share_excess", "deploy_maj_share_excess")


def bbse(logits_src, y_src, logits_tgt, n_classes):
    """Black-box shift estimation: q = p_src * w, w = C^{-1} mu, C[i, j] = P(pred = i, y = j) on labelled source data."""
    ps, yt = np.asarray(logits_src).argmax(1), np.asarray(y_src, int)
    C = np.zeros((n_classes, n_classes))
    np.add.at(C, (ps, yt), 1.0)
    C /= len(yt)
    p_src = np.bincount(yt, minlength=n_classes) / len(yt)
    mu = np.bincount(np.asarray(logits_tgt).argmax(1), minlength=n_classes) / len(logits_tgt)
    w = np.linalg.lstsq(C + 1e-6 * np.eye(n_classes), mu, rcond=None)[0]
    q = np.clip(w * p_src, 0, None)
    q = q / q.sum() if q.sum() > 0 else p_src
    return q, p_src, float(np.linalg.cond(C + 1e-6 * np.eye(n_classes)))


def analyse_run(r):
    d = Path(r["feat_dir"])
    ld = lambda s: dict(np.load(d / "{}.npz".format(s)))  # noqa: E731
    tr, te, dep = ld(r["id_train"]), ld(r["id_test"]), ld(r["primary_ood"])
    rep, nc, maj = r["rep"], int(r["n_classes"]), int(r["majority"])
    m = Q4.PooledMaha(tr[rep], tr["labels"], nc)
    prior = float((np.asarray(tr["labels"], int) == maj).mean())
    q, p_src, cond = bbse(te["logits"], te["labels"], dep["logits"], nc)
    rec = {k: r[k] for k in ("run_id", "task", "protocol", "family", "lambda", "seed")}
    rec.update({"pred_maj_frac_excess": float((np.asarray(dep["logits"]).argmax(1) == maj).mean()) - prior,
                "bbse_tv": float(0.5 * np.abs(q - p_src).sum()), "bbse_maj_ratio": float(q[maj] / max(p_src[maj], 1e-12)),
                "bbse_cond": cond, "bbse_q": q.tolist(), "p_src": p_src.tolist(),
                "q4_maj_share_excess": Q4.collapse_stats(tr[rep], tr["labels"], te[rep], None, maj, nc, m)["maj_share_excess"],
                "deploy_maj_share_excess": float((m.nearest(dep[rep]) == maj).mean()) - prior,
                "ood_auroc_primary": R.auroc(m.score(te[rep]), m.score(dep[rep]))})
    yd = np.asarray(dep["labels"], int)
    if (yd >= 0).all():
        q_true = np.bincount(yd, minlength=nc) / len(yd)
        rec["true_tv"] = float(0.5 * np.abs(q_true - p_src).sum())
        rec["bbse_abs_err_tv"] = float(0.5 * np.abs(q - q_true).sum())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Q4.OUT / "manifest.csv")
    ap.add_argument("--holdout_by", default=None)
    a = ap.parse_args()
    man = pd.read_csv(a.manifest)
    ready = man[man.status == "ready"]
    (OUT / "per_run").mkdir(parents=True, exist_ok=True)
    recs = []
    for r in ready.to_dict("records"):
        dest = OUT / "per_run" / "{}.json".format(r["run_id"].replace("/", "__"))
        rec = pd.read_json(dest, typ="series").to_dict() if dest.exists() else analyse_run(r)
        if not dest.exists():
            R.write_json(dest, rec)
            print("done", r["run_id"], flush=True)
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.drop(columns=["bbse_q", "p_src"], errors="ignore").to_csv(OUT / "summary.csv", index=False)
    cor = Q4.correlations(df, SIGNALS)
    cor.to_csv(OUT / "correlations.csv", index=False)
    by = a.holdout_by or ("task" if df.task.nunique() >= 2 else "family")
    tr = pd.concat([pd.DataFrame(Q4.transfer(df, s, by)).assign(signal=s) for s in SIGNALS], ignore_index=True)
    tr.to_csv(OUT / "transfer.csv", index=False)
    L = ["# Paper B R6 R4 — label-free warning signals on unlabeled deployment data", "",
         "Design: `decisions/paperB_r6_r4.md`. Commit `{}`. Ready runs: {} ({}). Deployment set = each run's primary OOD split; outcome = "
         "Mahalanobis AUROC there.".format(R.git_head(), len(ready), ", ".join("{} {}".format(k, v) for k, v in ready.family.value_counts().items())), "",
         "| subset | signal | n runs | ρ [95% CI] | partial ρ given λ |", "|---|---|---:|---|---|"]
    tex = []
    for x in cor.itertuples():
        L.append("| {} | {} | {} | {:.3f} [{:.3f}, {:.3f}] | {} |".format(x.subset, x.stat, x.n_runs, x.rho, x.ci_lo, x.ci_hi, R.fmt(x.partial_rho_given_lambda)))
        if x.subset == "all":
            tex.append("{} & {} & {:.2f} [{:.2f}, {:.2f}] \\\\".format(x.stat.replace("_", "\\_"), x.n_runs, x.rho, x.ci_lo, x.ci_hi))
    L += ["", "## Threshold transfer (outcome AUROC < 0.5; held out by {})".format(by), "",
          "| signal | held out | n | n inverted | threshold | sensitivity | specificity |", "|---|---|---:|---:|---|---|---|"]
    for x in tr.to_dict("records"):
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(x["signal"], x["heldout"], x["n"], x.get("n_inverted", "—"), R.fmt(x.get("threshold")),
                                                              R.fmt(x.get("sensitivity")), R.fmt(x.get("specificity"))))
    if "bbse_abs_err_tv" in df:
        L += ["", "BBSE validation (deployment labels used only here): true TV {} ; BBSE TV error {} (mean ± s.d. over runs).".format(
            R.fmt(df.true_tv.mean(), df.true_tv.std(ddof=1)), R.fmt(df.bbse_abs_err_tv.mean(), df.bbse_abs_err_tv.std(ddof=1)))]
    a_ = cor[cor.subset == "all"].set_index("stat")
    if len(a_):
        L += ["", "## Plain reading", "", "Across runs, Spearman ρ with deployment Mahalanobis AUROC: " + "; ".join(
            "{} {:.2f} [{:.2f}, {:.2f}]".format(s, a_.loc[s, "rho"], a_.loc[s, "ci_lo"], a_.loc[s, "ci_hi"]) for s in SIGNALS if s in a_.index) + "."]
    L += ["", "## LaTeX rows (signal & n & ρ [CI]), all runs", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
