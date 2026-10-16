#!/usr/bin/env python3
"""Paper B R5 two-tier warning signal (decisions/paperB_r5_warning.md). CPU.

Tier 1 (ID-only): centroid_collapse_excess on ISIC val and test.
Tier 2 (deployment): pred_maj_frac_excess and bbse_tv on pad_heldout.
Threshold: 95th percentile over λ = 0 runs only, written before OOD/rule-A labels are used for flagging.
Primary: per-run sens/spec vs seed invert (Mahalanobis AUROC < 0.5; bootstrap CI entirely below 0.5 when present)
and Spearman with OOD AUROC. Leave-one-setting-out secondary (labelled weak).

Usage: r5_warning_signal.py [--manifest ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r5_q4_collapse as Q4  # noqa: E402
import r5_q4_manifest as Man  # noqa: E402
import r6_r4_warning_signals as R4  # noqa: E402

OUT = R.R5 / "warning"
STATS = (
    "tier1_val_centroid_collapse_excess",
    "tier1_test_centroid_collapse_excess",
    "tier2_pred_maj_frac_excess",
    "tier2_bbse_tv",
)
PRIMARY_TIER1 = "tier1_test_centroid_collapse_excess"
RNG = 20261005


def corpus_rows():
    """Old recipe 35 + A4 15 (+ extras later via manifest scan)."""
    rows = []
    rows += Man.scan(R.FEAT_L, "old_recipe", "isic_lesion_split",
                     r"runB_orth1_ladv([0-9p]+)_s(\d+)", source="r2/item4")
    rows += Man.scan(R.R5 / "p8" / "main" / "features", "a4_p8", "isic_lesion_split",
                     r"main_ladv([0-9p]+)_s(\d+)", source="r5/p8/main last.ckpt")
    for fam, root, pat, src in (
            ("p9", R.R5 / "p9" / "features", r".*_ladv([0-9p]+)_s(\d+).*", "r5/p9"),
            ("p11", R.R5 / "p11" / "features", r".*_ladv([0-9p]+)_s(\d+).*", "r5/p11"),
            ("p10", R.R5 / "p10" / "features", r".*_ladv([0-9p]+)_s(\d+).*", "r5/p10"),
    ):
        if root.exists():
            rows += Man.scan(root, fam, "pending_or_isic", pat, source=src)
    for r in rows:
        r["status"] = Man.status(r)
        # need isic_val for tier-1 val
        if r["status"] == "ready" and not (Path(r["feat_dir"]) / "isic_val.npz").exists():
            r["status"] = "pending_no_val"
    return rows


def analyse(r):
    d = Path(r["feat_dir"])
    ld = lambda s: dict(np.load(d / "{}.npz".format(s)))  # noqa: E731
    tr, va, te, dep = ld("isic_train"), ld("isic_val"), ld(r["id_test"]), ld(r["primary_ood"])
    rep, nc, maj = r["rep"], int(r["n_classes"]), int(r["majority"])
    m = Q4.PooledMaha(tr[rep], tr["labels"], nc)
    prior = float((np.asarray(tr["labels"], int) == maj).mean())
    q, p_src, cond = R4.bbse(te["logits"], te["labels"], dep["logits"], nc)
    t1_val = Q4.collapse_stats(tr[rep], tr["labels"], va[rep], None, maj, nc, m)["maj_share_excess"]
    t1_te = Q4.collapse_stats(tr[rep], tr["labels"], te[rep], None, maj, nc, m)["maj_share_excess"]
    sid = m.score(te[rep])
    so = m.score(dep[rep])
    auroc = R.auroc(sid, so)
    rec = {k: r[k] for k in ("run_id", "task", "protocol", "family", "lambda", "seed", "source")}
    rec.update({
        "tier1_val_centroid_collapse_excess": t1_val,
        "tier1_test_centroid_collapse_excess": t1_te,
        "tier2_pred_maj_frac_excess": float((np.asarray(dep["logits"]).argmax(1) == maj).mean()) - prior,
        "tier2_bbse_tv": float(0.5 * np.abs(q - p_src).sum()),
        "bbse_cond": cond,
        "ood_auroc_maha": auroc,
        "inverts_proxy_auroc_lt_0p5": bool(auroc < 0.5),
        "setting": r["family"],
    })
    # Fitzpatrick Mahalanobis if present
    fz = d / "fitzpatrick17k.npz"
    if fz.exists():
        rec["ood_auroc_maha_fitz"] = R.auroc(sid, m.score(ld("fitzpatrick17k")[rep]))
    return rec


def boot_rho(x, y, n=2000, seed=RNG):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2:
            v.append(spearmanr(x[i], y[i])[0])
    if not v:
        return None, None, None
    return float(spearmanr(x, y)[0]), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def sens_spec(flag, y):
    flag, y = np.asarray(flag, bool), np.asarray(y, bool)
    tp = int((flag & y).sum())
    tn = int((~flag & ~y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "sensitivity": sens, "specificity": spec,
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None, help="optional; default rebuilds old+A4 corpus")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir(exist_ok=True)
    if a.manifest and a.manifest.exists():
        man = pd.read_csv(a.manifest)
        rows = [r for r in man.to_dict("records") if r.get("status") == "ready"]
    else:
        rows = [r for r in corpus_rows() if r["status"] == "ready"]
        pd.DataFrame(corpus_rows()).to_csv(OUT / "manifest.csv", index=False)

    # --- pass 1: statistics only (no invert labels used for threshold) ---
    recs = []
    for r in rows:
        rec = analyse(r)
        recs.append(rec)
        (OUT / "per_run" / (rec["run_id"].replace("/", "__") + ".json")).write_text(
            json.dumps(rec, indent=1, default=float) + "\n")

    df = pd.DataFrame(recs)
    if df.empty:
        print("no ready runs")
        return
    lam0 = df[np.isclose(df["lambda"].astype(float), 0.0)]
    thresholds = {
        "rule": "95th percentile over λ = 0 runs only; fixed before reading invert labels for flagging",
        "n_lambda0": int(len(lam0)),
        "stats": {},
    }
    for s in STATS:
        thr = float(np.quantile(lam0[s].to_numpy(float), 0.95)) if len(lam0) else None
        thresholds["stats"][s] = thr
    (OUT / "thresholds_lambda0.json").write_text(json.dumps(thresholds, indent=1) + "\n")
    print("wrote thresholds", OUT / "thresholds_lambda0.json", flush=True)

    # --- pass 2: flags vs invert ---
    y = df["inverts_proxy_auroc_lt_0p5"].to_numpy(bool)
    ss_rows, corr_rows = [], []
    for s in STATS:
        thr = thresholds["stats"][s]
        flag = df[s].to_numpy(float) > thr if thr is not None else np.zeros(len(df), bool)
        df["flag_" + s] = flag
        block = sens_spec(flag, y)
        block.update({"statistic": s, "threshold": thr, "tier": "1" if s.startswith("tier1") else "2"})
        ss_rows.append(block)
        rho, lo, hi = boot_rho(df[s], df["ood_auroc_maha"])
        corr_rows.append({"statistic": s, "rho_maha_auroc": rho, "ci_lo": lo, "ci_hi": hi, "n": len(df)})

    # leave-one-setting-out (weak)
    loso = []
    settings = sorted(df["setting"].unique())
    for hold in settings:
        train = df[df["setting"] != hold]
        test = df[df["setting"] == hold]
        lam0_t = train[np.isclose(train["lambda"].astype(float), 0.0)]
        if len(lam0_t) < 3 or len(test) < 3:
            loso.append({"holdout_setting": hold, "note": "too few runs", "weak": True})
            continue
        for s in STATS:
            thr = float(np.quantile(lam0_t[s].to_numpy(float), 0.95))
            flag = test[s].to_numpy(float) > thr
            block = sens_spec(flag, test["inverts_proxy_auroc_lt_0p5"].to_numpy(bool))
            block.update({"holdout_setting": hold, "statistic": s, "threshold_from_other_lambda0": thr, "weak": True})
            loso.append(block)

    df.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(ss_rows).to_csv(OUT / "sens_spec.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "correlations.csv", index=False)
    pd.DataFrame(loso).to_csv(OUT / "leave_one_setting_out.csv", index=False)

    L = [
        "# Paper B R5 — two-tier warning signal",
        "",
        "Design: `decisions/paperB_r5_warning.md`. Thresholds from λ = 0 only "
        "(n = {}), written before flagging.".format(thresholds["n_lambda0"]),
        "",
        "## Thresholds (95th percentile, λ = 0)",
        "",
    ]
    for s, thr in thresholds["stats"].items():
        L.append("- `{}`: {:.6f}".format(s, thr) if thr is not None else "- `{}`: —".format(s))
    L += ["", "## Primary: sensitivity / specificity vs invert (Mahalanobis AUROC < 0.5 proxy)", "",
          "| statistic | tier | threshold | sens | spec | n_pos | n_neg |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for b in ss_rows:
        L.append("| {} | {} | {:.4f} | {:.3f} | {:.3f} | {} | {} |".format(
            b["statistic"], b["tier"], b["threshold"], b["sensitivity"], b["specificity"],
            b["n_pos"], b["n_neg"]))
    L += ["", "## Spearman with Mahalanobis AUROC", "",
          "| statistic | ρ | 95% CI | n |", "|---|---:|---|---:|"]
    for c in corr_rows:
        ci = "—" if c["rho_maha_auroc"] is None else "[{:.3f}, {:.3f}]".format(c["ci_lo"], c["ci_hi"])
        L.append("| {} | {} | {} | {} |".format(
            c["statistic"],
            "—" if c["rho_maha_auroc"] is None else "{:.3f}".format(c["rho_maha_auroc"]),
            ci, c["n"]))
    L += ["", "## Leave-one-setting-out (secondary, labelled **weak**)", "",
          "Settings: {}. Few settings — not the primary claim.".format(", ".join(settings)), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", OUT / "REPORT.md", "n_runs", len(df), flush=True)


if __name__ == "__main__":
    main()
