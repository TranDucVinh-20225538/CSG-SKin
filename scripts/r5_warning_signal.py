#!/usr/bin/env python3
"""Paper B R5 two-tier warning signal (decisions/paperB_r5_warning.md). CPU.

Tier 1 (ID-only): centroid_collapse_excess on ISIC val and test.
Tier 2 (deployment): pred_maj_frac_excess and bbse_tv on pad_heldout.
Threshold: frozen 95th percentile over λ = 0 from thresholds_lambda0.json (never refit when present).
Primary: sens vs invert; specificity on λ > 0 non-inverting runs only; n beside every cell.
Invert labels: pad_heldout and Fitzpatrick (Mahalanobis AUROC < 0.5 proxy).

Usage: r5_warning_signal.py [--manifest ...]
"""

from __future__ import annotations

import argparse
import json
import re
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
RNG = 20261005
# Pre-committed primary table (old+A4, original all-non-invert specificity) — keep visible.
PRECOMMITTED = {
    "verdict": "tier 1 failed",
    "note": "Original primary on old_recipe+A4 only; specificity used all non-inverting runs (incl. λ = 0).",
    "rows": [
        {"statistic": "tier1_val_centroid_collapse_excess", "tier": "1", "threshold": 0.0597,
         "sens": 0.194, "spec": 0.857, "n_pos": 36, "n_neg": 14},
        {"statistic": "tier1_test_centroid_collapse_excess", "tier": "1", "threshold": 0.0493,
         "sens": 0.389, "spec": 0.857, "n_pos": 36, "n_neg": 14},
        {"statistic": "tier2_pred_maj_frac_excess", "tier": "2", "threshold": -0.3158,
         "sens": 1.000, "spec": 0.643, "n_pos": 36, "n_neg": 14},
        {"statistic": "tier2_bbse_tv", "tier": "2", "threshold": 0.8725,
         "sens": 0.000, "spec": 0.929, "n_pos": 36, "n_neg": 14},
    ],
}


def scan_r1(root):
    """R1 three-stream: shift 0% and 100% (any λ present; caller filters)."""
    rows = []
    root = Path(root)
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"r1_cur_shift(0|100)_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        rows.append({
            **Man.ISIC,
            "run_id": "r1/{}".format(d.name),
            "protocol": "isic_lesion_split_r1_three_stream",
            "family": "r1_shift{}".format(m.group(1)),
            "lambda": float(m.group(2).replace("p", ".")),
            "seed": int(m.group(3)),
            "feat_dir": str(d),
            "source": "r6/r1/cur features",
            "shift_pct": int(m.group(1)),
        })
    return rows


def corpus_rows():
    """Old recipe 35 + A4 15 + R1 shift0/100 + R2/P9/P11/P10 as features arrive."""
    rows = []
    rows += Man.scan(R.FEAT_L, "old_recipe", "isic_lesion_split",
                     r"runB_orth1_ladv([0-9p]+)_s(\d+)", source="r2/item4")
    rows += Man.scan(R.R5 / "p8" / "main" / "features", "a4_p8", "isic_lesion_split",
                     r"main_ladv([0-9p]+)_s(\d+)", source="r5/p8/main last.ckpt")
    # R1: shift 0% and 100% only (λ = 2 present now; other λ added as dirs appear)
    for r in scan_r1(R.RES / "r6" / "r1" / "cur" / "features"):
        rows.append(r)
    for fam, root, pat, src in (
            ("r2", R.RES / "r6" / "r2" / "cur" / "features", r"r2_cur_.*_ladv([0-9p]+)_s(\d+)", "r6/r2"),
            ("p9", R.R5 / "p9" / "features", r".*_ladv([0-9p]+)_s(\d+).*", "r5/p9"),
            ("p11", R.R5 / "p11" / "features", r".*_ladv([0-9p]+)_s(\d+).*", "r5/p11"),
            ("p10", R.R5 / "p10" / "features", r".*_ladv([0-9p]+)_s(\d+).*", "r5/p10"),
    ):
        if root.exists():
            rows += Man.scan(root, fam, "pending_or_isic", pat, source=src)
    for r in rows:
        r["status"] = Man.status(r)
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
    yd = np.asarray(dep["labels"], int)
    rec = {k: r[k] for k in ("run_id", "task", "protocol", "family", "lambda", "seed", "source") if k in r}
    rec["shift_pct"] = r.get("shift_pct")
    rec.update({
        "tier1_val_centroid_collapse_excess": t1_val,
        "tier1_test_centroid_collapse_excess": t1_te,
        "tier2_pred_maj_frac_excess": float((np.asarray(dep["logits"]).argmax(1) == maj).mean()) - prior,
        "tier2_bbse_tv": float(0.5 * np.abs(q - p_src).sum()),
        "bbse_cond": cond,
        "bbse_q": q.tolist(),
        "p_src": p_src.tolist(),
        "ood_auroc_maha": auroc,
        "inverts_pad_heldout": bool(auroc < 0.5),
        "inverts_proxy_auroc_lt_0p5": bool(auroc < 0.5),  # alias for pad_heldout
        "setting": r["family"],
    })
    if (yd >= 0).all():
        q_true = np.bincount(yd, minlength=nc) / max(len(yd), 1)
        rec["true_tv"] = float(0.5 * np.abs(q_true - p_src).sum())
        rec["bbse_abs_err_tv"] = float(0.5 * np.abs(q - q_true).sum())
    fz = d / "fitzpatrick17k.npz"
    if fz.exists():
        a_fz = R.auroc(sid, m.score(ld("fitzpatrick17k")[rep]))
        rec["ood_auroc_maha_fitz"] = a_fz
        rec["inverts_fitz"] = bool(a_fz < 0.5)
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


def sens_spec_fair(flag, y_invert, lam):
    """Sensitivity on all inverting runs; specificity on λ > 0 non-inverting only."""
    flag = np.asarray(flag, bool)
    y = np.asarray(y_invert, bool)
    lam = np.asarray(lam, float)
    pos = y
    neg = (~y) & (lam > 0)
    tp = int((flag & pos).sum())
    fn = int((~flag & pos).sum())
    tn = int((~flag & neg).sum())
    fp = int((flag & neg).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
        "n_neg_definition": "lambda>0 and not inverting",
    }


def load_frozen_thresholds():
    p = OUT / "thresholds_lambda0.json"
    if not p.exists():
        raise FileNotFoundError(
            "thresholds_lambda0.json missing — refusing to fit a new threshold. "
            "Restore the frozen file before running.")
    thr = json.loads(p.read_text())
    for s in STATS:
        if s not in thr.get("stats", {}) or thr["stats"][s] is None:
            raise ValueError("frozen thresholds incomplete: missing {}".format(s))
    return thr


def bbse_audit(df):
    """Check BBSE TV sign/implementation. Returns markdown lines + dict; no code change unless bug."""
    lines = [
        "## BBSE TV audit (2026-10-16)",
        "",
        "Checked against prediction that `bbse_tv` should track inversion (higher when AUROC falls).",
        "",
    ]
    tv = df["tier2_bbse_tv"].to_numpy(float)
    auroc = df["ood_auroc_maha"].to_numpy(float)
    bugs = []
    if (tv < -1e-12).any():
        bugs.append("NEGATIVE_TV")
    # TV formula is 0.5 * L1(q, p_src) — always ≥ 0 by construction in analyse().
    rho, lo, hi = boot_rho(tv, auroc)
    lines.append("- TV values: min={:.4f}, max={:.4f} (all ≥ 0 → unsigned distance; no sign flip possible).".format(
        float(tv.min()), float(tv.max())))
    lines.append("- Spearman(bbse_tv, Maha pad_heldout AUROC): ρ = {:.3f} CI [{:.3f}, {:.3f}] (n = {}).".format(
        rho, lo, hi, len(df)))
    # Stratify by λ
    by = df.groupby(df["lambda"].astype(float).round(3))[["tier2_bbse_tv", "ood_auroc_maha"]].mean()
    lines.append("- Mean by λ (bbse_tv, auroc): " + ", ".join(
        "λ={:g}→({:.3f},{:.3f})".format(float(i), r.tier2_bbse_tv, r.ood_auroc_maha) for i, r in by.iterrows()))
    if "true_tv" in df.columns and df["true_tv"].notna().sum() >= 5:
        sub = df.dropna(subset=["true_tv", "bbse_abs_err_tv"])
        r_true, _, _ = boot_rho(sub["tier2_bbse_tv"], sub["true_tv"])
        r_err, _, _ = boot_rho(sub["bbse_abs_err_tv"], sub["ood_auroc_maha"])
        def _f(x):
            return "—" if x is None else "{:.3f}".format(x)
        lines.append("- Among labelled pad_heldout runs (n = {}): Spearman(bbse_tv, true_tv) = {}; "
                     "Spearman(bbse_abs_err_tv, auroc) = {}.".format(len(sub), _f(r_true), _f(r_err)))
    lines += [
        "",
        "**Reading:** positive ρ is not a sign bug. At λ = 0, PAD has real class-mix shift so BBSE TV is large "
        "while AUROC is high; under strong λ the confusion matrix C collapses, BBSE TV shrinks even as AUROC "
        "falls — so TV covaries with AUROC rather than with inversion. `pred_maj_frac_excess` keeps the "
        "predicted negative ρ. Implementation matches R6 R4 (`q = clip(w)·p_src`, TV = ½‖q−p_src‖₁). "
        "**No code change.**",
        "",
    ]
    return lines, {"rho_bbse_auroc": rho, "ci": [lo, hi], "bugs": bugs, "code_changed": False}


def fmt_ss_table(ss_rows, label_name):
    L = [
        "### Invert label: {} (Mahalanobis AUROC < 0.5)".format(label_name),
        "",
        "Specificity = among **λ > 0 non-inverting** runs only. Sensitivity = among all inverting runs. "
        "`n_pos` / `n_neg` are the denominators for sens / spec.",
        "",
        "| statistic | tier | threshold | sens (n_pos) | spec (n_neg) |",
        "|---|---:|---:|---|---|",
    ]
    for b in ss_rows:
        L.append("| {} | {} | {:.4f} | {:.3f} (n={}) | {:.3f} (n={}) |".format(
            b["statistic"], b["tier"], b["threshold"],
            b["sensitivity"], b["n_pos"], b["specificity"], b["n_neg"]))
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None, help="optional; default rebuilds corpus")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir(exist_ok=True)

    frozen = load_frozen_thresholds()
    # Never rewrite thresholds — keep file bytes conceptually frozen (rewrite identical content only).
    (OUT / "thresholds_lambda0.json").write_text(json.dumps(frozen, indent=1) + "\n")

    if a.manifest and a.manifest.exists():
        man = pd.read_csv(a.manifest)
        rows = [r for r in man.to_dict("records") if r.get("status") == "ready"]
    else:
        all_rows = corpus_rows()
        pd.DataFrame(all_rows).to_csv(OUT / "manifest.csv", index=False)
        rows = [r for r in all_rows if r["status"] == "ready"]

    recs = []
    for r in rows:
        rec = analyse(r)
        recs.append(rec)
        slim = {k: v for k, v in rec.items() if k not in ("bbse_q", "p_src")}
        (OUT / "per_run" / (rec["run_id"].replace("/", "__") + ".json")).write_text(
            json.dumps(slim, indent=1, default=float) + "\n")

    df = pd.DataFrame(recs)
    if df.empty:
        print("no ready runs")
        return
    lam = df["lambda"].astype(float).to_numpy()

    ss_by_label = {}
    corr_rows = []
    for label_key, label_col, label_name in (
            ("pad_heldout", "inverts_pad_heldout", "pad_heldout"),
            ("fitzpatrick17k", "inverts_fitz", "fitzpatrick17k"),
    ):
        if label_col not in df.columns or df[label_col].isna().all():
            continue
        y = df[label_col].fillna(False).to_numpy(bool)
        ss_rows = []
        for s in STATS:
            thr = frozen["stats"][s]
            flag = df[s].to_numpy(float) > thr
            df["flag_" + s + "_" + label_key] = flag
            block = sens_spec_fair(flag, y, lam)
            block.update({
                "statistic": s, "threshold": thr,
                "tier": "1" if s.startswith("tier1") else "2",
                "invert_label": label_key,
            })
            ss_rows.append(block)
        ss_by_label[label_key] = ss_rows
        auroc_col = "ood_auroc_maha" if label_key == "pad_heldout" else "ood_auroc_maha_fitz"
        if auroc_col in df.columns:
            for s in STATS:
                rho, lo, hi = boot_rho(df[s], df[auroc_col])
                corr_rows.append({
                    "statistic": s, "ood": label_key,
                    "rho_maha_auroc": rho, "ci_lo": lo, "ci_hi": hi, "n": int(df[auroc_col].notna().sum()),
                })

    # leave-one-setting-out (weak) — still uses frozen global thresholds? Decision said LOSO refits
    # from other settings' λ0. Keep that weak secondary with refit-on-train only.
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
            block = sens_spec_fair(
                flag, test["inverts_pad_heldout"].to_numpy(bool), test["lambda"].astype(float).to_numpy())
            block.update({"holdout_setting": hold, "statistic": s, "threshold_from_other_lambda0": thr, "weak": True})
            loso.append(block)

    audit_md, audit = bbse_audit(df)

    drop_cols = [c for c in ("bbse_q", "p_src") if c in df.columns]
    df.drop(columns=drop_cols).to_csv(OUT / "summary.csv", index=False)
    ss_all = []
    for rows_ in ss_by_label.values():
        ss_all.extend(rows_)
    pd.DataFrame(ss_all).to_csv(OUT / "sens_spec.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "correlations.csv", index=False)
    pd.DataFrame(loso).to_csv(OUT / "leave_one_setting_out.csv", index=False)
    (OUT / "bbse_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    (OUT / "precommitted_primary.json").write_text(json.dumps(PRECOMMITTED, indent=1) + "\n")

    fam_counts = df["family"].value_counts().to_dict()
    L = [
        "# Paper B R5 — two-tier warning signal",
        "",
        "Design: `decisions/paperB_r5_warning.md`. **Frozen** thresholds from `thresholds_lambda0.json` "
        "(n_lambda0 = {}); **not refit** on the expanded corpus.".format(frozen.get("n_lambda0")),
        "",
        "## Thresholds (95th percentile, λ = 0) — frozen",
        "",
    ]
    for s, thr in frozen["stats"].items():
        L.append("- `{}`: {:.6f}".format(s, thr))
    L += [
        "",
        "## Pre-committed primary verdict (unchanged)",
        "",
        "**{}.** {}".format(PRECOMMITTED["verdict"], PRECOMMITTED["note"]),
        "",
        "| statistic | tier | threshold | sens | spec | n_pos | n_neg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in PRECOMMITTED["rows"]:
        L.append("| {} | {} | {:.4f} | {:.3f} | {:.3f} | {} | {} |".format(
            b["statistic"], b["tier"], b["threshold"], b["sens"], b["spec"], b["n_pos"], b["n_neg"]))
    L += [
        "",
        "## Updated primary (fairer specificity; expanded corpus)",
        "",
        "Corpus ready runs: n = {}. Families: {}.".format(
            len(df), ", ".join("{}={}".format(k, v) for k, v in sorted(fam_counts.items()))),
        "R1 shift 0%/100% included when features exist; R2/P9/P11/P10 added as directories arrive.",
        "",
    ]
    for label_key, label_name in (("pad_heldout", "pad_heldout"), ("fitzpatrick17k", "fitzpatrick17k")):
        if label_key in ss_by_label:
            L += fmt_ss_table(ss_by_label[label_key], label_name)
    L += ["## Spearman with Mahalanobis AUROC", "",
          "| statistic | OOD | ρ | 95% CI | n |", "|---|---|---:|---|---:|"]
    for c in corr_rows:
        ci = "—" if c["rho_maha_auroc"] is None else "[{:.3f}, {:.3f}]".format(c["ci_lo"], c["ci_hi"])
        L.append("| {} | {} | {} | {} | {} |".format(
            c["statistic"], c["ood"],
            "—" if c["rho_maha_auroc"] is None else "{:.3f}".format(c["rho_maha_auroc"]),
            ci, c["n"]))
    L += [""] + audit_md
    L += ["## Leave-one-setting-out (secondary, labelled **weak**)", "",
          "Settings: {}. Few settings — not the primary claim. LOSO still refits λ = 0 p95 on the "
          "complement; primary flags use the frozen thresholds above.".format(", ".join(settings)), ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", OUT / "REPORT.md", "n_runs", len(df), flush=True)


if __name__ == "__main__":
    main()
