#!/usr/bin/env python3
"""Paper B R5 P4 (CPU part): robustness of the lesion-level inversion to tail truncation and covariance shrinkage
(decisions/paperB_r5_p4.md). Cached features of the 35 lesion-level runs.

Variants of the Table 1 Mahalanobis (z_lesion_norm primary, backbone_raw_lesion secondary; OOD = pad_heldout, Fitzpatrick17k):
  tail truncation (Phase 14 A2 rule): drop the pooled (ID ∪ OOD) upper p90 / p95 / p99 of scores, AUROC on the bulk;
                  plus an ID-only rule: drop ID scores above the ID p95 / p99 (OOD untouched)
  trimmed fit:    refit means / covariance after removing the 5 % ISIC-train points with the largest own-class distance
  shrinkage:      ridge eps ∈ {0, 1e-6, 1e-4, 1e-3 (Table 1), 1e-2, 1e-1, 1}, Ledoit-Wolf, OAS, per-class covariances (QDA)
Pooled bootstrap (PREREGISTER draws) and rule A at λ = 2 for the headline variants (LW, p95 pooled drop).
Usage: r5_p4_robustness.py [--report_only]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "p4"
REPS = ("z_lesion_norm", "backbone_raw_lesion")
OODS = ("pad_heldout", "fitzpatrick17k")
RIDGE = (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
HEADLINE = ("lw", "drop_pooled_p95")


class Maha:
    def __init__(self, z, y, cov="ridge", eps=1e-3, per_class=False):
        z, y = np.asarray(z, np.float64), np.asarray(y, int)
        self.means = np.stack([z[y == c].mean(0) for c in range(8)])
        if per_class:
            self.precs = [np.linalg.pinv(np.cov(z[y == c].T) + 1e-3 * np.eye(z.shape[1])) for c in range(8)]
            return
        cen = z - self.means[y]
        if cov == "ridge":
            S = cen.T @ cen / max(len(z) - 8, 1) + eps * np.eye(z.shape[1])
            P = np.linalg.pinv(S) if eps == 0 else np.linalg.inv(S)
        elif cov == "lw":
            P = LedoitWolf().fit(cen).precision_
        elif cov == "oas":
            P = OAS().fit(cen).precision_
        self.precs = [P] * 8

    def dists(self, z):
        z = np.asarray(z, np.float64)
        if all(P is self.precs[0] for P in self.precs):
            P = self.precs[0]
            zp = z @ P
            mp = self.means @ P
            return np.maximum(np.einsum("nd,nd->n", zp, z)[:, None] + np.einsum("cd,cd->c", mp, self.means)[None]
                              - 2.0 * zp @ self.means.T, 0.0)
        out = []
        for m, P in zip(self.means, self.precs):
            d = z - m
            out.append(np.einsum("nd,nd->n", d @ P, d))
        return np.stack(out, 1)

    def score(self, z):
        return self.dists(z).min(1)


def drop(sid, so, pct, pooled=True):
    thr = np.quantile(np.r_[sid, so] if pooled else sid, pct / 100.0)
    ki, ko = sid <= thr, (so <= thr) if pooled else np.ones(len(so), bool)
    return R.auroc(sid[ki], so[ko]), int((~ki).sum()), int((~ko).sum())


def variants(tr, te, oo, rep):
    z, y = tr[rep], tr["labels"].astype(int)
    out = {}
    for eps in RIDGE:
        m = Maha(z, y, "ridge", eps)
        out["ridge_{:g}".format(eps)] = (m.score(te[rep]), {o: m.score(v[rep]) for o, v in oo.items()})
    for name, kw in (("lw", {"cov": "lw"}), ("oas", {"cov": "oas"}), ("per_class_cov", {"per_class": True})):
        m = Maha(z, y, **kw)
        out[name] = (m.score(te[rep]), {o: m.score(v[rep]) for o, v in oo.items()})
    base = Maha(z, y)
    own = base.dists(z)[np.arange(len(y)), y]
    keep = own <= np.quantile(own, 0.95)
    m = Maha(z[keep], y[keep])
    out["trimmed_fit_5pct"] = (m.score(te[rep]), {o: m.score(v[rep]) for o, v in oo.items()})
    return out, (base.score(te[rep]), {o: base.score(v[rep]) for o, v in oo.items()})


def analyse_run(lam, seed, tag):
    tr, te = R.load(tag, "isic_train"), R.load(tag, "isic_test")
    oo = {o: R.load(tag, o) for o in OODS}
    rec = {"lambda": lam, "seed": seed, "run": tag}
    for rep in REPS:
        var, (sid, so) = variants(tr, te, oo, rep)
        for o in OODS:
            blk = {k: R.auroc(v[0], v[1][o]) for k, v in var.items()}
            for pct in (90, 95, 99):
                blk["drop_pooled_p{}".format(pct)], blk["n_id_dropped_p{}".format(pct)], blk["n_ood_dropped_p{}".format(pct)] = drop(sid, so[o], pct)
            for pct in (95, 99):
                blk["drop_id_only_p{}".format(pct)] = drop(sid, so[o], pct, pooled=False)[0]
            rec["{}|{}".format(rep, o)] = blk
    return rec


def headline_bootstrap():
    """Rule A ingredients at λ = 2 for LW and pooled-p95 drop (z_lesion_norm, pad_heldout)."""
    pre = R.prereg()
    rng = np.random.default_rng(int(pre["rng_seed"]))
    n = int(pre["n_resamples"])
    out = {}
    for o in OODS:
        idd, odd = R.draws(R.id_test_groups(), n, rng), R.draws(R.ood_groups(o), n, rng)
        per_lw, per_drop = {}, {}
        for lam, s, tag in R.runs():
            if lam != 2.0:
                continue
            tr, te, oo = R.load(tag, "isic_train"), R.load(tag, "isic_test"), R.load(tag, o)
            m = Maha(tr["z_lesion_norm"], tr["labels"].astype(int), "lw")
            per_lw[s] = (m.score(te["z_lesion_norm"]), m.score(oo["z_lesion_norm"]))
            b = Maha(tr["z_lesion_norm"], tr["labels"].astype(int))
            sid, so = b.score(te["z_lesion_norm"]), b.score(oo["z_lesion_norm"])
            thr = np.quantile(np.r_[sid, so], 0.95)
            per_drop[s] = (np.where(sid <= thr, sid, np.nan), np.where(so <= thr, so, np.nan))
        out["lw|" + o] = R.pooled_bootstrap(per_lw, idd, odd)
        out["drop_pooled_p95|" + o] = bulk_bootstrap(per_drop, idd, odd)
    return out


def bulk_bootstrap(per, idd, odd):
    """pooled_bootstrap on the kept (non-NaN) points of each resample."""
    clean = {}
    dr = {}
    for s, (a, b) in per.items():
        vals = []
        for i, j in zip(idd, odd):
            x, y = a[i], b[j]
            vals.append(R.B.auroc_from_scores(x[~np.isnan(x)], y[~np.isnan(y)]))
        dr[s] = np.array(vals)
        clean[s] = R.B.auroc_from_scores(a[~np.isnan(a)], b[~np.isnan(b)])
    lo, hi, _ = R.B.percentile_ci(np.mean(np.stack(list(dr.values())), 0))
    k = int(sum(np.quantile(v, 0.975) < 0.5 for v in dr.values()))
    return {"mean": float(np.mean(list(clean.values()))), "pooled_ci_lo": lo, "pooled_ci_hi": hi,
            "k_seeds_ci_below_0.5": k, "rule_A": bool(hi < 0.5 and k >= 3), "n_seeds": len(clean)}


def report():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "per_run").glob("*.json"))]
    flat = []
    for r in rows:
        for key in [k for k in r if "|" in k]:
            flat.append({"lambda": r["lambda"], "seed": r["seed"], "block": key, **r[key]})
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "summary.csv", index=False)
    cols = (["ridge_0.001"] + ["ridge_{:g}".format(e) for e in RIDGE if e != 1e-3] + ["lw", "oas", "per_class_cov", "trimmed_fit_5pct",
            "drop_pooled_p90", "drop_pooled_p95", "drop_pooled_p99", "drop_id_only_p95", "drop_id_only_p99"])
    L = ["# Paper B R5 P4 (CPU) — tail truncation and shrinkage, lesion-level runs", "",
         "Design: `decisions/paperB_r5_p4.md`. Commit `{}`. ridge_0.001 is the Table 1 estimator. Mean ± s.d. over seeds; "
         "'k<0.5' = seeds with point AUROC below 0.5.".format(R.git_head()), ""]
    tex = []
    for block in sorted(df.block.unique(), key=lambda b: (b.split("|")[1] != "pad_heldout", b.split("|")[0] != "z_lesion_norm")):
        d = df[df.block == block]
        L += ["## " + block, "", "| variant | " + " | ".join("λ={:g}".format(l) for l in sorted(d["lambda"].unique())) + " |",
              "|---|" + "---|" * d["lambda"].nunique()]
        for c in cols:
            cells = []
            for lam in sorted(d["lambda"].unique()):
                v = d[d["lambda"] == lam][c].to_numpy(float)
                cells.append("{} (k<0.5: {}/{})".format(R.fmt(v.mean(), v.std(ddof=1)), int((v < 0.5).sum()), len(v)))
            L.append("| {} | {} |".format(c, " | ".join(cells)))
            if block == "z_lesion_norm|pad_heldout":
                v0, v2 = d[d["lambda"] == 0.0][c], d[d["lambda"] == 2.0][c]
                tex.append("{} & {} & {} \\\\".format(c.replace("_", "\\_"), R.tex_pm(v0.mean(), v0.std(ddof=1)), R.tex_pm(v2.mean(), v2.std(ddof=1))))
        L.append("")
    hb = json.loads((OUT / "headline_bootstrap.json").read_text()) if (OUT / "headline_bootstrap.json").exists() else {}
    if hb:
        L += ["## Headline variants at λ = 2 (z_lesion^norm), PREREGISTER bootstrap", "", "| variant | OOD | mean | pooled 95% CI | k seeds CI < 0.5 | rule A |",
              "|---|---|---|---|---|---|"]
        for k, v in hb.items():
            var, o = k.split("|")
            L.append("| {} | {} | {:.3f} | [{:.3f}, {:.3f}] | {}/{} | {} |".format(var, o, v["mean"], v["pooled_ci_lo"], v["pooled_ci_hi"],
                                                                         v["k_seeds_ci_below_0.5"], v["n_seeds"], "yes" if v["rule_A"] else "no"))
        L.append("")
    d = df[df.block == "z_lesion_norm|pad_heldout"]
    if len(d):
        rng = [d[d["lambda"] == 2.0][c].mean() for c in cols]
        L += ["## Plain reading", "", "Across the {} estimator and truncation variants, the mean λ = 2 Mahalanobis AUROC on pad_heldout "
              "(z_lesion^norm) ranges from {:.3f} to {:.3f}; the Table 1 estimator gives {:.3f}.".format(
                  len(cols), min(rng), max(rng), d[d["lambda"] == 2.0]["ridge_0.001"].mean()), ""]
    L += ["## LaTeX rows (variant & λ=0 & λ=2), z_lesion^norm, pad_heldout", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    (OUT / "per_run").mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        rr = R.runs()
        assert len(rr) == 35, len(rr)
        for lam, seed, tag in rr:
            dest = OUT / "per_run" / "{}.json".format(tag)
            if not dest.exists():
                R.write_json(dest, analyse_run(lam, seed, tag))
                print("done", tag, flush=True)
        R.write_json(OUT / "headline_bootstrap.json", headline_bootstrap())
    report()


if __name__ == "__main__":
    main()
