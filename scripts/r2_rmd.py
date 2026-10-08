"""R2 RMD: Relative Mahalanobis Distance on cached Phase 13 features. Pre-commit: results/paperB/r2/PRECOMMIT_RMD.json."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

OUT = C.R2 / "rmd"
OODS = ("pad_heldout", "fitzpatrick17k")
REPS = (("z_lesion_norm", "z_lesion^norm (16-d), primary"), ("backbone_raw_lesion", "Lesion backbone, pre-projection (1536-d), secondary"))
IDS = ("full", "disjoint")


class Gauss0:
    """Single Gaussian on ISIC train ignoring labels: global mean, full covariance + eps*I."""

    def __init__(self, ztr, eps=C.MAHA_EPS):
        z = np.asarray(ztr, np.float64)
        self.mu = z.mean(0)
        cen = z - self.mu
        self.prec = np.linalg.inv(cen.T @ cen / max(len(z) - 1, 1) + eps * np.eye(z.shape[1]))

    def score(self, z):
        c = np.asarray(z, np.float64) - self.mu
        return np.einsum("nd,nd->n", c @ self.prec, c)


def frac_removed(a_rmd, a_md):
    return None if a_md >= 0.5 else float((a_rmd - a_md) / (0.5 - a_md))


def main():
    pre = json.loads((C.R2 / "PRECOMMIT_RMD.json").read_text())
    disjoint, _groups, _te = C.isic_test_groups()
    rows = []
    for lam, seed, tag in C.runs_with_features():
        tr, te = C.load13(tag, "isic_train"), C.load13(tag, "isic_test")
        oo = {o: C.load13(tag, o) for o in OODS}
        assert len(te["labels"]) == len(disjoint)
        rec = {"lam": lam, "seed": seed, "tag": tag}
        for rep, _ in REPS:
            md = C.Maha(tr[rep], tr["labels"])
            g0 = Gauss0(tr[rep])

            def both(z):
                d = md.dists(z)
                return d.min(1), (d - g0.score(z)[:, None]).min(1)

            sid_md, sid_rmd = both(te[rep])
            for o in OODS:
                so_md, so_rmd = both(oo[o][rep])
                for idv in IDS:
                    m = np.ones(len(sid_md), bool) if idv == "full" else disjoint
                    rec[(rep, o, idv, "MD")] = C.auroc(sid_md[m], so_md)
                    rec[(rep, o, idv, "RMD")] = C.auroc(sid_rmd[m], so_rmd)
        rows.append(rec)
        print(tag, {o: round(rec[("z_lesion_norm", o, "full", "RMD")], 4) for o in OODS}, flush=True)

    lams = sorted({r["lam"] for r in rows})
    agg = {}
    for rep, _ in REPS:
        for o in OODS:
            for idv in IDS:
                for lam in lams:
                    rs = [r for r in rows if r["lam"] == lam]
                    md = C.mean_sd([r[(rep, o, idv, "MD")] for r in rs])
                    rm = C.mean_sd([r[(rep, o, idv, "RMD")] for r in rs])
                    fs = [frac_removed(r[(rep, o, idv, "RMD")], r[(rep, o, idv, "MD")]) for r in rs]
                    agg[(rep, o, idv, lam)] = {"MD": md, "RMD": rm, "f_from_means": frac_removed(rm[0], md[0]), "f_per_seed": C.mean_sd(fs)}

    # pre-committed test: primary rep, ID full, lambda=2
    pred = pre["prediction"]
    test = {}
    for o in OODS:
        a = agg[("z_lesion_norm", o, "full", 2.0)]
        test[o] = {"MD": a["MD"][0], "RMD": a["RMD"][0], "f": a["f_from_means"], "predicted_f": pred["f_{}".format(o)]}
    fs = [test[o]["f"] for o in OODS]
    rmds = [test[o]["RMD"] for o in OODS]
    if any(f is None for f in fs):
        verdict = "f undefined (MD not below chance at λ=2) -> no inversion to remove in the primary cell"
    elif all(f >= 1 for f in fs) and all(x > 0.60 for x in rmds):
        verdict = "RMD removes the inversion and more (AUROC > 0.60) -> contradicts the prediction"
    elif all(0.25 <= f <= 0.75 for f in fs) and all(x <= 0.60 for x in rmds):
        verdict = "As predicted: RMD removes part of the inversion and does not repair the collapse"
    elif all(f < 0.25 for f in fs):
        verdict = "RMD does not remove the inversion (f < 0.25 on both OOD sets) -> prediction not met"
    elif all(f > 0.75 for f in fs):
        verdict = "RMD removes more of the inversion than predicted (f > 0.75 on both OOD sets){}".format(
            "; collapse not repaired (AUROC ≤ 0.60)" if all(x <= 0.60 for x in rmds) else "")
    else:
        verdict = "Mixed across OOD sets -> prediction not met as stated"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rmd_results.json").write_text(json.dumps({
        "per_run": [{k if isinstance(k, str) else "|".join(map(str, k)): v for k, v in r.items()} for r in rows],
        "aggregate": {"|".join(map(str, k)): v for k, v in agg.items()},
        "precommitted_test": test, "verdict": verdict, "commit": C.git_head()}, indent=1))

    L = ["# R2 — Relative Mahalanobis Distance (RMD)", "",
         "Commit: `{}`. Pre-commit: `results/paperB/r2/PRECOMMIT_RMD.json` (`958684a`).".format(C.git_head()), "",
         "**Verdict:** {}".format(verdict), "",
         "**RMD applies only to Mahalanobis. kNN, MSP, Energy and cosine also collapsed with λ and have no RMD analogue; whatever RMD recovers is a partial fix of the monitoring failure, not a fix.**", "",
         "Score: min_k (MD_k − MD_0). MD_k: class means, pooled within-class covariance + 1e-3 I. MD_0: one Gaussian on ISIC train ignoring labels (full covariance + 1e-3 I). All statistics fit on ISIC train only.", "",
         "## Pre-committed test (z_lesion^norm, ID = full ISIC test, λ = 2, mean of 5 seeds)", "",
         "f = (AUROC_RMD − AUROC_MD) / (0.5 − AUROC_MD): fraction of the below-chance gap removed. Predicted f = 0.5 on both sets; correct if 0.25 ≤ f ≤ 0.75 and AUROC_RMD ≤ 0.60.", "",
         "| OOD set | AUROC MD | AUROC RMD | f observed | f predicted |", "|---|---|---|---|---|"]
    for o in OODS:
        t = test[o]
        L.append("| {} | {:.3f} | {:.3f} | {} | {:.2f} |".format(o, t["MD"], t["RMD"], "—" if t["f"] is None else "{:.2f}".format(t["f"]), t["predicted_f"]))
    for rep, rname in REPS:
        for o in OODS:
            L += ["", "## {} — {} (mean ± s.d. across seeds)".format(rname, o), "",
                  "| λ | n | MD, ID full | RMD, ID full | f (per-seed mean) | MD, ID lesion-disjoint | RMD, ID lesion-disjoint |",
                  "|---:|---:|---|---|---|---|---|"]
            for lam in lams:
                a, b = agg[(rep, o, "full", lam)], agg[(rep, o, "disjoint", lam)]
                f = a["f_per_seed"]
                L.append("| {:g} | {} | {} | {} | {} | {} | {} |".format(
                    lam, a["MD"][2], C.fmt(*a["MD"][:2]), C.fmt(*a["RMD"][:2]),
                    C.fmt(*f[:2], nd=2) if f[2] else "—", C.fmt(*b["MD"][:2]), C.fmt(*b["RMD"][:2])))
    L += ["", "## Caveats", "",
          "1. f is defined only where MD is below 0.5; it is computed from seed means for the verdict and per seed in the tables (seeds with MD ≥ 0.5 excluded from the per-seed mean).",
          "2. The ISIC test split is image-level; 3041 of 5067 test images share a lesion with train or val. The lesion-disjoint columns (n = 2026) remove that overlap from the ID side.",
          "3. backbone_raw_lesion received adversarial gradient; it is reported for completeness, not as a remedy."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(verdict)


if __name__ == "__main__":
    main()
