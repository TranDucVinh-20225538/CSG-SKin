"""R2: is the lesion-disjoint AUROC shift (0.427 -> 0.515 at λ=2) lesion leakage or class composition?

Rule stated before running: reweight the disjoint ID set to the full-test class mix.
  stays near the disjoint value (closer to 0.515 than to 0.427)   -> (a) leakage drives the shift
  returns near the full value (closer to 0.427)                    -> (b) class composition only
Converse: reweight the full ID set to the disjoint class mix.
Midpoints are computed per OOD set and per detector from the unweighted means.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

OUT = C.R2 / "disjoint_reweight"
OODS = ("pad_heldout", "fitzpatrick17k")
REP = "z_lesion_norm"


def wauroc(sid, wid, sood):
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=np.r_[wid, np.ones(len(sood))]))


def weights(lab, p_target):
    p_src = np.bincount(lab, minlength=8) / len(lab)
    w = np.zeros(8)
    nz = p_src > 0
    w[nz] = p_target[nz] / p_src[nz]
    return w[lab]


def main():
    disjoint, _g, _te = C.isic_test_groups()
    rows = []
    lab_ref = None
    for lam, seed, tag in C.runs_with_features():
        tr, te = C.load13(tag, "isic_train"), C.load13(tag, "isic_test")
        lab = te["labels"].astype(int)
        if lab_ref is None:
            lab_ref = lab
        assert np.array_equal(lab, lab_ref)
        p_full = np.bincount(lab, minlength=8) / len(lab)
        p_dis = np.bincount(lab[disjoint], minlength=8) / disjoint.sum()
        dets = {"Mahalanobis": C.Maha(tr[REP], tr["labels"]), "kNN k=50": C.KNN(tr[REP])}
        for dn, det in dets.items():
            sid = det.score(te[REP])
            for o in OODS:
                so = det.score(C.load13(tag, o)[REP])
                r = {"lam": lam, "seed": seed, "det": dn, "ood": o,
                     "full": C.auroc(sid, so), "disjoint": C.auroc(sid[disjoint], so),
                     "disjoint_rw_to_full": wauroc(sid[disjoint], weights(lab[disjoint], p_full), so),
                     "full_rw_to_disjoint": wauroc(sid, weights(lab, p_dis), so)}
                for c in range(8):
                    for nm, m in (("full", np.ones(len(lab), bool)), ("disjoint", disjoint)):
                        mm = m & (lab == c)
                        r["class{}_{}".format(c, nm)] = C.auroc(sid[mm], so) if mm.sum() >= 20 else None
                rows.append(r)
        print(tag, flush=True)

    p_full = np.bincount(lab_ref, minlength=8) / len(lab_ref)
    p_dis = np.bincount(lab_ref[disjoint], minlength=8) / disjoint.sum()
    n_full = np.bincount(lab_ref, minlength=8)
    n_dis = np.bincount(lab_ref[disjoint], minlength=8)
    lams = sorted({r["lam"] for r in rows})
    keys = ("full", "disjoint", "disjoint_rw_to_full", "full_rw_to_disjoint")

    def agg(det, o, lam, k):
        return C.mean_sd([r[k] for r in rows if r["det"] == det and r["ood"] == o and r["lam"] == lam])

    verdicts = {}
    for det in ("Mahalanobis", "kNN k=50"):
        for o in OODS:
            f, d = agg(det, o, 2.0, "full")[0], agg(det, o, 2.0, "disjoint")[0]
            a, b = agg(det, o, 2.0, "disjoint_rw_to_full")[0], agg(det, o, 2.0, "full_rw_to_disjoint")[0]
            mid = (f + d) / 2
            share = (d - a) / (d - f) if d != f else float("nan")
            verdicts[(det, o)] = {"full": f, "disjoint": d, "disjoint_rw_to_full": a, "full_rw_to_disjoint": b,
                                  "midpoint": mid, "share_of_shift_explained_by_class_mix": share,
                                  "reading": "(a) leakage" if abs(a - d) < abs(a - f) else "(b) class composition"}

    L = ["# R2 — lesion-disjoint shift: leakage or class composition?", "",
         "Commit: `{}`. Representation: z_lesion^norm (16-d). Scores fit on ISIC train only.".format(C.git_head()), "",
         "Rule (stated before running): reweight the 2026-image lesion-disjoint ID set to the class mix of the full 5067-image test set. "
         "If AUROC stays closer to the disjoint value than to the full value → (a) leakage drives the shift; if it returns closer to the full value → (b) class composition. "
         "Converse check: reweight the full set to the disjoint class mix. Weights per ID image w_c = p_target(c) / p_source(c); OOD weight 1.", "",
         "## 1. Class distribution", "", "| Class | Full n | Full % | Disjoint n | Disjoint % |", "|---|---:|---:|---:|---:|"]
    for c in range(8):
        L.append("| {} | {} | {:.1f} | {} | {:.1f} |".format(C.LABELS[c], n_full[c], 100 * p_full[c], n_dis[c], 100 * p_dis[c]))
    L.append("| total | {} | 100 | {} | 100 |".format(n_full.sum(), n_dis.sum()))
    L += ["", "## 2–3. λ = 2 (5 seeds, mean): reweighting both ways", "",
          "share = (disjoint − disjoint reweighted to full mix) / (disjoint − full): fraction of the full→disjoint shift attributable to class mix.", "",
          "| Detector | OOD | Full | Disjoint | Disjoint → full mix | Full → disjoint mix | share (class mix) | Reading |",
          "|---|---|---|---|---|---|---|---|"]
    for (det, o), v in verdicts.items():
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {} |".format(
            det, o, v["full"], v["disjoint"], v["disjoint_rw_to_full"], v["full_rw_to_disjoint"],
            v["share_of_shift_explained_by_class_mix"], v["reading"]))
    for det in ("Mahalanobis", "kNN k=50"):
        for o in OODS:
            L += ["", "### {} — {}, all λ (mean ± s.d.)".format(det, o), "",
                  "| λ | n | Full | Disjoint | Disjoint → full mix | Full → disjoint mix |", "|---:|---:|---|---|---|---|"]
            for lam in lams:
                ms = [agg(det, o, lam, k) for k in keys]
                L.append("| {:g} | {} | {} |".format(lam, ms[0][2], " | ".join(C.fmt(m, s) for m, s, _ in ms)))
    L += ["", "## Composition-free check: per-class AUROC (ID class c vs OOD), Mahalanobis, λ = 2, 5 seeds", "",
          "Within a class, composition cannot differ. If full and disjoint agree within each class, the aggregate shift is composition; if disjoint is higher within classes, it is leakage.", ""]
    for o in OODS:
        L += ["**{}**".format(o), "", "| Class | Full | Disjoint | Δ |", "|---|---|---|---|"]
        for c in range(8):
            a = agg("Mahalanobis", o, 2.0, "class{}_full".format(c))
            b = agg("Mahalanobis", o, 2.0, "class{}_disjoint".format(c))
            if a[2] and b[2]:
                L.append("| {} | {} | {} | {:+.3f} |".format(C.LABELS[c], C.fmt(*a[:2]), C.fmt(*b[:2]), b[0] - a[0]))
            else:
                L.append("| {} | {} | — (n<20) | — |".format(C.LABELS[c], C.fmt(*a[:2]) if a[2] else "—"))
        L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "results.json").write_text(json.dumps({"rows": rows, "verdicts_lambda2": {"|".join(k): v for k, v in verdicts.items()},
                                                  "p_full": p_full.tolist(), "p_disjoint": p_dis.tolist(), "commit": C.git_head()}, indent=1))
    for k, v in verdicts.items():
        print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
