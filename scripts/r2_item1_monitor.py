#!/usr/bin/env python3
"""R2 Item 1: unsupervised OOD monitor on an unseen domain. CPU, cached features.

Rules fixed in results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C

OUT = C.R2 / "item1"
R50 = C.R2 / "features" / "imagenet_resnet50"
SEEDED = ("z_context", "backbone_raw_lesion", "z_lesion_norm")
REP_LABEL = {
    "imagenet_resnet50": "Frozen ImageNet ResNet-50 (2048-d)",
    "z_context": "Context branch z_c (64-d)",
    "backbone_raw_lesion": "Lesion backbone, pre-projection (1536-d)",
    "z_lesion_norm": "z_lesion^norm (16-d)",
}
GRL = {"imagenet_resnet50": "no", "z_context": "no", "backbone_raw_lesion": "yes", "z_lesion_norm": "yes"}
OODS = ("pad_heldout", "fitzpatrick17k")
IDS = ("full", "disjoint")
DETS = ("maha", "knn")
THRESH = 0.90


def score_all(ztr, ytr, zid, zoods):
    m = C.Maha(ztr, ytr)
    k = C.KNN(ztr)
    out = {"maha": {"id": m.score(zid)}, "knn": {"id": k.score(zid)}}
    for name, z in zoods.items():
        out["maha"][name] = m.score(z)
        out["knn"][name] = k.score(z)
    return out


def cells(scores, disjoint):
    res = {}
    for det in DETS:
        sid = scores[det]["id"]
        for ood in OODS:
            res[(det, ood, "full")] = C.auroc(sid, scores[det][ood])
            res[(det, ood, "disjoint")] = C.auroc(sid[disjoint], scores[det][ood])
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    disjoint, groups, te = C.isic_test_groups()
    hold_meta = C.pad_heldout_meta()

    # Frozen ImageNet ResNet-50: one lambda-independent value.
    r50 = {s: dict(np.load(R50 / "{}.npz".format(s))) for s in ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k")}
    assert (r50["isic_test"]["labels"] == te.label_idx.values).all()
    sc = score_all(r50["isic_train"]["z"], r50["isic_train"]["labels"], r50["isic_test"]["z"],
                   {o: r50[o]["z"] for o in OODS})
    base = cells(sc, disjoint)
    ood_groups = {"pad_heldout": hold_meta.patient_id.values,
                  "fitzpatrick17k": np.arange(len(r50["fitzpatrick17k"]["z"]))}
    base_ci = {}
    for (det, ood, idv), _v in base.items():
        sid = sc[det]["id"]
        gid = groups
        if idv == "disjoint":
            sid, gid = sid[disjoint], groups[disjoint]
        base_ci[(det, ood, idv)] = C.cluster_bootstrap_auroc(sid, gid, sc[det][ood], ood_groups[ood])
        print("imagenet", det, ood, idv, round(base[(det, ood, idv)], 4), base_ci[(det, ood, idv)], flush=True)

    # Seeded representations from the Phase 3 sweep.
    per_run = []
    for lam, seed, tag in C.runs_with_features():
        tr, idp = C.load13(tag, "isic_train"), C.load13(tag, "isic_test")
        oo = {o: C.load13(tag, o) for o in OODS}
        rec = {"lambda": lam, "seed": seed, "run": tag}
        for rep in SEEDED:
            s = score_all(tr[rep], tr["labels"], idp[rep], {o: oo[o][rep] for o in OODS})
            rec[rep] = {"|".join(k): v for k, v in cells(s, disjoint).items()}
        per_run.append(rec)
        print(tag, {r: round(rec[r]["maha|fitzpatrick17k|full"], 3) for r in SEEDED}, flush=True)

    agg = defaultdict(dict)
    for rep in SEEDED:
        for lam in C.LAMS:
            rs = [r for r in per_run if r["lambda"] == lam]
            for key in rs[0][rep]:
                agg[rep][(lam, key)] = C.mean_sd([r[rep][key] for r in rs])

    stays_high = all(base[("maha", o, i)] >= THRESH for o in OODS for i in IDS)
    verdict = (
        "Frozen ImageNet stays high on Fitzpatrick and pad_heldout -> recommendation stands: monitor on a "
        "representation that never received adversarial gradient; z_c is supporting evidence only (modality caveat)."
        if stays_high
        else "Frozen ImageNet degrades -> there is no cheap fix."
    )

    head = C.git_head()
    js = {
        "commit": head,
        "precommit": "results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json",
        "n_id_full": int(len(disjoint)),
        "n_id_disjoint": int(disjoint.sum()),
        "n_id_disjoint_null_lesion": int(te[disjoint].lesion_id.isna().sum()),
        "imagenet": {"|".join(k): {"auroc": v, "ci95": base_ci[k]} for k, v in base.items()},
        "per_run": per_run,
        "aggregate": {rep: {"{}|{}".format(l, k): v for (l, k), v in d.items()} for rep, d in agg.items()},
        "stays_high": stays_high,
        "verdict": verdict,
    }
    (OUT / "item1_results.json").write_text(json.dumps(js, indent=2) + "\n")

    L = []
    L.append("# R2 Item 1 — unsupervised OOD monitor on an unseen domain\n")
    L.append("Commit: `{}`. Pre-commit: `results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json` (`e131a42`).\n".format(head))
    L.append("**Verdict:** {}\n".format(verdict))
    L.append("Statistics fit on ISIC train only; no domain labels in the scoring path. Mahalanobis: class-conditional, "
             "pooled covariance + 1e-3 I. kNN: k=50, L2-normalised, mean distance. ID full = ISIC test (n={}); "
             "ID lesion-disjoint = ISIC test rows whose `lesion_id` is absent from ISIC train and val (n={}, of which {} "
             "have null `lesion_id`, kept as their own group).\n".format(
                 len(disjoint), int(disjoint.sum()), js["n_id_disjoint_null_lesion"]))
    L.append("## Primary: frozen ImageNet ResNet-50 (λ-independent baseline)\n")
    L.append("Decision rule: Mahalanobis AUROC ≥ 0.90 in all four cells → stays high.\n")
    L.append("| Detector | OOD set | ID full | 95% CI | ID lesion-disjoint | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for det in DETS:
        for o in OODS:
            a, b = base[(det, o, "full")], base[(det, o, "disjoint")]
            ca, cb = base_ci[(det, o, "full")], base_ci[(det, o, "disjoint")]
            L.append("| {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.4f} | [{:.4f}, {:.4f}] |".format(
                "Mahalanobis" if det == "maha" else "kNN k=50", o, a, ca[0], ca[1], b, cb[0], cb[1]))
    L.append("")
    L.append("CI: cluster bootstrap (2000). ID resampled by lesion group; pad_heldout by patient; Fitzpatrick17k by image.\n")

    for det in DETS:
        for o in OODS:
            L.append("## {} — {}, by representation and λ (mean ± s.d. across seeds)\n".format(
                "Mahalanobis" if det == "maha" else "kNN k=50", o))
            L.append("| Representation | Adversarial gradient | λ | n | ID full | ID lesion-disjoint |")
            L.append("|---|---|---:|---:|---|---|")
            L.append("| {} | no | all | — | {:.3f} | {:.3f} |".format(
                REP_LABEL["imagenet_resnet50"], base[(det, o, "full")], base[(det, o, "disjoint")]))
            for rep in ("z_context", "backbone_raw_lesion", "z_lesion_norm"):
                for lam in C.LAMS:
                    mf, sf, n = agg[rep][(lam, "{}|{}|full".format(det, o))]
                    md, sd, _ = agg[rep][(lam, "{}|{}|disjoint".format(det, o))]
                    L.append("| {} | {} | {:g} | {} | {} | {} |".format(
                        REP_LABEL[rep], GRL[rep], lam, n, C.fmt(mf, sf), C.fmt(md, sd)))
            L.append("")

    L.append("## Caveats required by the work order\n")
    L.append("1. Frozen ImageNet is the primary test; it has never seen any of this data. z_c is secondary: "
             "Fitzpatrick17k is clinical photography, the same modality as PAD, and z_c was trained to separate clinical "
             "photography from dermoscopy, so a high z_c score may reflect learned PAD-likeness rather than generic monitoring.")
    L.append("2. The ISIC test split is image-level and lesions recur between train and test ({} of {} test images share a "
             "`lesion_id` with train or val). The ID side is therefore reported on the lesion-disjoint subset as well as on the full split.".format(
                 int((~disjoint).sum()), len(disjoint)))
    L.append("3. Fitzpatrick17k has no patient identifiers; its bootstrap is over images and the interval is anti-conservative.")
    L.append("4. The lesion backbone (pre-projection) received adversarial gradient; it is not a remedy whatever this item returns.")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "verdict:", verdict, flush=True)


if __name__ == "__main__":
    main()
