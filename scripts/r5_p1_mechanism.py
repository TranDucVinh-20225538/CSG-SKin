#!/usr/bin/env python3
"""Paper B R5 P1: the R2 item-2 / Phase 13 / Phase 3.5 mechanism analyses on the 35 lesion-level runs
(decisions/paperB_r5_p1.md). CPU, cached features only.

Per run, per representation (z_lesion_norm primary, backbone_raw_lesion secondary), per OOD set
(pad_heldout primary, fitzpatrick17k secondary):
  (a) nevus-nearest fraction (OOD and ID), AUROC on the NV-nearest OOD subset vs the rest (r2_item2_nv_centroid.run_block)
  (b) distances: class-wise median Mahalanobis ratio OOD->c / ID-c->c (item 2) and the Phase 3.5 distance family
      (maha / euclidean-to-nearest-mean / kNN / norm) as median ratios vs ISIC test
  (c) per-class AUROC both ways: OOD images of PAD class c vs all ID; ID class c vs all OOD
  (d) decomposition check (R3 A4 identity): AUROC with ID reweighted to the OOD class mix vs sum_c w_c AUROC(ID c, OOD),
      the NV-only counterfactual and the fraction of the drop it reproduces.
Usage: r5_p1_mechanism.py [--report_only]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r2_item2_nv_centroid as I2  # noqa: E402
from eval_ood_dual_branch import fit_class_means  # noqa: E402

OUT = R.R5 / "p1"
REPS = ("z_lesion_norm", "backbone_raw_lesion")
OODS = ("pad_heldout", "fitzpatrick17k")
SIX = tuple(int(c) for c in R.PAD_CLASSES)


def nearest_l2(z, means, present):
    """eval_final_gpu.nearest_centroid_l2, chunked (1536-d backbone)."""
    prot = means[list(present)]
    z = np.asarray(z, np.float64)
    return np.concatenate([np.linalg.norm(z[i:i + 1024, None, :] - prot[None], axis=2).min(1)
                           for i in range(0, len(z), 1024)])


def distance_family(tr, te, oo, rep, maha):
    """Phase 3.5 (eval_final_gpu) distance family; medians and ratio to ISIC test medians."""
    means, present = fit_class_means(np.asarray(tr[rep]), tr["labels"].astype(int), 8)
    knn = R.C.KNN(tr[rep])
    out = {}
    for name, z in (("isic_test", te[rep]), ("ood", oo[rep])):
        out[name] = {"maha": float(np.median(maha.score(z))), "euc": float(np.median(nearest_l2(z, means, present))),
                     "knn": float(np.median(knn.score(z))), "norm": float(np.median(np.linalg.norm(z, axis=1)))}
    out["ratio_ood_over_id"] = {k: out["ood"][k] / out["isic_test"][k] for k in out["ood"]}
    return out


def per_class_and_decomposition(sid, yid, sood, yood):
    rec = {"ood_class_vs_id": {}, "id_class_vs_ood": {}}
    for c in range(8):
        if (yood == c).sum() >= 5:
            rec["ood_class_vs_id"][R.LABELS[c]] = {"n": int((yood == c).sum()), "auroc": R.auroc(sid, sood[yood == c])}
        if (yid == c).any():
            rec["id_class_vs_ood"][R.LABELS[c]] = R.auroc(sid[yid == c], sood)
    p_id = np.bincount(yid, minlength=8) / len(yid)
    p_ood = np.bincount(yood, minlength=8) / len(yood)
    w = np.array([p_ood[c] / p_id[c] if p_id[c] > 0 else 0.0 for c in yid])
    reweighted = R.auroc(sid, sood, w_id=w)
    shares = {c: p_ood[c] for c in SIX if p_ood[c] > 0}
    tot = sum(shares.values())
    identity = float(sum(shares[c] / tot * R.auroc(sid[yid == c], sood) for c in shares))
    unweighted = R.auroc(sid, sood)
    n = np.bincount(yid, minlength=8).astype(float)
    rest = n.sum() - n[R.NV]
    sh_nv = {c: (p_ood[R.NV] if c == R.NV else (1 - p_ood[R.NV]) * n[c] / rest) for c in range(8)}
    nv_only = float(sum(sh_nv[c] * R.auroc(sid[yid == c], sood) for c in range(8) if n[c] > 0))
    drop = unweighted - reweighted
    rec["decomposition"] = {
        "unweighted": unweighted, "reweighted_to_ood_mix": reweighted, "identity_sum_w_auroc_c": identity,
        "identity_residual": identity - reweighted, "nv_only_counterfactual": nv_only,
        "fraction_of_reweight_change_from_nv_only": float((unweighted - nv_only) / drop) if abs(drop) > 1e-9 else None,
        "ood_mix": {R.LABELS[c]: float(p_ood[c]) for c in range(8)}, "id_mix": {R.LABELS[c]: float(p_id[c]) for c in range(8)}}
    return rec


def analyse_run(lam, seed, tag):
    tr, te = R.load(tag, "isic_train"), R.load(tag, "isic_test")
    yid = te["labels"].astype(int)
    rec = {"lambda": lam, "seed": seed, "run": tag}
    for o in OODS:
        oo = R.load(tag, o)
        for rep in REPS:
            blk = I2.run_block(tr, te, oo, rep)
            maha = R.C.Maha(tr[rep], tr["labels"].astype(int))
            blk["distance_family"] = distance_family(tr, te, oo, rep, maha)
            if o == "pad_heldout":
                sid, sood = maha.score(te[rep]), maha.score(oo[rep])
                blk.update(per_class_and_decomposition(sid, yid, sood, oo["labels"].astype(int)))
            rec["{}|{}".format(rep, o)] = blk
    return rec


def report():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "per_run").glob("*.json"))]
    flat = []
    for r in rows:
        for key in [k for k in r if "|" in k]:
            b = r[key]
            row = {"lambda": r["lambda"], "seed": r["seed"], "block": key, "auroc_all": b["auroc_all"],
                   "frac_nv_nearest_ood": b["frac_nv_nearest_ood"], "frac_nv_nearest_id": b["frac_nv_nearest_id"],
                   "auroc_nv_nearest": b["auroc_nv_nearest"], "auroc_non_nv_nearest": b["auroc_non_nv_nearest"],
                   "pred_nv_ood": b["pred_nv_ood"], "ratio_NV": b["ratio_NV"]}
            for k, v in b["distance_family"]["ratio_ood_over_id"].items():
                row["dist_ratio_" + k] = v
            if "decomposition" in b:
                for k in ("unweighted", "reweighted_to_ood_mix", "identity_sum_w_auroc_c", "identity_residual",
                          "nv_only_counterfactual", "fraction_of_reweight_change_from_nv_only"):
                    row["dec_" + k] = b["decomposition"][k]
                for c, v in b["ood_class_vs_id"].items():
                    row["oodcls_" + c] = v["auroc"]
                for c, v in b["id_class_vs_ood"].items():
                    row["idcls_" + c] = v
            flat.append(row)
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "summary.csv", index=False)
    lams = sorted(df["lambda"].unique())

    def ms(block, col, lam):
        v = df[(df.block == block) & (df["lambda"] == lam)][col].dropna().to_numpy(float)
        return R.mean_sd(v)

    L = ["# Paper B R5 P1 — mechanism on the lesion-level runs", "",
         "Design: `decisions/paperB_r5_p1.md`. Commit `{}`. {} runs. Mahalanobis class-conditional, pooled covariance + "
         "1e-3 I, fit on ISIC train (lesion-level). ID = lesion-level ISIC test (5,067). Mean ± s.d. over seeds.".format(
             R.git_head(), len(rows)), ""]
    tex = []
    for block in sorted(df.block.unique(), key=lambda b: (b.split("|")[1] != "pad_heldout", b.split("|")[0] != "z_lesion_norm")):
        L += ["## {}".format(block), "",
              "| λ | n | AUROC all | NV-nearest (OOD) | NV-nearest (ID) | AUROC NV-nearest | AUROC rest | ratio NV | dist ratio maha / euc / kNN / norm |",
              "|---:|---:|---|---|---|---|---|---|---|"]
        for lam in lams:
            g = lambda c: ms(block, c, lam)  # noqa: E731
            n = g("auroc_all")[2]
            L.append("| {:g} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam, n, R.fmt(*g("auroc_all")[:2]), R.fmt(*g("frac_nv_nearest_ood")[:2]), R.fmt(*g("frac_nv_nearest_id")[:2]),
                R.fmt(*g("auroc_nv_nearest")[:2]), R.fmt(*g("auroc_non_nv_nearest")[:2]), R.fmt(*g("ratio_NV")[:2], nd=2),
                " / ".join(R.fmt(g("dist_ratio_" + k)[0], nd=2) for k in ("maha", "euc", "knn", "norm"))))
            if block == "z_lesion_norm|pad_heldout":
                tex.append("{:g} & {} & {} & {} & {} & {} \\\\".format(
                    lam, n, R.tex_pm(*g("auroc_all")[:2]), R.tex_pm(*g("frac_nv_nearest_ood")[:2]),
                    R.tex_pm(*g("auroc_nv_nearest")[:2]), R.tex_pm(*g("auroc_non_nv_nearest")[:2])))
        L.append("")
        if "dec_unweighted" in df and df[df.block == block]["dec_unweighted"].notna().any():
            L += ["Per-class AUROC and decomposition ({}):".format(block), "",
                  "| λ | " + " | ".join("OOD {}".format(c) for c in ("MEL", "NV", "BCC", "AK", "BKL", "SCC")) + " | " +
                  " | ".join("ID {}".format(c) for c in R.LABELS) + " | unweighted | reweighted | Σ w_c AUROC_c | NV-only | frac of change from NV |",
                  "|---:|" + "---|" * (6 + 8 + 5)]
            for lam in lams:
                g = lambda c: ms(block, c, lam)  # noqa: E731
                L.append("| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                    lam, " | ".join(R.fmt(*g("oodcls_" + c)[:2]) for c in ("MEL", "NV", "BCC", "AK", "BKL", "SCC")),
                    " | ".join(R.fmt(*g("idcls_" + c)[:2]) for c in R.LABELS),
                    R.fmt(*g("dec_unweighted")[:2]), R.fmt(*g("dec_reweighted_to_ood_mix")[:2]),
                    R.fmt(*g("dec_identity_sum_w_auroc_c")[:2]), R.fmt(*g("dec_nv_only_counterfactual")[:2]),
                    R.fmt(*g("dec_fraction_of_reweight_change_from_nv_only")[:2], nd=2)))
            res = df[df.block == block]["dec_identity_residual"].abs().max()
            L += ["", "Max |identity residual| over runs: {:.1e}.".format(res), ""]
    P = "z_lesion_norm|pad_heldout"
    if (df.block == P).any():
        f0, f2 = ms(P, "frac_nv_nearest_ood", 0.0), ms(P, "frac_nv_nearest_ood", 2.0)
        a2n, a2r = ms(P, "auroc_nv_nearest", 2.0), ms(P, "auroc_non_nv_nearest", 2.0)
        L += ["## Plain reading (primary block)", "",
              "On z_lesion^norm with pad_heldout as OOD, the share of OOD images whose nearest class mean is NV is {} at "
              "λ = 0 and {} at λ = 2 (n = {}, {}). At λ = 2 the Mahalanobis AUROC on the NV-nearest OOD subset is {} and on "
              "the rest {}. The image-level values (R2 item 2) were 0.15 → 0.59 and NV-subset AUROC below the rest; the "
              "table above gives the lesion-level values for every λ.".format(
                  R.fmt(*f0[:2]), R.fmt(*f2[:2]), f0[2], f2[2], R.fmt(*a2n[:2]), R.fmt(*a2r[:2])), ""]
    L += ["## LaTeX rows (λ & n & AUROC & NV-nearest frac & AUROC NV-nearest & AUROC rest), z_lesion^norm, pad_heldout", "",
          "```latex"] + tex + ["```", ""]
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
            if dest.exists():
                continue
            R.write_json(dest, analyse_run(lam, seed, tag))
            print("done", tag, flush=True)
    report()


if __name__ == "__main__":
    main()
