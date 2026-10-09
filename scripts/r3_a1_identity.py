#!/usr/bin/env python3
"""R3 A1: per-class Mahalanobis AUROC for all eight ISIC classes (adds DF, VASC) and the identity
AUROC_pooled = sum_c w_c AUROC_c checked against the measured pooled value.

Same detector as R2 Item 1 (class-conditional, pooled covariance + 1e-3 I, fit on ISIC train only),
same representation (z_lesion^norm), same runs. Writes results/paperB/r3/a1/{REPORT.md,a1_results.json}."""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

OUT = C.PAPERB / "results" / "paperB" / "r3" / "a1"
SPLIT_REC = C.R2 / "item4" / "split_record.json"
LES_FEAT = C.R2 / "item4" / "features"
PHASE3 = C.PAPERB / "results" / "paperB" / "phase3_sweep"
OODS = ("pad_heldout", "fitzpatrick17k")
CLS = C.LABELS
CLS6 = ("MEL", "NV", "BCC", "AK", "BKL", "SCC")


def run_cells(tr, te, oo, masks):
    det = C.Maha(tr["z_lesion_norm"], tr["labels"].astype(int))
    sid = det.score(te["z_lesion_norm"])
    y = te["labels"].astype(int)
    out = {}
    for o in OODS:
        so = det.score(oo[o]["z_lesion_norm"])
        for idv, m in masks.items():
            rec = {"pooled": C.auroc(sid[m], so), "n": {}, "auroc": {}}
            for c in CLS:
                mm = m & (y == CLS.index(c))
                rec["n"][c] = int(mm.sum())
                rec["auroc"][c] = C.auroc(sid[mm], so) if mm.sum() > 0 else None
            out["{}|{}".format(o, idv)] = rec
    return out


def identity(rec, w):
    """sum_c w_c AUROC_c with w normalised over the classes present; also the 6-class-only version."""
    a = rec["auroc"]
    cs = [c for c in CLS if a[c] is not None and w.get(c, 0) > 0]
    W = np.array([w[c] for c in cs], float)
    s8 = float(np.dot(W / W.sum(), [a[c] for c in cs]))
    c6 = [c for c in CLS6 if a[c] is not None]
    W6 = np.array([w[c] for c in c6], float)
    s6 = float(np.dot(W6 / W6.sum(), [a[c] for c in c6]))
    return s8, s6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sr = json.loads(SPLIT_REC.read_text())
    w_split = {c: sr["test"]["label_counts"][c] / sr["test"]["n"] for c in CLS}
    disjoint, _g, _te = C.isic_test_groups()

    rows = []
    for lam, seed, tag in C.runs_with_features():
        tr, te = C.load13(tag, "isic_train"), C.load13(tag, "isic_test")
        oo = {o: C.load13(tag, o) for o in OODS}
        masks = {"full": np.ones(len(te["labels"]), bool), "disjoint": disjoint}
        cells = run_cells(tr, te, oo, masks)
        summ = json.loads((PHASE3 / tag / "summary.json").read_text()) if (PHASE3 / tag / "summary.json").exists() else None
        rows.append({"split": "image", "lambda": lam, "seed": seed, "run": tag, "cells": cells,
                     "published_pooled": {o: summ["ood"][o]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]
                                          for o in OODS} if summ else None})
        print(tag, flush=True)
    for d in sorted(LES_FEAT.glob("runB_orth1_ladv*_s*")):
        if not all((d / "{}.npz".format(s)).exists() for s in ("isic_train", "isic_test") + OODS):
            continue
        lam = float(d.name.split("ladv")[1].split("_s")[0].replace("p", "."))
        seed = int(d.name.split("_s")[-1])
        L = {s: dict(np.load(d / "{}.npz".format(s))) for s in ("isic_train", "isic_test") + OODS}
        cells = run_cells(L["isic_train"], L["isic_test"], {o: L[o] for o in OODS},
                          {"full": np.ones(len(L["isic_test"]["labels"]), bool)})
        sp = C.R2 / "item4" / "runs" / d.name / "summary.json"
        summ = json.loads(sp.read_text()) if sp.exists() else None
        rows.append({"split": "lesion", "lambda": lam, "seed": seed, "run": d.name, "cells": cells,
                     "published_pooled": {o: summ["ood"][o]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]
                                          for o in OODS} if summ else None})
        print("lesion", d.name, flush=True)

    # Identity per run: actual ID-set weights (exact) and split_record weights (as ordered).
    for r in rows:
        for key, rec in r["cells"].items():
            w_act = {c: rec["n"][c] / sum(rec["n"].values()) for c in CLS}
            s8a, s6a = identity(rec, w_act)
            s8s, s6s = identity(rec, w_split)
            rec["identity"] = {"actual_w_8": s8a, "actual_w_6": s6a, "split_record_w_8": s8s, "split_record_w_6": s6s}

    def agg(split, lam, key, f):
        xs = [f(r["cells"][key]) for r in rows if r["split"] == split and r["lambda"] == lam and key in r["cells"]]
        return C.mean_sd(xs)

    head = C.git_head()
    lines = ["# R3 A1 — per-class Mahalanobis AUROC, all eight classes, and the pooled identity", "",
             "Commit: `{}`. Detector: class-conditional Mahalanobis, pooled covariance + 1e-3 I, fit on ISIC train only; "
             "representation z_lesion^norm; OOD = positives. Image-level runs: Phase 13 features. Lesion-level runs: R2 Item 4 "
             "features. Seeds per λ: {}.".format(head, "; ".join(
                 "{} λ={:g}: {}".format(sp, l, "/".join(str(r["seed"]) for r in rows if r["split"] == sp and r["lambda"] == l))
                 for sp in ("image", "lesion") for l in sorted({r["lambda"] for r in rows if r["split"] == sp}))), ""]

    # Verdict: max |residual| with all eight classes and the actual ID-set weights.
    res8 = [abs(rec["identity"]["actual_w_8"] - rec["pooled"]) for r in rows for rec in r["cells"].values()]
    res8s = [abs(rec["identity"]["split_record_w_8"] - rec["pooled"]) for r in rows if r["split"] == "lesion"
             for rec in r["cells"].values()]
    res6 = [abs(rec["identity"]["actual_w_6"] - rec["pooled"]) for r in rows for rec in r["cells"].values()]
    mx8, mx8s, mx6 = max(res8), max(res8s) if res8s else float("nan"), max(res6)
    verdict = ("Residual falls to the third decimal -> the manuscript states the decomposition as exact and uses it only "
               "to present the per-class structure." if mx8 < 0.0005 else
               "A residual above 0.005 survives with all eight classes -> pipeline problem; see below." if mx8 > 0.005 else
               "Residual between 0.0005 and 0.005 with all eight classes; see below.")
    lines += ["**Verdict:** {}".format(verdict), "",
              "Max |Σ_c w_c AUROC_c − pooled| over every run, λ, OOD set and ID set: all eight classes with the ID set's own "
              "class weights {:.1e}; six PAD-shared classes only (renormalised) {:.4f}; lesion-level runs with "
              "`r2/item4/split_record.json` test weights {:.1e}.".format(mx8, mx6, mx8s), ""]

    for split, idvs in (("image", ("full", "disjoint")), ("lesion", ("full",))):
        lams = sorted({r["lambda"] for r in rows if r["split"] == split})
        for o in OODS:
            for idv in idvs:
                key = "{}|{}".format(o, idv)
                lines += ["## {}-level split — {} — ID {}".format(split.capitalize(), o,
                                                                    "lesion-disjoint subset" if idv == "disjoint" else "full test"), ""]
                lines += ["| λ | n | " + " | ".join(CLS) + " | Σ w_c AUROC_c (8, own w) | pooled (same features) | residual | "
                          "Σ (6 classes only) | published pooled |",
                          "|---:|---:|" + "---|" * len(CLS) + "---|---|---|---|---|"]
                for lam in lams:
                    rs = [r for r in rows if r["split"] == split and r["lambda"] == lam]
                    pc = [C.fmt(*agg(split, lam, key, lambda rec, c=c: rec["auroc"][c])[:2]) for c in CLS]
                    s8 = agg(split, lam, key, lambda rec: rec["identity"]["actual_w_8"])
                    pl = agg(split, lam, key, lambda rec: rec["pooled"])
                    rd = agg(split, lam, key, lambda rec: rec["identity"]["actual_w_8"] - rec["pooled"])
                    s6 = agg(split, lam, key, lambda rec: rec["identity"]["actual_w_6"])
                    pub = C.mean_sd([r["published_pooled"][o] for r in rs if r["published_pooled"]]) if idv == "full" else None
                    lines.append("| {:g} | {} | {} | {} | {} | {:+.1e} | {} | {} |".format(
                        lam, len(rs), " | ".join(pc), C.fmt(*s8[:2], nd=4), C.fmt(*pl[:2], nd=4), rd[0],
                        C.fmt(*s6[:2], nd=4), C.fmt(*pub[:2], nd=4) if pub and pub[2] else "—"))
                n_row = next(r for r in rows if r["split"] == split)["cells"][key]["n"]
                lines += ["", "ID class counts: " + ", ".join("{} {}".format(c, n_row[c]) for c in CLS) + ".", ""]

    if res8s:
        lines += ["## Lesion-level split, weights from `r2/item4/split_record.json`", "",
                  "| λ | OOD | Σ w_c AUROC_c (split_record w) | pooled | residual |", "|---:|---|---|---|---|"]
        for lam in sorted({r["lambda"] for r in rows if r["split"] == "lesion"}):
            for o in OODS:
                key = "{}|full".format(o)
                s = agg("lesion", lam, key, lambda rec: rec["identity"]["split_record_w_8"])
                p = agg("lesion", lam, key, lambda rec: rec["pooled"])
                d = agg("lesion", lam, key, lambda rec: rec["identity"]["split_record_w_8"] - rec["pooled"])
                lines.append("| {:g} | {} | {} | {} | {:+.1e} |".format(lam, o, C.fmt(*s[:2], nd=4), C.fmt(*p[:2], nd=4), d[0]))
        lines.append("")

    lines += ["## Caveats", "",
              "- The decomposition is an identity (AUROC averages over ID samples), not evidence. The content is the per-class values.",
              "- 'pooled (same features)' is the pooled AUROC recomputed from the same cached features and detector; 'published "
              "pooled' is the run's summary.json value (training-time evaluation pipeline), which may differ from the cached-feature "
              "value by a few images' worth of AUROC.",
              "- Image-level split_record weights do not exist; `r2/item4/split_record.json` describes the lesion-level split, "
              "so its weights are applied to the lesion-level runs only."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "a1_results.json").write_text(json.dumps({"commit": head, "verdict": verdict, "max_abs_residual_8_actual": mx8,
                                                     "max_abs_residual_6_only": mx6, "max_abs_residual_8_split_record_lesion": mx8s,
                                                     "w_split_record": w_split, "runs": rows}, indent=1) + "\n")
    print(verdict, mx8, mx6, mx8s)


if __name__ == "__main__":
    main()
