#!/usr/bin/env python3
"""R2 Item 2: does the inversion run through the NV centroid? CPU, cached features.

Rules fixed in results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json. The Mahalanobis score is not redefined:
the nearest centroid is the argmin of the standard class-conditional score.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C

OUT = C.R2 / "item2"
REPS = ("z_lesion_norm", "backbone_raw_lesion")
OODS = ("pad_heldout", "fitzpatrick17k")
P25 = C.PAPERB / "results" / "paperB" / "phase2_5"


def run_block(tr, idp, ood, rep):
    m = C.Maha(tr[rep], tr["labels"])
    did, dood = m.dists(idp[rep]), m.dists(ood[rep])
    sid, sood = did.min(1), dood.min(1)
    nn_id, nn_ood = did.argmin(1), dood.argmin(1)
    nv = nn_ood == C.NV
    rec = {
        "auroc_all": C.auroc(sid, sood),
        "frac_nv_nearest_ood": float(nv.mean()),
        "frac_nv_nearest_id": float((nn_id == C.NV).mean()),
        "n_nv_nearest_ood": int(nv.sum()),
        "auroc_nv_nearest": C.auroc(sid, sood[nv]) if nv.sum() >= 5 else None,
        "auroc_non_nv_nearest": C.auroc(sid, sood[~nv]) if (~nv).sum() >= 5 else None,
        "pred_nv_ood": int((ood["logits"].argmax(1) == C.NV).sum()),
    }
    yid = idp["labels"]
    for c, name in enumerate(C.LABELS):
        ref = np.median(did[yid == c, c])
        rec["med_ood_to_{}".format(name)] = float(np.median(dood[:, c]))
        rec["med_id{}_to_{}".format(name, name)] = float(ref)
        rec["ratio_{}".format(name)] = float(np.median(dood[:, c]) / ref)
    return rec


def gap_rule(a, b):
    """True if mean(a) - mean(b) > 2*sqrt(var_a/n_a + var_b/n_b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return bool(a.mean() - b.mean() > 2 * se), float(a.mean() - b.mean()), float(2 * se)


def weighted_share_auroc(sid, yid, sood, shares):
    return float(sum(shares[c] * C.auroc(sid[yid == c], sood) for c in shares if shares[c] > 0))


def part_c():
    pri = json.loads((P25 / "class_priors.json").read_text())
    pad_p = {int(k): v for k, v in pri["pad"].items()}
    isic_p = {int(k): v for k, v in pri["isic_test"].items()}
    rows = []
    for seed in (42, 52, 62, 72, 82):
        z = np.load(P25 / "features" / "runB_orth1_s{}.npz".format(seed))
        m = C.Maha(z["train_z"], z["train_y"])
        sid, sood, yid = m.score(z["id_z"]), m.score(z["pad_z"]), z["id_y"]
        w = np.zeros(len(sid))
        for c, pp in pad_p.items():
            if isic_p.get(c, 0) > 0:
                w[yid == c] = pp / isic_p[c]
        y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
        s = np.r_[sid, sood]
        observed = float(roc_auc_score(y, s, sample_weight=np.r_[w, np.ones(len(sood))]))
        unweighted = C.auroc(sid, sood)
        n = np.bincount(yid, minlength=8).astype(float)
        sh_isic = {c: n[c] / n.sum() for c in range(8)}
        wsum = np.array([w[yid == c].sum() for c in range(8)])
        sh_pad = {c: wsum[c] / wsum.sum() for c in range(8)}
        decomposed = weighted_share_auroc(sid, yid, sood, sh_pad)
        rest = n.sum() - n[C.NV]
        sh_nv = {c: (pad_p[C.NV] if c == C.NV else (1 - pad_p[C.NV]) * n[c] / rest) for c in range(8)}
        nv_only = weighted_share_auroc(sid, yid, sood, sh_nv)
        per_class = {C.LABELS[c]: C.auroc(sid[yid == c], sood) for c in range(8)}
        drop = unweighted - observed
        rows.append({
            "seed": seed, "unweighted": unweighted, "reweighted_observed": observed,
            "reweighted_decomposed": decomposed, "nv_only_counterfactual": nv_only,
            "fraction_of_drop_from_nv_only": float((unweighted - nv_only) / drop) if drop != 0 else None,
            "per_class_auroc_ood_vs_id_class": per_class,
            "share_isic": {C.LABELS[c]: sh_isic[c] for c in range(8)},
            "share_pad": {C.LABELS[c]: sh_pad[c] for c in range(8)},
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_run = []
    for lam, seed, tag in C.runs_with_features():
        tr, idp = C.load13(tag, "isic_train"), C.load13(tag, "isic_test")
        rec = {"lambda": lam, "seed": seed, "run": tag}
        for o in OODS:
            ood = C.load13(tag, o)
            for rep in REPS:
                rec["{}|{}".format(rep, o)] = run_block(tr, idp, ood, rep)
        per_run.append(rec)
        b = rec["z_lesion_norm|pad_heldout"]
        print(tag, "frac_nv", round(b["frac_nv_nearest_ood"], 3), "auc", round(b["auroc_all"], 3),
              "nv", b["auroc_nv_nearest"], "non", b["auroc_non_nv_nearest"], "ratioNV", round(b["ratio_NV"], 3), flush=True)

    def col(key, lam, field):
        return [r[key][field] for r in per_run if r["lambda"] == lam and r[key][field] is not None]

    # Criteria on the primary block.
    P = "z_lesion_norm|pad_heldout"
    c1_parts = {l: gap_rule(col(P, l, "frac_nv_nearest_ood"), col(P, 0.0, "frac_nv_nearest_ood")) for l in (0.25, 2.0)}
    c1 = all(v[0] for v in c1_parts.values())
    c2_parts = {}
    for l in C.LAMS[1:]:
        nvm = np.mean(col(P, l, "auroc_nv_nearest")) if col(P, l, "auroc_nv_nearest") else float("nan")
        nnm = np.mean(col(P, l, "auroc_non_nv_nearest")) if col(P, l, "auroc_non_nv_nearest") else float("nan")
        c2_parts[l] = (bool(nvm < 0.5 and nvm < nnm), float(nvm), float(nnm))
    c2 = all(v[0] for v in c2_parts.values())
    c3_parts = {l: gap_rule(col(P, 0.0, "ratio_NV"), col(P, l, "ratio_NV")) for l in (0.25, 2.0)}
    c3 = all(v[0] for v in c3_parts.values())
    held = [n for n, ok in (("C1 NV-nearest fraction rises", c1), ("C2 inversion concentrated in NV-nearest subset", c2),
                            ("C3 OOD→NV distance shrinks vs class-matched reference", c3)) if ok]
    failed = [n for n, ok in (("C1 NV-nearest fraction rises", c1), ("C2 inversion concentrated in NV-nearest subset", c2),
                              ("C3 OOD→NV distance shrinks vs class-matched reference", c3)) if not ok]
    if len(held) == 3:
        verdict = "All three hold -> the mechanism is identified."
    elif held:
        verdict = "Partial support -> do not claim the mechanism; failing criterion: {}.".format("; ".join(failed))
    else:
        verdict = "None hold -> a seventh refuted account."

    xs = [r[P]["frac_nv_nearest_ood"] for r in per_run]
    ys = [r[P]["pred_nv_ood"] for r in per_run]
    rho = spearmanr(xs, ys)
    pc = part_c()
    pc_mean = {k: C.mean_sd([r[k] for r in pc]) for k in ("unweighted", "reweighted_observed", "reweighted_decomposed",
                                                          "nv_only_counterfactual", "fraction_of_drop_from_nv_only")}
    pc_consistent = pc_mean["fraction_of_drop_from_nv_only"][0] >= 0.5

    head = C.git_head()
    js = {"commit": head, "per_run": per_run, "criteria": {
        "C1": {"holds": c1, "parts": {str(k): v for k, v in c1_parts.items()}},
        "C2": {"holds": c2, "parts": {str(k): v for k, v in c2_parts.items()}},
        "C3": {"holds": c3, "parts": {str(k): v for k, v in c3_parts.items()}}},
        "spearman_frac_nv_vs_pred_nv": [float(rho.statistic), float(rho.pvalue)],
        "part_c": pc, "part_c_mean": pc_mean, "part_c_consistent": pc_consistent, "verdict": verdict}
    (OUT / "item2_results.json").write_text(json.dumps(js, indent=2, default=float) + "\n")

    L = ["# R2 Item 2 — does the inversion run through the NV centroid?\n",
         "Commit: `{}`. Pre-commit: `results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json` (`e131a42`).\n".format(head),
         "**Verdict:** {}\n".format(verdict),
         "Standard class-conditional Mahalanobis (pooled covariance + 1e-3 I, fit on ISIC train). Nearest centroid = argmin of "
         "the same 8 distances. ID = full ISIC test. Distances are squared Mahalanobis. Mean ± s.d. across seeds.\n"]

    def table_a(key, title):
        L.append("## (a) {}\n".format(title))
        L.append("| λ | n | AUROC all | NV-nearest frac (OOD) | NV-nearest frac (ID) | AUROC NV-nearest | AUROC non-NV-nearest | predicted NV (OOD) |")
        L.append("|---:|---:|---|---|---|---|---|---|")
        for l in C.LAMS:
            g = lambda f: C.mean_sd(col(key, l, f))
            n = g("auroc_all")[2]
            L.append("| {:g} | {} | {} | {} | {} | {} | {} | {:.0f} |".format(
                l, n, C.fmt(*g("auroc_all")[:2]), C.fmt(*g("frac_nv_nearest_ood")[:2]), C.fmt(*g("frac_nv_nearest_id")[:2]),
                C.fmt(*g("auroc_nv_nearest")[:2]), C.fmt(*g("auroc_non_nv_nearest")[:2]), g("pred_nv_ood")[0]))
        L.append("")

    def table_b(key, title):
        L.append("## (b) {}\n".format(title))
        L.append("Ratio = median distance(OOD → class-c centroid) / median distance(ID test images of class c → class-c centroid).\n")
        L.append("| λ | " + " | ".join("ratio {}".format(n) for n in C.LABELS) + " | med OOD→NV | med ID-NV→NV |")
        L.append("|---:|" + "---|" * (len(C.LABELS) + 2))
        for l in C.LAMS:
            cells_ = [C.fmt(*C.mean_sd(col(key, l, "ratio_{}".format(n)))[:2], nd=2) for n in C.LABELS]
            L.append("| {:g} | {} | {} | {} |".format(l, " | ".join(cells_),
                     C.fmt(C.mean_sd(col(key, l, "med_ood_to_NV"))[0], nd=1), C.fmt(C.mean_sd(col(key, l, "med_idNV_to_NV"))[0], nd=1)))
        L.append("")

    table_a(P, "z_lesion^norm, OOD = pad_heldout (primary)")
    table_b(P, "z_lesion^norm, OOD = pad_heldout (primary)")
    L.append("## Pre-committed criteria (primary block)\n")
    L.append("| Criterion | Holds | Detail |")
    L.append("|---|---|---|")
    L.append("| C1 NV-nearest fraction rises | {} | {} |".format(
        "yes" if c1 else "no", "; ".join("λ={:g}: Δ={:.3f} vs 2SE={:.3f}".format(k, v[1], v[2]) for k, v in c1_parts.items())))
    L.append("| C2 inversion concentrated in NV-nearest subset | {} | {} |".format(
        "yes" if c2 else "no", "; ".join("λ={:g}: NV {:.3f} / non-NV {:.3f}".format(k, v[1], v[2]) for k, v in c2_parts.items())))
    L.append("| C3 OOD→NV distance shrinks vs class-matched reference | {} | {} |".format(
        "yes" if c3 else "no", "; ".join("λ={:g}: Δratio={:.3f} vs 2SE={:.3f}".format(k, v[1], v[2]) for k, v in c3_parts.items())))
    L.append("")
    L.append("Spearman(NV-nearest fraction, predicted-NV count on pad_heldout) across all runs: ρ = {:.3f} (p = {:.2g}).\n".format(
        rho.statistic, rho.pvalue))

    table_a("backbone_raw_lesion|pad_heldout", "repeat on pre-projection backbone, OOD = pad_heldout")
    table_b("backbone_raw_lesion|pad_heldout", "repeat on pre-projection backbone, OOD = pad_heldout")
    table_a("z_lesion_norm|fitzpatrick17k", "z_lesion^norm, OOD = Fitzpatrick17k (secondary)")
    table_b("z_lesion_norm|fitzpatrick17k", "z_lesion^norm, OOD = Fitzpatrick17k (secondary)")
    table_a("backbone_raw_lesion|fitzpatrick17k", "pre-projection backbone, OOD = Fitzpatrick17k (secondary)")

    L.append("## (c) Re-reading Table 4 (ID reweighted to PAD class mix)\n")
    L.append("Phase 2.5 runB_orth1 features, OOD = pad_full, 5 seeds. Weighted AUROC = Σ_c share_c · AUROC(OOD vs ID class c).\n")
    L.append("| Quantity | mean ± s.d. |")
    L.append("|---|---|")
    for k, lab in (("unweighted", "Unweighted (Table 4 col 1)"), ("reweighted_observed", "Reweighted, observed (Table 4 col 3)"),
                   ("reweighted_decomposed", "Reweighted, from per-class decomposition"),
                   ("nv_only_counterfactual", "Only NV share moved to PAD level (10.6%)"),
                   ("fraction_of_drop_from_nv_only", "Fraction of the drop reproduced by the NV-only change")):
        L.append("| {} | {} |".format(lab, C.fmt(*pc_mean[k][:2])))
    L.append("")
    L.append("| Class | ISIC share | PAD-weighted share | AUROC(OOD vs ID class), mean ± s.d. |")
    L.append("|---|---|---|---|")
    for n in C.LABELS:
        L.append("| {} | {:.3f} | {:.3f} | {} |".format(n, pc[0]["share_isic"][n], pc[0]["share_pad"][n],
                 C.fmt(*C.mean_sd([r["per_class_auroc_ood_vs_id_class"][n] for r in pc])[:2])))
    L.append("")
    L.append("Magnitude consistent with the NV hypothesis (pre-committed: NV-only change reproduces ≥ half the drop): **{}**.\n".format(
        "yes" if pc_consistent else "no"))
    L.append("## Caveats\n")
    L.append("- The class-composition test in Table 4 reweighted the ID set only; (a)–(b) here locate the OOD samples in feature space.")
    L.append("- Fitzpatrick17k is secondary; it does not enter the criteria.")
    L.append('- **The NV ratio stays above 1 at every λ; this sets the wording of the mechanism claim.** On z_lesion^norm / pad_heldout the NV ratio falls from 6.63 to 2.37 at λ=2 but never reaches 1: OOD images never get closer to the NV centroid than real ID nevi are. Per class in (c) (Phase 2.5 features, OOD = pad_full), AUROC(OOD vs ID-NV) is 0.540, near chance and not inverted, while AUROC(OOD vs every non-NV ID class) is 0.130–0.364. The inversion therefore arises because OOD moves into the NV region, where it looks less atypical than ID images of the non-NV classes look relative to their own centroids. It does not arise because OOD lands closer to the NV centroid than nevi do. The claim should be worded as "OOD collapses toward the NV centroid", not "OOD becomes more NV-like than nevi".')
    L.append('- **C1 is close to tautological with the prediction shift.** With a shared covariance, the nearest Mahalanobis centroid is a linear-discriminant classification, so the NV-nearest fraction largely restates how often OOD images are predicted NV (105 → 424 of 716 on pad_heldout; Spearman ρ = 0.982 across runs). Two counting conventions exist and answer slightly different questions. **The paper uses the 6-class PAD-restricted argmax** (113 → 426, `phase6_xfer`): how often a PAD image is called a nevus when only PAD's six diagnoses are allowed. **This item uses the 8-class argmax** (105 → 424, Phase 13 features): how often NV wins among all eight ISIC classes, the same label space as the Mahalanobis centroids. The 6-class count recomputed from Phase 13 features is 113.6 → 425.6, within one image of `phase6_xfer` (separate evaluation passes on the same checkpoints). C1 confirms the shift but adds little independent evidence; C2, C3 and (c) carry the mechanism.')
    L.append('- **At the pre-projection backbone the shift is broad, not NV-specific.** From λ=0 to λ=2 the OOD-to-centroid ratio shrinks for every class, not only NV: NV 2.76 → 1.48, MEL 1.61 → 0.88, BCC 1.07 → 0.69, AK 0.85 → 0.59, SCC 0.84 → 0.63. The NV-nearest fraction rises only from 0.48 ± 0.15 to 0.60. The NV-specific concentration is a property of the 16-d projection z_lesion^norm, not of the backbone.')
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "verdict:", verdict, flush=True)


if __name__ == "__main__":
    main()
