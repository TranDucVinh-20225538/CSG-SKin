#!/usr/bin/env python3
"""R3 A4: provenance of the Phase 2.5b reweighted AUROC (0.240) and the per-class identity under its exact protocol.

Protocol (scripts/eval_phase25.py, weighted_auroc): Mahalanobis class-conditional on z_lesion (16-d, = z_lesion^norm),
fit on ISIC train; ID = ISIC image-level test (8 classes); OOD = PAD full (2298); sklearn roc_auc_score with
sample_weight w_i = p_pad(c_i) / p_isic(c_i) on ID, 1 on OOD; DF/VASC get weight 0 (p_pad = 0). No resampling.
Model = runB_orth1 (Phase 1 checkpoints, lambda_adv = 2), seeds 42/52/62/72/82."""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

RES = C.PAPERB / "results" / "paperB"
P25 = RES / "phase2_5"
OUT = RES / "r3" / "a4"
SEEDS = (42, 52, 62, 72, 82)
SIX = tuple(int(c) for c in C.PAD_CLASSES)


def per_class(sid, yid, sood):
    return {c: C.auroc(sid[yid == c], sood) for c in range(8) if (yid == c).any()}


def ident(pc, w):
    cs = [c for c in SIX if w.get(c, 0) > 0]
    W = np.array([w[c] for c in cs], float)
    return float(np.dot(W / W.sum(), [pc[c] for c in cs]))


def two_group(pc, yid, w_target):
    """NV vs one lumped non-NV value, the lump weighted by ISIC proportions over the five other shared classes."""
    other = [c for c in SIX if c != C.NV]
    n = np.array([(yid == c).sum() for c in other], float)
    lump = float(np.dot(n / n.sum(), [pc[c] for c in other]))
    wnv = w_target[C.NV] / sum(w_target[c] for c in SIX)
    return wnv * pc[C.NV] + (1 - wnv) * lump


def weighted(sid, yid, sood, w_target):
    p_isic = {c: (yid == c).mean() for c in range(8)}
    wi = np.array([w_target.get(c, 0.0) / p_isic[c] if p_isic[c] > 0 else 0.0 for c in yid])
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=np.r_[wi, np.ones(len(sood))]))


def mix(y):
    return {c: float((y == c).mean()) for c in range(8)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    priors = json.loads((P25 / "class_priors.json").read_text())
    pad_full_p = {int(k): v for k, v in priors["pad"].items()}
    hold_y = C.load13("runB_orth1_ladv2_s42", "pad_heldout")["labels"].astype(int)
    pad_hold_p = mix(hold_y)

    rows = []
    for s in SEEDS:
        stored = json.loads((P25 / "per_seed" / "runB_orth1_s{}.json".format(s)).read_text())
        st = stored["ood_sets"]["pad"]["mahalanobis_classcond"]
        f = np.load(P25 / "features" / "runB_orth1_s{}.npz".format(s))
        det = C.Maha(f["train_z"], f["train_y"].astype(int))
        sid, yid, sfull = det.score(f["id_z"]), f["id_y"].astype(int), det.score(f["pad_z"])
        pc_full = per_class(sid, yid, sfull)
        # pad_heldout under the same (Phase 1) checkpoint is not in phase2_5; use Phase 13 lambda=2 features of the same seed.
        tag = "runB_orth1_ladv2_s{}".format(s)
        tr, te, ho = C.load13(tag, "isic_train"), C.load13(tag, "isic_test"), C.load13(tag, "pad_heldout")
        d13 = C.Maha(tr["z_lesion_norm"], tr["labels"].astype(int))
        sid13, y13, sho = d13.score(te["z_lesion_norm"]), te["labels"].astype(int), d13.score(ho["z_lesion_norm"])
        pc_hold = per_class(sid13, y13, sho)
        rows.append({
            "seed": s,
            "stored_unrestricted": st["unrestricted"]["AUROC"], "stored_reweighted": st["id_reweighted_to_pad"],
            "recomputed_unrestricted": C.auroc(sid, sfull), "recomputed_reweighted_pad_full": weighted(sid, yid, sfull, pad_full_p),
            "identity_pad_full_w_pad_full_auroc": ident(pc_full, pad_full_p),
            "two_group_pad_full": two_group(pc_full, yid, pad_full_p),
            "per_class_pad_full": {C.LABELS[c]: v for c, v in pc_full.items()},
            "measured_reweighted_pad_heldout": weighted(sid13, y13, sho, pad_hold_p),
            "identity_pad_heldout": ident(pc_hold, pad_hold_p),
            "two_group_pad_heldout": two_group(pc_hold, y13, pad_hold_p),
            "per_class_pad_heldout": {C.LABELS[c]: v for c, v in pc_hold.items()},
            "identity_pad_heldout_w_on_pad_full_auroc": ident(pc_full, pad_hold_p),
            "identity_pad_full_w_on_pad_heldout_auroc": ident(pc_hold, pad_full_p),
        })
        print(s, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)

    def a(k):
        return C.mean_sd([r[k] for r in rows])

    m_st = a("stored_reweighted")
    m_id = a("identity_pad_full_w_pad_full_auroc")
    resid = max(abs(r["identity_pad_full_w_pad_full_auroc"] - r["stored_reweighted"]) for r in rows)
    verdict = ("The identity reproduces 0.240 -> the manuscript states that the per-class decomposition reproduces all three "
               "manipulations; the wrong explanation is deleted outright." if resid < 5e-4 else
               "A residual survives ({:.4f}); see table.".format(resid))
    head = C.git_head()
    pc = lambda key: {c: C.mean_sd([r[key][c] for r in rows]) for c in C.LABELS if c in rows[0][key]}
    L = ["# R3 A4 — PAD-reweight discrepancy", "", "Commit: `{}`.".format(head), "", "**Verdict:** {}".format(verdict), "",
         "## Provenance of 0.240 (Phase 2.5b)", "",
         "| Question | Answer |", "|---|---|",
         "| Source | `results/paperB/phase2_5/per_seed/runB_orth1_s*.json` → `ood_sets.pad.mahalanobis_classcond.id_reweighted_to_pad`; "
         "aggregated in `PHASE2_5_REPORT.md` 2.5b (0.240 ± 0.014); code `scripts/eval_phase25.py::weighted_auroc` |",
         "| PAD mix | **pad_full** (n = 2298; `phase2_5/class_priors.json`), not pad_heldout (716) |",
         "| Model / λ | `runB_orth1` Phase 1 checkpoints, λ_adv = 2, image-level ISIC split |",
         "| Seeds | 42, 52, 62, 72, 82 (n = 5) |",
         "| Weighting | exact `sample_weight` in `roc_auc_score`, w = p_pad(c) / p_isic(c) per ID image, OOD weight 1; no resampling |",
         "| Six-class restriction | implicit, inside the weighting: DF and VASC get w = 0 because p_pad = 0 (no separate restriction step) |",
         "| Seed s.d. | {:.3f} (n = 5) |".format(m_st[1]), "",
         "PAD class mixes: pad_full " + ", ".join("{} {:.3f}".format(C.LABELS[c], pad_full_p[c]) for c in SIX) +
         "; pad_heldout " + ", ".join("{} {:.3f}".format(C.LABELS[c], pad_hold_p[c]) for c in SIX) + ".", "",
         "## Identity under the exact protocol (pad_full), mean ± s.d. over 5 seeds", "",
         "| Quantity | Value |", "|---|---|",
         "| Stored reweighted AUROC (the 0.240) | {} |".format(C.fmt(*m_st[:2], nd=4)),
         "| Recomputed reweighted AUROC, same features | {} |".format(C.fmt(*a("recomputed_reweighted_pad_full")[:2], nd=4)),
         "| Σ_c w_c AUROC_c, six classes, pad_full weights, per-class AUROC on pad_full | {} |".format(C.fmt(*m_id[:2], nd=4)),
         "| max per-seed \\|identity − stored\\| | {:.1e} |".format(resid),
         "| Two-group form (NV vs non-NV lumped at ISIC proportions), pad_full weights | {} |".format(C.fmt(*a("two_group_pad_full")[:2], nd=4)),
         "| Stored unrestricted AUROC on pad_full | {} |".format(C.fmt(*a("stored_unrestricted")[:2], nd=4)), "",
         "## Where 0.2585 comes from", "",
         "The 0.2585 prediction combines pad_heldout weights with per-class AUROCs on pad_heldout and is compared with a "
         "measurement on pad_full. Same five seeds, Phase 13 λ = 2 features for pad_heldout:", "",
         "| Weights | Per-class AUROCs on | Σ w_c AUROC_c | Measured reweighted AUROC (same OOD set) |", "|---|---|---|---|",
         "| pad_full | pad_full | {} | {} |".format(C.fmt(*m_id[:2], nd=4), C.fmt(*m_st[:2], nd=4)),
         "| pad_heldout | pad_heldout | {} | {} |".format(C.fmt(*a("identity_pad_heldout")[:2], nd=4), C.fmt(*a("measured_reweighted_pad_heldout")[:2], nd=4)),
         "| pad_heldout | pad_full | {} | — |".format(C.fmt(*a("identity_pad_heldout_w_on_pad_full_auroc")[:2], nd=4)),
         "| pad_full | pad_heldout | {} | — |".format(C.fmt(*a("identity_pad_full_w_on_pad_heldout_auroc")[:2], nd=4)),
         "| two-group, pad_heldout weights | pad_heldout | {} | — |".format(C.fmt(*a("two_group_pad_heldout")[:2], nd=4)), "",
         "## Per-class Mahalanobis AUROC (ID class c vs OOD), mean ± s.d.", "",
         "| Class | vs pad_full (Phase 1 ckpt) | vs pad_heldout (Phase 13 λ=2) |", "|---|---|---|"]
    pf, ph = pc("per_class_pad_full"), pc("per_class_pad_heldout")
    for c in C.LABELS:
        L.append("| {} | {} | {} |".format(c, C.fmt(*pf[c][:2]), C.fmt(*ph[c][:2])))
    L += ["", "## Caveats", "",
          "- Reweighting the ID set changes only w_c; model, scores and OOD set are unchanged, so every AUROC_c is unchanged. "
          "The manuscript sentence attributing the gap to shifted per-class AUROCs is deleted in every outcome.",
          "- The pad_heldout rows use Phase 13 features of the λ = 2 sweep runs (the same Phase 1 checkpoints re-extracted); "
          "the pad_full rows use the Phase 2.5 features."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "a4_results.json").write_text(json.dumps({"commit": head, "verdict": verdict, "pad_full_mix": pad_full_p,
                                                     "pad_heldout_mix": pad_hold_p, "per_seed": rows}, indent=1) + "\n")
    print(verdict)


if __name__ == "__main__":
    main()
