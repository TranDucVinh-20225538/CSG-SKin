#!/usr/bin/env python3
"""R3 A2: pre-registered bootstrap (PREREGISTER_BOOTSTRAP.json, unmodified) on the lesion-level ISIC split runs.

Only change required by the split: the ID side is resampled by lesion group (null lesion_id = own group) instead of by
image. pad_heldout: patients. Fitzpatrick17k: images (anti-conservative). Shared draws across seeds within a draw."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
import bootstrap_ood_ci as B  # noqa: E402
import r2_common as C  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402

FEAT = C.R2 / "item4" / "features"
OUT = C.PAPERB / "results" / "paperB" / "r3" / "a2"
DET_LABEL = {"mahalanobis_classcond": "Mahalanobis", "knn_k50": "kNN k=50", "cosine_max": "Cosine"}


def idx_groups(g):
    _, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


def runs():
    out = {}
    for d in sorted(FEAT.glob("runB_orth1_ladv*_s*")):
        m = re.fullmatch(r"runB_orth1_ladv([0-9p]+)_s(\d+)", d.name)
        if m and all((d / "{}.npz".format(s)).exists() for s in ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k")):
            out.setdefault(float(m.group(1).replace("p", ".")), []).append(int(m.group(2)))
    return {k: sorted(v) for k, v in sorted(out.items())}


def main():
    pre = json.loads(B.PREREG.read_text())
    n_boot, rng = int(pre["n_resamples"]), np.random.default_rng(int(pre["rng_seed"]))
    _rec, (_tr, _va, test) = T.split_record()
    lesion_g, _null = T.lesion_groups(test)
    id_groups = idx_groups(lesion_g)
    hold = C.pad_heldout_meta()
    pat_groups = idx_groups(hold.patient_id.astype(str).to_numpy())
    assert len(hold) == 716 and len(pat_groups) == 412

    avail = runs()
    results = {"pad_heldout": {}, "fitzpatrick17k": {}}
    for lam, seeds in avail.items():
        sc = {det: {} for det in pre["detectors"]}
        for s in seeds:
            d = FEAT / "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), s)
            tr, te = B.load_npz(d, "isic_train"), B.load_npz(d, "isic_test")
            assert np.array_equal(te["labels"].astype(int), test.label_idx.to_numpy().astype(int)), d
            zo = {"pad": B.load_npz(d, "pad_heldout")["z_lesion_norm"], "fitz": B.load_npz(d, "fitzpatrick17k")["z_lesion_norm"]}
            for det in pre["detectors"]:
                sc[det][s] = {}
                for key, z in zo.items():
                    if det == "knn_k50":
                        sc[det][s][key] = B.knn50_scores(tr["z_lesion_norm"], te["z_lesion_norm"], z)
                    else:
                        sc[det][s][key] = B.DET_FNS[det](tr["z_lesion_norm"], tr["labels"], te["z_lesion_norm"], z)
        for ood, key in (("pad_heldout", "pad"), ("fitzpatrick17k", "fitz")):
            id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))]) for _ in range(n_boot)]
            if ood == "pad_heldout":
                ood_draws = [np.concatenate([pat_groups[j] for j in rng.integers(0, len(pat_groups), len(pat_groups))]) for _ in range(n_boot)]
            else:
                n_o = len(sc[pre["detectors"][0]][seeds[0]]["fitz"][1])
                ood_draws = list(rng.integers(0, n_o, size=(n_boot, n_o)))
            block = {}
            for det in pre["detectors"]:
                per = {}
                draws = {}
                for s in seeds:
                    sid, so = sc[det][s][key]
                    draws[s] = np.array([B.auroc_from_scores(sid[id_draws[b]], so[ood_draws[b]]) for b in range(n_boot)])
                    lo, hi = float(np.quantile(draws[s], 0.025)), float(np.quantile(draws[s], 0.975))
                    per[str(s)] = {"point": B.auroc_from_scores(sid, so), "ci_lo": lo, "ci_hi": hi, "below_chance_entire_ci": hi < 0.5}
                lo, hi, mu = B.percentile_ci(np.mean(np.stack([draws[s] for s in seeds]), 0))
                block[det] = {"point_mean_over_seeds": float(np.mean([per[str(s)]["point"] for s in seeds])),
                              "mean_across_seeds_bootstrap_ci": {"lo": lo, "hi": hi, "mean": mu},
                              "n_seeds_below_chance_ci": int(sum(p["below_chance_entire_ci"] for p in per.values())),
                              "n_seeds": len(seeds), "seeds": seeds, "per_seed": per}
                print(lam, ood, det, round(block[det]["point_mean_over_seeds"], 3), (round(lo, 3), round(hi, 3)), flush=True)
            results[ood]["{:g}".format(lam)] = block

    m = results["pad_heldout"].get("2", {}).get("mahalanobis_classcond")
    if m is None:
        verdict, rule = "λ = 2 lesion-level features missing; no decision.", None
    else:
        ci, k, n = m["mean_across_seeds_bootstrap_ci"], m["n_seeds_below_chance_ci"], m["n_seeds"]
        if ci["hi"] < 0.5:
            rule = "A" if k >= 3 else "B"
            verdict = ("Interval entirely below 0.5 -> the word 'inverts' survives the move to the lesion-level split "
                       "(pre-registered rule {}: {}/{} seeds' own CI below 0.5).".format(rule, k, n))
        elif ci["lo"] < 0.5 < ci["hi"]:
            rule = "C"
            verdict = "Interval straddles 0.5 -> 'inverts' comes out of the title and abstract (pre-registered rule C)."
        else:
            rule = "none_above"
            verdict = "Interval entirely above 0.5 -> 'inverts' comes out of the title and abstract."
    head = C.git_head()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bootstrap_lesion_results.json").write_text(json.dumps(
        {"commit": head, "preregister": "results/paperB/PREREGISTER_BOOTSTRAP.json", "id_resample_unit": "lesion group",
         "n_id_groups": len(id_groups), "n_pad_patients": len(pat_groups), "rule": rule, "verdict": verdict,
         "results": results}, indent=1) + "\n")
    L = ["# R3 A2 — pre-registered bootstrap on the lesion-level split", "",
         "Commit: `{}`. Protocol: `PREREGISTER_BOOTSTRAP.json` unmodified ({} resamples, percentile 95%, both sides resampled, "
         "draws shared across seeds). ID side resampled by lesion group ({} groups in the lesion-level ISIC test, null "
         "`lesion_id` = own group); pad_heldout by patient ({} patients, 716 images); Fitzpatrick17k by image.".format(
             head, n_boot, len(id_groups), len(pat_groups)), "",
         "**Verdict (Mahalanobis, pad_heldout, λ = 2):** {}".format(verdict), ""]
    for ood in ("pad_heldout", "fitzpatrick17k"):
        L += ["## {}".format(ood), "", "| λ | seeds | " + " | ".join(
            "{} mean [95% CI] | k below 0.5".format(DET_LABEL[d]) for d in pre["detectors"]) + " |",
              "|---:|---|" + "---|---|" * len(pre["detectors"])]
        for lam, blk in results[ood].items():
            cells = []
            for d in pre["detectors"]:
                b = blk[d]
                c = b["mean_across_seeds_bootstrap_ci"]
                cells.append("{:.3f} [{:.3f}, {:.3f}] | {}/{}".format(b["point_mean_over_seeds"], c["lo"], c["hi"],
                                                                     b["n_seeds_below_chance_ci"], b["n_seeds"]))
            L.append("| {} | {} | {} |".format(lam, "/".join(map(str, blk[pre["detectors"][0]]["seeds"])), " | ".join(cells)))
        L.append("")
    L += ["## Caveats", "",
          "- Fitzpatrick17k has no patient identifiers; its bootstrap is over images and the interval is anti-conservative.",
          "- The decision applies to Mahalanobis, the abstract's number; kNN and cosine are reported alongside.",
          "- Seeds per λ are listed in the table; λ ∈ {0, 2} reach n = 5 only once B1 (seeds 72, 82) lands."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print(verdict)


if __name__ == "__main__":
    main()
