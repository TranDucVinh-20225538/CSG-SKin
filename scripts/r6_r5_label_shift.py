#!/usr/bin/env python3
"""Paper B R6 R5: class distributions and label shift for HAM↔BCN next to ISIC↔PAD and ISIC↔Fitzpatrick17k, with the
observed outcome of each pair (decisions/paperB_r6_r5.md). CPU, metadata and existing summary files only.

Class counts
  HAM, BCN      results/paperB/phase16/bcn_ham_class_counts.json (ISIC 2019 metadata join, verified 2026-10-02)
  ISIC          lesion-level ISIC train (train_r2_item4_lesion_split.split_record; source of the Table 1 runs) and all ISIC
  PAD           PAD-UFES-20 full / pad_adv / pad_heldout (load_master + phase2 patient split)
  Fitzpatrick   the 3,887 OOD images, labels mapped where an ISIC class exists (MAP below; pyogenic granuloma → VASC as a
                sensitivity row); unmapped images are reported as a count
Shift metrics on the classes both sides can express (support = classes present in the target label space):
  TV = ½ Σ |p − q|; KL(q‖p) and KL(p‖q) with ε = 1e-4 added before renormalising; majority share in source and target;
  majority ratio q_maj / p_maj.
Outcomes (existing results, not recomputed): Mahalanobis OOD AUROC at λ = 0, at λ = 2 and the minimum over λ, mean over the
seeds present: HAM→BCN (phase16/b1_rev), BCN→HAM (phase16/b1), ISIC→pad_heldout and ISIC→Fitzpatrick (r4 lesion Table 1).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R6 / "r5"
P16 = R.RES / "phase16"
LAB = list(R.LABELS)
MAP = {"melanoma": "MEL", "basal cell carcinoma": "BCC", "squamous cell carcinoma": "SCC", "actinic keratosis": "AK",
       "dermatofibroma": "DF", "congenital nevus": "NV"}
MAP_SENS = {**MAP, "granuloma pyogenic": "VASC"}
EPS = 1e-4
P7A = R.RES / "r5" / "p7a"
P7A_SETS = {"mednode": "MED-NODE", "sd198": "SD-198", "scin": "SCIN"}
SCIN_DIR = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/data/unseen_clinical/scin")


def scin_mapping_sensitivity():
    from r5_p7a_sets import SCIN_MAP
    cases = pd.read_csv(SCIN_DIR / "scin_cases.csv", dtype={"case_id": str})
    labels = pd.read_csv(SCIN_DIR / "scin_labels.csv", dtype={"case_id": str}).set_index("case_id")
    n_img = cases[["image_1_path", "image_2_path", "image_3_path"]].notna().sum(axis=1)
    top_any = any_lab = unlab = img_top = 0
    tops = defaultdict(int)
    for cid, ni in zip(cases.case_id, n_img):
        w = labels.weighted_skin_condition_label.get(cid)
        if not (isinstance(w, str) and w.strip() not in ("", "{}")):
            unlab += 1
            continue
        d = ast.literal_eval(w)
        k = max(d.items(), key=lambda x: x[1])[0]
        tops[k] += 1
        if k in SCIN_MAP:
            top_any += 1
            img_top += int(ni)
        any_lab += int(any(x in SCIN_MAP for x in d))
    return {"cases": int(len(cases)), "cases_unlabelled": unlab, "cases_top_any_weight": top_any,
            "cases_any_label": any_lab, "images_top_any_weight": img_top,
            "top_labels": sorted(tops.items(), key=lambda x: -x[1])[:10]}


def vec(counts):
    v = np.array([float(counts.get(c, 0)) for c in LAB])
    return v


def shift(p_counts, q_counts, support):
    idx = [LAB.index(c) for c in support]
    p, q = vec(p_counts)[idx], vec(q_counts)[idx]
    p, q = p / p.sum(), q / q.sum()
    ps, qs = (p + EPS) / (p + EPS).sum(), (q + EPS) / (q + EPS).sum()
    maj = int(np.argmax(p))
    return {"support": support, "tv": float(0.5 * np.abs(p - q).sum()), "kl_q_p": float(np.sum(qs * np.log(qs / ps))),
            "kl_p_q": float(np.sum(ps * np.log(ps / qs))), "source_majority": support[maj], "p_maj": float(p[maj]),
            "q_maj": float(q[maj]), "maj_ratio": float(q[maj] / p[maj]), "p": dict(zip(support, p.round(4).tolist())),
            "q": dict(zip(support, q.round(4).tolist()))}


def phase16_outcomes(root, prefix):
    by = defaultdict(list)
    for d in sorted(root.glob(prefix + "_ladv*_s*")):
        m = re.fullmatch(prefix + r"_ladv([0-9p]+)_s(\d+)", d.name)
        if m and (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            by[float(m.group(1).replace("p", "."))].append(s["ood_auroc_maha"]["unrestricted"])
    return {lam: (float(np.mean(v)), len(v)) for lam, v in sorted(by.items())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cc = json.loads((P16 / "bcn_ham_class_counts.json").read_text())["crosstab_BCN_HAM"]
    train, _val, test = R.lesion_frames()
    spec = json.loads(R.T.p3.P2_SPLIT.read_text())
    df = R.T.p2.load_master()
    isic, pad = df[df.domain == "isic"], df[df.domain == "pad_ufes"]
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))]
    adv = pad[~pad.path.isin(set(spec["pad_heldout_paths"]))]
    fz = pd.read_csv(R.RES / "phase11_third_domain" / "fitzpatrick17k.csv").drop_duplicates("md5hash").set_index("md5hash").loc[R.fitz_stems()]
    counts = {"HAM": cc["HAM"], "BCN": cc["BCN"], "ISIC_lesion_train": train.label.value_counts().to_dict(),
              "ISIC_all": isic.label.value_counts().to_dict(), "PAD_full": pad.label.value_counts().to_dict(),
              "PAD_adv": adv.label.value_counts().to_dict(), "PAD_heldout": hold.label.value_counts().to_dict(),
              "Fitz_mapped": fz.label.map(MAP).value_counts().to_dict(), "Fitz_mapped_sens": fz.label.map(MAP_SENS).value_counts().to_dict()}
    n_fz_mapped = int(fz.label.map(MAP).notna().sum())
    p7a = {}
    for s, name in P7A_SETS.items():
        d = pd.read_csv(P7A / "sets" / "{}.csv".format(s), dtype={"group": str})
        m = d[d.mapped.notna() & (d.mapped != "")]
        counts[name + "_mapped"] = m.mapped.value_counts().to_dict()
        p7a[name] = {"images": int(len(d)), "groups": int(d.group.nunique()), "mapped_images": int(len(m)),
                     "mapped_groups": int(m.group.nunique())}
    p7a["SCIN"].update(scin_mapping_sensitivity())
    pad_sup = [c for c in LAB if counts["PAD_full"].get(c, 0) > 0]
    fz_sup = [c for c in LAB if counts["Fitz_mapped"].get(c, 0) > 0]
    fz_sup_s = [c for c in LAB if counts["Fitz_mapped_sens"].get(c, 0) > 0]
    pairs = {"HAM→BCN": ("HAM", "BCN", LAB), "BCN→HAM": ("BCN", "HAM", LAB),
             "ISIC→PAD (heldout)": ("ISIC_lesion_train", "PAD_heldout", pad_sup), "ISIC→PAD (adv target)": ("ISIC_lesion_train", "PAD_adv", pad_sup),
             "ISIC→Fitzpatrick (mapped)": ("ISIC_lesion_train", "Fitz_mapped", fz_sup),
             "ISIC→Fitzpatrick (mapped + VASC)": ("ISIC_lesion_train", "Fitz_mapped_sens", fz_sup_s)}
    for name in P7A_SETS.values():
        pairs["ISIC→{} (mapped)".format(name)] = ("ISIC_lesion_train", name + "_mapped",
                                                  [c for c in LAB if counts[name + "_mapped"].get(c, 0) > 0])
    sh = {k: shift(counts[a], counts[b], sup) for k, (a, b, sup) in pairs.items()}
    r4 = json.loads((R.RES / "r4" / "r4_results.json").read_text())["lesion_table1"]
    oc = {"HAM→BCN": phase16_outcomes(P16 / "b1_rev", "ham_bcn"), "BCN→HAM": phase16_outcomes(P16 / "b1", "bcn_ham"),
          "ISIC→PAD (heldout)": {float(k): (v["maha"]["mean"], v["maha"]["n"]) for k, v in r4.items()},
          "ISIC→Fitzpatrick (mapped)": {float(k): (v["fitz"]["mean"], v["fitz"]["n"]) for k, v in r4.items()}}
    ps = pd.read_csv(P7A / "summary.csv")
    for s, name in P7A_SETS.items():
        g = ps[ps.set == s]
        oc["ISIC→{} (mapped)".format(name)] = {float(r.Index): (float(r.mahalanobis_classcond_mean), 5)
                                               for r in g.set_index("lambda").itertuples()}
    rows = []
    for k, s in sh.items():
        o = oc.get(k, {})
        lam_min = min(o, key=lambda l: o[l][0]) if o else None
        rows.append({"pair": k, "support_k": len(s["support"]), "tv": s["tv"], "kl_q_p": s["kl_q_p"], "kl_p_q": s["kl_p_q"],
                     "source_majority": s["source_majority"], "p_maj": s["p_maj"], "q_maj": s["q_maj"], "maj_ratio": s["maj_ratio"],
                     "maha_lam0": o.get(0.0, (np.nan,))[0], "maha_lam2": o.get(2.0, (np.nan,))[0],
                     "maha_min": o[lam_min][0] if o else np.nan, "lam_at_min": lam_min,
                     "drop_lam0_to_min": (o.get(0.0, (np.nan,))[0] - o[lam_min][0]) if o else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "summary.csv", index=False)
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "counts": counts, "shift": sh, "outcomes": {k: {str(l): v for l, v in o.items()} for k, o in oc.items()},
                                        "fitz_mapped_n": n_fz_mapped, "fitz_total": int(len(fz)), "map": MAP, "map_sensitivity": MAP_SENS,
                                        "p7a_mapping": p7a})
    L = ["# Paper B R6 R5 — label shift: HAM↔BCN vs ISIC↔PAD vs ISIC↔Fitzpatrick17k", "",
         "Design: `decisions/paperB_r6_r5.md`. Commit `{}`. Metadata only; outcomes copied from existing results.".format(R.git_head()), "",
         "## Class distributions (counts)", "", "| set | n | " + " | ".join(LAB) + " |", "|---|---:|" + "---:|" * len(LAB)]
    for k, c in counts.items():
        L.append("| {} | {} | {} |".format(k, int(sum(c.values())), " | ".join(str(int(c.get(x, 0))) for x in LAB)))
    L += ["", "Fitzpatrick17k: {} of {} images map to an ISIC class (strict map); BKL has no counterpart.".format(n_fz_mapped, len(fz)), "",
          "## Share of the P7(a) unseen clinical sets that maps to the 8 ISIC classes", "",
          "Maps as in `scripts/r5_p7a_sets.py`. The outcome columns below are AUROCs on the whole set (mapped and unmapped "
          "images); the shift rows use the mapped images only. SCIN: a case maps if its top weighted dermatologist label is an "
          "ISIC class with weight ≥ 0.5 (used in P7(a)); sensitivity rows count a case if its top label maps at any weight, or "
          "if any label in its differential maps.", "",
          "| set | images | groups | mapped images | mapped groups | mapped share (images) |", "|---|---:|---:|---:|---:|---:|"]
    for name, v in p7a.items():
        L.append("| {} | {} | {} | {} | {} | {:.1%} |".format(name, v["images"], v["groups"], v["mapped_images"],
                                                            v["mapped_groups"], v["mapped_images"] / v["images"]))
    sc = p7a["SCIN"]
    L += ["", "SCIN cases: {} total, {} without any dermatologist label; top label maps at weight ≥ 0.5: {} ({:.1%}); top label "
          "maps at any weight: {} ({:.1%}); any label in the differential maps: {} ({:.1%}). Images in cases whose top label "
          "maps at any weight: {} ({:.1%}).".format(
              sc["cases"], sc["cases_unlabelled"], sc["mapped_groups"], sc["mapped_groups"] / sc["cases"], sc["cases_top_any_weight"],
              sc["cases_top_any_weight"] / sc["cases"], sc["cases_any_label"], sc["cases_any_label"] / sc["cases"],
              sc["images_top_any_weight"], sc["images_top_any_weight"] / sc["images"]),
          "Most frequent SCIN top labels: " + "; ".join("{} {}".format(k, v) for k, v in sc["top_labels"]) + ".", "",
          "## Shift and outcome", "", "| pair (source→target) | k classes | TV | KL(q‖p) | KL(p‖q) | source majority p → q | Maha λ=0 | Maha λ=2 | min over λ (λ) |",
          "|---|---:|---:|---:|---:|---|---|---|---|"]
    tex = []
    for x in tab.itertuples():
        L.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} {:.3f} → {:.3f} | {} | {} | {} |".format(
            x.pair, x.support_k, x.tv, x.kl_q_p, x.kl_p_q, x.source_majority, x.p_maj, x.q_maj, R.fmt(x.maha_lam0), R.fmt(x.maha_lam2),
            "{} ({:g})".format(R.fmt(x.maha_min), x.lam_at_min) if x.lam_at_min is not None and np.isfinite(x.maha_min) else "—"))
        tex.append("{} & {:.2f} & {:.2f} & {:.2f}$\\to${:.2f} & {} & {} \\\\".format(x.pair.replace("→", "$\\to$"), x.tv, x.kl_q_p, x.p_maj, x.q_maj,
                                                                                R.fmt(x.maha_lam0), R.fmt(x.maha_min)))
    t = tab.dropna(subset=["maha_min"]).sort_values("tv")
    L += ["", "## Plain reading", "", "Ordered by TV: " + "; ".join("{} TV {:.3f} (majority {:.2f}→{:.2f}), Mahalanobis {}→{}".format(
        x.pair, x.tv, x.p_maj, x.q_maj, R.fmt(x.maha_lam0), R.fmt(x.maha_min)) for x in t.itertuples()) + ".",
          "", "## LaTeX rows (pair & TV & KL(q‖p) & majority p→q & Maha λ=0 & min Maha)", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
