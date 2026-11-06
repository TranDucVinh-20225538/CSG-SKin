#!/usr/bin/env python3
"""Append Paper B R11 (X1–X5) entries to paper/number_manifest.json.

Reads the printed values' sources under results/paperB/r11 and refuses to
write an entry whose source does not round (half up) to `written`.
Replaces any existing id that starts with ``r11_``; every other entry is kept.

Does not edit the manuscript.
"""
from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paperB"
MANIFEST = ROOT / "paper" / "number_manifest.json"


def rnd(v, d):
    return float(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def get(data, key):
    cur = data
    for part in key:
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def entry(eid, written, file, key, label=None, context=None, near=None, note=""):
    e = {"id": eid, "written": written, "file": file, "key": list(key)}
    if label:
        e["label"] = label
    if context:
        e["context"] = context
    if near:
        e["near"] = list(near)
    if note:
        e["note"] = note
    return e


def build():
    """One entry per load-bearing number introduced for X1–X5 / app:r11."""
    x1 = "r11/x1/X1.json"
    x2 = "r11/x2/RESULTS.json"
    x3 = "r11/x3/RESULTS.json"
    x3cfg = "r11/x3/runs/x3_ladv2_shift100_s42/config.json"
    x4 = "r11/x4/RESULTS.json"
    x5 = "r11/x5/RESULTS.json"
    out = []

    def add(*a, **k):
        out.append(entry(*a, **k))

    # --- X1 counts. Parallel "Fitzpatrick17k and SD-198 ... (in: a and b; ...)".
    counts = [
        ("fitz_in", "586", "fitzpatrick17k", "in-ISIC", "fitzpatrick"),
        ("sd_in", "525", "sd198", "in-ISIC", "sd198"),
        ("fitz_out", "3{,}241", "fitzpatrick17k", "out-of-ISIC", "fitzpatrick"),
        ("sd_out", "5{,}950", "sd198", "out-of-ISIC", "sd198"),
        ("fitz_amb", "60", "fitzpatrick17k", "ambiguous", "fitzpatrick"),
        ("sd_amb", "109", "sd198", "ambiguous", "sd198"),
    ]
    for eid, written, ds, subset, label in counts:
        add("r11_x1_n_" + eid, written, x1,
            [ds, "counts", subset], label=label, context=r"ISIC class",
            note="X1 split size, {}".format(subset))

    # λ = 0 in/out means. "0.755/0.706 (Fitzpatrick17k) and 0.723/0.708 (SD-198)".
    lam0 = [
        ("fitz_in", "0.755", "fitzpatrick17k", "in-ISIC", "fitzpatrick"),
        ("fitz_out", "0.706", "fitzpatrick17k", "out-of-ISIC", "fitzpatrick"),
        ("sd_in", "0.723", "sd198", "in-ISIC", "sd198"),
        ("sd_out", "0.708", "sd198", "out-of-ISIC", "sd198"),
    ]
    for eid, written, ds, subset, label in lam0:
        add("r11_x1_l0_" + eid, written, x1,
            [ds, "by_lambda", "0.0", "subsets", subset, "mean"],
            label=label, context=r"in/out AUROC",
            note="X1 Mahalanobis mean, λ=0 {}".format(subset))

    # λ = 2 in/out means and pooled CIs. The sentence continues the same pair.
    lam2 = [
        ("fitz_in", "0.341", "fitzpatrick17k", "in-ISIC", "mean", "fitzpatrick"),
        ("fitz_in_lo", "0.322", "fitzpatrick17k", "in-ISIC", "ci_lo", "fitzpatrick"),
        ("fitz_in_hi", "0.362", "fitzpatrick17k", "in-ISIC", "ci_hi", "fitzpatrick"),
        ("fitz_out", "0.417", "fitzpatrick17k", "out-of-ISIC", "mean", "fitzpatrick"),
        ("fitz_out_lo", "0.403", "fitzpatrick17k", "out-of-ISIC", "ci_lo", "fitzpatrick"),
        ("fitz_out_hi", "0.431", "fitzpatrick17k", "out-of-ISIC", "ci_hi", "fitzpatrick"),
        ("sd_in", "0.302", "sd198", "in-ISIC", "mean", "sd198"),
        ("sd_in_lo", "0.283", "sd198", "in-ISIC", "ci_lo", "sd198"),
        ("sd_in_hi", "0.323", "sd198", "in-ISIC", "ci_hi", "sd198"),
        ("sd_out", "0.382", "sd198", "out-of-ISIC", "mean", "sd198"),
        ("sd_out_lo", "0.368", "sd198", "out-of-ISIC", "ci_lo", "sd198"),
        ("sd_out_hi", "0.395", "sd198", "out-of-ISIC", "ci_hi", "sd198"),
    ]
    for eid, written, ds, subset, stat, label in lam2:
        # 0.382 is also an M3 flag share; keep this occurrence on the λ=2 sentence.
        # 0.382 is also the M3 flag share ~760 characters later. The slash
        # form "$0.323]$/$0.382$" occurs only on the λ=2 SD-198 out mean.
        ctx = r"\\lam=2\$ it is" if written != "0.382" else r"/\$0\.382"
        add("r11_x1_l2_" + eid, written, x1,
            [ds, "by_lambda", "2.0", "subsets", subset, stat],
            label=label, context=ctx,
            note="X1 λ=2 {} {}".format(subset, stat))

    add("r11_x1_l2_paired_fitz", "-0.075", x1,
        ["fitzpatrick17k", "by_lambda", "2.0", "paired_in_minus_out", "mean"],
        label="fitzpatrick", context=r"paired in",
        note="X1 paired in−out at λ=2, Fitzpatrick17k")
    add("r11_x1_l2_paired_sd", "-0.079", x1,
        ["sd198", "by_lambda", "2.0", "paired_in_minus_out", "mean"],
        label="sd198", context=r"paired in",
        note="X1 paired in−out at λ=2, SD-198")

    # 5/5 on both parts of both datasets. Each cell is its own key; the prose
    # prints one fraction. No single dataset label: the sentence says both.
    for ds, subset in (
        ("fitzpatrick17k", "in-ISIC"), ("fitzpatrick17k", "out-of-ISIC"),
        ("sd198", "in-ISIC"), ("sd198", "out-of-ISIC"),
    ):
        tag = ds[:4] + "_" + ("in" if subset.startswith("in") else "out")
        for stat in ("k_below", "n_seeds"):
            add("r11_x1_l2_{}_{}".format(tag, stat), "5", x1,
                [ds, "by_lambda", "2.0", "subsets", subset, stat],
                context=r"both parts meet the criterion",
                note="X1 λ=2 {} {} {}".format(ds, subset, stat))

    # --- X2 monitors at λ=2. List order is pad, Fitzpatrick, MED-NODE, SD-198.
    m1 = [
        ("pad", "0.381", "pad_heldout", "pad_heldout"),
        ("fitz", "0.405", "fitzpatrick17k", "fitzpatrick"),
        ("med", "0.288", "mednode", "mednode"),
        ("sd", "0.376", "sd198", "sd198"),
    ]
    m2 = [
        ("pad", "0.708", "pad_heldout", "pad_heldout"),
        ("fitz", "0.802", "fitzpatrick17k", "fitzpatrick"),
        ("med", "0.518", "mednode", "mednode"),
        ("sd", "0.785", "sd198", "sd198"),
    ]
    m3 = [
        ("pad", "0.792", "pad_heldout", "pad_heldout"),
        ("fitz", "0.953", "fitzpatrick17k", "fitzpatrick"),
        ("med", "0.740", "mednode", "mednode"),
        ("sd", "0.943", "sd198", "sd198"),
    ]
    for mon, rows, ctx in (
        ("M1", m1, r"M1\s+meets the criterion"),
        ("M2", m2, r"M2 meets it on none"),
        ("M3", m3, r"nor does M3"),
    ):
        for eid, written, ds, label in rows:
            add("r11_x2_{}_{}".format(mon.lower(), eid), written, x2,
                ["cells", "{}|lam2|{}".format(mon, ds), "mahalanobis_classcond", "mean"],
                label=label, context=ctx,
                note="X2 {} λ=2 Mahalanobis {}".format(mon, ds))

    flags = [
        ("m1_fitz", "0.000", "M1|fitzpatrick17k", "fitzpatrick", r"M1 flags"),
        ("m1_sd", "0.002", "M1|sd198", "sd198", r"M1 flags"),
        ("m2_fitz", "0.046", "M2|fitzpatrick17k", "fitzpatrick", r"TPR, M2"),
        ("m2_sd", "0.037", "M2|sd198", "sd198", r"TPR, M2"),
        ("m3_fitz", "0.382", "M3|fitzpatrick17k", "fitzpatrick", r"M3 \$0\.382"),
        ("m3_sd", "0.335", "M3|sd198", "sd198", r"M3 \$0\.382"),
    ]
    for eid, written, cell, label, ctx in flags:
        add("r11_x2_flag_" + eid, written, x2,
            ["clinical", cell, "share_mean"],
            label=label, context=ctx,
            note="X2 share of MEL→NV (conf>0.9) flagged at 95% TPR")

    # --- X3 frozen-head adversary.
    add("r11_x3_latent", "16", x3cfg, ["latent"],
        context=r"two-layer head",
        note="X3 head width, latent in the run config")
    add("r11_x3_erm_ref", "0.558", x3, ["erm_ref"],
        context=r"ERM ID balanced accuracy|ERM:",
        note="X3 ERM reference ID balanced accuracy; same figure as the main-text ERM")
    add("r11_x3_invariant_thr", "0.60", x3, ["invariant_v1"],
        context=r"stayed above",
        note="X3 invariance threshold recorded with the run")
    add("r11_x3_v1_l0", "0.613", x3, ["cells", "lam0|shift100", "v1", "mean"],
        context=r"every cell",
        note="X3 held-out leakage at λ=0, shift 100%; also the minimum cell mean")
    add("r11_x3_v1_l0_sd", "0.043", x3, ["cells", "lam0|shift100", "v1", "sd"],
        context=r"0\.613",
        note="X3 seed s.d. of that leakage")
    add("r11_x3_v1_l2", "0.767", x3, ["cells", "lam2|shift100", "v1", "mean"],
        context=r"0\.767",
        note="X3 held-out leakage at λ=2, shift 100%")
    add("r11_x3_v1_l2_sd", "0.033", x3, ["cells", "lam2|shift100", "v1", "sd"],
        context=r"0\.767",
        note="X3 seed s.d. at λ=2")

    # --- X4 BCN→HAM.
    pair = ["bcn", "ham"]
    add("r11_x4_tv", "0.361", x4, ["tv_priors", "tv_bcn_train_vs_ham_adv"],
        label="ham", context=r"between BCN", near=pair,
        note="TV of class priors, BCN train vs HAM adversary")
    add("r11_x4_tv_2dp", "0.36", x4, ["tv_priors", "tv_bcn_train_vs_ham_adv"],
        label="ham", context=r"prior total variation", near=pair,
        note="same TV rounded to two decimals in the discussion")
    add("r11_x4_id", "0.366", x4, ["cells", "lam2|shift100", "id_bal", "mean"],
        label="bcn", context=r"ID balanced accuracy", near=pair,
        note="X4 ID balanced accuracy, λ=2, full label shift")
    add("r11_x4_id_sd", "0.017", x4, ["cells", "lam2|shift100", "id_bal", "sd"],
        label="bcn", context=r"ID balanced accuracy", near=pair,
        note="X4 seed s.d. of ID balanced accuracy")
    add("r11_x4_v1", "0.599", x4, ["cells", "lam2|shift100", "v1", "mean"],
        context=r"leakage", near=pair,
        note="X4 held-out leakage V1 at λ=2; discussion and appendix")
    add("r11_x4_v1_sd", "0.041", x4, ["cells", "lam2|shift100", "v1", "sd"],
        context=r"0\.599", near=pair,
        note="X4 seed s.d. of held-out leakage")

    # --- X5 dose grid and the printed Fitzpatrick / SD-198 ranges at λ=2.
    for i, (written, note) in enumerate((
        ("0", "t=0"), ("0.25", "t=0.25"), ("0.5", "t=0.5"),
        ("0.75", "t=0.75"), ("1", "t=1"),
    )):
        add("r11_x5_t_" + written.replace(".", "p"), written, x5, ["t_levels", str(i)],
            context=r"t\\in",
            note="X5 mixing grid {}".format(note))
    add("r11_x5_fitz_lo", "0.672", x5,
        ["cells", "lam2|t1", "point_maha", "fitzpatrick17k", "mean"],
        label="fitzpatrick", context=r"Fitzpatrick17k \$0\.672",
        note="X5 min over t of Fitzpatrick17k Mahalanobis at λ=2 (attained at t=1)")
    add("r11_x5_fitz_hi", "0.707", x5,
        ["cells", "lam2|t0.5", "point_maha", "fitzpatrick17k", "mean"],
        label="fitzpatrick", context=r"0\.672\$--\$0\.707",
        note="X5 max over t of Fitzpatrick17k Mahalanobis at λ=2 (attained at t=0.5)")
    add("r11_x5_sd_lo", "0.668", x5,
        ["cells", "lam2|t1", "point_maha", "sd198", "mean"],
        label="sd198", context=r"SD-198 \$0\.668",
        note="X5 min over t of SD-198 Mahalanobis at λ=2 (attained at t=1)")
    add("r11_x5_sd_hi", "0.698", x5,
        ["cells", "lam2|t0.25", "point_maha", "sd198", "mean"],
        label="sd198", context=r"0\.668\$--\$0\.698",
        note="X5 max over t of SD-198 Mahalanobis at λ=2 (attained at t=0.25)")
    return out


def main():
    fresh = build()
    ids = [e["id"] for e in fresh]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate r11 id")
    cache = {}
    for e in fresh:
        data = cache.setdefault(e["file"], json.loads((RES / e["file"]).read_text()))
        got = float(get(data, e["key"]))
        written = e["written"]
        d = len(written.split(".")[1]) if "." in written else 0
        want = float(written.replace("{,}", ""))
        if rnd(got, d) != rnd(want, d):
            raise SystemExit("{} source {} rounds to {}, not {}".format(
                e["id"], got, rnd(got, d), written))
    manifest = json.loads(MANIFEST.read_text())
    kept = [e for e in manifest if not str(e.get("id", "")).startswith("r11_")]
    if len(kept) != len(manifest):
        print("replacing {} existing r11_ entries".format(len(manifest) - len(kept)))
    # Historical writing-r6 manifest is the prefix. Do not drop or rewrite it.
    if len(kept) < 1120:
        raise SystemExit("refusing to shrink the pre-R11 manifest ({} < 1120)".format(len(kept)))
    old_ids = [e["id"] for e in manifest if not str(e.get("id", "")).startswith("r11_")]
    new_ids = [e["id"] for e in kept]
    if old_ids != new_ids:
        raise SystemExit("pre-R11 entry order changed")
    MANIFEST.write_text(json.dumps(kept + fresh, indent=1) + "\n")
    print("wrote {} entries ({} previous + {} r11)".format(len(kept) + len(fresh), len(kept), len(fresh)))


if __name__ == "__main__":
    main()
