#!/usr/bin/env python3
"""Report generator for Paper B R5 P8, R5 P7(b), R6 R1 and R6 R2 (CPU; reads features and summaries only).

Writes <item dir>/summary.csv, cells.json, REPORT.md (markdown tables, LaTeX rows in Table 1 column order, one plain
paragraph per result with the prediction from the decisions file next to the observed numbers). P8 also writes
RECIPE_DECISION.json (competence flag and the extension gate) once its λ ∈ {0, 2} cells have 5 seeds.

Usage: r6_report.py --item {p8,r1,r2,p7b} [--recipe current|p8|auto] [--no_boot]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

LES_RUNS = R.C.R2 / "item4" / "runs"
P8_MAIN = R.P8_DIR / "main"
SHIFT_LABEL = {100: "shift 100%", 50: "shift 50%", 0: "shift 0%"}
PRED = {
    "p8": "selected recipe raises λ = 0 val bal acc to 0.60–0.70 (current 0.47 on val); ID test bal acc at λ = 0 "
          "0.60–0.72; Mahalanobis pad_heldout at λ = 0 0.75–0.85; at λ = 2 0.38–0.48 with rule A met (inversion "
          "survives a stronger classifier); Fitzpatrick at λ = 2 below 0.5.",
    "r1": "inversion weakens as shift -> 0 and AUROC returns above 0.5 at 0% (work order). Numerically, Mahalanobis "
          "pad_heldout at λ = 2: shift 100% 0.38–0.45 (rule A met), 50% 0.45–0.60, 0% 0.55–0.75 (rule A not met); "
          "pad_heldout NV-nearest fraction falls with shift; Table 1 leakage at 0% within 0.05 of 100%.",
    "r2": "CDAN: inversion persists at λ = 2 (Mahalanobis pad_heldout 0.38–0.50). IWDAN / IWCDAN: Mahalanobis "
          "pad_heldout 0.55–0.80 at λ = 2, leakage below the λ = 0 value by ≥ 0.10 but above DANN's, AK recall above "
          "DANN's at the same λ, estimated PAD NV share 0.08–0.25 (true pad_adv 0.099), AK share underestimated.",
    "p7b": "with PAD outside the adversary: Mahalanobis on all of PAD ≥ 0.60 at every λ, drop from λ = 0 to λ = 2 "
           "≤ 0.15, rule A not met; Fitzpatrick likewise above 0.5; BCN/HAM leakage falls with λ.",
}


def item_dir(item, rt):
    return {"p8": R.P8_DIR, "r1": R.RES / "r6" / "r1" / rt, "r2": R.RES / "r6" / "r2" / rt, "p7b": R.RES / "r5" / "p7b" / rt}[item]


def existing(pairs):
    return {s: (f, r) for s, (f, r) in pairs.items() if (Path(f) / "isic_test.npz").exists()}


def ref_runs(rt, lam):
    """Same-recipe two-stream DANN runs (λ = 0 baselines and DANN references)."""
    if rt == "cur":
        seeds = R.SEEDS if lam in (0.0, 2.0) else (42, 43, 44, 45, 46)
        tag = "runB_orth1_ladv{}_s{}"
        return existing({s: (R.ITEM4_FEAT / tag.format(R.lam_tag(lam), s), LES_RUNS / tag.format(R.lam_tag(lam), s))
                         for s in seeds})
    tag = "main_ladv{}_s{}"
    return existing({s: (P8_MAIN / "features" / tag.format(R.lam_tag(lam), s), P8_MAIN / "runs" / tag.format(R.lam_tag(lam), s))
                     for s in R.SEEDS})


def own_runs(root, fmt, lam, feat="features", **kw):
    return existing({s: (root / feat / fmt.format(lam=R.lam_tag(lam), s=s, **kw),
                         root / "runs" / fmt.format(lam=R.lam_tag(lam), s=s, **kw)) for s in R.SEEDS})


def metrics(cells_src, site_test=None, from_features=False):
    cells = []
    for label, lam, pairs in cells_src:
        ms = {}
        for s, (f, r) in pairs.items():
            m, _ = R.metrics_for(f, None if from_features else r, site_test=site_test)
            if m is not None and "ood" in m:
                ms[s] = m
        cells.append((label, lam, ms))
    return cells


def boots_for(cells_src, ood_names, data, id_groups, dets=None):
    out = {}
    for name in ood_names:
        try:
            og, unit = R.ood_groups(name, data)
        except FileNotFoundError:
            continue
        for label, lam, pairs in cells_src:
            if pairs:
                b = R.bootstrap_cell({s: f for s, (f, _r) in pairs.items()}, name, id_groups, og, dets=dets)
                if b is not None:
                    b["unit_ood"] = unit
                    out[(label, lam, name)] = b
    return out


def fmt_ci(b, det="mahalanobis_classcond"):
    if b is None:
        return "—"
    x = b[det]
    sd = "" if x["sd"] is None else " ± {:.3f}".format(x["sd"])
    return "{:.3f}{} [{:.3f}, {:.3f}], {}/{} seed CIs below 0.5, rule A {}".format(
        x["mean"], sd, x["ci_lo"], x["ci_hi"], x["k_below"], x["n_seeds"], "met" if b["rule_A"] else "not met")


def mean_of(cell, key_fn):
    vals = [key_fn(m) for m in cell[2].values()]
    return R.ms(vals)


def flag_lines(cells, primary="pad_heldout", leak=lambda m: m["leakage"]["bal_acc_mean"], leak_name="Table 1 leakage"):
    """Pre-registered flags (flag, never drop): λ = 0 Mahalanobis in [0.45, 0.55] -> 'no headroom'; λ > 0 cell whose
    mean leakage is not at least 0.05 below the λ = 0 cell's -> 'adversary not engaged'."""
    base = next((c for c in cells if c[1] == 0.0 and c[2]), None)
    b_leak = mean_of(base, leak) if base else None
    out = ["### Flags ({} vs the λ = 0 cell; Mahalanobis {} headroom)".format(leak_name, primary), "",
           "| cell | λ | leakage | Δ vs λ = 0 | flag |", "|---|---:|---|---|---|"]
    for c in cells:
        lk = mean_of(c, leak)
        maha = mean_of(c, lambda m: m["ood"][primary]["mahalanobis_classcond"])
        if c[1] == 0.0:
            flag = "no headroom" if maha and 0.45 <= maha["mean"] <= 0.55 else "—"
            delta = "—"
        else:
            d = None if (lk is None or b_leak is None) else lk["mean"] - b_leak["mean"]
            delta = "—" if d is None else "{:+.3f}".format(d)
            flag = "adversary not engaged" if d is not None and d > -0.05 else "—"
        out.append("| {} | {:g} | {} | {} | {} |".format(c[0], c[1], R.pm(lk), delta, flag))
    return out + [""]


def standard_sections(L, tex_all, cells, boots, ood_sets, primary="pad_heldout", title="", **flag_kw):
    md, tex, _agg = R.cell_table(cells, primary)
    L += flag_lines(cells, primary, **flag_kw)
    L += ["## Table 1 columns{}".format(title), "",
          "Leakage = logistic domain probe on z_lesion_norm, ISIC test vs all of PAD (Table 1 definition). Mean ± s.d. "
          "over seeds; n = seeds present.", ""] + md + [""]
    tex_all += tex
    for o in ood_sets:
        if any(o in m.get("ood", {}) for c in cells for m in c[2].values()):
            L += ["### Detectors on {} (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap){}".format(o, title), ""]
            L += R.detector_lines(cells, boots, o) + [""]
    for o in ("pad_heldout", "pad_full"):
        if any(o in m.get("ood", {}) for c in cells for m in c[2].values()):
            L += ["### Per-class Mahalanobis AUROC and decomposition check on {}{}".format(o, title), "",
                  "ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the "
                  "OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the "
                  "reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs "
                  "the OOD images of that class group.", ""]
            L += R.decomposition_lines(cells, o) + [""]


def label_shift_extras(L, cells, title=""):
    L += ["### Cross-domain behaviour on pad_heldout{}".format(title), "",
          "| cell | λ | AK recall (6-cls) | NV recall | NV predicted share | ID NV-nearest | pad_heldout NV-nearest | "
          "leakage (class-matched ISIC subsample) |", "|---|---:|---|---|---|---|---|---|"]
    for c in cells:
        L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} |".format(
            c[0], c[1], R.pm(mean_of(c, lambda m: m["xdom_pad_heldout"]["recall_6cls"]["AK"])),
            R.pm(mean_of(c, lambda m: m["xdom_pad_heldout"]["recall_6cls"]["NV"])),
            R.pm(mean_of(c, lambda m: m["xdom_pad_heldout"]["pred_share_6cls"]["NV"])),
            R.pm(mean_of(c, lambda m: m.get("id_nv_nearest_frac"))),
            R.pm(mean_of(c, lambda m: m["ood"]["pad_heldout"]["nv_nearest_frac"])),
            R.pm(mean_of(c, lambda m: m.get("leakage_classmatched", {}).get("bal_acc_mean")))))
    L.append("")


def write(item_root, L, tex, cells_by_table, item, extra_json=None):
    import pandas as pd

    frames = [R.summary_rows(item, cells, prim) for cells, prim in cells_by_table if cells]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    item_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(item_root / "summary.csv", index=False)
    L += ["## LaTeX rows (cell & λ & n & leakage & ID bal acc & ID ECE & OOD ECE & OOD AUROC & Fitz AUROC & X-dom)", "",
          "```latex"] + tex + ["```", ""]
    (item_root / "REPORT.md").write_text("\n".join(L) + "\n")
    cj = {"commit": R.C.git_head(), "cells": {"{}|{:g}".format(c[0], c[1]): {str(s): m for s, m in c[2].items()}
                                              for cells, _p in cells_by_table for c in cells}}
    if extra_json:
        cj.update(extra_json)
    (item_root / "cells.json").write_text(json.dumps(cj, indent=1, default=float) + "\n")
    print("wrote", item_root / "REPORT.md", len(df), "runs")


# ----------------------------------------------------------------------------------------------------------------------
def report_p8(a, data, idg):
    root = R.P8_DIR
    L = ["# Paper B R5 P8 — stronger dual-encoder base model", "",
         "Design: `decisions/paperB_r5_p8.md`. Commit `{}`. Code: `scripts/train_r5_p8.py`, `scripts/r6_common.py`, "
         "this report `scripts/r6_report.py --item p8`.".format(R.C.git_head()), ""]
    # stage A/B grid (validation split only)
    rows = []
    for p in sorted((root / "grid" / "runs").glob("*/summary.json")):
        s = json.loads(p.read_text())
        rows.append((s["recipe"]["name"], s["seed"], s["metrics"]["val"]))
    L += ["## Recipe search (λ = 0, lesion-level validation split only)", "",
          "| recipe | latent | cls LR × | balance | epochs | seeds | val bal acc | val acc | val ECE | min class recall |",
          "|---|---:|---:|---|---:|---|---|---|---|---|"]
    names = sorted({r[0] for r in rows})
    for n in names:
        rs = [r for r in rows if r[0] == n]
        rec = R.resolve_recipe(n)
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            n, rec["latent"], rec["cls_lr_mult"], rec["balance"], rec["max_epochs"], "/".join(str(r[1]) for r in rs),
            R.pm(R.ms([r[2]["bal_acc"] for r in rs])), R.pm(R.ms([r[2]["acc"] for r in rs])),
            R.pm(R.ms([r[2]["ece"] for r in rs])), R.pm(R.ms([min(r[2]["recall"].values()) for r in rs]))))
    erm = []
    for p in sorted((root / "erm_ref").glob("single_dann_ladv0_s*/summary.json")):
        e = json.loads(p.read_text())
        if e.get("checkpoint_policy") == R.CKPT_POLICY:
            erm.append(e)
    if not erm:
        erm = [json.loads(p.read_text()) for p in sorted((root / "erm_ref").glob("single_dann_ladv0_s*/summary.json"))]
    erm_ms = R.ms([e["id_balanced_acc"] for e in erm])
    L += ["", "Lesion-level single-encoder ERM reference (ID test bal acc, last.ckpt, seeds {}): {}. "
          "Image-level single-encoder ERM in the paper: {:.3f} (leakage-inflated on the image-level split; "
          "the 0.70 competence threshold was set from that figure; original ≥ 0.70 verdicts stay visible; "
          "this lesion-level mean ± SD is the competence reference). Full detector AUROCs: "
          "`results/paperB/r5/p8/erm_ref/REPORT.md`.".format(
              "/".join(str(e["seed"]) for e in erm) or "—", R.pm(erm_ms), R.ERM_SINGLE_ENCODER_IMAGE_LEVEL), ""]
    sel = json.loads((root / "selection_stageB.json").read_text()) if (root / "selection_stageB.json").exists() else None
    selA = json.loads((root / "selection_stageA.json").read_text()) if (root / "selection_stageA.json").exists() else None
    if selA:
        L += ["Stage A winner: {} (within-tie set {}); rule: {}.".format(selA["winner"], selA["within_tie"], selA["rule"])]
    if sel:
        gain = sel.get("gain_e80_minus_e40")
        if gain is None:
            L += ["Selected recipe: {} ({}).".format(sel["recipe"], sel["rule"]), ""]
        else:
            L += ["Selected recipe: {} (80-epoch gain {:+.3f}; {}).".format(sel["recipe"], gain, sel["rule"]), ""]
    rule = json.loads((root / "MAIN_RECIPE_RULE.json").read_text()) if (root / "MAIN_RECIPE_RULE.json").exists() else None
    if rule:
        L += ["Main-recipe rule (λ = 0, lesion-level val, last.ckpt, 5 seeds): D = {:+.4f}, 95% paired t CI "
              "[{:+.4f}, {:+.4f}]; outcome **{}** (RECIPE=auto → {}); P8 ext: {}.".format(
                  rule["D"], rule["ci"][0], rule["ci"][1], rule["outcome"], rule["main_recipe"], rule["p8_ext"]), ""]
    tex, tables = [], []
    paragraphs = []
    if sel:
        oods_boot = ["pad_heldout", "fitzpatrick17k"]
        oods_all = oods_boot + ["pad_full"] + list(R.P7A_NAMES)
        boots_by_ck, cells_by_ck = {}, {}
        for ck, feat, from_feat, title in (
                ("last.ckpt", "features", False, " — A4 last.ckpt (primary)"),
                ("best-val", "features_bestval", True, " — A4 best-val (secondary)")):
            src = [("A4 ({})".format(ck), lam, own_runs(root / "main", "main_ladv{lam}_s{s}", lam, feat=feat))
                   for lam in (0.0, 0.5, 2.0)]
            src = [c for c in src if c[2]]
            if not src:
                continue
            boots = {} if a.no_boot else boots_for(src, oods_boot, data, idg, dets=R.DETS)
            cells = metrics(src, from_features=from_feat)
            boots_by_ck[ck], cells_by_ck[ck] = boots, cells
            standard_sections(L, tex, cells, boots, oods_all, title=title)
            label_shift_extras(L, cells, title)
            tables.append((cells, "pad_heldout"))
            L += ["### Bootstrap units{}".format(title), "",
                  "ID: lesion. pad_heldout: patient. Fitzpatrick17k: {}.".format(
                      boots.get((src[0][0], src[0][1], "fitzpatrick17k"), {}).get("unit_ood", "—")), ""]
            for lam in (0.5, 2.0):
                for o in oods_boot:
                    b = boots.get(("A4 ({})".format(ck), lam, o))
                    if b is None:
                        continue
                    L.append("A4 λ = {:g}, {}, {}: Mahalanobis {}; rule A {}; units: ID lesion, OOD {}.".format(
                        lam, ck, o, fmt_ci(b), "met" if b["rule_A"] else "not met", b.get("unit_ood", "—")))
            L.append("")
        boots_last, cells_last = boots_by_ck.get("last.ckpt", {}), cells_by_ck.get("last.ckpt", [])
        c0 = next((c for c in cells_last if c[1] == 0.0), None)
        idb = mean_of(c0, lambda m: m["id"]["bal_acc"]) if c0 else None
        b2 = boots_last.get(("A4 (last.ckpt)", 2.0, "pad_heldout"))
        competent = bool(idb and idb["n"] == 5 and idb["mean"] >= R.COMPETENCE_TARGET)
        dec = {"competent": competent, "id_bal_acc_lambda0": idb, "target": R.COMPETENCE_TARGET,
               "inversion_survives": bool(b2 and b2["rule_A"]), "rule_A_block": b2 and b2["mahalanobis_classcond"],
               "recipe": sel["recipe"], "main_recipe_rule": rule,
               "note": "competence/inversion flags only; main recipe is MAIN_RECIPE_RULE.json"}
        (root / "RECIPE_DECISION.json").write_text(json.dumps(dec, indent=1, default=float) + "\n")
        paragraphs.append(
            "Competence. Under A4 (last.ckpt) the λ = 0 model reaches ID test balanced accuracy {} (n = {}), against "
            "the target {:.2f} (set from the image-level single-encoder ERM {:.3f}, which is leakage-inflated on the "
            "image-level split; the lesion-level single-encoder ERM reference, last.ckpt, is {}). "
            "Competence flag (≥ {:.2f}): {}.".format(
                R.pm(idb), idb["n"] if idb else 0, R.COMPETENCE_TARGET, R.ERM_SINGLE_ENCODER_IMAGE_LEVEL, R.pm(erm_ms),
                R.COMPETENCE_TARGET, "yes" if competent else "no"))
        paragraphs.append(
            "Main-recipe rule. D = {:+.4f} [{:+.4f}, {:+.4f}]; outcome {}; A4 reported as a robustness check at 3 λ; "
            "P8 ext {}.".format(rule["D"], rule["ci"][0], rule["ci"][1], rule["outcome"], rule["p8_ext"]) if rule else
            "Main-recipe rule file missing.")
        paragraphs.append("Inversion under A4 last.ckpt. Mahalanobis on pad_heldout at λ = 0: {}; at λ = 2: {}. "
                          "Fitzpatrick at λ = 2: {}. Prediction: {}".format(
                              fmt_ci(boots_last.get(("A4 (last.ckpt)", 0.0, "pad_heldout"))), fmt_ci(b2),
                              fmt_ci(boots_last.get(("A4 (last.ckpt)", 2.0, "fitzpatrick17k"))), PRED["p8"]))
    # current-recipe reference, separate table (recipes never mixed)
    src_c = [("current recipe (R4 Table 1)", lam, ref_runs("cur", lam)) for lam in (0.0, 0.5, 2.0)]
    cells_c = metrics([c for c in src_c if c[2]])
    md, tex_c, _ = R.cell_table(cells_c)
    L += ["## Reference: current recipe, lesion-level (separate table; recipes never mixed)", ""] + md + [""]
    tables.append((cells_c, "pad_heldout"))
    L += ["## Results in plain words", ""] + [p + "\n" for p in paragraphs]
    write(root, L, tex + ["% current recipe reference"] + tex_c, tables, "p8")


def report_r1(a, data, idg, rt):
    root = item_dir("r1", rt)
    fmt = "r1_" + rt + "_shift{sh}_ladv{lam}_s{s}"
    src = [("λ=0 baseline (shift 100%)", 0.0, own_runs(root, fmt, 0.0, sh=100))]
    for lam in (0.5, 2.0):
        for sh in (100, 50, 0):
            src.append((SHIFT_LABEL[sh], lam, own_runs(root, fmt, lam, sh=sh)))
    src = [c for c in src if c[2]]
    oods = ["pad_heldout", "fitzpatrick17k"] + list(R.P7A_NAMES)
    boots = {} if a.no_boot else boots_for(src, oods, data, idg)
    cells = metrics(src)
    L = ["# Paper B R6 R1 — causal dose test (recipe: {})".format(rt), "",
         "Design: `decisions/paperB_r6_r1.md`. Commit `{}`. Three streams per step: ISIC classifier stream (full ISIC "
         "train, natural order), ISIC adversary stream resampled to the shift level, PAD adversary stream unchanged. PAD "
         "labels set only the resampling proportions and enter no loss.".format(R.C.git_head()), ""]
    tex = []
    standard_sections(L, tex, cells, boots, oods)
    label_shift_extras(L, cells)
    L += ["## Dose table (Mahalanobis, pad_heldout)", "",
          "| λ | shift | AUROC mean [95% CI] / rule A | Fitzpatrick | leakage | leakage class-matched | NV-nearest |",
          "|---:|---|---|---|---|---|---|"]
    paragraphs = []
    for lam in (0.5, 2.0):
        parts = []
        for sh in (100, 50, 0):
            c = next((x for x in cells if x[0] == SHIFT_LABEL[sh] and x[1] == lam), None)
            if c is None:
                continue
            b = boots.get((SHIFT_LABEL[sh], lam, "pad_heldout"))
            L.append("| {:g} | {} | {} | {} | {} | {} | {} |".format(
                lam, SHIFT_LABEL[sh], fmt_ci(b), fmt_ci(boots.get((SHIFT_LABEL[sh], lam, "fitzpatrick17k"))),
                R.pm(mean_of(c, lambda m: m["leakage"]["bal_acc_mean"])),
                R.pm(mean_of(c, lambda m: m["leakage_classmatched"]["bal_acc_mean"])),
                R.pm(mean_of(c, lambda m: m["ood"]["pad_heldout"]["nv_nearest_frac"]))))
            parts.append("{}: {}, leakage {}, NV-nearest {}".format(
                SHIFT_LABEL[sh], fmt_ci(b), R.pm(mean_of(c, lambda m: m["leakage"]["bal_acc_mean"])),
                R.pm(mean_of(c, lambda m: m["ood"]["pad_heldout"]["nv_nearest_frac"]))))
        if parts:
            paragraphs.append("At λ = {:g}, Mahalanobis on pad_heldout by shift level — {}. Prediction: {}".format(
                lam, "; ".join(parts), PRED["r1"]))
    L.append("")
    c0 = next((x for x in cells if x[1] == 0.0), None)
    if c0:
        paragraphs.append("λ = 0 baseline (three-stream, shift 100%): Mahalanobis pad_heldout {}, ID bal acc {}.".format(
            fmt_ci(boots.get((c0[0], 0.0, "pad_heldout"))), R.pm(mean_of(c0, lambda m: m["id"]["bal_acc"]))))
    ref = metrics([("two-stream DANN (same recipe)", lam, ref_runs(rt, lam)) for lam in (0.0, 0.5, 2.0)])
    md, tex_r, _ = R.cell_table([c for c in ref if c[2]])
    L += ["## Continuity reference: two-stream runs of the same recipe (separate table)", ""] + md + [""]
    L += ["## Results in plain words", ""] + [p + "\n" for p in paragraphs]
    write(root, L, tex + ["% two-stream reference"] + tex_r, [(cells, "pad_heldout"), ([c for c in ref if c[2]], "pad_heldout")], "r6_r1")


def report_r2(a, data, idg, rt):
    root = item_dir("r2", rt)
    true_adv = R.class_mix(data["pad_adv"].label_idx)
    true_hold = R.class_mix(data["pad_heldout"].label_idx)
    src = [("λ=0 baseline", 0.0, ref_runs(rt, 0.0))]
    for lam in (0.5, 2.0):
        src.append(("DANN", lam, ref_runs(rt, lam)))
        for m in ("cdan", "iwdan", "iwcdan"):
            src.append((m.upper(), lam, own_runs(root, "r2_" + rt + "_" + m + "_ladv{lam}_s{s}", lam)))
    src = [c for c in src if c[2]]
    oods = ["pad_heldout", "fitzpatrick17k"] + list(R.P7A_NAMES)
    boots = {} if a.no_boot else boots_for(src, oods, data, idg)
    cells = metrics(src)
    L = ["# Paper B R6 R2 — label-shift-aware alignment (recipe: {})".format(rt), "",
         "Design: `decisions/paperB_r6_r2.md`. Commit `{}`. λ = 0 baseline and DANN rows are the same recipe's "
         "existing runs (current: R2 item 4 / R4 lesion-level runs, DANN λ = 0.5 seeds 42–46; p8: P8 main runs).".format(
             R.C.git_head()), ""]
    tex = []
    standard_sections(L, tex, cells, boots, oods)
    label_shift_extras(L, cells)
    L += ["## Estimated vs true PAD class proportions", "",
          "IW = final-epoch training estimate (w ⊙ p_S, renormalised; target = pad_adv, no target labels). BBSE = "
          "post-hoc estimate from the final checkpoint (confusion on ISIC val, predictions on pad_heldout). True mixes "
          "from PAD labels, evaluation only. TV = total variation distance.", "",
          "| cell | λ | IW est NV | IW est AK | IW TV vs pad_adv | BBSE est NV (pad_heldout) | BBSE est AK | BBSE TV vs pad_heldout |",
          "|---|---:|---|---|---|---|---|---|"]
    est_json = {}
    for label, lam, pairs in src:
        iw = []
        for s, (_f, r) in pairs.items():
            p = Path(r) / "iw_log.jsonl"
            if p.exists() and p.read_text().strip():
                iw.append(np.asarray(json.loads(p.read_text().strip().splitlines()[-1])["est_target_mix_from_w"]))
        c = next(x for x in cells if x[0] == label and x[1] == lam)
        bb = [np.asarray(m["bbse"]["pad_heldout"]["est_mix"]) for m in c[2].values() if "bbse" in m]
        est_json["{}|{:g}".format(label, lam)] = {"iw_est": [x.tolist() for x in iw], "bbse_est": [x.tolist() for x in bb],
                                                  "true_pad_adv": true_adv.tolist(), "true_pad_heldout": true_hold.tolist()}
        L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} |".format(
            label, lam, R.pm(R.ms([x[R.NV] for x in iw])), R.pm(R.ms([x[R.AK] for x in iw])),
            R.pm(R.ms([0.5 * np.abs(x - true_adv).sum() for x in iw])), R.pm(R.ms([x[R.NV] for x in bb])),
            R.pm(R.ms([x[R.AK] for x in bb])), R.pm(R.ms([0.5 * np.abs(x - true_hold).sum() for x in bb]))))
    L += ["", "True mixes: pad_adv NV {:.3f}, AK {:.3f}; pad_heldout NV {:.3f}, AK {:.3f}.".format(
        true_adv[R.NV], true_adv[R.AK], true_hold[R.NV], true_hold[R.AK]), ""]
    paragraphs = []
    for lam in (0.5, 2.0):
        for label in ("DANN", "CDAN", "IWDAN", "IWCDAN"):
            c = next((x for x in cells if x[0] == label and x[1] == lam), None)
            if c is None:
                continue
            paragraphs.append("{} at λ = {:g}: leakage {}, Mahalanobis pad_heldout {}, Fitzpatrick {}, X-dom bal acc {}, "
                              "AK recall {}, NV predicted share {}.".format(
                                  label, lam, R.pm(mean_of(c, lambda m: m["leakage"]["bal_acc_mean"])),
                                  fmt_ci(boots.get((label, lam, "pad_heldout"))),
                                  fmt_ci(boots.get((label, lam, "fitzpatrick17k"))),
                                  R.pm(mean_of(c, lambda m: m["xdom_pad_heldout"]["bal_acc_6cls"])),
                                  R.pm(mean_of(c, lambda m: m["xdom_pad_heldout"]["recall_6cls"]["AK"])),
                                  R.pm(mean_of(c, lambda m: m["xdom_pad_heldout"]["pred_share_6cls"]["NV"]))))
    paragraphs.append("Prediction: " + PRED["r2"])
    L += ["## Results in plain words", ""] + [p + "\n" for p in paragraphs]
    write(root, L, tex, [(cells, "pad_heldout")], "r6_r2", {"label_shift_estimates": est_json})


def report_p7b(a, data, idg, rt):
    root = item_dir("p7b", rt)
    site = R.add_site(data["test"]).site.to_numpy()
    src = [("PAD held out of adversary", lam, own_runs(root, "p7b_" + rt + "_ladv{lam}_s{s}", lam)) for lam in (0.0, 0.5, 2.0)]
    src = [c for c in src if c[2]]
    oods = ["pad_full", "pad_heldout", "fitzpatrick17k"] + list(R.P7A_NAMES)
    boots = {} if a.no_boot else boots_for(src, oods, data, idg)
    cells = metrics(src, site_test=site)
    L = ["# Paper B R5 P7(b) — PAD fully held out of training (recipe: {})".format(rt), "",
         "Design: `decisions/paperB_r5_p7b.md`. Commit `{}`. Adversary and context-domain head: ISIC BCN vs HAM (from "
         "the ISIC train split). OOD AUROC column = Mahalanobis on all of PAD (2,298 images, bootstrap by patient).".format(
             R.C.git_head()), ""]
    tex = []
    standard_sections(L, tex, cells, boots, oods, primary="pad_full",
                      leak=lambda m: m["leakage_bcn_ham"]["bal_acc_mean"], leak_name="BCN-vs-HAM probe")
    label_shift_extras(L, cells)
    L += ["## Invariance pressure on the adversary's own domains", "",
          "| λ | BCN-vs-HAM probe bal acc (ISIC test) | ISIC-vs-PAD probe bal acc (Table 1 leakage) |", "|---:|---|---|"]
    paragraphs = []
    for c in cells:
        L.append("| {:g} | {} | {} |".format(c[1], R.pm(mean_of(c, lambda m: m["leakage_bcn_ham"]["bal_acc_mean"])),
                                            R.pm(mean_of(c, lambda m: m["leakage"]["bal_acc_mean"]))))
        paragraphs.append("λ = {:g}: Mahalanobis on all of PAD {}; pad_heldout {}; Fitzpatrick {}; BCN/HAM leakage {}.".format(
            c[1], fmt_ci(boots.get((c[0], c[1], "pad_full"))), fmt_ci(boots.get((c[0], c[1], "pad_heldout"))),
            fmt_ci(boots.get((c[0], c[1], "fitzpatrick17k"))),
            R.pm(mean_of(c, lambda m: m["leakage_bcn_ham"]["bal_acc_mean"]))))
    paragraphs.append("Prediction: " + PRED["p7b"])
    L += [""] + ["## Results in plain words", ""] + [p + "\n" for p in paragraphs]
    write(root, L, tex, [(cells, "pad_full")], "p7b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, choices=("p8", "r1", "r2", "p7b"))
    ap.add_argument("--recipe", default="current")
    ap.add_argument("--no_boot", action="store_true")
    a = ap.parse_args()
    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    idg = R.idx_groups(lesion_g)
    if a.item == "p8":
        return report_p8(a, data, idg)
    rt = R.recipe_tag(R.resolve_recipe(a.recipe))
    {"r1": report_r1, "r2": report_r2, "p7b": report_p7b}[a.item](a, data, idg, rt)


if __name__ == "__main__":
    main()
