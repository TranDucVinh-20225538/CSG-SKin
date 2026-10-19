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
# Every R1 shift cell is scored on these OOD sets (same five detectors; cluster/patient bootstrap).
R1_OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
R1_OOD_LAB = {
    "pad_heldout": "pad_heldout", "fitzpatrick17k": "Fitzpatrick17k",
    "mednode": "MED-NODE", "sd198": "SD-198", "scin": "SCIN",
}
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


def _ensure_nearest(m, feat_dir):
    """Backfill nearest_class_share / degraded from features when the summary predates those fields."""
    need = ("nearest_class_share" not in m.get("ood", {}).get("pad_heldout", {})
            or "id_nearest_class_share" not in m or "degraded" not in m)
    if not need:
        return m
    Fs = R.load_feats(feat_dir, ("isic_train", "isic_test", "pad_heldout"))
    if "isic_train" not in Fs or "isic_test" not in Fs:
        return m
    sc = R.Scorer(Fs["isic_train"])
    m = dict(m)
    m["id_nearest_class_share"] = sc.nearest_class_share(Fs["isic_test"]["z_lesion_norm"])
    m["id_nv_nearest_frac"] = m["id_nearest_class_share"]["NV"]
    ood = dict(m.get("ood", {}))
    for n, pack in Fs.items():
        if n.startswith("isic_"):
            continue
        if n not in ood:
            continue
        blk = dict(ood[n])
        shares = sc.nearest_class_share(pack["z_lesion_norm"])
        blk["nearest_class_share"] = shares
        blk["nv_nearest_frac"] = shares["NV"]
        ood[n] = blk
    m["ood"] = ood
    m["degraded"] = bool(m["id"]["bal_acc"] < R.DEGRADED_ID_BAL)
    return m


def metrics(cells_src, site_test=None, from_features=False):
    cells = []
    for label, lam, pairs in cells_src:
        ms = {}
        for s, (f, r) in pairs.items():
            m, _ = R.metrics_for(f, None if from_features else r, site_test=site_test)
            if m is not None and "ood" in m:
                ms[s] = _ensure_nearest(m, f)
        cells.append((label, lam, ms))
    return cells


def boots_for(cells_src, ood_names, data, id_groups, dets=None, exclude_by_cell=None):
    """exclude_by_cell: optional {(label, lam): set(seeds)} dropped from the pooled rule A verdict."""
    out = {}
    for name in ood_names:
        try:
            og, unit = R.ood_groups(name, data)
        except FileNotFoundError:
            continue
        for label, lam, pairs in cells_src:
            if pairs:
                ex = None if exclude_by_cell is None else exclude_by_cell.get((label, lam))
                b = R.bootstrap_cell({s: f for s, (f, _r) in pairs.items()}, name, id_groups, og, dets=dets,
                                     exclude_seeds=ex)
                if b is not None:
                    b["unit_ood"] = unit
                    out[(label, lam, name)] = b
    return out


def fmt_ci(b, det="mahalanobis_classcond"):
    if b is None:
        return "—"
    if b.get("n_seeds", 1) == 0 or det not in b:
        return "— (no seeds after exclusion)"
    x = b[det]
    sd = "" if x["sd"] is None else " ± {:.3f}".format(x["sd"])
    excl = b.get("excluded_seeds") or []
    suffix = "" if not excl else "; excluded degraded {}".format(excl)
    verdict = "—" if b.get("rule_A") is None else ("met" if b["rule_A"] else "not met")
    return "{:.3f}{} [{:.3f}, {:.3f}], {}/{} seed CIs below 0.5, rule A {}{}".format(
        x["mean"], sd, x["ci_lo"], x["ci_hi"], x["k_below"], x["n_seeds"], verdict, suffix)


def mean_of(cell, key_fn, skip_degraded=False):
    vals = [key_fn(m) for s, m in cell[2].items() if not (skip_degraded and m.get("degraded"))]
    return R.ms(vals)


def lambda0_id_bal(rt):
    """Matched-seed λ = 0 ID bal acc for the same recipe (R1 reuse / two-stream reference)."""
    out = {}
    for s, (f, r) in ref_runs(rt, 0.0).items():
        m, raw = R.metrics_for(f, r)
        if m is not None and "id" in m:
            out[s] = float(m["id"]["bal_acc"])
        elif raw is not None and "id_balanced_acc" in raw:
            out[s] = float(raw["id_balanced_acc"])
    return out


def run_flags(m, id_l0=None):
    """Return flag labels for one run (degraded / ID-drop); never silent-drop."""
    flags = []
    idb = m["id"]["bal_acc"]
    if idb < R.DEGRADED_ID_BAL or m.get("degraded"):
        flags.append("degraded (ID bal {:.3f} < {:.2f})".format(idb, R.DEGRADED_ID_BAL))
    if id_l0 is not None and idb < id_l0 - R.ID_DROP_FLAG:
        flags.append("ID drop vs λ=0 ({:.3f} − {:.3f} = {:+.3f})".format(idb, id_l0, idb - id_l0))
    return flags


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


def degraded_by_cell(cells):
    return {(lab, lam): {s for s, m in runs.items() if m.get("degraded") or m["id"]["bal_acc"] < R.DEGRADED_ID_BAL}
            for lab, lam, runs in cells}


def r1_run_flag_section(L, cells, id_l0):
    L += ["## Run quality flags (R1 deviation 2026-10-16 10:05)", "",
          "Degraded = ID bal acc < {:.2f} (2× chance {:.3f}): reported, flagged, **excluded from the main rule A "
          "verdict**; verdict also shown with them included. ID drop > {:.2f} below matched λ = 0: flagged, **kept** "
          "in the main verdict. No seed reruns unless > 1 of 5 degrade in a cell → replacement seeds {}.".format(
              R.DEGRADED_ID_BAL, R.CHANCE_ID_BAL, R.ID_DROP_FLAG, list(R.REPLACEMENT_SEEDS)), "",
          "| cell | λ | seed | ID bal | λ=0 ID bal | flags |",
          "|---|---:|---:|---:|---:|---|"]
    n_deg_cell = {}
    for lab, lam, runs in cells:
        deg = 0
        for s, m in sorted(runs.items()):
            fl = run_flags(m, id_l0.get(s))
            if any(x.startswith("degraded") for x in fl):
                deg += 1
            L.append("| {} | {:g} | {} | {:.3f} | {} | {} |".format(
                lab, lam, s, m["id"]["bal_acc"],
                "—" if s not in id_l0 else "{:.3f}".format(id_l0[s]),
                "; ".join(fl) if fl else "—"))
        n_deg_cell[(lab, lam)] = deg
        if deg > 1 and len(runs) >= 5:
            L.append("| {} | {:g} | — | — | — | **replacement trigger**: {}/{} degraded → run seeds {} |".format(
                lab, lam, deg, len(runs), list(R.REPLACEMENT_SEEDS)))
    L.append("")
    return n_deg_cell


def nearest_shift_section(L, cells):
    """Per shift × λ: pad_heldout nearest-centroid class share as Δ vs λ=0 pad_heldout and vs ID test (same shift)."""
    base = next((c for c in cells if c[1] == 0.0 and c[2]), None)
    L += ["## Where OOD images go (nearest ISIC class centroid, Mahalanobis)", "",
          "Share of pad_heldout whose nearest ISIC-train class mean (Mahalanobis) is each class. Reported as "
          "**Δ vs λ = 0** pad_heldout distribution (paper baseline NV share ≈ 0.332 at λ = 0) and **Δ vs ID test** "
          "at the same cell (classifier / ID test unchanged by adversary-stream resampling). Raw shares in "
          "`cells.json`. Means over non-degraded seeds when any remain; otherwise all seeds.", ""]
    hdr = "| cell | λ | vs | " + " | ".join(R.LABELS) + " |"
    L += [hdr, "|---|---:|---|" + "---|" * len(R.LABELS)]
    if base is None:
        L += ["*(λ = 0 cell not yet present; Δ vs λ = 0 omitted until baselines finish.)*", ""]
    base_share = None
    if base is not None:
        skip = bool(degraded_by_cell([base])[(base[0], base[1])])
        # mean share over seeds (prefer non-degraded)
        def mean_share(cell, key):
            deg = degraded_by_cell([cell])[(cell[0], cell[1])]
            use = {s: m for s, m in cell[2].items() if s not in deg} or cell[2]
            return {c: float(np.mean([key(m).get(c, 0.0) for m in use.values()])) for c in R.LABELS}

        base_share = mean_share(base, lambda m: m["ood"]["pad_heldout"]["nearest_class_share"])
        L.append("| {} | {:g} | raw (baseline) | {} |".format(
            base[0], base[1], " | ".join("{:.3f}".format(base_share[c]) for c in R.LABELS)))
    for lab, lam, runs in cells:
        if not runs or lam == 0.0:
            continue
        deg = degraded_by_cell([(lab, lam, runs)])[(lab, lam)]
        use = {s: m for s, m in runs.items() if s not in deg} or runs

        def avg(fn):
            return {c: float(np.mean([fn(m).get(c, 0.0) for m in use.values()])) for c in R.LABELS}

        ood = avg(lambda m: m["ood"]["pad_heldout"]["nearest_class_share"])
        id_ = avg(lambda m: m.get("id_nearest_class_share") or {})
        if base_share is not None:
            L.append("| {} | {:g} | Δ vs λ=0 pad_heldout | {} |".format(
                lab, lam, " | ".join("{:+.3f}".format(ood[c] - base_share[c]) for c in R.LABELS)))
        L.append("| {} | {:g} | Δ vs ID test (same cell) | {} |".format(
            lab, lam, " | ".join("{:+.3f}".format(ood[c] - id_.get(c, 0.0)) for c in R.LABELS)))
        L.append("| {} | {:g} | raw pad_heldout | {} |".format(
            lab, lam, " | ".join("{:.3f}".format(ood[c]) for c in R.LABELS)))
        L.append("| {} | {:g} | raw ID test | {} |".format(
            lab, lam, " | ".join("{:.3f}".format(id_.get(c, 0.0)) for c in R.LABELS)))
    L.append("")


def r1_key_comparison(L, cells, boots, boots_all, oods):
    """λ=2: R1 shift 0% vs R1's own shift 100% (last.ckpt), plus shift 50% when present — every OOD set."""
    L += ["## Key comparison: λ = 2 shift 0% vs R1 shift 100% (last.ckpt), all unseen sets", "",
          "Same three-stream recipe; primary = `last.ckpt`. Cells are R1's own shift runs (not two-stream item4). "
          "Shift 50% (seeds 42/52/62) is included when present. For each set: five detectors (point mean ± s.d.) and "
          "Mahalanobis / kNN / cosine rule-A bootstrap (cluster or patient units). Degraded seeds excluded from the "
          "primary rule A column; all-seeds column secondary.", ""]
    have = []
    for sh in (100, 50, 0):
        c = next((x for x in cells if x[0] == SHIFT_LABEL[sh] and x[1] == 2.0 and x[2]), None)
        if c is not None:
            have.append(sh)
    if not have:
        L += ["*(No λ = 2 shift cells with features yet.)*", ""]
        return
    for o in oods:
        if not any(o in m.get("ood", {}) for c in cells if c[1] == 2.0 for m in c[2].values()):
            continue
        L += ["### {} (λ = 2)".format(R1_OOD_LAB.get(o, o)), "",
              "| shift | n seeds | " + " | ".join(R.DET_LABEL[d] for d in R.DETS)
              + " | Maha [rule A excl. degraded] | Maha [all seeds] |",
              "|---|---:|" + "|".join(["---:"] * len(R.DETS)) + "|---|---|"]
        for sh in (100, 50, 0):
            c = next((x for x in cells if x[0] == SHIFT_LABEL[sh] and x[1] == 2.0), None)
            if c is None or not c[2]:
                continue
            n = sum(1 for m in c[2].values() if o in m.get("ood", {}))
            pts = []
            for d in R.DETS:
                pts.append(R.pm(mean_of(c, lambda m, _d=d, _o=o: m["ood"][_o][_d], skip_degraded=True)))
            L.append("| {} | {} | {} | {} | {} |".format(
                SHIFT_LABEL[sh], n, " | ".join(pts),
                fmt_ci(boots.get((SHIFT_LABEL[sh], 2.0, o))),
                fmt_ci(boots_all.get((SHIFT_LABEL[sh], 2.0, o)))))
        L.append("")
    # compact Δ(0% − 100%) for Mahalanobis point means (descriptive; CI contrast is in the rows above)
    if 0 in have and 100 in have:
        L += ["### λ = 2 Mahalanobis point Δ (shift 0% − shift 100%), excl. degraded", "",
              "| set | mean Δ | shift 0% | shift 100% | shift 50% |",
              "|---|---:|---:|---:|---:|"]
        for o in oods:
            c0 = next((x for x in cells if x[0] == SHIFT_LABEL[0] and x[1] == 2.0), None)
            c100 = next((x for x in cells if x[0] == SHIFT_LABEL[100] and x[1] == 2.0), None)
            c50 = next((x for x in cells if x[0] == SHIFT_LABEL[50] and x[1] == 2.0), None)
            m0 = mean_of(c0, lambda m, _o=o: m["ood"][_o]["mahalanobis_classcond"], skip_degraded=True) if c0 else None
            m100 = mean_of(c100, lambda m, _o=o: m["ood"][_o]["mahalanobis_classcond"], skip_degraded=True) if c100 else None
            m50 = mean_of(c50, lambda m, _o=o: m["ood"][_o]["mahalanobis_classcond"], skip_degraded=True) if c50 else None
            if m0 is None or m100 is None:
                continue
            L.append("| {} | {:+.3f} | {} | {} | {} |".format(
                R1_OOD_LAB.get(o, o), m0["mean"] - m100["mean"], R.pm(m0), R.pm(m100),
                "—" if m50 is None else R.pm(m50)))
        L.append("")


def r1_dose_by_set(L, cells, boots, boots_all, oods):
    """Dose tables: Mahalanobis (+ rule A) by shift for every λ and every unseen set."""
    L += ["## Dose tables (Mahalanobis by shift; every unseen set)", "",
          "Primary rule A excludes degraded seeds; all-seeds column secondary. Point detectors for each set are in "
          "the detector sections above.", ""]
    for o in oods:
        L += ["### Dose — {}".format(R1_OOD_LAB.get(o, o)), "",
              "| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |",
              "|---:|---|---|---|---|---|"]
        for lam in (0.5, 2.0):
            for sh in (100, 50, 0):
                c = next((x for x in cells if x[0] == SHIFT_LABEL[sh] and x[1] == lam), None)
                if c is None or not c[2]:
                    continue
                if not any(o in m.get("ood", {}) for m in c[2].values()):
                    continue
                L.append("| {:g} | {} | {} | {} | {} | {} |".format(
                    lam, SHIFT_LABEL[sh],
                    fmt_ci(boots.get((SHIFT_LABEL[sh], lam, o))),
                    fmt_ci(boots_all.get((SHIFT_LABEL[sh], lam, o))),
                    R.pm(mean_of(c, lambda m: m["leakage"]["bal_acc_mean"], skip_degraded=True)),
                    R.pm(mean_of(c, lambda m, _o=o: m["ood"][_o]["nv_nearest_frac"], skip_degraded=True))))
        L.append("")


def report_r1(a, data, idg, rt):
    root = item_dir("r1", rt)
    fmt = "r1_" + rt + "_shift{sh}_ladv{lam}_s{s}"
    src = [("λ=0 baseline (shift 100%)", 0.0, own_runs(root, fmt, 0.0, sh=100))]
    for lam in (0.5, 2.0):
        for sh in (100, 50, 0):
            src.append((SHIFT_LABEL[sh], lam, own_runs(root, fmt, lam, sh=sh)))
    src = [c for c in src if c[2]]
    oods = list(R1_OODS)
    cells = metrics(src)
    id_l0 = lambda0_id_bal(rt)
    # fall back: if R1 λ=0 cells exist, prefer those ID bals for the matched-seed flag
    c0 = next((x for x in cells if x[1] == 0.0), None)
    if c0:
        for s, m in c0[2].items():
            id_l0[s] = float(m["id"]["bal_acc"])
    excl = degraded_by_cell(cells)
    boots = {} if a.no_boot else boots_for(src, oods, data, idg, exclude_by_cell=excl)
    boots_all = {} if a.no_boot else boots_for(src, oods, data, idg)
    L = ["# Paper B R6 R1 — causal dose test (recipe: {})".format(rt), "",
         "Design: `decisions/paperB_r6_r1.md`. Commit `{}`. Three streams per step: ISIC classifier stream (full ISIC "
         "train, natural order), ISIC adversary stream resampled to the shift level, PAD adversary stream unchanged. PAD "
         "labels set only the resampling proportions and enter no loss. Primary checkpoint = `last.ckpt`. Every shift "
         "cell is scored on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN (same detectors + cluster/patient "
         "bootstrap).".format(R.C.git_head()), ""]
    tex = []
    r1_run_flag_section(L, cells, id_l0)
    r1_key_comparison(L, cells, boots, boots_all, oods)
    standard_sections(L, tex, cells, boots, oods)
    L += ["### Rule A with degraded seeds included (secondary; all unseen sets)", "",
          "| cell | λ | " + " | ".join(R1_OOD_LAB[o] for o in oods) + " |",
          "|---|---:|" + "|".join(["---"] * len(oods)) + "|"]
    for lab, lam, runs in cells:
        if not runs:
            continue
        L.append("| {} | {:g} | {} |".format(
            lab, lam, " | ".join(fmt_ci(boots_all.get((lab, lam, o))) for o in oods)))
    L.append("")
    label_shift_extras(L, cells)
    nearest_shift_section(L, cells)
    r1_dose_by_set(L, cells, boots, boots_all, oods)
    paragraphs = []
    for lam in (0.5, 2.0):
        parts = []
        for sh in (100, 50, 0):
            c = next((x for x in cells if x[0] == SHIFT_LABEL[sh] and x[1] == lam), None)
            if c is None:
                continue
            b = boots.get((SHIFT_LABEL[sh], lam, "pad_heldout"))
            ba = boots_all.get((SHIFT_LABEL[sh], lam, "pad_heldout"))
            parts.append("{}: {} (all: {})".format(SHIFT_LABEL[sh], fmt_ci(b), fmt_ci(ba)))
        if parts:
            paragraphs.append("At λ = {:g}, Mahalanobis on pad_heldout by shift — {}. Same shift ordering is tabulated "
                              "for Fitzpatrick / MED-NODE / SD-198 / SCIN above. Prediction: {}".format(
                                  lam, "; ".join(parts), PRED["r1"]))
    if c0:
        paragraphs.append("λ = 0 baseline (shift 100%): Mahalanobis pad_heldout {}, ID bal acc {}.".format(
            fmt_ci(boots.get((c0[0], 0.0, "pad_heldout"))), R.pm(mean_of(c0, lambda m: m["id"]["bal_acc"]))))
    ref = metrics([("two-stream DANN (same recipe)", lam, ref_runs(rt, lam)) for lam in (0.0, 0.5, 2.0)])
    md, tex_r, _ = R.cell_table([c for c in ref if c[2]])
    L += ["## Continuity reference: two-stream runs of the same recipe (separate table)", ""] + md + [""]
    L += ["## Results in plain words", ""] + [p + "\n" for p in paragraphs]
    extra = {"degraded_by_cell": {"{}|{:g}".format(k[0], k[1]): sorted(v) for k, v in excl.items() if v},
             "id_bal_lambda0_matched": id_l0,
             "key_comparison": "λ=2 shift 0% vs R1 shift 100% (last.ckpt) per set in R1_OODS; + shift 50% when present",
             "ood_sets": list(oods)}
    write(root, L, tex + ["% two-stream reference"] + tex_r,
          [(cells, "pad_heldout"), ([c for c in ref if c[2]], "pad_heldout")], "r6_r1", extra_json=extra)


def r1_shift100_runs(rt, lam):
    """R1 shift 100% / RECIPE=current / last.ckpt features+runs (DANN and λ=0 references for R2)."""
    r1 = item_dir("r1", rt)
    fmt = "r1_" + rt + "_shift100_ladv{lam}_s{s}"
    return own_runs(r1, fmt, lam)


def _score_packs(feat_dirs, ood_name, det="mahalanobis_classcond"):
    """{seed: (s_id, s_ood)} for one detector."""
    out = {}
    for s, d in feat_dirs.items():
        Fs = R.load_feats(d, ("isic_train", "isic_test", ood_name))
        if ood_name not in Fs:
            continue
        S = R.Scorer(Fs["isic_train"])
        a, b = S.scores(Fs["isic_test"]), S.scores(Fs[ood_name])
        if det not in a:
            continue
        out[int(s)] = (a[det], b[det])
    return out


def paired_delta_cluster_ci(feat_a, feat_b, ood_name, id_groups, o_groups, det="mahalanobis_classcond"):
    """Paired-by-seed Δ(AUROC_a − AUROC_b) with shared cluster/patient bootstrap of the seed-mean."""
    sa, sb = _score_packs(feat_a, ood_name, det), _score_packs(feat_b, ood_name, det)
    seeds = sorted(set(sa) & set(sb))
    if len(seeds) < 1:
        return None
    pre = json.loads(R.PREREG.read_text())
    n_boot = int(pre["n_resamples"])
    rng = np.random.default_rng(int(pre["rng_seed"]))
    id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))])
                for _ in range(n_boot)]
    oo_draws = [np.concatenate([o_groups[j] for j in rng.integers(0, len(o_groups), len(o_groups))])
                for _ in range(n_boot)]
    point = []
    boot = np.zeros(n_boot)
    per = {}
    for s in seeds:
        aid, ao = sa[s]
        bid, bo = sb[s]
        dpt = R.auroc(aid, ao) - R.auroc(bid, bo)
        point.append(dpt)
        da = np.array([R.auroc(aid[id_draws[b]], ao[oo_draws[b]]) for b in range(n_boot)])
        db = np.array([R.auroc(bid[id_draws[b]], bo[oo_draws[b]]) for b in range(n_boot)])
        boot += (da - db) / len(seeds)
        per[str(s)] = {"delta": float(dpt)}
    pts = np.asarray(point, float)
    return {"mean": float(pts.mean()), "sd": float(pts.std(ddof=1)) if len(pts) > 1 else 0.0,
            "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975)),
            "n_seeds": len(seeds), "per_seed": per, "det": det, "ood": ood_name}


# Locked in decisions/paperB_r6_r2.md (2026-10-17): drop seed if |λ0−DANN| below this; flag set if mean |denom| < SMALL.
GAP_DENOM_EPS = 1e-6
GAP_SMALL_DENOM = 0.05


def gap_recovered_seed_boot(feat_iw, feat_dann, feat_l0, ood_name, det="mahalanobis_classcond", n_boot=None):
    """Share of gap recovered: (method − DANN) / (λ0 − DANN), paired by seed; bootstrap CI of the seed-mean.

    λ0 and DANN are per-set Mahalanobis AUROCs from R1 shift100 last.ckpt (same recipe). Always also
    returns mean absolute Δ (method − DANN) and mean(λ0 − DANN) so small-denominator sets are not
    read from the ratio alone.
    """
    si, sd, s0 = (_score_packs(feat_iw, ood_name, det), _score_packs(feat_dann, ood_name, det),
                  _score_packs(feat_l0, ood_name, det))
    seeds = sorted(set(si) & set(sd) & set(s0))
    gaps, abs_deltas, denoms = [], [], []
    per = {}
    for s in seeds:
        iw = R.auroc(*si[s])
        dann = R.auroc(*sd[s])
        l0 = R.auroc(*s0[s])
        denom = l0 - dann
        abs_d = iw - dann
        denoms.append(denom)
        abs_deltas.append(abs_d)
        if abs(denom) < GAP_DENOM_EPS:
            g = float("nan")
        else:
            g = abs_d / denom
        gaps.append(g)
        per[str(s)] = {"iw": iw, "dann": dann, "lambda0": l0, "gap_recovered": g,
                       "delta_abs": abs_d, "denom": denom}
    gaps = np.asarray(gaps, float)
    ok = np.isfinite(gaps)
    if ok.sum() < 1:
        return None
    g = gaps[ok]
    den_arr = np.asarray(denoms, float)
    abs_arr = np.asarray(abs_deltas, float)
    pre = json.loads(R.PREREG.read_text())
    n_boot = int(pre["n_resamples"]) if n_boot is None else n_boot
    rng = np.random.default_rng(int(pre["rng_seed"]) + 17)
    boots = np.array([g[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)])
    mean_denom = float(den_arr.mean())
    return {"mean": float(g.mean()), "sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "n_seeds": int(ok.sum()), "per_seed": per, "det": det, "ood": ood_name,
            "mean_delta_abs": float(abs_arr.mean()),
            "sd_delta_abs": float(abs_arr.std(ddof=1)) if len(abs_arr) > 1 else 0.0,
            "mean_denom": mean_denom,
            "sd_denom": float(den_arr.std(ddof=1)) if len(den_arr) > 1 else 0.0,
            "small_denom": bool(abs(mean_denom) < GAP_SMALL_DENOM)}


def _fmt_delta(d):
    if d is None:
        return "—"
    return "{:+.3f} ± {:.3f} [{:.3f}, {:.3f}] (n={})".format(
        d["mean"], d["sd"], d["ci_lo"], d["ci_hi"], d["n_seeds"])


def _fmt_gap(d):
    """Gap-recovered ratio plus denom context; flags small |λ0−DANN|."""
    if d is None:
        return "—"
    base = _fmt_delta(d)
    extra = " · Δabs={:+.3f} · mean(λ0−DANN)={:+.3f}".format(
        d.get("mean_delta_abs", float("nan")), d.get("mean_denom", float("nan")))
    if d.get("small_denom"):
        extra += " · **small denom — prefer absolute Δ**"
    return base + extra


def report_r2(a, data, idg, rt):
    root = item_dir("r2", rt)
    true_adv = R.class_mix(data["pad_adv"].label_idx)
    true_hold = R.class_mix(data["pad_heldout"].label_idx)
    # Baselines: R1 shift 100% last.ckpt (not Table 1 / item4 best-val). Deviation 2026-10-17.
    src = [("λ=0 (R1 shift100 last.ckpt)", 0.0, r1_shift100_runs(rt, 0.0))]
    methods = ("cdan", "iwdan", "iwcdan", "iwdan_oracle")
    for lam in (0.5, 2.0):
        src.append(("DANN (R1 shift100 last.ckpt)", lam, r1_shift100_runs(rt, lam)))
        for m in methods:
            lab = "IWDAN_ORACLE" if m == "iwdan_oracle" else m.upper()
            src.append((lab, lam, own_runs(root, "r2_" + rt + "_" + m + "_ladv{lam}_s{s}", lam)))
    src = [c for c in src if c[2]]
    # SCIN: separate report (decisions/paperB_r6_r2_scin.md) — 0.6% ISIC-mapped; not in fix reading.
    oods = ["pad_heldout", "fitzpatrick17k", "mednode", "sd198"]
    boots = {} if a.no_boot else boots_for(src, oods, data, idg)
    cells = metrics(src)

    # Headline DANN λ=2 means (for the prose; matches decisions deviation)
    dann2 = next((c for c in cells if c[0].startswith("DANN") and c[1] == 2.0), None)
    l0 = next((c for c in cells if c[1] == 0.0), None)

    L = ["# Paper B R6 R2 — label-shift-aware alignment (recipe: {})".format(rt), "",
         "Design: `decisions/paperB_r6_r2.md`. Commit `{}`.".format(R.C.git_head()), "",
         "## Baseline and pairing (fixed 2026-10-17)", "",
         "- **DANN reference** and **λ = 0 reference** are **R1 shift 100%, RECIPE=`current`, primary=`last.ckpt`** "
         "(`results/paperB/r6/r1/{}/`), **not** Table 1 / item4 best-val.".format(rt),
         "- Headline DANN λ = 2 cell means (R1 shift100 last.ckpt): Maha pad_heldout **0.368**, Fitzpatrick **0.340**, "
         "leakage **0.634** (verify against the DANN row below).",
         "- **Paired-by-seed deltas are valid:** R2 and R1 use the **same seeds** `{}` and the **same data split** "
         "(lesion-level ISIC train/val/test from `train_r2_item4_lesion_split` / Phase 3; Phase 2 PAD patient split "
         "pad_adv / pad_heldout). R2 `default_mode(current)` is the two-stream Table 1 path; R1 shift 100% is "
         "three-stream with `adv_shift=1` (ISIC mix), so the **classifier stream and split match**; the DANN "
         "reference is the R1 shift-100% last.ckpt cell at the same λ/seed.".format(list(R.SEEDS)),
         "- Pre-committed fix reading (estimated vs oracle on Fitz/MED/SD) is **unchanged**; interpret only after "
         "oracle, CDAN, and λ = 0.5 cells are complete.",
         "- **SCIN** is scored separately (`REPORT_SCIN.md` / `decisions/paperB_r6_r2_scin.md`): same detectors + "
         "case-level cluster CIs; only 0.6% of images map to ISIC classes; does **not** enter the fix reading.",
         ""]
    tex = []
    standard_sections(L, tex, cells, boots, oods)
    label_shift_extras(L, cells)

    # Paired Δ and gap recovered vs DANN / λ0 (formula locked in decisions/paperB_r6_r2.md)
    L += ["## Paired-by-seed Δ vs DANN (R1 shift100 last.ckpt) and gap recovered", "",
          "Locked formula (decisions/paperB_r6_r2.md): "
          "**gap recovered** = (method − DANN) / (λ0 − DANN), Mahalanobis, paired by seed; "
          "λ0 = R1 shift100 λ = 0 `last.ckpt` **per unseen set**; DANN = R1 shift100 same λ `last.ckpt`. "
          "Δ = method − DANN with **cluster/patient bootstrap CI** of the seed-mean. "
          "Ratio uses **seed bootstrap** CI; `|λ0−DANN| < 1e-6` → seed dropped from the ratio. "
          "Always read absolute Δ with the ratio; if mean `|λ0−DANN| < 0.05`, flag "
          "**small denom — prefer absolute Δ** (MED-NODE is the smallest pre-committed denom among the four sets).",
          ""]
    gap_json, delta_json = {}, {}
    unseen = [o for o in ("pad_heldout", "fitzpatrick17k", "mednode", "sd198") if o in oods or True]
    L += ["| method | λ | set | Δ Maha vs DANN [cluster CI] | gap recovered (method−DANN)/(λ0−DANN) [seed boot CI] |",
          "|---|---:|---|---|---|"]
    for lam in (0.5, 2.0):
        dann_pairs = r1_shift100_runs(rt, lam)
        l0_pairs = r1_shift100_runs(rt, 0.0)
        if not dann_pairs or not l0_pairs:
            continue
        dann_feat = {s: f for s, (f, _r) in dann_pairs.items()}
        l0_feat = {s: f for s, (f, _r) in l0_pairs.items()}
        for m in methods:
            lab = "IWDAN_ORACLE" if m == "iwdan_oracle" else m.upper()
            pairs = own_runs(root, "r2_" + rt + "_" + m + "_ladv{lam}_s{s}", lam)
            if not pairs:
                continue
            feat = {s: f for s, (f, _r) in pairs.items()}
            for ood in unseen:
                try:
                    og, _unit = R.ood_groups(ood, data)
                except FileNotFoundError:
                    continue
                if a.no_boot:
                    dlt = gap = None
                else:
                    dlt = paired_delta_cluster_ci(feat, dann_feat, ood, idg, og)
                    gap = gap_recovered_seed_boot(feat, dann_feat, l0_feat, ood)
                key = "{}|{:g}|{}".format(lab, lam, ood)
                if dlt:
                    delta_json[key] = dlt
                if gap:
                    gap_json[key] = gap
                L.append("| {} | {:g} | {} | {} | {} |".format(
                    lab, lam, ood, _fmt_delta(dlt), _fmt_gap(gap)))
    L.append("")

    # leakage paired Δ (point + seed boot) for the DANN leakage baseline story
    L += ["### Leakage paired Δ vs DANN (pad_full probe; seed bootstrap of seed-mean)", "",
          "| method | λ | Δ leakage (method − DANN) |",
          "|---|---:|---|"]
    for lam in (0.5, 2.0):
        dann_c = next((c for c in cells if c[0].startswith("DANN") and c[1] == lam), None)
        if dann_c is None:
            continue
        for m in methods:
            lab = "IWDAN_ORACLE" if m == "iwdan_oracle" else m.upper()
            c = next((x for x in cells if x[0] == lab and x[1] == lam), None)
            if c is None:
                continue
            seeds = sorted(set(c[2]) & set(dann_c[2]))
            if not seeds:
                continue
            deltas = [c[2][s]["leakage"]["bal_acc_mean"] - dann_c[2][s]["leakage"]["bal_acc_mean"] for s in seeds]
            arr = np.asarray(deltas, float)
            pre = json.loads(R.PREREG.read_text())
            rng = np.random.default_rng(int(pre["rng_seed"]) + 23)
            leak_boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                                   for _ in range(int(pre["n_resamples"]))])
            L.append("| {} | {:g} | {:+.3f} ± {:.3f} [{:.3f}, {:.3f}] (n={}) |".format(
                lab, lam, arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0,
                float(np.quantile(leak_boots, 0.025)), float(np.quantile(leak_boots, 0.975)), len(arr)))
    L.append("")

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
    if dann2 is not None:
        paragraphs.append(
            "DANN reference (R1 shift 100% last.ckpt) at λ = 2: leakage {}, Mahalanobis pad_heldout {}, "
            "Fitzpatrick {} — baseline for all R2 comparisons (not Table 1 best-val).".format(
                R.pm(mean_of(dann2, lambda m: m["leakage"]["bal_acc_mean"])),
                fmt_ci(boots.get((dann2[0], 2.0, "pad_heldout"))),
                fmt_ci(boots.get((dann2[0], 2.0, "fitzpatrick17k")))))
    if l0 is not None:
        paragraphs.append(
            "λ = 0 reference (R1 shift 100% last.ckpt): leakage {}, Mahalanobis pad_heldout {}.".format(
                R.pm(mean_of(l0, lambda m: m["leakage"]["bal_acc_mean"])),
                fmt_ci(boots.get((l0[0], 0.0, "pad_heldout")))))
    for lam in (0.5, 2.0):
        for label in ("DANN (R1 shift100 last.ckpt)", "CDAN", "IWDAN", "IWCDAN", "IWDAN_ORACLE"):
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
    have_cdan05 = any(x[0] == "CDAN" and x[1] == 0.5 and len(x[2]) >= 5 for x in cells)
    have_ora05 = any(x[0] == "IWDAN_ORACLE" and x[1] == 0.5 and len(x[2]) >= 5 for x in cells)
    if have_cdan05 and have_ora05:
        paragraphs.append(
            "CDAN and IWDAN_ORACLE λ = 0.5 are complete (5 seeds each). Pre-committed reading unchanged. "
            "Prediction (decisions): " + PRED["r2"])
    else:
        paragraphs.append(
            "Interpretation deferred until oracle, CDAN, and λ = 0.5 cells finish. Pre-committed reading unchanged. "
            "Prediction (decisions): " + PRED["r2"])
    L += ["## Results in plain words", ""] + [p + "\n" for p in paragraphs]
    write(root, L, tex, [(cells, "pad_heldout")], "r6_r2", {
        "label_shift_estimates": est_json,
        "dann_baseline": "r1_shift100_last.ckpt",
        "paired_delta_vs_dann": delta_json,
        "gap_recovered": gap_json,
        "pairing_note": "R2 and R1 share SEEDS and lesion-level ISIC + Phase2 PAD split",
    })


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
