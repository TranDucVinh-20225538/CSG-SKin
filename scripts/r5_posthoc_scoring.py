#!/usr/bin/env python3
"""Paper B post hoc CPU: scoring-side mitigation / clinical consequence / nv_frac correlation.

Design: decisions/paperB_r5_posthoc_scoring.md (committed before this run). Label: post hoc.
Usage:
  r5_posthoc_scoring.py mitigation
  r5_posthoc_scoring.py clinical
  r5_posthoc_scoring.py nvfrac
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402
import r5cpu_common as R  # noqa: E402
import r2_rmd as RMD  # noqa: E402

OUT_ROOT = R.R5 / "posthoc"
OODS_CRIT = ("fitzpatrick17k", "mednode", "sd198")
OODS_ALL = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
MALIGN = ("MEL", "BCC", "SCC", "AK")
FITZ_MAP = {
    "melanoma": "MEL",
    "basal cell carcinoma": "BCC",
    "squamous cell carcinoma": "SCC",
    "actinic keratosis": "AK",
    "dermatofibroma": "DF",
    "congenital nevus": "NV",
}
ID_BAL_MIN = 0.25
EPS = C.MAHA_EPS
NCLS = 8
LABELS = list(C.LABELS)
NV = 1


# ----------------------------------------------------------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------------------------------------------------------
class StdMaha(C.Maha):
    """Pooled within-class Mahalanobis (paper standard)."""


class RelMaha:
    """Relative Mahalanobis (Ren et al. 2021): min_c (MD_c − MD_0)."""

    def __init__(self, ztr, ytr):
        self.md = StdMaha(ztr, ytr)
        self.g0 = RMD.Gauss0(ztr, eps=EPS)

    def score(self, z):
        return (self.md.dists(z) - self.g0.score(z)[:, None]).min(1)


class PerClassStdMaha:
    """Per-class mean + per-class covariance (+ eps I); score = min_c MD_c."""

    def __init__(self, ztr, ytr, n_classes=NCLS, eps=EPS):
        z = np.asarray(ztr, np.float64)
        y = np.asarray(ytr, int)
        self.means = np.zeros((n_classes, z.shape[1]), np.float64)
        self.precs = np.zeros((n_classes, z.shape[1], z.shape[1]), np.float64)
        self.present = np.zeros(n_classes, bool)
        eye = np.eye(z.shape[1])
        for c in range(n_classes):
            zc = z[y == c]
            if len(zc) < 2:
                continue
            self.present[c] = True
            self.means[c] = zc.mean(0)
            cen = zc - self.means[c]
            cov = cen.T @ cen / max(len(zc) - 1, 1) + eps * eye
            self.precs[c] = np.linalg.inv(cov)

    def score(self, z):
        z = np.asarray(z, np.float64)
        d = np.full((len(z), NCLS), np.inf)
        for c in range(NCLS):
            if not self.present[c]:
                continue
            cen = z - self.means[c]
            d[:, c] = np.einsum("nd,nd->n", cen @ self.precs[c], cen)
        return np.min(d, 1)


VARIANTS = {
    "standard_maha": ("Standard Mahalanobis", StdMaha),
    "relative_maha": ("Relative Mahalanobis (Ren et al. 2021)", RelMaha),
    "perclass_std_maha": ("Per-class standardised Mahalanobis", PerClassStdMaha),
}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None or isinstance(x, str):
        return x
    return x


def ms(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return None
    if a.size == 1:
        return {"mean": float(a[0]), "sd": None, "n": 1}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size)}


def pm(m):
    if m is None:
        return "—"
    if m.get("sd") is None:
        return "{:.3f}".format(m["mean"])
    return "{:.3f} ± {:.3f}".format(m["mean"], m["sd"])


# ----------------------------------------------------------------------------------------------------------------------
# Run inventory
# ----------------------------------------------------------------------------------------------------------------------
def _parse_old(name):
    m = re.fullmatch(r"runB_orth1_ladv([0-9p]+)_s(\d+)", name)
    return ("old_recipe", float(m.group(1).replace("p", ".")), int(m.group(2)), None) if m else None


def _parse_a4(name):
    m = re.fullmatch(r"main_ladv([0-9p]+)_s(\d+)", name)
    return ("a4", float(m.group(1).replace("p", ".")), int(m.group(2)), None) if m else None


def _parse_r1(name):
    m = re.fullmatch(r"r1_cur_shift(\d+)_ladv([0-9p]+)_s(\d+)", name)
    if not m:
        return None
    return ("r1", float(m.group(2).replace("p", ".")), int(m.group(3)), int(m.group(1)))


def _parse_main_last(name):
    """Table 1 dual-encoder last.ckpt features under r5/t1last/features_last/."""
    m = re.fullmatch(r"runB_orth1_ladv([0-9p]+)_s(\d+)", name)
    return ("main_last", float(m.group(1).replace("p", ".")), int(m.group(2)), None) if m else None


def inventory(need=("isic_train", "isic_test", "isic_val") + OODS_ALL):
    roots = [
        (R.FEAT_L, _parse_old),
        (R.R5 / "p8" / "main" / "features", _parse_a4),
        (R.R6 / "r1" / "cur" / "features", _parse_r1),
        (R.R5 / "t1last" / "features_last", _parse_main_last),
    ]
    rows = []
    for root, parse in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            meta = parse(d.name)
            if meta is None:
                continue
            recipe, lam, seed, shift = meta
            if not all((d / "{}.npz".format(n)).exists() for n in need):
                continue
            id_bal = None
            degraded = False
            if recipe == "r1":
                sj = R.R6 / "r1" / "cur" / "runs" / d.name / "summary.json"
                if sj.exists():
                    id_bal = float(json.loads(sj.read_text())["metrics"]["id"]["bal_acc"])
                    degraded = id_bal < ID_BAL_MIN
            rows.append({"recipe": recipe, "lambda": lam, "seed": seed, "shift": shift,
                         "feat_dir": d, "id_bal": id_bal, "degraded": degraded})
    return rows


def cell_key(r):
    if r["recipe"] == "r1":
        return (r["recipe"], r["shift"], r["lambda"])
    return (r["recipe"], r["lambda"])


def cell_label(k):
    if k[0] == "r1":
        return "R1 shift {}% λ={:g}".format(k[1], k[2])
    return "{} λ={:g}".format(k[0], k[1])


# ----------------------------------------------------------------------------------------------------------------------
# Labels / groups for OOD sets
# ----------------------------------------------------------------------------------------------------------------------
def ood_label_arrays():
    """ISIC-mapped labels (-1 if unmapped) aligned to feature row order for each OOD set."""
    out = {}
    # pad_heldout: labels already in npz are ISIC indices
    out["pad_heldout"] = None  # filled per-run from pack
    # fitz
    stems = R.fitz_stems()
    meta = pd.read_csv(R.RES / "phase11_third_domain" / "fitzpatrick17k.csv").drop_duplicates("md5hash").set_index("md5hash")
    lab = meta.loc[stems, "label"].astype(str).str.lower().map(FITZ_MAP)
    out["fitzpatrick17k"] = np.array([LABELS.index(x) if isinstance(x, str) and x in LABELS else -1 for x in lab], int)
    # p7a sets
    for name in ("mednode", "sd198", "scin"):
        df = pd.read_csv(R.R5 / "p7a" / "sets" / "{}.csv".format(name), keep_default_na=False)
        mapped = df["mapped"].astype(str)
        out[name] = np.array([LABELS.index(m) if m in LABELS else -1 for m in mapped], int)
    return out


def ood_group_arrays():
    g = {
        "pad_heldout": R.pad_heldout_patients(),
        "fitzpatrick17k": R.fitz_groups("cluster_union" if R.P13_CLUSTERS.exists() else "image"),
    }
    for name in ("mednode", "sd198", "scin"):
        df = pd.read_csv(R.R5 / "p7a" / "sets" / "{}.csv".format(name), keep_default_na=False)
        g[name] = df["group"].astype(str).to_numpy()
    return g


def class_balanced_auroc(sid, sood, yood):
    """Diagnostic: OOD reweighted to uniform over present mapped classes (needs labels)."""
    yood = np.asarray(yood, int)
    m = yood >= 0
    if m.sum() < 20:
        return None
    cls = [c for c in range(NCLS) if (yood == c).sum() >= 5]
    if len(cls) < 2:
        return None
    p = np.array([(yood == c).sum() for c in cls], float)
    p /= p.sum()
    target = np.full(len(cls), 1.0 / len(cls))
    w = np.zeros(len(yood))
    for c, a, b in zip(cls, target, p):
        w[yood == c] = a / b
    return R.auroc(sid, sood[m], w_ood=w[m])


# ----------------------------------------------------------------------------------------------------------------------
# (1) Mitigation
# ----------------------------------------------------------------------------------------------------------------------
def run_mitigation():
    out_dir = OUT_ROOT / "scoring_mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    n_boot = int(pre["n_resamples"])
    idg = R.id_test_groups()
    groups = ood_group_arrays()
    lab_fixed = ood_label_arrays()
    # cells used for criterion + tables
    rows = [r for r in inventory() if not r["degraded"]]
    # restrict: old all λ; a4 λ in {0,2}; r1 shift in {0,100} λ=2
    use = []
    for r in rows:
        if r["recipe"] == "old_recipe":
            use.append(r)
        elif r["recipe"] == "a4" and r["lambda"] in (0.0, 2.0):
            use.append(r)
        elif r["recipe"] == "r1" and r["shift"] in (0, 100) and r["lambda"] == 2.0:
            use.append(r)
    print("mitigation runs", len(use), flush=True)

    per_run = []
    # scores cache: (feat_dir, variant) -> {split: scores}
    for i, r in enumerate(use):
        Fs = {n: dict(np.load(r["feat_dir"] / "{}.npz".format(n)))
              for n in ("isic_train", "isic_test") + OODS_ALL}
        ztr, ytr = Fs["isic_train"]["z_lesion_norm"], Fs["isic_train"]["labels"].astype(int)
        scorers = {k: cls(ztr, ytr) for k, (_, cls) in VARIANTS.items()}
        sid = {k: sc.score(Fs["isic_test"]["z_lesion_norm"]) for k, sc in scorers.items()}
        rec = {"recipe": r["recipe"], "lambda": r["lambda"], "seed": r["seed"], "shift": r["shift"],
               "feat_dir": str(r["feat_dir"]), "ood": {}}
        for ood in OODS_ALL:
            yood = Fs[ood]["labels"].astype(int) if ood == "pad_heldout" else lab_fixed[ood]
            blk = {}
            for vk, sc in scorers.items():
                so = sc.score(Fs[ood]["z_lesion_norm"])
                entry = {"auroc": R.auroc(sid[vk], so)}
                entry["classbal_auroc_diagnostic"] = class_balanced_auroc(sid[vk], so, yood)
                blk[vk] = {"point": entry, "s_id": sid[vk], "s_ood": so}
            rec["ood"][ood] = {"yood": yood, "variants": blk}
        per_run.append(rec)
        if (i + 1) % 5 == 0:
            print("  scored {}/{}".format(i + 1, len(use)), flush=True)

    # bootstrap per cell × ood × variant
    by_cell = defaultdict(list)
    for rec in per_run:
        by_cell[cell_key({"recipe": rec["recipe"], "lambda": rec["lambda"], "shift": rec["shift"]})].append(rec)

    cells = {}
    summary_rows = []
    for ck, recs in sorted(by_cell.items(), key=lambda x: str(x[0])):
        seeds = [r["seed"] for r in recs]
        cell = {"label": cell_label(ck), "n_seeds": len(seeds), "seeds": seeds, "ood": {}}
        for ood in OODS_ALL:
            og = groups[ood]
            rng = np.random.default_rng(int(pre["rng_seed"]))
            id_draws = R.draws(idg, n_boot, rng)
            oo_draws = R.draws(og, n_boot, np.random.default_rng(int(pre["rng_seed"]) + 17))
            cell["ood"][ood] = {}
            for vk in VARIANTS:
                per = {r["seed"]: (r["ood"][ood]["variants"][vk]["s_id"],
                                   r["ood"][ood]["variants"][vk]["s_ood"]) for r in recs}
                boot = R.pooled_bootstrap(per, id_draws, oo_draws)
                pts = [r["ood"][ood]["variants"][vk]["point"]["auroc"] for r in recs]
                cbal = [r["ood"][ood]["variants"][vk]["point"]["classbal_auroc_diagnostic"] for r in recs]
                k_above = int(sum(1 for s in boot["per_seed"].values() if s["lo"] > 0.5))
                bshort = {
                    "mean": boot["mean"], "sd": boot["sd"],
                    "ci_lo": boot["pooled_ci_lo"], "ci_hi": boot["pooled_ci_hi"],
                    "k_seeds_ci_below_0.5": boot["k_seeds_ci_below_0.5"],
                    "k_seeds_ci_above_0.5": k_above,
                    "rule_A": boot["rule_A"],
                }
                block = {
                    "point": ms(pts),
                    "bootstrap": bshort,
                    "ci_excludes_0.5_above": bool(boot["pooled_ci_lo"] > 0.5),
                    "ci_excludes_0.5_below": bool(boot["pooled_ci_hi"] < 0.5),
                    "classbal_diagnostic": ms(cbal),
                }
                cell["ood"][ood][vk] = block
                summary_rows.append({
                    "cell": cell_label(ck), "recipe": ck[0],
                    "shift": ck[1] if ck[0] == "r1" else "",
                    "lambda": ck[2] if ck[0] == "r1" else ck[1],
                    "ood": ood, "variant": vk,
                    "auroc_mean": block["point"]["mean"] if block["point"] else None,
                    "auroc_sd": block["point"].get("sd") if block["point"] else None,
                    "ci_lo": bshort["ci_lo"], "ci_hi": bshort["ci_hi"],
                    "ci_excludes_0.5_above": block["ci_excludes_0.5_above"],
                    "classbal_mean": block["classbal_diagnostic"]["mean"] if block["classbal_diagnostic"] else None,
                })
        cells[cell_label(ck)] = cell
        print("bootstrapped", cell_label(ck), flush=True)

    # criterion on RMD, prefer old_recipe λ=2; also report a4 λ=2 and r1 shift100 λ=2
    def criterion_for(label):
        c = cells.get(label)
        if not c:
            return None
        hits = []
        for ood in OODS_CRIT:
            b = c["ood"][ood]["relative_maha"]
            ok = (b["point"] and b["point"]["mean"] > 0.5 and b["ci_excludes_0.5_above"])
            hits.append({"ood": ood, "ok": bool(ok), "mean": b["point"]["mean"] if b["point"] else None,
                         "ci": [b["bootstrap"]["ci_lo"], b["bootstrap"]["ci_hi"]] if b["bootstrap"] else None})
        # λ=0 drop vs standard
        lab0 = label.replace("λ=2", "λ=0").replace("λ=2.0", "λ=0")
        # for r1 no λ=0 — use a4 λ=0
        if label.startswith("R1"):
            lab0 = "a4 λ=0"
        drops = []
        c0 = cells.get(lab0) or cells.get("old_recipe λ=0") or cells.get("a4 λ=0")
        if c0:
            for ood in OODS_CRIT:
                m0s = c0["ood"][ood]["standard_maha"]["point"]
                m0r = c0["ood"][ood]["relative_maha"]["point"]
                if m0s and m0r:
                    drops.append({"ood": ood, "drop": m0s["mean"] - m0r["mean"],
                                  "ok": (m0s["mean"] - m0r["mean"]) <= 0.05})
        n_hit = sum(1 for h in hits if h["ok"])
        return {"hits": hits, "n_hit": n_hit, "pass_ood_limb": n_hit >= 2,
                "lambda0_drops": drops,
                "pass_lambda0_limb": all(d["ok"] for d in drops) if drops else None,
                "pass_criterion": (n_hit >= 2) and (all(d["ok"] for d in drops) if drops else False)}

    criteria = {lab: criterion_for(lab) for lab in ("old_recipe λ=2", "a4 λ=2", "R1 shift 100% λ=2", "R1 shift 0% λ=2")}

    # strip heavy arrays from per_run for JSON
    per_run_light = []
    for rec in per_run:
        lite = {k: rec[k] for k in ("recipe", "lambda", "seed", "shift", "feat_dir")}
        lite["ood"] = {}
        for ood, blk in rec["ood"].items():
            lite["ood"][ood] = {vk: blk["variants"][vk]["point"] for vk in VARIANTS}
        per_run_light.append(lite)

    payload = {
        "commit": R.git_head(),
        "decision": "decisions/paperB_r5_posthoc_scoring.md",
        "label": "post hoc",
        "variants": {k: v[0] for k, v in VARIANTS.items()},
        "criterion": criteria,
        "cells": cells,
        "per_run": per_run_light,
    }
    (out_dir / "results.json").write_text(json.dumps(_jsonable(payload), indent=1))
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    _write_mitigation_report(out_dir, payload)
    print("wrote", out_dir, flush=True)


def _write_mitigation_report(out_dir, payload):
    L = [
        "# Paper B post hoc — scoring-side Mahalanobis mitigation",
        "",
        "Design: `decisions/paperB_r5_posthoc_scoring.md`. Commit `{}`. **Label: post hoc.** "
        "No retraining. Fit on ISIC train only. Primary variant = Relative Mahalanobis (Ren et al. 2021).".format(
            payload["commit"]),
        "",
        "## Pre-committed criterion (Relative Mahalanobis)",
        "",
        "At λ = 2: AUROC > 0.5 and cluster CI excludes 0.5 on ≥ 2 of {Fitz, MED-NODE, SD-198}; "
        "at λ = 0: drop vs standard ≤ 0.05 on those three. pad_heldout / SCIN reported separately.",
        "",
    ]
    for lab, crit in payload["criterion"].items():
        if crit is None:
            L += ["### {} — null (cell absent)".format(lab), ""]
            continue
        L += [
            "### {}".format(lab),
            "",
            "- OOD limb: {}/3 → **{}**".format(crit["n_hit"], "PASS" if crit["pass_ood_limb"] else "FAIL"),
            "- λ = 0 limb: **{}**".format(
                "PASS" if crit["pass_lambda0_limb"] else ("FAIL" if crit["pass_lambda0_limb"] is False else "null")),
            "- Criterion: **{}**".format("PASS" if crit["pass_criterion"] else "FAIL"),
            "",
            "| set | RMD mean | CI | hit |",
            "|---|---:|---|:---:|",
        ]
        for h in crit["hits"]:
            ci = "—" if not h["ci"] else "[{:.3f}, {:.3f}]".format(*h["ci"])
            L.append("| {} | {} | {} | {} |".format(
                h["ood"], "—" if h["mean"] is None else "{:.3f}".format(h["mean"]), ci, "yes" if h["ok"] else "no"))
        if crit["lambda0_drops"]:
            L += ["", "| set | λ0 drop (std − RMD) | ≤0.05 |", "|---|---:|:---:|"]
            for d in crit["lambda0_drops"]:
                L.append("| {} | {:.3f} | {} |".format(d["ood"], d["drop"], "yes" if d["ok"] else "no"))
        L.append("")

    # main tables per variant
    for vk, (vname, _) in VARIANTS.items():
        L += ["## {}".format(vname), ""]
        for ood in OODS_ALL:
            L += [
                "### {}".format(ood),
                "",
                "| cell | n | AUROC | 95% CI | CI > 0.5 |",
                "|---|---:|---|---|:---:|",
            ]
            for lab, cell in sorted(payload["cells"].items()):
                b = cell["ood"][ood][vk]
                ci = "—" if not b["bootstrap"] else "[{:.3f}, {:.3f}]".format(
                    b["bootstrap"]["ci_lo"], b["bootstrap"]["ci_hi"])
                L.append("| {} | {} | {} | {} | {} |".format(
                    lab, cell["n_seeds"], pm(b["point"]), ci,
                    "yes" if b["ci_excludes_0.5_above"] else "no"))
            L.append("")

    L += [
        "## Evaluation diagnostic — class-balanced AUROC (needs OOD labels; not deployable)",
        "",
        "| cell | ood | standard | relative | per-class std |",
        "|---|---|---|---|---|",
    ]
    for lab, cell in sorted(payload["cells"].items()):
        for ood in ("pad_heldout", "mednode", "sd198"):
            vals = [pm(cell["ood"][ood][vk]["classbal_diagnostic"]) for vk in VARIANTS]
            L.append("| {} | {} | {} | {} | {} |".format(lab, ood, *vals))
    L += ["", "Reading: see criterion blocks above; every variant is reported, not only the best.", ""]
    (out_dir / "REPORT.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------------------------------------------------
# (2) Clinical
# ----------------------------------------------------------------------------------------------------------------------
def run_clinical(out_name="clinical", recipes=None):
    """Clinical consequence table. Default: best-val main (old_recipe) + a4 + R1.

    For Table 1 last.ckpt use out_name='clinical_t1last' and recipes=('main_last',).
    """
    out_dir = OUT_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    n_boot = int(pre["n_resamples"])
    lab_fixed = ood_label_arrays()
    groups = ood_group_arrays()
    need = ("isic_train", "isic_val", "fitzpatrick17k", "sd198", "pad_heldout")
    rows = []
    for r in inventory(need=need):
        if r["degraded"]:
            continue
        if recipes is not None:
            if r["recipe"] in recipes and r["lambda"] in (0.0, 2.0):
                rows.append(r)
            continue
        if r["recipe"] == "old_recipe" and r["lambda"] in (0.0, 2.0):
            rows.append(r)
        elif r["recipe"] == "a4" and r["lambda"] in (0.0, 2.0):
            rows.append(r)
        elif r["recipe"] == "r1" and r["shift"] in (0, 100) and r["lambda"] == 2.0:
            rows.append(r)
    print("clinical runs", len(rows), "->", out_dir, flush=True)

    per_run2 = []
    for r in rows:
        Fs = {n: dict(np.load(r["feat_dir"] / "{}.npz".format(n)))
              for n in ("isic_train", "isic_val", "fitzpatrick17k", "sd198", "pad_heldout")}
        md = StdMaha(Fs["isic_train"]["z_lesion_norm"], Fs["isic_train"]["labels"].astype(int))
        tau = float(np.quantile(md.score(Fs["isic_val"]["z_lesion_norm"]), 0.95))
        rec = {"recipe": r["recipe"], "lambda": r["lambda"], "seed": r["seed"], "shift": r["shift"],
               "tau_95_val": tau, "feat_dir": str(r["feat_dir"]), "sets": {}}
        for ood, yood in (("fitzpatrick17k", lab_fixed["fitzpatrick17k"]), ("sd198", lab_fixed["sd198"])):
            so = md.score(Fs[ood]["z_lesion_norm"])
            logits = Fs[ood]["logits"]
            conf = R.softmax(logits).max(1)
            pred = logits.argmax(1)
            accepted = so <= tau
            pred_nv09 = (pred == NV) & (conf > 0.9)
            blk = {}
            for cname in MALIGN:
                c = LABELS.index(cname)
                m = yood == c
                n = int(m.sum())
                blk[cname] = {
                    "n": n,
                    "share_accepted_id": float(accepted[m].mean()) if n else None,
                    "share_pred_nv_conf09": float(pred_nv09[m].mean()) if n else None,
                    "accepted_cls": accepted[m].astype(np.float64) if n else np.array([]),
                    "nv09_cls": pred_nv09[m].astype(np.float64) if n else np.array([]),
                }
            rec["sets"][ood] = blk
            rec["n_unmapped_{}".format(ood)] = int((yood < 0).sum())
            rec["n_mapped_{}".format(ood)] = int((yood >= 0).sum())
        rec["pad_heldout_n"] = {LABELS[c]: int((Fs["pad_heldout"]["labels"].astype(int) == c).sum())
                                for c in range(NCLS)}
        per_run2.append(rec)
        print("clinical", r["feat_dir"].name, "tau={:.3f}".format(tau), flush=True)

    by_cell = defaultdict(list)
    for rec in per_run2:
        by_cell[cell_key(rec)].append(rec)

    cells = {}
    summary_rows = []
    for ck, recs in sorted(by_cell.items(), key=lambda x: str(x[0])):
        cell = {"label": cell_label(ck), "n_seeds": len(recs), "tau": ms([r["tau_95_val"] for r in recs]),
                "pad_heldout_n": recs[0]["pad_heldout_n"], "sets": {}}
        for ood in ("fitzpatrick17k", "sd198"):
            og = groups[ood]
            yood = lab_fixed[ood]
            cell["sets"][ood] = {"n_unmapped": recs[0]["n_unmapped_{}".format(ood)],
                                 "n_mapped": recs[0]["n_mapped_{}".format(ood)], "classes": {}}
            for cname in MALIGN:
                c = LABELS.index(cname)
                cls_idx = np.flatnonzero(yood == c)
                n = len(cls_idx)
                if n == 0:
                    cell["sets"][ood]["classes"][cname] = {"n": 0}
                    continue
                g_cls = og[cls_idx]
                rng = np.random.default_rng(
                    int(pre["rng_seed"]) + LABELS.index(cname) * 13 + (0 if ood.startswith("f") else 99))
                gdraws = R.draws(g_cls, n_boot, rng)
                mat_acc = np.zeros((n_boot, len(recs)))
                mat_nv = np.zeros((n_boot, len(recs)))
                for j, r in enumerate(recs):
                    a = r["sets"][ood][cname]["accepted_cls"]
                    v = r["sets"][ood][cname]["nv09_cls"]
                    for b, dd in enumerate(gdraws):
                        mat_acc[b, j] = a[dd].mean()
                        mat_nv[b, j] = v[dd].mean()
                pooled_acc = mat_acc.mean(1)
                pooled_nv = mat_nv.mean(1)
                acc_ms = ms([r["sets"][ood][cname]["share_accepted_id"] for r in recs])
                nv_ms = ms([r["sets"][ood][cname]["share_pred_nv_conf09"] for r in recs])
                block = {
                    "n": n,
                    "share_accepted_id": acc_ms,
                    "share_pred_nv_conf09": nv_ms,
                    "accepted_ci": [float(np.quantile(pooled_acc, 0.025)), float(np.quantile(pooled_acc, 0.975))],
                    "nv09_ci": [float(np.quantile(pooled_nv, 0.025)), float(np.quantile(pooled_nv, 0.975))],
                }
                cell["sets"][ood]["classes"][cname] = block
                summary_rows.append({
                    "cell": cell_label(ck), "ood": ood, "class": cname, "n": n,
                    "share_accepted_id": acc_ms["mean"] if acc_ms else None,
                    "accepted_ci_lo": block["accepted_ci"][0], "accepted_ci_hi": block["accepted_ci"][1],
                    "share_pred_nv_conf09": nv_ms["mean"] if nv_ms else None,
                    "nv09_ci_lo": block["nv09_ci"][0], "nv09_ci_hi": block["nv09_ci"][1],
                })
        cells[cell_label(ck)] = cell
        print("clinical cell", cell_label(ck), flush=True)

    per_light = []
    for r in per_run2:
        lite = {k: r[k] for k in ("recipe", "lambda", "seed", "shift", "tau_95_val", "feat_dir", "pad_heldout_n")}
        lite["sets"] = {}
        for ood in ("fitzpatrick17k", "sd198"):
            lite["sets"][ood] = {c: {kk: vv for kk, vv in r["sets"][ood][c].items()
                                     if kk not in ("accepted_cls", "nv09_cls")}
                                 for c in MALIGN}
        per_light.append(lite)

    payload = {
        "commit": R.git_head(),
        "decision": "decisions/paperB_r5_posthoc_scoring.md",
        "label": "post hoc",
        "label_maps": {
            "fitzpatrick17k": FITZ_MAP,
            "sd198": "SD198_MAP in scripts/r5_p7a_sets.py",
            "malignant": list(MALIGN),
            "threshold": "95th percentile of ISIC val standard Mahalanobis (per run, per λ)",
        },
        "cells": cells,
        "per_run": per_light,
    }
    (out_dir / "results.json").write_text(json.dumps(_jsonable(payload), indent=1))
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    _write_clinical_report(out_dir, payload)
    print("wrote", out_dir, flush=True)


def _write_clinical_report(out_dir, payload):
    L = [
        "# Paper B post hoc — clinical consequence of 95% ID-acceptance",
        "",
        "Design: `decisions/paperB_r5_posthoc_scoring.md`. Commit `{}`. **Label: post hoc.**".format(
            payload["commit"]),
        "",
        "Threshold τ = 95th percentile of each run’s ISIC **validation** standard Mahalanobis scores "
        "(λ = 0 and λ = 2 separately). Accept as ID if score ≤ τ. Malignant = MEL, BCC, SCC, AK.",
        "",
        "## Label maps",
        "",
        "- **Fitzpatrick17k:** `label` → " + ", ".join("{}→{}".format(k, v) for k, v in FITZ_MAP.items()) + ".",
        "- **SD-198:** `source_label` folder map in `scripts/r5_p7a_sets.py` (`SD198_MAP`).",
        "- **pad_heldout:** native PAD labels; **n only** (MEL = {}).".format(
            next(iter(payload["cells"].values()))["pad_heldout_n"].get("MEL", "null")),
        "",
    ]
    for lab, cell in sorted(payload["cells"].items()):
        L += [
            "## {}".format(lab),
            "",
            "τ (mean ± s.d.): {}. pad_heldout n: {}.".format(pm(cell["tau"]), cell["pad_heldout_n"]),
            "",
        ]
        for ood in ("fitzpatrick17k", "sd198"):
            s = cell["sets"][ood]
            L += [
                "### {}".format(ood),
                "",
                "mapped / unmapped images: {} / {}.".format(s["n_mapped"], s["n_unmapped"]),
                "",
                "| class | n | share accepted as ID [CI] | share pred NV & conf>0.9 [CI] |",
                "|---|---:|---|---|",
            ]
            for cname in MALIGN:
                b = s["classes"].get(cname, {"n": 0})
                if b.get("n", 0) == 0:
                    L.append("| {} | 0 | — | — |".format(cname))
                    continue
                L.append("| {} | {} | {} [{:.3f}, {:.3f}] | {} [{:.3f}, {:.3f}] |".format(
                    cname, b["n"], pm(b["share_accepted_id"]), *b["accepted_ci"],
                    pm(b["share_pred_nv_conf09"]), *b["nv09_ci"]))
            L.append("")
    (out_dir / "REPORT.md").write_text("\n".join(L))


# ----------------------------------------------------------------------------------------------------------------------
# (3) nv_frac correlation
# ----------------------------------------------------------------------------------------------------------------------
def run_nvfrac():
    out_dir = OUT_ROOT / "nvfrac_corr"
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_fixed = ood_label_arrays()
    rows = [r for r in inventory(need=("isic_train", "isic_test") + OODS_ALL) if not r["degraded"]]
    # include all old, a4, r1 with features
    print("nvfrac runs", len(rows), flush=True)
    panel = []
    for r in rows:
        Fs = {n: dict(np.load(r["feat_dir"] / "{}.npz".format(n)))
              for n in ("isic_train", "isic_test") + OODS_ALL}
        md = StdMaha(Fs["isic_train"]["z_lesion_norm"], Fs["isic_train"]["labels"].astype(int))
        sid = md.score(Fs["isic_test"]["z_lesion_norm"])
        # leakage if present in summary
        leak = None
        if r["recipe"] == "old_recipe":
            # from phase13 per_run json
            pj = R.RES / "phase13" / "per_run" / "{}.json".format(r["feat_dir"].name)
            if not pj.exists():
                pj = R.RES / "r2" / "item4" / "runs" / r["feat_dir"].name / "summary.json"
        elif r["recipe"] == "a4":
            pj = R.R5 / "p8" / "main" / "runs" / r["feat_dir"].name / "summary.json"
        else:
            pj = R.R6 / "r1" / "cur" / "runs" / r["feat_dir"].name / "summary.json"
        if pj.exists():
            s = json.loads(pj.read_text())
            # try common paths
            for path in (
                lambda s: s.get("leakage", {}).get("auroc_mean") if isinstance(s.get("leakage"), dict) else s.get("leakage"),
                lambda s: s.get("metrics", {}).get("leakage", {}).get("auroc_mean"),
                lambda s: s.get("ood", {}).get("pad_heldout", {}).get("leakage"),
            ):
                try:
                    v = path(s)
                    if isinstance(v, (int, float)):
                        leak = float(v)
                        break
                except Exception:
                    pass
        # nearest-centroid NV share
        # use Maha dists argmin
        dist = md.dists(Fs["isic_test"]["z_lesion_norm"])  # unused; need ood
        for ood in OODS_ALL:
            so = md.score(Fs[ood]["z_lesion_norm"])
            auroc = R.auroc(sid, so)
            yood = Fs[ood]["labels"].astype(int) if ood == "pad_heldout" else lab_fixed[ood]
            mapped = yood >= 0
            nv_true = float((yood[mapped] == NV).mean()) if mapped.any() else None
            nc = md.dists(Fs[ood]["z_lesion_norm"]).argmin(1)
            nv_near = float((nc == NV).mean())
            panel.append({
                "recipe": r["recipe"], "lambda": r["lambda"], "seed": r["seed"], "shift": r["shift"],
                "ood": ood, "maha_auroc": auroc, "nv_frac_true": nv_true, "nv_frac_nearest": nv_near,
                "leakage": leak, "n_mapped": int(mapped.sum()) if ood != "pad_heldout" else int(len(yood)),
            })
        print("nvfrac", r["feat_dir"].name, flush=True)

    df = pd.DataFrame(panel)
    df.to_csv(out_dir / "panel.csv", index=False)
    corr_rows = []
    results = {"commit": R.git_head(), "label": "post hoc", "per_ood": {}}
    for ood in OODS_ALL:
        sub = df[df.ood == ood].copy()
        block = {}
        for nvcol in ("nv_frac_true", "nv_frac_nearest"):
            m = sub[nvcol].notna() & sub["maha_auroc"].notna()
            if m.sum() < 5:
                block[nvcol] = None
                continue
            rho, p = spearmanr(sub.loc[m, nvcol], sub.loc[m, "maha_auroc"])
            block[nvcol] = {"spearman_rho": float(rho), "p_value": float(p), "n": int(m.sum())}
            corr_rows.append({"ood": ood, "x": nvcol, "spearman_rho": float(rho), "p": float(p), "n": int(m.sum())})
        # R^2 descriptive
        m = sub["nv_frac_true"].notna() & sub["maha_auroc"].notna()
        m2 = m & sub["leakage"].notna()
        r2 = {}
        if m.sum() >= 5:
            X = sub.loc[m, ["nv_frac_true"]].to_numpy()
            y = sub.loc[m, "maha_auroc"].to_numpy()
            r2["nv_frac_true"] = float(LinearRegression().fit(X, y).score(X, y))
        if m2.sum() >= 5:
            X = sub.loc[m2, ["leakage"]].to_numpy()
            y = sub.loc[m2, "maha_auroc"].to_numpy()
            r2["leakage"] = float(LinearRegression().fit(X, y).score(X, y))
            X2 = sub.loc[m2, ["nv_frac_true", "leakage"]].to_numpy()
            r2["nv_frac_true+leakage"] = float(LinearRegression().fit(X2, y).score(X2, y))
        block["r2_descriptive"] = r2
        block["n_rows"] = int(len(sub))
        results["per_ood"][ood] = block
        corr_rows.append({"ood": ood, "x": "r2_nv_frac_true", "spearman_rho": r2.get("nv_frac_true"),
                          "p": None, "n": int(m.sum())})
        corr_rows.append({"ood": ood, "x": "r2_leakage", "spearman_rho": r2.get("leakage"),
                          "p": None, "n": int(m2.sum())})

    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=1))
    pd.DataFrame(corr_rows).to_csv(out_dir / "summary.csv", index=False)
    L = [
        "# Paper B post hoc — nv_frac vs Mahalanobis AUROC (descriptive)",
        "",
        "Design: `decisions/paperB_r5_posthoc_scoring.md`. Commit `{}`. **Label: post hoc.** "
        "Correlation / variance explained only — **not** mediation. Not the pre-committed R1 dose test (a).".format(
            results["commit"]),
        "",
        "| OOD set | Spearman(nv_frac true, Maha) | p | n | Spearman(nv_frac nearest, Maha) | R²(nv_frac) | R²(leakage) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ood in OODS_ALL:
        b = results["per_ood"][ood]
        t = b.get("nv_frac_true") or {}
        n = b.get("nv_frac_nearest") or {}
        r2 = b.get("r2_descriptive") or {}
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            ood,
            "—" if not t else "{:.3f}".format(t["spearman_rho"]),
            "—" if not t else "{:.2g}".format(t["p_value"]),
            "—" if not t else t["n"],
            "—" if not n else "{:.3f}".format(n["spearman_rho"]),
            "—" if r2.get("nv_frac_true") is None else "{:.3f}".format(r2["nv_frac_true"]),
            "—" if r2.get("leakage") is None else "{:.3f}".format(r2["leakage"]),
        ))
    L += ["", "Panel: `panel.csv` (one row per run × OOD set).", ""]
    (out_dir / "REPORT.md").write_text("\n".join(L))
    print("wrote", out_dir, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("mitigation", "clinical", "clinical_t1last", "nvfrac", "all"))
    a = ap.parse_args()
    if a.mode in ("mitigation", "all"):
        run_mitigation()
    if a.mode in ("clinical", "all"):
        run_clinical()
    if a.mode == "clinical_t1last":
        run_clinical(out_name="clinical_t1last", recipes=("main_last",))
    if a.mode in ("nvfrac", "all"):
        run_nvfrac()


if __name__ == "__main__":
    main()
