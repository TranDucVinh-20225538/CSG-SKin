#!/usr/bin/env python3
"""SD-198 P13-style near-duplicate clusters: image vs cluster_union CI / rule A comparison.

Clusters already built by r5_d1_domain_direction.sd198_clusters (pHash Hamming ≤ 8 ∪ ImageNet-R50
cosine ≥ 0.95). This script copies them to r5/p13/, re-boots Mahalanobis (and clinical rates) with
image vs cluster_union resampling for P7(a), R1, detmech_shift, R2, S1/S2, clinical; writes
results/paperB/r5/p13_sd198/REPORT.md (old vs new side by side; flags verdict flips).

R1 (a)/(b): documented only — (a) wild-cluster OLS by setting cell; (b) seed bootstrap of point
AUROCs; neither resamples SD-198 images, so image→cluster does not change those CIs.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R5  # noqa: E402
import r6_common as R6  # noqa: E402

SRC_CL = R5.R5 / "d1_domain" / "sd198_clusters.csv"
SRC_ST = R5.R5 / "d1_domain" / "sd198_cluster_stats.json"
DST_CL = R5.R5 / "p13" / "sd198_clusters.csv"
DST_ST = R5.R5 / "p13" / "sd198_cluster_stats.json"
OUT = R5.R5 / "p13_sd198"
DET = "mahalanobis_classcond"
SETS_CSV = R5.R5 / "p7a" / "sets" / "sd198.csv"


def ensure_clusters():
    OUT.mkdir(parents=True, exist_ok=True)
    (R5.R5 / "p13").mkdir(parents=True, exist_ok=True)
    assert SRC_CL.exists(), "missing {}; run r5_d1_domain_direction.sd198_clusters first".format(SRC_CL)
    shutil.copy2(SRC_CL, DST_CL)
    if SRC_ST.exists():
        shutil.copy2(SRC_ST, DST_ST)
    cl = pd.read_csv(DST_CL, keep_default_na=False)
    meta = pd.read_csv(SETS_CSV, keep_default_na=False)
    assert cl.path.tolist() == meta.path.tolist(), "cluster CSV path order != p7a/sets/sd198.csv"
    stats = json.loads(DST_ST.read_text()) if DST_ST.exists() else {
        "n_images": len(cl), "n_clusters": int(cl.cluster_union.nunique())}
    return cl, stats


def groups_image_and_cluster(cl):
    # P7a group column is 1:1 with images for SD-198 (class/stem).
    g_img = pd.read_csv(SETS_CSV, keep_default_na=False).group.astype(str).to_numpy()
    g_cl = cl.cluster_union.astype(str).to_numpy()
    assert len(g_img) == len(g_cl) == 6584
    return g_img, g_cl


def maha_pair(feat_dir):
    """Return (s_id, s_ood) Mahalanobis scores for one run directory with isic_train/test + sd198."""
    Fs = R6.load_feats(Path(feat_dir), ("isic_train", "isic_test", "sd198"))
    if "sd198" not in Fs:
        return None
    S = R6.Scorer(Fs["isic_train"])
    return S.scores(Fs["isic_test"])[DET], S.scores(Fs["sd198"])[DET]


def boot(per_seed, id_draws, ood_draws):
    if not per_seed:
        return None
    return R5.pooled_bootstrap(per_seed, id_draws, ood_draws)


def fmt(b):
    if b is None:
        return "—"
    return "{:.3f} [{:.3f}, {:.3f}] k={}/{} ruleA={}".format(
        b["mean"], b["pooled_ci_lo"], b["pooled_ci_hi"],
        b["k_seeds_ci_below_0.5"], b["n_seeds"], "met" if b["rule_A"] else "not")


def short(b):
    if b is None:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "k": None, "n": 0, "rule_A": None}
    return {"mean": b["mean"], "ci_lo": b["pooled_ci_lo"], "ci_hi": b["pooled_ci_hi"],
            "k": b["k_seeds_ci_below_0.5"], "n": b["n_seeds"], "rule_A": bool(b["rule_A"])}


def compare_cell(label, per_seed, id_draws, draws_img, draws_cl, exclude=None):
    per = {s: v for s, v in per_seed.items() if exclude is None or s not in exclude}
    bi = boot(per, id_draws, draws_img)
    bc = boot(per, id_draws, draws_cl)
    flip = None
    if bi is not None and bc is not None and bi["rule_A"] != bc["rule_A"]:
        flip = "{} → {}".format("met" if bi["rule_A"] else "not", "met" if bc["rule_A"] else "not")
    return {"label": label, "n_seeds": len(per), "excluded": sorted(exclude or ()),
            "image": short(bi), "cluster": short(bc), "verdict_flip": flip,
            "image_fmt": fmt(bi), "cluster_fmt": fmt(bc)}


def make_draws(idg, g_img, g_cl, n_boot, seed):
    rng_i = np.random.default_rng(seed)
    rng_c = np.random.default_rng(seed)  # same seed → comparable protocol; independent streams per unit
    id_draws = R5.draws(idg, n_boot, rng_i)
    # Rebuild rng for OOD so ID draws match PREREGISTER shared-stream style per unit independently.
    rng_oi = np.random.default_rng(seed + 1)
    rng_oc = np.random.default_rng(seed + 1)
    return id_draws, R5.draws(g_img, n_boot, rng_oi), R5.draws(g_cl, n_boot, rng_oc)


def section_p7a(idg, g_img, g_cl, n_boot, seed):
    id_draws, di, dc = make_draws(idg, g_img, g_cl, n_boot, seed)
    rows = []
    for lam in R5.LAMS:
        per = {}
        for s in sorted(int(d.name.rsplit("_s", 1)[1]) for d in R5.FEAT_L.glob(
                "runB_orth1_ladv{}_s*".format(R5.C.lam_tag(lam)))):
            pair = maha_pair(R5.FEAT_L / "runB_orth1_ladv{}_s{}".format(R5.C.lam_tag(lam), s))
            if pair:
                per[s] = pair
        rows.append(compare_cell("P7a λ={:g}".format(lam), per, id_draws, di, dc))
        print("p7a", lam, rows[-1]["image_fmt"], "vs", rows[-1]["cluster_fmt"], flush=True)
    return rows


def _r1_id_bal(run_dir):
    sj = R6.RES / "r6" / "r1" / "cur" / "runs" / run_dir.name / "summary.json"
    if not sj.exists():
        return None
    m = json.loads(sj.read_text()).get("metrics", {})
    return float((m.get("id") or {}).get("bal_acc")) if m.get("id") else None


def section_r1(idg, g_img, g_cl, n_boot, seed):
    root = R6.RES / "r6" / "r1" / "cur" / "features"
    id_draws, di, dc = make_draws(idg, g_img, g_cl, n_boot, seed)
    cells = defaultdict(dict)  # (sh, lam) -> {seed: pair}
    deg = defaultdict(set)
    for d in sorted(root.glob("r1_cur_shift*_ladv*_s*")):
        m = re.fullmatch(r"r1_cur_shift(\d+)_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        sh, lam, s = int(m.group(1)), float(m.group(2).replace("p", ".")), int(m.group(3))
        pair = maha_pair(d)
        if pair is None:
            continue
        cells[(sh, lam)][s] = pair
        ib = _r1_id_bal(d)
        if ib is not None and ib < R6.DEGRADED_ID_BAL:
            deg[(sh, lam)].add(s)
    rows = []
    for (sh, lam), per in sorted(cells.items()):
        lab = "R1 shift {}% λ={:g}".format(sh, lam)
        rows.append(compare_cell(lab + " [excl degraded]", per, id_draws, di, dc, exclude=deg[(sh, lam)]))
        rows.append(compare_cell(lab + " [all seeds]", per, id_draws, di, dc))
        print(lab, rows[-2]["verdict_flip"], rows[-1]["verdict_flip"], flush=True)
    return rows


def section_r2(idg, g_img, g_cl, n_boot, seed):
    root = R6.RES / "r6" / "r2" / "cur" / "features"
    id_draws, di, dc = make_draws(idg, g_img, g_cl, n_boot, seed)
    # also R1 shift100 as DANN / λ0 baselines
    r1 = R6.RES / "r6" / "r1" / "cur" / "features"
    cells = defaultdict(dict)
    for d in sorted(root.glob("r2_cur_*_ladv*_s*")):
        m = re.fullmatch(r"r2_cur_([a-z0-9_]+)_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        method, lam, s = m.group(1), float(m.group(2).replace("p", ".")), int(m.group(3))
        pair = maha_pair(d)
        if pair:
            cells[("R2 " + method.upper(), lam)][s] = pair
    for d in sorted(r1.glob("r1_cur_shift100_ladv*_s*")):
        m = re.fullmatch(r"r1_cur_shift100_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        lam, s = float(m.group(1).replace("p", ".")), int(m.group(2))
        pair = maha_pair(d)
        if pair:
            lab = "λ=0 (R1 shift100)" if abs(lam) < 1e-12 else "DANN (R1 shift100)"
            cells[(lab, lam)][s] = pair
    rows = []
    for (lab, lam), per in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0])):
        rows.append(compare_cell("{} λ={:g}".format(lab, lam), per, id_draws, di, dc))
        print(rows[-1]["label"], rows[-1]["verdict_flip"], flush=True)
    return rows


def section_detmech(idg, g_img, g_cl, n_boot, seed):
    """Recompute the three Mahalanobis cells used in detmech_shift REPORT (healthy seeds)."""
    id_draws, di, dc = make_draws(idg, g_img, g_cl, n_boot, seed)
    specs = [
        ("A4 λ=0 (baseline)", R6.P8_DIR / "main" / "features",
         sorted((R6.P8_DIR / "main" / "features").glob("main_ladv0_s*"))),
        ("R1 λ=2 shift 0%", R6.RES / "r6" / "r1" / "cur" / "features",
         sorted((R6.RES / "r6" / "r1" / "cur" / "features").glob("r1_cur_shift0_ladv2_s*"))),
        ("R1 λ=2 shift 100%", R6.RES / "r6" / "r1" / "cur" / "features",
         sorted((R6.RES / "r6" / "r1" / "cur" / "features").glob("r1_cur_shift100_ladv2_s*"))),
    ]
    rows = []
    for lab, _root, dirs in specs:
        per, excl = {}, set()
        for d in dirs:
            s = int(d.name.rsplit("_s", 1)[1])
            pair = maha_pair(d)
            if pair is None:
                continue
            # healthy: ID bal ≥ 0.25 when summary exists
            if "r1_cur" in d.name:
                ib = _r1_id_bal(d)
                if ib is not None and ib < R6.DEGRADED_ID_BAL:
                    excl.add(s)
                    continue
            per[s] = pair
        rows.append(compare_cell("detmech " + lab, per, id_draws, di, dc))
        print(lab, rows[-1]["verdict_flip"], flush=True)
    return rows


def section_sflip(idg, g_img, g_cl, n_boot, seed):
    id_draws, di, dc = make_draws(idg, g_img, g_cl, n_boot, seed)
    rows = []
    for lam in R5.LAMS:
        per = {}
        for s in sorted(int(d.name.rsplit("_s", 1)[1]) for d in R5.FEAT_L.glob(
                "runB_orth1_ladv{}_s*".format(R5.C.lam_tag(lam)))):
            pair = maha_pair(R5.FEAT_L / "runB_orth1_ladv{}_s{}".format(R5.C.lam_tag(lam), s))
            if pair:
                per[s] = pair
        rows.append(compare_cell("S1/S2 (Table1) λ={:g}".format(lam), per, id_draws, di, dc))
        print("sflip", lam, rows[-1]["verdict_flip"], flush=True)
    return rows


def section_clinical(g_img, g_cl, n_boot, seed):
    """Acceptance-rate CIs for malignant classes on SD-198 (same cells as posthoc clinical)."""
    import r5_posthoc_scoring as PH  # noqa: E402

    lab_fixed = PH.ood_label_arrays()
    yood = lab_fixed["sd198"]
    rows_inv = []
    for r in PH.inventory():
        if r["degraded"]:
            continue
        if r["recipe"] == "old_recipe" and r["lambda"] in (0.0, 2.0):
            rows_inv.append(r)
        elif r["recipe"] == "a4" and r["lambda"] in (0.0, 2.0):
            rows_inv.append(r)
        elif r["recipe"] == "r1" and r["shift"] in (0, 100) and r["lambda"] == 2.0:
            rows_inv.append(r)

    per_run = []
    for r in rows_inv:
        Fs = {n: dict(np.load(r["feat_dir"] / "{}.npz".format(n)))
              for n in ("isic_train", "isic_val", "sd198")}
        md = PH.StdMaha(Fs["isic_train"]["z_lesion_norm"], Fs["isic_train"]["labels"].astype(int))
        tau = float(np.quantile(md.score(Fs["isic_val"]["z_lesion_norm"]), 0.95))
        so = md.score(Fs["sd198"]["z_lesion_norm"])
        accepted = so <= tau
        blk = {}
        for cname in PH.MALIGN:
            c = PH.LABELS.index(cname)
            m = yood == c
            blk[cname] = accepted[m].astype(np.float64) if m.any() else np.array([])
        per_run.append({"key": PH.cell_key(r), "label": PH.cell_label(PH.cell_key(r)),
                        "seed": r["seed"], "acc": blk})

    by = defaultdict(list)
    for rec in per_run:
        by[rec["label"]].append(rec)

    out_rows = []
    for lab, recs in sorted(by.items()):
        for cname in PH.MALIGN:
            c = PH.LABELS.index(cname)
            cls_idx = np.flatnonzero(yood == c)
            if len(cls_idx) == 0:
                continue
            point = float(np.mean([r["acc"][cname].mean() for r in recs if len(r["acc"][cname])]))
            cis = {}
            for unit, g in (("image", g_img), ("cluster", g_cl)):
                g_cls = g[cls_idx]
                rng = np.random.default_rng(seed + PH.LABELS.index(cname) * 13 + 99)
                gdraws = R5.draws(g_cls, n_boot, rng)
                mat = np.zeros((n_boot, len(recs)))
                for j, r in enumerate(recs):
                    a = r["acc"][cname]
                    for b, dd in enumerate(gdraws):
                        mat[b, j] = a[dd].mean()
                pooled = mat.mean(1)
                cis[unit] = [float(np.quantile(pooled, 0.025)), float(np.quantile(pooled, 0.975))]
            # clinical has no rule A; flag CI-cover change for the point (whether 0.5 in interval) N/A —
            # flag if intervals' relation to each other differs qualitatively: width ratio / endpoint move.
            out_rows.append({
                "label": "{} | {}".format(lab, cname),
                "n": int(len(cls_idx)), "point": point,
                "image_ci": cis["image"], "cluster_ci": cis["cluster"],
                "verdict_flip": None,  # no rule A
            })
        print("clinical", lab, flush=True)
    return out_rows


def section_ab_note():
    reg = R6.RES / "r6" / "r1" / "cur" / "leakage_matched" / "regression.json"
    mat = R6.RES / "r6" / "r1" / "cur" / "leakage_matched" / "matched.json"
    return {
        "note": (
            "R1 (a) wild-cluster bootstrap clusters = setting cells (recipe|λ|shift_level), not SD-198 "
            "images/near-duplicates; y = seed-level Mahalanobis point AUROC from summary.json. "
            "R1 (b) is a seed bootstrap of those same point AUROCs. Neither path resamples SD-198, so "
            "switching SD-198 from image to cluster_union does not change (a) or (b) CIs or verdicts."
        ),
        "regression_exists": reg.exists(),
        "matched_exists": mat.exists(),
        "code_a": "scripts/r6_r1_leakage_matched.py:244-247 cluster = recipe|lambda|shift_level",
        "code_b": "scripts/r6_r1_leakage_matched.py:283-296 seed bootstrap of point AUROCs",
        "auroc_source": "maha_of(summary.json) ← metrics.ood.sd198.mahalanobis_classcond (point, no resample)",
    }


def write_report(stats, sections, clinical, ab):
    L = ["# SD-198 image vs P13-style cluster_union bootstrap", "",
         "Clusters: `results/paperB/r5/p13/sd198_clusters.csv` (copied from `r5/d1_domain/`; method = "
         "DCT pHash Hamming ≤ 8 ∪ ImageNet-R50 cosine ≥ 0.95, connected components of the union — same as "
         "Fitzpatrick P13). Commit `{}`.".format(R5.git_head()), "",
         "## Cluster count", "",
         "| | |", "|---|---:|",
         "| images | {} |".format(stats.get("n_images", 6584)),
         "| cluster_union | {} |".format(stats.get("n_clusters")),
         "| singletons | {} |".format(stats.get("n_singletons", "—")),
         "| largest cluster | {} |".format(stats.get("largest", "—")),
         "| effective n | {:.1f} |".format(stats.get("effective_n", float("nan"))),
         "| n_edges_phash | {} |".format(stats.get("n_edges_phash", "—")),
         "| n_edges_embed | {} |".format(stats.get("n_edges_embed", "—")),
         "",
         "PREREGISTER 2,000 draws, rng 20261005; ID = lesion; Mahalanobis class-conditional on "
         "`z_lesion_norm`. Rule A = pooled CI hi < 0.5 and k_below ≥ 3.", ""]

    flips = []
    for name, rows in sections:
        L += ["## {}".format(name), "",
              "| cell | image (old) | cluster_union (new) | verdict flip |",
              "|---|---|---|---|"]
        for r in rows:
            L.append("| {} | {} | {} | {} |".format(
                r["label"], r["image_fmt"], r["cluster_fmt"], r["verdict_flip"] or "—"))
            if r["verdict_flip"]:
                flips.append("{}: {}".format(r["label"], r["verdict_flip"]))
        L.append("")

    L += ["## Clinical (acceptance rates; no rule A)", "",
          "| cell | class | n | point | image CI | cluster CI |",
          "|---|---|---:|---:|---|---|"]
    for r in clinical:
        cell, cls = r["label"].rsplit(" | ", 1)
        L.append("| {} | {} | {} | {:.3f} | [{:.3f}, {:.3f}] | [{:.3f}, {:.3f}] |".format(
            cell, cls, r["n"], r["point"],
            r["image_ci"][0], r["image_ci"][1], r["cluster_ci"][0], r["cluster_ci"][1]))

    L += ["", "## R1 (a) / (b)", "", ab["note"], "",
          "- (a) code: `{}`".format(ab["code_a"]),
          "- (b) code: `{}`".format(ab["code_b"]),
          "- AUROC inputs: `{}`".format(ab["auroc_source"]),
          "",
          "## Verdict flips (rule A image → cluster)", ""]
    if flips:
        for f in flips:
            L.append("- **{}**".format(f))
    else:
        L.append("- none")
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", flush=True)


def main():
    cl, stats = ensure_clusters()
    g_img, g_cl = groups_image_and_cluster(cl)
    print("SD-198 clusters: {} / {} images".format(stats.get("n_clusters"), stats.get("n_images")), flush=True)
    pre = R5.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    idg = R5.id_test_groups()

    sections = [
        ("P7(a) — Table 1 runs", section_p7a(idg, g_img, g_cl, n_boot, seed)),
        ("R1 — dose cells", section_r1(idg, g_img, g_cl, n_boot, seed)),
        ("detmech_shift", section_detmech(idg, g_img, g_cl, n_boot, seed)),
        ("R2", section_r2(idg, g_img, g_cl, n_boot, seed)),
        ("S1/S2 (lesion Table-1 block)", section_sflip(idg, g_img, g_cl, n_boot, seed)),
    ]
    clinical = section_clinical(g_img, g_cl, n_boot, seed)
    ab = section_ab_note()

    payload = {
        "commit": R5.git_head(),
        "cluster_stats": stats,
        "cluster_csv": str(DST_CL),
        "sections": {name: rows for name, rows in sections},
        "clinical": clinical,
        "r1_ab": ab,
        "flips": [r for _, rows in sections for r in rows if r.get("verdict_flip")],
    }
    (OUT / "comparison.json").write_text(json.dumps(payload, indent=1, default=float) + "\n")
    write_report(stats, sections, clinical, ab)


if __name__ == "__main__":
    main()
