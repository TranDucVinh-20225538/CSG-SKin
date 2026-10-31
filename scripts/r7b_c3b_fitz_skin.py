#!/usr/bin/env python3
"""Paper B R7b C3b: Fitzpatrick17k AUROC by skin type (decisions/paperB_r7b_c3b.md).

Table 1 last.ckpt; Mahalanobis + kNN; ID = full ISIC test. Groups 1–2 / 3–4 / 5–6 / unknown(−1).
Reweighted AUROC to full-csv Fitzpatrick mix, and to type × ISIC-mapped class.
Bootstrap by P13 cluster_union. No image download.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import train_phase3_sweep as p3  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7B / "c3b"
FEAT = R.R5 / "t1last" / "features_last"
SUM = R.R5 / "t1last" / "summary.csv"
CSV = R.PAPERB / "results" / "paperB" / "phase11_third_domain" / "fitzpatrick17k.csv"
LMAP = CSV.parent / "fitzpatrick17k_label_map.json"
P13 = R.R5 / "p13" / "fitz_clusters.csv"
REP = "z_lesion_norm"
DETS = ("mahalanobis_classcond", "knn_k50")
GROUPS = (
    ("types_1_2", (1, 2)),
    ("types_3_4", (3, 4)),
    ("types_5_6", (5, 6)),
    ("unknown", (-1,)),
)


def table1_runs():
    rows = []
    with open(SUM, newline="") as f:
        for r in csv.DictReader(f):
            if r["checkpoint"] != "last":
                continue
            rows.append(r)
    return rows


def load_pack(tag, name):
    p = FEAT / tag / "{}.npz".format(name)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def fitz_meta():
    """Align sorted FITZ jpgs (feature order) with csv + P13 clusters + ISIC-mapped class."""
    paths = sorted(p3.FITZ_DIR.glob("*.jpg"))
    full = pd.read_csv(CSV)
    kept_csv = full.set_index("md5hash")
    cl = pd.read_csv(P13).set_index("md5hash")["cluster_union"]
    rule = {str(k).lower(): v for k, v in json.loads(LMAP.read_text())["mapping_rule"].items()}
    scales, clusters, isic = [], [], []
    for p in paths:
        md5 = p.stem
        scales.append(int(kept_csv.loc[md5, "fitzpatrick_scale"]))
        clusters.append(str(cl[md5]))
        raw = str(kept_csv.loc[md5, "label"]).lower()
        isic.append(rule.get(raw, "UNMAPPED"))
    scales = np.asarray(scales, int)
    clusters = np.asarray(clusters)
    isic = np.asarray(isic)
    assert len(paths) == 3887
    # full vs kept type shares
    full_s = full["fitzpatrick_scale"].astype(int).to_numpy()
    types = sorted(set(full_s.tolist()) | set(scales.tolist()))
    full_share = {t: float((full_s == t).mean()) for t in types}
    kept_share = {t: float((scales == t).mean()) for t in types}
    w_type = np.array([full_share[t] / kept_share[t] if kept_share[t] > 0 else 0.0 for t in scales])
    # type × ISIC-mapped class
    def key(t, c):
        return (int(t), str(c))
    full_isic = full["label"].astype(str).str.lower().map(lambda x: rule.get(x, "UNMAPPED")).to_numpy()
    full_keys = [key(t, c) for t, c in zip(full_s, full_isic)]
    kept_keys = [key(t, c) for t, c in zip(scales, isic)]
    from collections import Counter
    nf, nk = Counter(full_keys), Counter(kept_keys)
    n_full, n_kept = len(full_keys), len(kept_keys)
    w_txc = np.array([
        (nf[k] / n_full) / (nk[k] / n_kept) if nk[k] > 0 else 0.0 for k in kept_keys
    ])
    group_idx = {}
    group_stats = {}
    for name, vals in GROUPS:
        idx = np.where(np.isin(scales, vals))[0]
        group_idx[name] = idx
        group_stats[name] = {
            "n_images": int(len(idx)),
            "n_p13_clusters": int(len(set(clusters[idx].tolist()))) if len(idx) else 0,
            "types": list(vals),
        }
    return {
        "scales": scales, "clusters": clusters, "isic": isic,
        "w_type": w_type, "w_txc": w_txc,
        "group_idx": group_idx, "group_stats": group_stats,
        "full_share_type": full_share, "kept_share_type": kept_share,
        "n_full": int(len(full)), "n_kept": int(len(paths)),
    }


def scores_maha_knn(tr, pack):
    det = R.Detectors(tr[REP], tr["labels"].astype(int), 8)
    sc, _ = det.scores(pack[REP], pack["logits"])
    return {d: np.asarray(sc[d], float) for d in DETS}


def auroc_cell(sid, sood, w_ood=None):
    if len(sood) == 0:
        return None
    return R.auroc(sid, sood, w_ood)


def bootstrap_auroc(per_seed, id_groups, ood_groups, rng, n_boot, w_per_seed=None):
    """per_seed: {seed: (sid, sood)}; optional w_per_seed: {seed: w_ood aligned to sood}.
    OOD resampled by ood_groups (index lists into sood)."""
    gi = list(id_groups)
    go = list(ood_groups)
    if not go or not per_seed:
        return None
    di = [np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))]) for _ in range(n_boot)]
    do = [np.concatenate([go[j] for j in rng.integers(0, len(go), len(go))]) for _ in range(n_boot)]
    draws, per = {}, {}
    for s, (sid, so) in per_seed.items():
        w = None if w_per_seed is None else w_per_seed[s]
        pts = []
        for b in range(n_boot):
            ww = None if w is None else w[do[b]]
            pts.append(R.auroc(sid[di[b]], so[do[b]], ww))
        draws[s] = np.asarray(pts, float)
        lo, hi = np.quantile(draws[s], [0.025, 0.975])
        pt = auroc_cell(sid, so, None if w is None else w)
        per[str(s)] = {"point": pt, "ci_lo": float(lo), "ci_hi": float(hi),
                       "ci_entirely_below_0.5": bool(hi < 0.5)}
    pooled = np.mean(np.stack(list(draws.values())), 0)
    lo, hi = np.quantile(pooled, [0.025, 0.975])
    k = int(sum(p["ci_entirely_below_0.5"] for p in per.values()))
    rule = ("A" if k >= 3 else "B") if hi < 0.5 else ("C" if lo < 0.5 else "above")
    pts = [p["point"] for p in per.values() if p["point"] is not None]
    return {"mean": float(np.mean(pts)) if pts else None,
            "sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0,
            "pooled_ci": [float(lo), float(hi)], "k_below": k, "n_seeds": len(per),
            "rule": rule, "per_seed": per}


def ood_groups_from_clusters(cluster_ids, idx):
    """P13 groups restricted to image indices idx (positions in full kept array)."""
    sub = cluster_ids[idx]
    # map local positions 0..len(idx)-1
    return R.idx_groups(sub.tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = fitz_meta()
    id_groups = R.isic_test_groups()
    n_boot, seed = R.prereg()
    rng = np.random.default_rng(seed)

    # collect per (lam, seed) score vectors
    by_lam = defaultdict(dict)  # lam -> seed -> {det: (sid, sood_full)}
    for r in table1_runs():
        tag, lam, s = r["run"], float(r["lambda"]), int(r["seed"])
        tr = load_pack(tag, "isic_train")
        te = load_pack(tag, "isic_test")
        oo = load_pack(tag, "fitzpatrick17k")
        if tr is None or te is None or oo is None:
            print("missing", tag)
            continue
        sid = scores_maha_knn(tr, te)
        sood = scores_maha_knn(tr, oo)
        assert len(next(iter(sood.values()))) == meta["n_kept"]
        by_lam[lam][s] = {d: (sid[d], sood[d]) for d in DETS}

    summary_rows = []
    boot = {"group_stats": meta["group_stats"],
            "full_share_type": meta["full_share_type"],
            "kept_share_type": meta["kept_share_type"],
            "n_full": meta["n_full"], "n_kept": meta["n_kept"],
            "cells": {}}

    for lam in sorted(by_lam):
        seeds = by_lam[lam]
        for det in DETS:
            # per group
            for gname, _ in GROUPS:
                idx = meta["group_idx"][gname]
                per = {s: (v[det][0], v[det][1][idx]) for s, v in seeds.items()}
                go = ood_groups_from_clusters(meta["clusters"], idx) if len(idx) else []
                cell = bootstrap_auroc(per, id_groups, go, rng, n_boot) if len(idx) else None
                key = "lam{:g}|{}|{}".format(lam, gname, det)
                boot["cells"][key] = cell
                st = meta["group_stats"][gname]
                summary_rows.append({
                    "lambda": lam, "setting": gname, "detector": det,
                    "n_images": st["n_images"], "n_p13_clusters": st["n_p13_clusters"],
                    "auroc_mean": None if cell is None else cell["mean"],
                    "auroc_sd": None if cell is None else cell["sd"],
                    "pooled_ci_lo": None if cell is None else cell["pooled_ci"][0],
                    "pooled_ci_hi": None if cell is None else cell["pooled_ci"][1],
                    "k_below": None if cell is None else cell["k_below"],
                    "rule": None if cell is None else cell["rule"],
                    "n_seeds": None if cell is None else cell["n_seeds"],
                })
            # reweighted: all kept
            go_all = R.idx_groups(meta["clusters"].tolist())
            for wname, wvec in (("reweight_type", meta["w_type"]),
                                ("reweight_type_x_isic", meta["w_txc"])):
                per = {s: (v[det][0], v[det][1]) for s, v in seeds.items()}
                wps = {s: wvec for s in seeds}
                cell = bootstrap_auroc(per, id_groups, go_all, rng, n_boot, w_per_seed=wps)
                key = "lam{:g}|{}|{}".format(lam, wname, det)
                boot["cells"][key] = cell
                summary_rows.append({
                    "lambda": lam, "setting": wname, "detector": det,
                    "n_images": meta["n_kept"], "n_p13_clusters": int(len(set(meta["clusters"].tolist()))),
                    "auroc_mean": cell["mean"], "auroc_sd": cell["sd"],
                    "pooled_ci_lo": cell["pooled_ci"][0], "pooled_ci_hi": cell["pooled_ci"][1],
                    "k_below": cell["k_below"], "rule": cell["rule"], "n_seeds": cell["n_seeds"],
                })
            # unrestricted kept (reference)
            per = {s: (v[det][0], v[det][1]) for s, v in seeds.items()}
            cell = bootstrap_auroc(per, id_groups, go_all, rng, n_boot)
            key = "lam{:g}|kept_all|{}".format(lam, det)
            boot["cells"][key] = cell
            summary_rows.append({
                "lambda": lam, "setting": "kept_all", "detector": det,
                "n_images": meta["n_kept"], "n_p13_clusters": int(len(set(meta["clusters"].tolist()))),
                "auroc_mean": cell["mean"], "auroc_sd": cell["sd"],
                "pooled_ci_lo": cell["pooled_ci"][0], "pooled_ci_hi": cell["pooled_ci"][1],
                "k_below": cell["k_below"], "rule": cell["rule"], "n_seeds": cell["n_seeds"],
            })

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(boot, indent=2) + "\n")

    L = ["# Paper B R7b C3b: Fitzpatrick17k AUROC by skin type", "",
         "Design: `decisions/paperB_r7b_c3b.md`. Commit `{}`.".format(R.git_head()), "",
         "Kept {} / full {} images. ID = full ISIC test. Bootstrap: P13 `cluster_union`.".format(
             meta["n_kept"], meta["n_full"]), "",
         "## Group sizes (kept)", "",
         "| group | n images | n P13 clusters |",
         "|---|---:|---:|"]
    for gname, _ in GROUPS:
        st = meta["group_stats"][gname]
        L.append("| {} | {} | {} |".format(gname, st["n_images"], st["n_p13_clusters"]))
    L += ["", "## AUROC (Mahalanobis; Rule A as Table 1)", "",
          "| λ | setting | n_img | n_P13 | mean±sd | pooled CI | k_below | rule |",
          "|---:|---|---:|---:|---|---|---:|---|"]
    for _, r in df[df.detector == "mahalanobis_classcond"].iterrows():
        mean = "—" if pd.isna(r.auroc_mean) else "{:.3f} ± {:.3f}".format(r.auroc_mean, r.auroc_sd)
        ci = "—" if pd.isna(r.pooled_ci_lo) else "[{:.3f}, {:.3f}]".format(r.pooled_ci_lo, r.pooled_ci_hi)
        L.append("| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.lambda, r.setting, int(r.n_images), int(r.n_p13_clusters), mean, ci,
            "—" if pd.isna(r.k_below) else int(r.k_below), r.rule if isinstance(r.rule, str) else "—"))
    L += ["", "## AUROC (kNN)", "",
          "| λ | setting | mean±sd | pooled CI | k_below | rule |",
          "|---:|---|---|---|---:|---|"]
    for _, r in df[df.detector == "knn_k50"].iterrows():
        mean = "—" if pd.isna(r.auroc_mean) else "{:.3f} ± {:.3f}".format(r.auroc_mean, r.auroc_sd)
        ci = "—" if pd.isna(r.pooled_ci_lo) else "[{:.3f}, {:.3f}]".format(r.pooled_ci_lo, r.pooled_ci_hi)
        L.append("| {:g} | {} | {} | {} | {} | {} |".format(
            r.lambda, r.setting, mean, ci,
            "—" if pd.isna(r.k_below) else int(r.k_below), r.rule if isinstance(r.rule, str) else "—"))
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "rows", len(df))


if __name__ == "__main__":
    main()
