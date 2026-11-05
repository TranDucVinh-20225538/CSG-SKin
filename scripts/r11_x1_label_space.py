#!/usr/bin/env python3
"""R11 X1: OOD split by label space on Fitzpatrick17k and SD-198 (cached Table 1 last.ckpt features)."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_ckpt_audit as A  # noqa: E402
import r6_common as R  # noqa: E402

OUT = R.RES / "r11" / "x1"
FEAT = R.RES / "r5" / "t1last" / "features_last"
DET = "mahalanobis_classcond"
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_BOOT = 2000
RNG = 20261005

# Fitz in-ISIC / ambiguous (lowercase condition names from phase11)
IN_FITZ = {
    "melanoma", "malignant melanoma", "superficial spreading melanoma ssm", "lentigo maligna",
    "nevocytic nevus", "congenital nevus", "halo nevus",
    "basal cell carcinoma", "basal cell carcinoma morpheiform", "solid cystic basal cell carcinoma",
    "squamous cell carcinoma", "actinic keratosis", "seborrheic keratosis", "dermatofibroma",
    "pyogenic granuloma",
}
AMB_FITZ = {
    "epidermal nevus", "naevus comedonicus", "becker nevus", "nevus sebaceous of jadassohn",
    "kaposi sarcoma",
}

# SD-198 mapping locked in decisions/paperB_r11_x1.md
SD_IN = {
    "Actinic_solar_Damage(Actinic_Keratosis)", "Basal_Cell_Carcinoma", "Congenital_Nevus",
    "Dermatofibroma", "Halo_Nevus", "Lentigo_Maligna_Melanoma", "Malignant_Melanoma",
    "Pyogenic_Granuloma", "Seborrheic_Keratosis",
}
SD_AMB = {
    "Becker's_Nevus", "Epidermal_Nevus", "Linear_Epidermal_Nevus", "Nevus_Comedonicus",
    "Nevus_Sebaceous_of_Jadassohn",
}


def fitz_buckets():
    frame = pd.read_csv(R.RES / "r5" / "frozen" / "frames" / "fitzpatrick17k.csv")
    src = pd.read_csv(R.RES / "phase11_third_domain" / "fitzpatrick17k.csv")
    frame["md5"] = frame.path.map(lambda p: Path(p).stem)
    m = frame.merge(src[["md5hash", "label"]], left_on="md5", right_on="md5hash", how="left")
    # feature order = sorted jpg paths (FolderJpegDataset)
    paths = sorted(R.p3.FITZ_DIR.glob("*.jpg"))
    assert len(paths) == len(frame)
    by_md5 = m.set_index("md5")
    labels = [str(by_md5.loc[p.stem, "label"]).strip().lower() for p in paths]
    buckets = []
    for lab in labels:
        if lab in IN_FITZ:
            buckets.append("in-ISIC")
        elif lab in AMB_FITZ:
            buckets.append("ambiguous")
        else:
            buckets.append("out-of-ISIC")
    cl = pd.read_csv(R.RES / "r5" / "p13" / "fitz_clusters.csv").set_index("md5hash")["cluster_union"]
    clusters = [str(cl[p.stem]) for p in paths]
    return np.array(buckets), np.array(clusters), {
        "in-ISIC": int(sum(b == "in-ISIC" for b in buckets)),
        "out-of-ISIC": int(sum(b == "out-of-ISIC" for b in buckets)),
        "ambiguous": int(sum(b == "ambiguous" for b in buckets)),
        "total": len(buckets),
    }


def sd_buckets():
    meta = pd.read_csv(R.P7A_SETS / "sd198.csv", keep_default_na=False)
    # feature order must match extract order = meta.path order in p7a
    buckets = []
    for lab in meta.source_label.astype(str):
        if lab in SD_IN:
            buckets.append("in-ISIC")
        elif lab in SD_AMB:
            buckets.append("ambiguous")
        else:
            buckets.append("out-of-ISIC")
    SD_CL = R.RES / "r5" / "p13" / "sd198_clusters.csv"
    if not SD_CL.exists():
        SD_CL = R.RES / "r5" / "d1_domain" / "sd198_clusters.csv"
    cl = pd.read_csv(SD_CL, keep_default_na=False)
    assert cl.path.tolist() == meta.path.tolist()
    return np.array(buckets), cl.cluster_union.astype(str).to_numpy(), {
        "in-ISIC": int(sum(b == "in-ISIC" for b in buckets)),
        "out-of-ISIC": int(sum(b == "out-of-ISIC" for b in buckets)),
        "ambiguous": int(sum(b == "ambiguous" for b in buckets)),
        "total": len(buckets),
    }


def idx_groups_from(labels):
    return R.idx_groups(np.asarray(labels).astype(str))


def bootstrap_subset(feat_dirs, ood_name, id_groups, ood_idx, ood_group_labels, exclude_seeds=None):
    """Bootstrap Mahalanobis on a subset of OOD indices; unit = groups among those indices."""
    exclude_seeds = set(exclude_seeds or ())
    # restrict groups to subset
    sub_groups = ood_group_labels[ood_idx]
    # remap to local indices 0..len(ood_idx)-1
    local = np.arange(len(ood_idx))
    # group membership in local coords
    gmap = {}
    for i, g in enumerate(sub_groups):
        gmap.setdefault(str(g), []).append(local[i])
    o_groups = [np.array(v, dtype=int) for v in gmap.values()]
    pre = json.loads(R.PREREG.read_text())
    n_boot = int(pre["n_resamples"])
    rng = np.random.default_rng(int(pre["rng_seed"]))
    sc = {}
    for s, d in feat_dirs.items():
        if s in exclude_seeds:
            continue
        Fs = R.load_feats(d, ("isic_train", "isic_test", ood_name))
        S = R.Scorer(Fs["isic_train"])
        sid = S.scores(Fs["isic_test"])[DET]
        so_full = S.scores(Fs[ood_name])[DET]
        so = so_full[ood_idx]
        sc[s] = (sid, so)
    if len(sc) < 1:
        return None
    id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))])
                for _ in range(n_boot)]
    oo_draws = [np.concatenate([o_groups[j] for j in rng.integers(0, len(o_groups), len(o_groups))])
                for _ in range(n_boot)]
    seeds = sorted(sc)
    per, draws = {}, {}
    for s in seeds:
        sid, so = sc[s]
        draws[s] = np.array([R.auroc(sid[id_draws[b]], so[oo_draws[b]]) for b in range(n_boot)])
        lo, hi = float(np.quantile(draws[s], 0.025)), float(np.quantile(draws[s], 0.975))
        per[str(s)] = {"point": float(R.auroc(sid, so)), "ci_lo": lo, "ci_hi": hi,
                       "below_chance_entire_ci": hi < 0.5}
    pooled = np.mean(np.stack([draws[s] for s in seeds]), 0)
    pts = [per[str(s)]["point"] for s in seeds]
    block = {
        "mean": float(np.mean(pts)),
        "sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0,
        "ci_lo": float(np.quantile(pooled, 0.025)),
        "ci_hi": float(np.quantile(pooled, 0.975)),
        "k_below": int(sum(p["below_chance_entire_ci"] for p in per.values())),
        "n_seeds": len(seeds), "seeds": seeds, "per_seed": per,
    }
    block["rule_A"] = bool(block["ci_hi"] < 0.5 and block["k_below"] >= 3)
    return block


def paired_diff_ci(per_in, per_out, n_boot=N_BOOT, seed=RNG):
    """Seed-paired (in − out) with seed-bootstrap CI."""
    seeds = sorted(set(per_in) & set(per_out), key=int)
    deltas = np.array([per_in[s]["point"] - per_out[s]["point"] for s in seeds], float)
    rng = np.random.default_rng(seed)
    boots = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(n_boot)])
    return {
        "seeds": [int(s) for s in seeds],
        "per_seed_delta": {s: float(deltas[i]) for i, s in enumerate(seeds)},
        "mean": float(deltas.mean()),
        "sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
    }


def runs_by_lam():
    out = {lam: {} for lam in LAMS}
    for tag in A.table1_runs():
        lam = float(A.lam_of(tag))
        s = int(tag.rsplit("_s", 1)[1])
        d = FEAT / tag
        if not (d / "isic_train.npz").exists():
            continue
        # degraded check
        sp = R.RES / "r5" / "t1last" / "per_run" / f"{tag}.json"
        if sp.exists():
            bal = json.loads(sp.read_text())["last"]["id_bal_acc"]
            if bal < R.DEGRADED_ID_BAL:
                print("degraded exclude", tag, bal, flush=True)
                continue
        out[lam][s] = d
    return out


def case_for(in_b, out_b, paired):
    """Reading rule at λ=2."""
    ra_in = bool(in_b["rule_A"])
    ra_out = bool(out_b["rule_A"])
    diff_below = paired["ci_hi"] < 0
    if ra_in and (not ra_out) and diff_below:
        return "Label space carries it"
    if ra_in and ra_out:
        return "Not label space"
    return "Anything else (reported as found)"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = R.load_data()
    idg = R.idx_groups(R.T.lesion_groups(data["test"])[0])
    fb, fcl, fcounts = fitz_buckets()
    sb, scl, scounts = sd_buckets()
    print("fitz counts", fcounts, flush=True)
    print("sd counts", scounts, flush=True)
    by_lam = runs_by_lam()
    results = {"fitzpatrick17k": {"counts": fcounts, "by_lambda": {}},
               "sd198": {"counts": scounts, "by_lambda": {}}}
    latex = []
    for ood, buckets, groups, name in (
        ("fitzpatrick17k", fb, fcl, "fitzpatrick17k"),
        ("sd198", sb, scl, "sd198"),
    ):
        for lam in LAMS:
            feat_dirs = by_lam[lam]
            if len(feat_dirs) < 3:
                print("skip", ood, lam, "n", len(feat_dirs), flush=True)
                continue
            cell = {"lambda": lam, "n_seeds": len(feat_dirs), "subsets": {}}
            per_pts = {}
            for subset in ("in-ISIC", "out-of-ISIC", "ambiguous"):
                idx = np.where(buckets == subset)[0]
                if len(idx) < 5:
                    cell["subsets"][subset] = {"n": int(len(idx)), "note": "too few"}
                    continue
                # need ood npz
                ok = {s: d for s, d in feat_dirs.items() if (d / f"{ood}.npz").exists()}
                b = bootstrap_subset(ok, ood, idg, idx, groups)
                cell["subsets"][subset] = {"n": int(len(idx)), **b}
                per_pts[subset] = b["per_seed"]
                print(f"{ood} λ={lam:g} {subset}: {b['mean']:.3f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}] "
                      f"k={b['k_below']}/{b['n_seeds']} RuleA={b['rule_A']}", flush=True)
            if "in-ISIC" in per_pts and "out-of-ISIC" in per_pts:
                cell["paired_in_minus_out"] = paired_diff_ci(per_pts["in-ISIC"], per_pts["out-of-ISIC"])
                print(f"  paired Δ mean={cell['paired_in_minus_out']['mean']:.3f} "
                      f"CI=[{cell['paired_in_minus_out']['ci_lo']:.3f},"
                      f"{cell['paired_in_minus_out']['ci_hi']:.3f}]", flush=True)
            results[name]["by_lambda"][str(lam)] = cell

    # cases at λ=2
    cases = {}
    for name in ("fitzpatrick17k", "sd198"):
        cell = results[name]["by_lambda"].get("2.0") or results[name]["by_lambda"].get("2")
        if not cell:
            cases[name] = "missing λ=2"
            continue
        inn = cell["subsets"].get("in-ISIC")
        out = cell["subsets"].get("out-of-ISIC")
        paired = cell.get("paired_in_minus_out")
        cases[name] = case_for(inn, out, paired)
    results["cases_lambda2"] = cases
    results["commit"] = R.C.git_head()
    results["cpu_hours"] = (time.time() - t0) / 3600.0
    (OUT / "X1.json").write_text(json.dumps(results, indent=1) + "\n")
    # markdown
    md = ["# R11 X1 — OOD split by label space", "",
          f"Commit `{results['commit']}`. CPU h {results['cpu_hours']:.3f}.", "",
          "## Counts", "",
          f"- Fitzpatrick17k: {fcounts}", "",
          f"- SD-198: {scounts}", "",
          "## Cases (λ = 2 decides)", "",
          f"- Fitzpatrick17k: **{cases.get('fitzpatrick17k')}**", "",
          f"- SD-198: **{cases.get('sd198')}**", "",
          "## Tables", ""]
    for name in ("fitzpatrick17k", "sd198"):
        md += [f"### {name}", "",
               "| λ | subset | mean±sd | pooled CI | k | Rule A |",
               "|---:|---|---:|---|---:|---|"]
        for lam_s, cell in sorted(results[name]["by_lambda"].items(), key=lambda kv: float(kv[0])):
            for subset, b in cell["subsets"].items():
                if "mean" not in b:
                    md.append(f"| {float(lam_s):g} | {subset} | — | — | — | {b.get('note','')} |")
                    continue
                md.append(
                    f"| {float(lam_s):g} | {subset} | {b['mean']:.3f}±{b['sd']:.3f} | "
                    f"[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}] | {b['k_below']}/{b['n_seeds']} | "
                    f"{'met' if b['rule_A'] else 'not met'} |"
                )
            if "paired_in_minus_out" in cell:
                p = cell["paired_in_minus_out"]
                md.append(
                    f"| {float(lam_s):g} | paired in−out | {p['mean']:.3f}±{p['sd']:.3f} | "
                    f"[{p['ci_lo']:.3f}, {p['ci_hi']:.3f}] | — | — |"
                )
        md.append("")
    (OUT / "X1.md").write_text("\n".join(md))
    print("cases", cases, "cpu_h", results["cpu_hours"], flush=True)


if __name__ == "__main__":
    main()
