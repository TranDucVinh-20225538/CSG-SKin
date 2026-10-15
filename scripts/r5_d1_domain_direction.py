#!/usr/bin/env python3
"""Paper B R5 D1: domain-direction test (decisions/paperB_r5_d1.md). CPU.

Per OOD image: the λ = 0 model's linear domain score (ISIC train vs pad_adv; phase16: ID-site train vs adversary pool)
against the drop of its Mahalanobis distance to the nearest ID centroid from λ = 0 to λ = 2 (matched seeds), with the
size of the deviation orthogonal to the domain direction as the control. Patient / cluster / case / lesion bootstrap.
Usage: r5_d1_domain_direction.py [--extract_only | --report_only]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
from r5_p6_leakage_probe import pad_heldout_mask  # noqa: E402

OUT = R.R5 / "d1_domain"
S16 = R.R5 / "s_flip" / "features"
P7A = R.R5 / "p7a" / "sets"
SEEDS = (42, 52, 62, 72, 82)
LESION_SETS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
REPS = {"latent": "z_lesion_norm", "backbone": "backbone_raw_lesion"}
DET_REP = "z_lesion_norm"
P16 = {"BCN→HAM": ("b1", "bcn_ham", "ham_adv"), "HAM→BCN": ("b1_rev", "ham_bcn", "bcn_adv")}
SET_ORDER = LESION_SETS + tuple(P16)
UNIT = {"pad_heldout": "patient", "fitzpatrick17k": "P13 cluster (pHash ∪ ImageNet-R50 cosine)",
        "mednode": "image (no grouping exists)", "sd198": "P13-method cluster (pHash ∪ ImageNet-R50 cosine)",
        "scin": "case", "BCN→HAM": "lesion", "HAM→BCN": "lesion"}
SET_LABEL = {"pad_heldout": "pad_heldout", "fitzpatrick17k": "Fitzpatrick17k", "mednode": "MED-NODE",
             "sd198": "SD-198", "scin": "SCIN", "BCN→HAM": "BCN→HAM (HAM held-out)", "HAM→BCN": "HAM→BCN (BCN held-out)"}
R6R5_PAIR = {"pad_heldout": "ISIC→PAD (heldout)", "fitzpatrick17k": "ISIC→Fitzpatrick (mapped)",
             "mednode": "ISIC→MED-NODE (mapped)", "sd198": "ISIC→SD-198 (mapped)", "scin": "ISIC→SCIN (mapped)",
             "BCN→HAM": "BCN→HAM", "HAM→BCN": "HAM→BCN"}


# ----------------------------------------------------------------------------------------------------------------------
# Phase16 adversary-pool features (the only forward pass)
# ----------------------------------------------------------------------------------------------------------------------


def extract_adv_pools():
    import torch

    import r5_s_extract as SX

    torch.set_num_threads(8)
    for name, (sub, prefix, split) in P16.items():
        run = "{}_ladv0_s42".format(prefix)
        out = OUT / "features_adv" / run / "adv_pool.npz"
        if out.exists():
            print("skip_done", out, flush=True)
            continue
        summary = json.loads((SX.P16 / sub / run / "summary.json").read_text())
        _df, splits, _meta, _hyg = (SX.b1 if sub == "b1" else SX.b1r).load_or_create_splits()
        frame = splits[split]
        lit = SX.SingleDannLightning.load_from_checkpoint(summary["best_checkpoint"], strict=False, map_location="cpu")
        net = lit.net.eval()
        with torch.no_grad():
            p = SX.collect_all(net, SX.p2.make_loader(SX.SkinDataset(frame, transform=SX.build_eval_tf(448)), 32, 8, False),
                               torch.device("cpu"))
        assert len(p["labels"]) == len(frame)
        p["group"] = frame.lesion_id.astype(str).to_numpy()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **p)
        print("features", name, run, split, p["z"].shape, flush=True)


def sd198_clusters():
    """SD-198 near-duplicate clusters with the P13 method (r5_p13_fitz_clusters): DCT pHash Hamming ≤ 8 ∪ cosine ≥ 0.95
    on frozen ImageNet ResNet-50 features (r2_extract_imagenet_r50 recipe), connected components of the union."""
    out = OUT / "sd198_clusters.csv"
    if out.exists():
        return
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet50_Weights, resnet50

    import r5_p13_fitz_clusters as P13
    from eval_fitz_domain_axis import FolderJpegDataset
    from src.datasets.skin_dataset import build_val_transform_robust

    torch.set_num_threads(8)
    paths = pd.read_csv(P7A / "sd198.csv", keep_default_na=False).path.tolist()
    H = np.stack([P13.phash(p) for p in paths])
    rp, cp = P13.edges_hamming(H, P13.HAMMING)
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    zs = []
    with torch.no_grad():
        for batch in DataLoader(FolderJpegDataset(paths, build_val_transform_robust()), batch_size=64, num_workers=8):
            zs.append(net(batch[0]).float().numpy())
    z = np.concatenate(zs).astype(np.float64)
    np.savez_compressed(OUT / "sd198_imagenet_r50.npz", z=z.astype(np.float32))
    re_, ce = P13.edges_cosine(z, P13.COS)
    n = len(paths)
    df = pd.DataFrame({"path": paths, "phash": ["".join("{:02x}".format(b) for b in h) for h in H]})
    df["cluster_phash"] = P13.components(n, rp, cp)
    df["cluster_embed"] = P13.components(n, re_, ce)
    df["cluster_union"] = P13.components(n, np.r_[rp, re_], np.r_[cp, ce])
    df.to_csv(out, index=False)
    sz = df.cluster_union.value_counts()
    R.write_json(OUT / "sd198_cluster_stats.json", {
        "n_images": n, "n_edges_phash": int(len(rp)), "n_edges_embed": int(len(re_)), "n_clusters": int(len(sz)),
        "n_singletons": int((sz == 1).sum()), "largest": int(sz.max()), "effective_n": float(n ** 2 / (sz ** 2).sum())})
    print("sd198 clusters", len(sz), "of", n, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Per-image quantities
# ----------------------------------------------------------------------------------------------------------------------


def domain_fit(X0, X1):
    sc = StandardScaler().fit(np.r_[X0, X1])
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=0)
    clf.fit(sc.transform(np.r_[X0, X1]), np.r_[np.zeros(len(X0)), np.ones(len(X1))])
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    mu = sc.transform(X0).mean(0)

    def score(X):
        u = sc.transform(X)
        dev = u - mu
        orth = dev - np.outer(dev @ w, w)
        return clf.decision_function(u), np.linalg.norm(orth, axis=1)

    return clf, sc, score


def valid_bal_acc(clf, sc, Xid, Xood):
    y = np.r_[np.zeros(len(Xid)), np.ones(len(Xood))]
    return float(balanced_accuracy_score(y, clf.predict(sc.transform(np.r_[Xid, Xood]))))


def pulls(sid0, sood0, sid2, sood2):
    d = [np.sqrt(np.maximum(x, 0.0)) for x in (sid0, sood0, sid2, sood2)]
    med0, med2 = np.median(d[0]), np.median(d[2])
    P0 = np.searchsorted(np.sort(d[0]), d[1], side="right") / len(d[0])
    P2 = np.searchsorted(np.sort(d[2]), d[3], side="right") / len(d[2])
    return {"d0": d[1], "d2": d[3], "P0": P0, "P2": P2, "pull": d[1] / med0 - d[3] / med2, "pull_pct": P0 - P2,
            "pull_raw": d[1] - d[3], "auroc0": R.auroc(sid0, sood0), "auroc2": R.auroc(sid2, sood2)}


def set_groups(name, n):
    if name == "pad_heldout":
        g = R.pad_heldout_patients()
    elif name == "fitzpatrick17k":
        g = R.fitz_groups("cluster_union")
    elif name == "sd198":
        c = pd.read_csv(OUT / "sd198_clusters.csv", keep_default_na=False)
        assert c.path.tolist() == pd.read_csv(P7A / "sd198.csv", keep_default_na=False).path.tolist()
        g = c.cluster_union.to_numpy()
    else:
        g = pd.read_csv(P7A / "{}.csv".format(name), keep_default_na=False).group.astype(str).to_numpy()
    assert len(g) == n, (name, len(g), n)
    return np.asarray(g).astype(str)


def lesion_frames_per_seed(seed, hmask):
    t0, t2 = R.run_tag(0.0, seed), R.run_tag(2.0, seed)
    A = {s: R.load(t0, s) for s in ("isic_train", "isic_test", "pad_full") + LESION_SETS}
    Bq = {s: R.load(t2, s) for s in ("isic_train", "isic_test") + LESION_SETS}
    rows, meta = {}, {"seed": seed, "lambda0": t0, "lambda2": t2}
    ytr0, ytr2 = A["isic_train"]["labels"].astype(int), Bq["isic_train"]["labels"].astype(int)
    for rep_name, rep in REPS.items():
        X0, X1 = A["isic_train"][rep], A["pad_full"][rep][~hmask]
        clf, sc, score = domain_fit(X0, X1)
        meta["valid_bal_acc_" + rep_name] = valid_bal_acc(clf, sc, A["isic_test"][rep], A["pad_heldout"][rep])
        for s in LESION_SETS:
            dom, orth = score(A[s][rep])
            rows.setdefault(s, {})["dom_" + rep_name] = dom
            rows[s]["orth_" + rep_name] = orth
    for s in LESION_SETS:
        sid0, sood0 = R.B.maha_scores(A["isic_train"][DET_REP], ytr0, A["isic_test"][DET_REP], A[s][DET_REP])
        sid2, sood2 = R.B.maha_scores(Bq["isic_train"][DET_REP], ytr2, Bq["isic_test"][DET_REP], Bq[s][DET_REP])
        rows[s].update(pulls(sid0, sood0, sid2, sood2))
    return rows, meta


def phase16_frames(name):
    sub, prefix, _split = P16[name]
    r0, r2 = "{}_ladv0_s42".format(prefix), "{}_ladv2_s42".format(prefix)
    L0 = {k: dict(np.load(S16 / r0 / "{}.npz".format(k))) for k in ("id_train", "id_test", "ood")}
    L2 = {k: dict(np.load(S16 / r2 / "{}.npz".format(k))) for k in ("id_train", "id_test", "ood")}
    adv = dict(np.load(OUT / "features_adv" / r0 / "adv_pool.npz"))
    assert np.array_equal(L0["ood"]["group"], L2["ood"]["group"]), "ood row order λ0 vs λ2"
    clf, sc, score = domain_fit(L0["id_train"]["z"], adv["z"])
    meta = {"seed": 42, "lambda0": r0, "lambda2": r2, "n_adv_pool": int(len(adv["z"])),
            "valid_bal_acc_latent": valid_bal_acc(clf, sc, L0["id_test"]["z"], L0["ood"]["z"])}
    dom, orth = score(L0["ood"]["z"])
    row = {"dom_latent": dom, "orth_latent": orth}
    sid0, sood0 = R.B.maha_scores(L0["id_train"]["z"], L0["id_train"]["labels"].astype(int), L0["id_test"]["z"],
                                  L0["ood"]["z"])
    sid2, sood2 = R.B.maha_scores(L2["id_train"]["z"], L2["id_train"]["labels"].astype(int), L2["id_test"]["z"],
                                  L2["ood"]["z"])
    row.update(pulls(sid0, sood0, sid2, sood2))
    return row, meta, np.asarray(L0["ood"]["group"]).astype(str)


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------


def _r(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def _partial(rx, ry, rz):
    xy, xz, yz = _r(rx, ry), _r(rx, rz), _r(ry, rz)
    return (xy - xz * yz) / np.sqrt(max((1 - xz ** 2) * (1 - yz ** 2), 1e-12))


def stats_one(row, idx, reps):
    """Spearman statistics on the rows idx: {key: value} for one seed."""
    rk = {k: rankdata(row[k][idx]) for k in ("pull", "P0")}
    out = {}
    for rep in reps:
        rd, ro = rankdata(row["dom_" + rep][idx]), rankdata(row["orth_" + rep][idx])
        out["rho_dom_" + rep] = _r(rd, rk["pull"])
        out["rho_orth_" + rep] = _r(ro, rk["pull"])
        out["delta_" + rep] = out["rho_dom_" + rep] - abs(out["rho_orth_" + rep])
        out["prho_dom_" + rep] = _partial(rd, rk["pull"], rk["P0"])
        out["prho_orth_" + rep] = _partial(ro, rk["pull"], rk["P0"])
    return out


def analyse_set(per_seed_rows, groups, reps, rng, n_boot):
    seeds = sorted(per_seed_rows)
    n = len(groups)
    full = np.arange(n)
    pts = {s: stats_one(per_seed_rows[s], full, reps) for s in seeds}
    keys = list(next(iter(pts.values())))
    point = {k: float(np.mean([pts[s][k] for s in seeds])) for k in keys}
    draws = R.draws(groups, n_boot, rng)
    mat = np.array([[np.mean([stats_one(per_seed_rows[s], ii, reps)[k] for s in seeds]) for k in keys] for ii in draws])
    lo, hi = np.nanpercentile(mat, [2.5, 97.5], axis=0)
    ci = {k: [float(lo[j]), float(hi[j])] for j, k in enumerate(keys)}
    sec = {}
    for rep in reps:
        for pk in ("pull_pct", "pull_raw"):
            for kind in ("dom", "orth"):
                sec["rho_{}_{}_{}".format(kind, rep, pk)] = float(np.mean(
                    [_r(rankdata(per_seed_rows[s]["{}_{}".format(kind, rep)]), rankdata(per_seed_rows[s][pk]))
                     for s in seeds]))
        sec["mean_dom_" + rep] = float(np.mean([per_seed_rows[s]["dom_" + rep].mean() for s in seeds]))
    a0 = [per_seed_rows[s]["auroc0"] for s in seeds]
    a2 = [per_seed_rows[s]["auroc2"] for s in seeds]
    crit = {rep: bool(point["rho_dom_" + rep] > 0 and ci["delta_" + rep][0] > 0) for rep in reps}
    return {"n": int(n), "n_groups": int(len(np.unique(groups))), "n_seeds": len(seeds), "point": point, "ci": ci,
            "per_seed": {str(s): pts[s] for s in seeds}, "secondary": sec, "auroc0": float(np.mean(a0)),
            "auroc2": float(np.mean(a2)), "delta_auroc": float(np.mean(a2) - np.mean(a0)), "criterion": crit}


# ----------------------------------------------------------------------------------------------------------------------
# Main / report
# ----------------------------------------------------------------------------------------------------------------------


def analyse():
    pre = R.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    hmask = pad_heldout_mask()
    per_set = {s: {} for s in LESION_SETS}
    metas = []
    for sd in SEEDS:
        rows, meta = lesion_frames_per_seed(sd, hmask)
        metas.append(meta)
        for s in LESION_SETS:
            per_set[s][sd] = rows[s]
            pd.DataFrame(rows[s]).drop(columns=["auroc0", "auroc2"]).to_csv(
                OUT / "per_image" / "{}_s{}.csv.gz".format(s, sd), index=False)
        print("seed", sd, meta, flush=True)
    res = {"meta_lesion": metas, "sets": {}}
    for s in LESION_SETS:
        g = set_groups(s, len(per_set[s][SEEDS[0]]["pull"]))
        res["sets"][s] = analyse_set(per_set[s], g, ("latent", "backbone"), np.random.default_rng(seed), n_boot)
        print(s, res["sets"][s]["point"], flush=True)
    res["meta_phase16"] = {}
    for name in P16:
        row, meta, g = phase16_frames(name)
        res["meta_phase16"][name] = meta
        pd.DataFrame(row).drop(columns=["auroc0", "auroc2"]).to_csv(OUT / "per_image" / "{}_s42.csv.gz".format(
            P16[name][1]), index=False)
        res["sets"][name] = analyse_set({42: row}, g, ("latent",), np.random.default_rng(seed), n_boot)
        print(name, meta, res["sets"][name]["point"], flush=True)
    res.update({"commit": R.git_head(), "n_boot": n_boot, "rng_seed": seed})
    R.write_json(OUT / "D1_RESULTS.json", res)
    return res


def fci(p, c, k):
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(p[k], *c[k])


def report(res):
    mix = pd.read_csv(R.R6 / "r5" / "summary.csv").set_index("pair")
    rows, L = [], ["# Paper B R5 D1: domain direction vs per-image pull", "",
                   "Decisions: `decisions/paperB_r5_d1.md`. Commit {}. Bootstrap {} draws, rng seed {}; CI = percentile 95% "
                   "of the seed mean (lesion-level: 5 matched seeds; BCN↔HAM: seed 42 only).".format(
                       res["commit"], res["n_boot"], res["rng_seed"]), ""]
    va = pd.DataFrame(res["meta_lesion"])
    L += ["Domain classifier validity (balanced accuracy, ISIC test vs pad_heldout, λ = 0): latent {}, backbone {}; "
          "BCN↔HAM site classifiers (ID test vs OOD held-out): {}.".format(
              R.fmt(*R.mean_sd(va.valid_bal_acc_latent)), R.fmt(*R.mean_sd(va.valid_bal_acc_backbone)),
              ", ".join("{} {:.3f} (adversary pool n = {})".format(k, m["valid_bal_acc_latent"], m["n_adv_pool"])
                        for k, m in res["meta_phase16"].items())), "",
          "## Per-image test (primary pull = d₀/med₀ − d₂/med₂; Spearman, mean over seeds)", "",
          "Bootstrap unit per set is in the table; within a draw the same resample is applied to all 5 seed pairs "
          "(lesion-level) and the statistic is the seed mean (pooled over the 5 pairs).", "",
          "| set | unit | n (groups) | rep | ρ_dom [CI] | ρ_orth [CI] | Δ = ρ_dom − abs(ρ_orth) [CI] | criterion | partial ρ_dom given P₀ "
          "[CI] | partial ρ_orth [CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SET_ORDER:
        r = res["sets"][s]
        for rep in ("latent", "backbone"):
            if "rho_dom_" + rep not in r["point"]:
                continue
            p, c = r["point"], r["ci"]
            L.append("| {} | {} | {} ({}) | {} | {} | {} | {} | {} | {} | {} |".format(
                SET_LABEL[s], UNIT[s], r["n"], r["n_groups"], rep, fci(p, c, "rho_dom_" + rep), fci(p, c, "rho_orth_" + rep),
                fci(p, c, "delta_" + rep), "met" if r["criterion"][rep] else "not met", fci(p, c, "prho_dom_" + rep),
                fci(p, c, "prho_orth_" + rep)))
            rows.append({"set": s, "rep": rep, "n": r["n"], "n_groups": r["n_groups"], "n_seeds": r["n_seeds"],
                         **{k: p[k] for k in p if k.endswith(rep)},
                         **{k + "_lo": c[k][0] for k in c if k.endswith(rep)},
                         **{k + "_hi": c[k][1] for k in c if k.endswith(rep)},
                         **{k: v for k, v in r["secondary"].items() if rep in k},
                         "criterion": r["criterion"][rep], "auroc0": r["auroc0"], "auroc2": r["auroc2"],
                         "delta_auroc": r["delta_auroc"]})
    k_lat = sum(res["sets"][s]["criterion"]["latent"] for s in SET_ORDER)
    k_bb = sum(res["sets"][s]["criterion"].get("backbone", False) for s in LESION_SETS)
    verdict = "supported at the per-image level" if k_lat >= 4 else "not supported"
    neg = ["{} {}".format(SET_LABEL[s], rep) for s in SET_ORDER for rep in ("latent", "backbone")
           if "prho_orth_" + rep in res["sets"][s]["point"] and res["sets"][s]["point"]["prho_orth_" + rep] < 0]
    L += ["", "Partial ρ (controlling for the initial distance percentile P₀) is secondary and has no pass/fail rule. "
          "Negative partial ρ_orth ({}): reported as is; any reading of the sign was formed after the seed-42 smoke test "
          "and is labelled as such.".format(", ".join(neg) or "none")]
    L += ["", "Criterion (ρ_dom > 0 and Δ CI entirely above 0) met in **{} of 7** sets (latent) → hypothesis {} (rule fixed "
          "in decisions: ≥ 4 of 7). Backbone: {} of 5 lesion-level sets.".format(k_lat, verdict, k_bb), "",
          "## Secondary pulls (points, mean over seeds)", "",
          "| set | rep | ρ_dom pct | ρ_orth pct | ρ_dom raw | ρ_orth raw |", "|---|---|---:|---:|---:|---:|"]
    for s in SET_ORDER:
        sec = res["sets"][s]["secondary"]
        for rep in ("latent", "backbone"):
            if "mean_dom_" + rep in sec:
                L.append("| {} | {} | {:+.3f} | {:+.3f} | {:+.3f} | {:+.3f} |".format(
                    SET_LABEL[s], rep, *(sec["rho_{}_{}_{}".format(k, rep, pk)] for pk in ("pull_pct", "pull_raw")
                                         for k in ("dom", "orth"))))
    L += ["", "## Across sets (descriptive, no test) with class mix (R6-R5)", "",
          "| set | mean domain score latent | mean domain score backbone | Maha AUROC λ0 | λ2 | ΔAUROC | ρ_dom latent "
          "| TV (class mix) | majority ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    acr = []
    for s in SET_ORDER:
        r = res["sets"][s]
        sec = r["secondary"]
        tv = mix.loc[R6R5_PAIR[s], "tv"] if R6R5_PAIR[s] in mix.index else np.nan
        mr = mix.loc[R6R5_PAIR[s], "maj_ratio"] if R6R5_PAIR[s] in mix.index else np.nan
        acr.append({"set": s, "dom_lat": sec["mean_dom_latent"], "dom_bb": sec.get("mean_dom_backbone", np.nan),
                    "dauroc": r["delta_auroc"], "rho": r["point"]["rho_dom_latent"], "tv": tv})
        L.append("| {} | {:+.2f} | {} | {:.3f} | {:.3f} | {:+.3f} | {:+.3f} | {:.3f} | {:.2f} |".format(
            SET_LABEL[s], sec["mean_dom_latent"], "{:+.2f}".format(sec["mean_dom_backbone"]) if "mean_dom_backbone" in sec
            else "—", r["auroc0"], r["auroc2"], r["delta_auroc"], r["point"]["rho_dom_latent"], tv, mr))
    A = pd.DataFrame(acr)

    def sp(a, b):
        return _r(rankdata(a), rankdata(b))

    les = A[A.set.isin(LESION_SETS)]
    L += ["", "Spearman over sets (descriptive): mean latent domain score vs −ΔAUROC: {:+.2f} (5 lesion-level sets), "
          "{:+.2f} (all 7; BCN↔HAM scores come from a different classifier); mean backbone domain score vs −ΔAUROC "
          "(5 lesion sets): {:+.2f}; TV vs −ΔAUROC (7 sets): {:+.2f}; ρ_dom vs −ΔAUROC (7 sets): {:+.2f}.".format(
              sp(les.dom_lat, -les.dauroc), sp(A.dom_lat, -A.dauroc), sp(les.dom_bb, -les.dauroc),
              sp(A.tv, -A.dauroc), sp(A.rho, -A.dauroc)), "",
          "## LaTeX rows (set & ρ_dom & ρ_orth & Δ [CI] & ΔAUROC & TV)", "", "```latex"]
    for s in SET_ORDER:
        r = res["sets"][s]
        p, c = r["point"], r["ci"]
        L.append("{} & {:+.2f} & {:+.2f} & {:+.2f} [{:+.2f}, {:+.2f}] & {:+.3f} & {:.2f} \\\\".format(
            SET_LABEL[s].replace("→", "$\\to$"), p["rho_dom_latent"], p["rho_orth_latent"], p["delta_latent"],
            *c["delta_latent"], r["delta_auroc"], A.set_index("set").tv[s]))
    L += ["```", ""]
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_only", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    (OUT / "per_image").mkdir(parents=True, exist_ok=True)
    if a.report_only:
        report(json.loads((OUT / "D1_RESULTS.json").read_text()))
        return
    extract_adv_pools()
    sd198_clusters()
    if a.extract_only:
        return
    report(analyse())


if __name__ == "__main__":
    main()
