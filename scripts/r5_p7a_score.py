#!/usr/bin/env python3
"""Paper B R5 P7(a): Table 1 detectors on the unseen clinical sets, all 35 lesion-level runs (decisions/paperB_r5_p7a.md).

Usage: r5_p7a_score.py [set ...]   (default: every set whose 35 feature files exist), then writes the report from
whatever per-set JSON exists.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
import bootstrap_ood_ci as B  # noqa: E402
import r2_common as C  # noqa: E402
import r3_a2_bootstrap_lesion as A2  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402
from src.datasets.constants import LABEL_TO_INDEX  # noqa: E402
from src.utils import ood_metrics  # noqa: E402

FEAT = C.R2 / "item4" / "features"
OUT = C.PAPERB / "results" / "paperB" / "r5" / "p7a"
SETS = {"mednode": "MED-NODE", "sd198": "SD-198", "scin": "SCIN"}
UNIT = {"mednode": "image", "sd198": "image (no patient IDs; anti-conservative)", "scin": "case"}
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BOOT_DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")
DETS = BOOT_DETS + ("MSP", "Energy_T1")
DET_LABEL = {"mahalanobis_classcond": "Mahalanobis", "knn_k50": "kNN k=50", "cosine_max": "Cosine", "MSP": "MSP",
             "Energy_T1": "Energy"}
CLASSES = sorted(LABEL_TO_INDEX, key=LABEL_TO_INDEX.get)
NV = LABEL_TO_INDEX["NV"]


def seeds(lam):
    return sorted(int(d.name.rsplit("_s", 1)[1]) for d in FEAT.glob("runB_orth1_ladv{}_s*".format(C.lam_tag(lam))))


def run_dir(lam, s):
    return FEAT / "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), s)


def scores(tr, te, oo):
    """Higher = more OOD for every detector (Table 1 / R4 detector_block convention)."""
    z_tr, z_te, z_oo = tr["z_lesion_norm"], te["z_lesion_norm"], oo["z_lesion_norm"]
    out = {"mahalanobis_classcond": B.maha_scores(z_tr, tr["labels"], z_te, z_oo),
           "cosine_max": B.cosine_scores(z_tr, tr["labels"], z_te, z_oo),
           "knn_k50": B.knn50_scores(z_tr, z_te, z_oo)}
    p_te = torch.softmax(torch.from_numpy(te["logits"]).float(), 1).numpy().max(1)
    p_oo = torch.softmax(torch.from_numpy(oo["logits"]).float(), 1).numpy().max(1)
    e_te = torch.logsumexp(torch.from_numpy(te["logits"]).float(), 1).numpy()
    e_oo = torch.logsumexp(torch.from_numpy(oo["logits"]).float(), 1).numpy()
    out["MSP"] = (-p_te.astype(np.float64), -p_oo.astype(np.float64))
    out["Energy_T1"] = (-e_te.astype(np.float64), -e_oo.astype(np.float64))
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(z_tr, tr["labels"], 8, 1e-3)
    diff = z_oo[:, None, :].astype(np.float64) - means[None].astype(np.float64)
    d2 = np.einsum("nkd,de,nke->nk", diff, prec.astype(np.float64), diff)
    return out, d2.argmin(1)


def auroc_w(sid, sood, w_ood=None):
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    w = None if w_ood is None else np.r_[np.ones(len(sid)), w_ood]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=w))


def score_set(name):
    pre = json.loads(B.PREREG.read_text())
    n_boot, rng = int(pre["n_resamples"]), np.random.default_rng(int(pre["rng_seed"]))
    meta = pd.read_csv(OUT / "sets" / "{}.csv".format(name), keep_default_na=False)
    _rec, (_tr, _va, test) = T.split_record()
    lesion_g, _ = T.lesion_groups(test)
    id_groups = A2.idx_groups(lesion_g)
    ood_groups = A2.idx_groups(meta.group.astype(str).to_numpy())
    id_mix = test.label.value_counts(normalize=True)
    mapped = meta.mapped.to_numpy()
    res = {"set": name, "n_images": int(len(meta)), "n_groups": len(ood_groups), "resample_unit": UNIT[name],
           "mapped_counts": meta[meta.mapped != ""].mapped.value_counts().to_dict(), "lambda": {}}
    # reweighting to the ISIC test class mix over the mapped OOD classes
    mk = [c for c in res["mapped_counts"] if res["mapped_counts"][c] >= 20]
    sub = np.isin(mapped, mk)
    w_rw = np.zeros(len(meta))
    if len(mk) >= 2:
        p_id = id_mix[mk] / id_mix[mk].sum()
        p_oo = pd.Series(mapped[sub]).value_counts(normalize=True)[mk]
        for c in mk:
            w_rw[mapped == c] = p_id[c] / p_oo[c]
    for lam in LAMS:
        ss = seeds(lam)
        sc, nearest = {}, {}
        for s in ss:
            d = run_dir(lam, s)
            tr, te, oo = (B.load_npz(d, k) for k in ("isic_train", "isic_test", name))
            assert np.array_equal(te["labels"].astype(int), test.label_idx.to_numpy().astype(int)), d
            assert len(oo["z_lesion_norm"]) == len(meta), (d, name)
            sc[s], nearest[s] = scores(tr, te, oo)
        id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))])
                    for _ in range(n_boot)]
        oo_draws = [np.concatenate([ood_groups[j] for j in rng.integers(0, len(ood_groups), len(ood_groups))])
                    for _ in range(n_boot)]
        blk = {}
        for det in DETS:
            per = {}
            draws = {}
            for s in ss:
                sid, so = sc[s][det]
                per[str(s)] = {"point": auroc_w(sid, so)}
                if det in BOOT_DETS:
                    draws[s] = np.array([B.auroc_from_scores(sid[id_draws[b]], so[oo_draws[b]]) for b in range(n_boot)])
                    lo, hi = float(np.quantile(draws[s], 0.025)), float(np.quantile(draws[s], 0.975))
                    per[str(s)].update(ci_lo=lo, ci_hi=hi, below_chance_entire_ci=hi < 0.5)
            pts = [per[str(s)]["point"] for s in ss]
            b = {"mean": float(np.mean(pts)), "sd": float(np.std(pts, ddof=1)), "per_seed": per, "seeds": ss}
            if det in BOOT_DETS:
                lo, hi, _mu = B.percentile_ci(np.mean(np.stack([draws[s] for s in ss]), 0))
                b.update(ci_lo=lo, ci_hi=hi, k_below=int(sum(p["below_chance_entire_ci"] for p in per.values())))
            blk[det] = b
        m = blk["mahalanobis_classcond"]
        blk["rule_A"] = bool(m["ci_hi"] < 0.5 and m["k_below"] >= 3)
        blk["nv_nearest_frac"] = {"mean": float(np.mean([np.mean(nearest[s] == NV) for s in ss])),
                                  "sd": float(np.std([np.mean(nearest[s] == NV) for s in ss], ddof=1))}
        pc = {}
        for c, n in res["mapped_counts"].items():
            sel = mapped == c
            pts = [auroc_w(sc[s]["mahalanobis_classcond"][0], sc[s]["mahalanobis_classcond"][1][sel]) for s in ss]
            pc[c] = {"n": int(n), "mean": float(np.mean(pts)), "sd": float(np.std(pts, ddof=1))}
        blk["per_class_maha"] = pc
        if len(mk) >= 2:
            a_all = [auroc_w(sc[s]["mahalanobis_classcond"][0], sc[s]["mahalanobis_classcond"][1][sub]) for s in ss]
            a_rw = [auroc_w(sc[s]["mahalanobis_classcond"][0], sc[s]["mahalanobis_classcond"][1][sub], w_rw[sub])
                    for s in ss]
            blk["decomposition"] = {"classes": mk, "mapped_all": float(np.mean(a_all)),
                                    "reweighted_to_isic_mix": float(np.mean(a_rw))}
        res["lambda"]["{:g}".format(lam)] = blk
        print(name, lam, "maha {:.3f} [{:.3f}, {:.3f}] k={} ruleA={}".format(m["mean"], m["ci_lo"], m["ci_hi"],
                                                                         m["k_below"], blk["rule_A"]), flush=True)
    (OUT / "{}.json".format(name)).write_text(json.dumps(res, indent=1) + "\n")


def report():
    rows, L, tex = [], [], []
    res = {n: json.loads((OUT / "{}.json".format(n)).read_text()) for n in SETS if (OUT / "{}.json".format(n)).exists()}
    for n, r in res.items():
        for lam, b in r["lambda"].items():
            row = {"set": n, "lambda": lam, "n_images": r["n_images"], "n_groups": r["n_groups"],
                   "rule_A": b["rule_A"], "nv_nearest": b["nv_nearest_frac"]["mean"]}
            for d in DETS:
                row[d + "_mean"], row[d + "_sd"] = b[d]["mean"], b[d]["sd"]
                if d in BOOT_DETS:
                    row[d + "_ci_lo"], row[d + "_ci_hi"], row[d + "_k_below"] = b[d]["ci_lo"], b[d]["ci_hi"], b[d]["k_below"]
            for c, p in b["per_class_maha"].items():
                row["maha_" + c] = p["mean"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    L += ["# Paper B R5 P7(a): truly unseen clinical sets (evaluation only)", "",
          "Design: `decisions/paperB_r5_p7a.md`. 35 lesion-level runs (7 λ × 5 seeds, the Table 1 runs), detectors fit "
          "on ISIC train, ID = lesion-level ISIC test (5,067 images, resampled by lesion). Bootstrap: "
          "PREREGISTER_BOOTSTRAP.json (2,000 draws, percentile 95%, draws shared across seeds). Rule A per set: at λ = 2 "
          "the mean-over-seeds CI of Mahalanobis entirely below 0.5 and ≥ 3/5 per-seed CIs below 0.5.", ""]
    for n, r in res.items():
        L += ["## {} ({} images, {} resample groups; unit: {})".format(SETS[n], r["n_images"], r["n_groups"],
                                                                      r["resample_unit"]), "",
              "| λ | " + " | ".join(DET_LABEL[d] for d in DETS) + " | Maha k<0.5 | rule A | NV-nearest |",
              "|---:|" + "---|" * (len(DETS) + 3)]
        for lam, b in r["lambda"].items():
            cells = []
            for d in DETS:
                x = b[d]
                cells.append("{:.3f} ± {:.3f}".format(x["mean"], x["sd"]) +
                             (" [{:.3f}, {:.3f}]".format(x["ci_lo"], x["ci_hi"]) if d in BOOT_DETS else ""))
            m = b["mahalanobis_classcond"]
            L.append("| {} | {} | {}/5 | {} | {:.3f} |".format(lam, " | ".join(cells), m["k_below"],
                                                               "yes" if b["rule_A"] else "no", b["nv_nearest_frac"]["mean"]))
            tex.append("{} & {} & {} \\\\".format(SETS[n], lam, " & ".join(
                "{:.3f}\\,$\\pm$\\,{:.3f}".format(b[d]["mean"], b[d]["sd"]) for d in DETS)))
        L += ["", "Per-class Mahalanobis AUROC (ID test vs the OOD images of one mapped class; mean ± s.d. over seeds):", "",
              "| λ | " + " | ".join("{} (n={})".format(c, k) for c, k in r["mapped_counts"].items()) + " | decomposition: mapped / reweighted |",
              "|---:|" + "---|" * (len(r["mapped_counts"]) + 1)]
        for lam, b in r["lambda"].items():
            dc = b.get("decomposition")
            L.append("| {} | {} | {} |".format(lam, " | ".join("{:.3f} ± {:.3f}".format(p["mean"], p["sd"])
                                                            for p in b["per_class_maha"].values()),
                                               "{:.3f} / {:.3f}".format(dc["mapped_all"], dc["reweighted_to_isic_mix"])
                                               if dc else "—"))
        L.append("")
    L += ["## LaTeX rows (set & λ & Mahalanobis & kNN & cosine & MSP & Energy)", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")


def main():
    want = sys.argv[1:] or [n for n in SETS if all((run_dir(l, s) / "{}.npz".format(n)).exists()
                                                    for l in LAMS for s in seeds(l))]
    for n in want:
        score_set(n)
    report()


if __name__ == "__main__":
    main()
