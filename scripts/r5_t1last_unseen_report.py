#!/usr/bin/env python3
"""Table 1 last.ckpt: unseen clinical Maha CIs (image vs group), paired λ2−λ0 ID bal, ID/OOD ECE.

Writes results/paperB/r5/t1last/UNSEEN_ECE_REPORT.md (+ JSON/CSV).
CPU only. Extracts mednode/sd198/scin into features_last/<run>/ if missing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_ckpt_audit as A  # noqa: E402
import r5cpu_common as R5  # noqa: E402
import r6_common as R  # noqa: E402
import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import build_val_transform_robust  # noqa: E402

OUT = R.RES / "r5" / "t1last"
FEAT = OUT / "features_last"
UNSEEN = ("mednode", "sd198", "scin")
DET = "mahalanobis_classcond"
SD_CL = R.RES / "r5" / "p13" / "sd198_clusters.csv"
if not SD_CL.exists():
    SD_CL = R.RES / "r5" / "d1_domain" / "sd198_clusters.csv"


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def p7a_paths(name):
    df = pd.read_csv(R.P7A_SETS / "{}.csv".format(name), keep_default_na=False)
    return df.path.tolist(), df.group.astype(str).to_numpy()


def ensure_unseen_features(tag):
    """Forward last.ckpt for mednode/sd198/scin into features_last/<tag>/ if missing."""
    out = FEAT / tag
    out.mkdir(parents=True, exist_ok=True)
    need = [n for n in UNSEEN if not (out / "{}.npz".format(n)).exists()]
    if not need:
        return out
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    use_cuda = os.environ.get("T1LAST_USE_CUDA", "").strip() in ("1", "true", "yes")
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    tf = build_val_transform_robust()
    net = fin.load_csg(A.ITEM4_CKPT / tag / "last.ckpt", device)
    bs = 128 if device.type == "cuda" else 64
    for name in need:
        paths, _ = p7a_paths(name)
        ds = FolderJpegDataset(paths, tf)
        # nw=0: NFS-safe (avoids DataLoader tempdir .nfs EBUSY cleanup noise)
        pack = x13.collect_depths(net, DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0),
                                  device)
        np.savez_compressed(out / "{}.npz".format(name), **{k: pack[k] for k in R.FEAT_KEYS})
        print("features", tag, name, pack["labels"].shape, flush=True)
    return out


def group_units(name):
    """Return [(unit_label, group_array)] for bootstrap. Image always; plus set-specific group."""
    paths, g_p7a = p7a_paths(name)
    g_img = np.arange(len(paths)).astype(str)  # one group per image
    out = [("image", g_img)]
    if name == "sd198":
        cl = pd.read_csv(SD_CL, keep_default_na=False)
        assert cl.path.tolist() == paths
        out.append(("cluster_union", cl.cluster_union.astype(str).to_numpy()))
    elif name == "scin":
        out.append(("case", g_p7a))
    elif name == "mednode":
        # no near-dup clusters; P7a group == image
        out.append(("group(=image)", g_p7a))
    return out


def maha_pair(feat_dir, ood):
    Fs = R.load_feats(Path(feat_dir), ("isic_train", "isic_test", ood))
    if ood not in Fs:
        return None
    S = R.Scorer(Fs["isic_train"])
    return S.scores(Fs["isic_test"])[DET], S.scores(Fs[ood])[DET]


def boot_cell(per_seed, idg, g, n_boot, seed):
    rng = np.random.default_rng(seed)
    id_draws = R5.draws(idg, n_boot, rng)
    rng2 = np.random.default_rng(seed + 1)
    oo_draws = R5.draws(g, n_boot, rng2)
    return R5.pooled_bootstrap(per_seed, id_draws, oo_draws)


def ece_of(feat_dir):
    Fs = R.load_feats(Path(feat_dir), ("isic_train", "isic_test", "pad_heldout"))
    m = R.run_metrics(Fs, ood_names=["pad_heldout"])
    xdom = float(m["xdom_pad_heldout"]["bal_acc_6cls"])
    return float(m["id"]["ece"]), float(m["ood_ece_pad_heldout"]), float(m["id"]["bal_acc"]), xdom


def paired_delta_ci(deltas, n_boot, seed):
    d = np.asarray(deltas, float)
    point = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return {"mean": point, "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "ci_lo": lo, "ci_hi": hi, "n": int(len(d)), "per_seed": {str(i): float(x) for i, x in enumerate(d)}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None, help="extract unseen features for one Table1 run index")
    ap.add_argument("--report_only", action="store_true", help="skip extract; bootstrap+report from existing features")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    runs = A.table1_runs()
    if a.task is not None:
        tag = runs[a.task]
        print("extract task", a.task, tag, flush=True)
        ensure_unseen_features(tag)
        return

    pre = R5.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    idg = R5.id_test_groups()
    print("runs", len(runs), flush=True)

    # 1) ensure features + score
    per_maha = {o: {} for o in UNSEEN}  # ood -> (lam, seed) -> (sid, sood)
    ece_rows = []
    id_bal = {}  # (lam, seed) -> bal
    for tag in runs:
        lam = A.lam_of(tag)
        s = int(tag.rsplit("_s", 1)[1])
        if a.report_only:
            fl = FEAT / tag
            missing = [n for n in UNSEEN if not (fl / "{}.npz".format(n)).exists()]
            if missing:
                raise FileNotFoundError("{} missing {}".format(fl, missing))
        else:
            fl = ensure_unseen_features(tag)
        assert (fl / "isic_test.npz").exists(), fl
        id_ece, ood_ece, bal, xdom = ece_of(fl)
        ece_rows.append({"run": tag, "lambda": lam, "seed": s,
                         "id_bal_acc": bal, "id_ece": id_ece, "ood_ece_pad_heldout": ood_ece,
                         "xdom_bal_acc": xdom})
        id_bal[(lam, s)] = bal
        for ood in UNSEEN:
            pair = maha_pair(fl, ood)
            if pair is None:
                raise FileNotFoundError(fl / "{}.npz".format(ood))
            per_maha[ood][(lam, s)] = pair
        print("scored", tag, flush=True)

    # 2) bootstrap per λ × set × unit
    boot_rows = []
    boot_json = {}
    for ood in UNSEEN:
        units = group_units(ood)
        for lam in R5.LAMS:
            per = {s: per_maha[ood][(lam, s)] for (l, s) in per_maha[ood] if abs(l - lam) < 1e-12}
            for unit, g in units:
                b = boot_cell(per, idg, g, n_boot, seed)
                key = "{}|{:g}|{}".format(ood, lam, unit)
                boot_json[key] = {
                    "mean": b["mean"], "ci_lo": b["pooled_ci_lo"], "ci_hi": b["pooled_ci_hi"],
                    "k_below": b["k_seeds_ci_below_0.5"], "n_seeds": b["n_seeds"],
                    "rule_A": b["rule_A"], "n_groups": int(len(set(g))),
                }
                boot_rows.append({"ood": ood, "lambda": lam, "unit": unit, **boot_json[key]})
                print(key, boot_json[key]["mean"], boot_json[key]["k_below"], boot_json[key]["rule_A"], flush=True)

    # 3) paired λ=2 − λ=0 ID bal (matched seeds)
    seeds0 = sorted(s for (l, s) in id_bal if abs(l) < 1e-12)
    seeds2 = sorted(s for (l, s) in id_bal if abs(l - 2.0) < 1e-12)
    common = sorted(set(seeds0) & set(seeds2))
    deltas = [id_bal[(2.0, s)] - id_bal[(0.0, s)] for s in common]
    paired = paired_delta_ci(deltas, n_boot, seed)
    paired["seeds"] = common
    paired["per_seed"] = {str(s): id_bal[(2.0, s)] - id_bal[(0.0, s)] for s in common}

    # 4) ECE summary per λ
    edf = pd.DataFrame(ece_rows)
    ece_sum = []
    for lam, d in edf.groupby("lambda"):
        ece_sum.append({
            "lambda": float(lam), "n": int(len(d)),
            "id_bal_mean": float(d.id_bal_acc.mean()), "id_bal_sd": float(d.id_bal_acc.std(ddof=1)),
            "id_ece_mean": float(d.id_ece.mean()), "id_ece_sd": float(d.id_ece.std(ddof=1)),
            "ood_ece_mean": float(d.ood_ece_pad_heldout.mean()),
            "ood_ece_sd": float(d.ood_ece_pad_heldout.std(ddof=1)),
            "xdom_mean": float(d.xdom_bal_acc.mean()),
            "xdom_sd": float(d.xdom_bal_acc.std(ddof=1)),
        })

    payload = {
        "commit": R.git_head() if hasattr(R, "git_head") else R5.git_head(),
        "checkpoint": "last.ckpt",
        "n_runs": len(runs),
        "bootstrap": boot_json,
        "paired_id_bal_l2_minus_l0": paired,
        "ece_per_run": ece_rows,
        "ece_by_lambda": ece_sum,
        "sd198_clusters": str(SD_CL),
    }
    (OUT / "UNSEEN_ECE.json").write_text(json.dumps(_jsonable(payload), indent=1) + "\n")
    pd.DataFrame(boot_rows).to_csv(OUT / "UNSEEN_bootstrap.csv", index=False)
    edf.to_csv(OUT / "UNSEEN_ece_per_run.csv", index=False)

    # REPORT
    L = ["# Table 1 last.ckpt — unseen Maha CIs, paired ID bal Δ, ECE", "",
         "Checkpoint: `last.ckpt` of the 35 Table 1 runs (`r2/item4`). Features under "
         "`results/paperB/r5/t1last/features_last/`. Commit `{}`. PREREGISTER 2,000 draws, rng 20261005; "
         "ID = lesion; Mahalanobis class-conditional on `z_lesion_norm`.".format(payload["commit"]), "",
         "Units: MED-NODE image (P7a group = image); SD-198 image and P13-style `cluster_union` "
         "(pHash Hamming ≤ 8 ∪ ImageNet-R50 cosine ≥ 0.95; 6016 clusters / 6584 images); "
         "SCIN image and case (P7a `group`).", "",
         "## (1) Mahalanobis AUROC — image vs group/cluster", ""]
    for ood in UNSEEN:
        L += ["### {}".format(ood), "",
              "| λ | n | unit | n_groups | mean | 95% CI | k CI<0.5 | rule A |",
              "|---:|---:|---|---:|---:|---|---:|---|"]
        for lam in R5.LAMS:
            for unit, _g in group_units(ood):
                b = boot_json["{}|{:g}|{}".format(ood, lam, unit)]
                L.append("| {:g} | {} | {} | {} | {:.3f} | [{:.3f}, {:.3f}] | {}/{} | {} |".format(
                    lam, b["n_seeds"], unit, b["n_groups"], b["mean"], b["ci_lo"], b["ci_hi"],
                    b["k_below"], b["n_seeds"], "met" if b["rule_A"] else "not met"))
        L.append("")

    L += ["## (2) Paired ID bal acc: λ = 2 − λ = 0", "",
          "Matched seeds: {}. Mean Δ = {:.4f} ± {:.4f}; seed-bootstrap 95% CI [{:.4f}, {:.4f}] "
          "(2,000 draws, rng 20261005).".format(
              common, paired["mean"], paired["sd"], paired["ci_lo"], paired["ci_hi"]), "",
          "| seed | ID bal λ=0 | ID bal λ=2 | Δ |", "|---:|---:|---:|---:|"]
    for s in common:
        L.append("| {} | {:.4f} | {:.4f} | {:+.4f} |".format(
            s, id_bal[(0.0, s)], id_bal[(2.0, s)], id_bal[(2.0, s)] - id_bal[(0.0, s)]))
    L += ["", "## (3) ID ECE, OOD ECE (pad_heldout), X-dom bal acc by λ (last.ckpt)", "",
          "| λ | n | ID bal acc | ID ECE | OOD ECE (pad_heldout) | X-dom bal acc |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in ece_sum:
        L.append("| {:g} | {} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} |".format(
            r["lambda"], r["n"], r["id_bal_mean"], r["id_bal_sd"],
            r["id_ece_mean"], r["id_ece_sd"], r["ood_ece_mean"], r["ood_ece_sd"],
            r["xdom_mean"], r["xdom_sd"]))
    L.append("")
    (OUT / "UNSEEN_ECE_REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "UNSEEN_ECE_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
