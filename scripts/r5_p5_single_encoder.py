#!/usr/bin/env python3
"""Paper B R5 P5: single-encoder GRL (Phase 15.1) at λ = 4 — n, per-detector AUROC with bootstrap CI, seeds below 0.5
(decisions/paperB_r5_p5.md).

Stage 1 (CPU, loads checkpoints; one array task per run): r5_p5_single_encoder.py --extract TASK_ID
  Features for the 25 Phase 15.1 runs (all λ; λ = 4 is the reported cell, the rest feed Q4/R4) on the image-level
  split the runs were trained on: isic_train, isic_test, pad_full, pad_heldout, fitzpatrick17k -> z (post-BN feature
  the detectors use), logits, labels. Output results/paperB/r5/p5/features/<run>/<split>.npz.
Stage 2 (CPU): r5_p5_single_encoder.py --score
  Five detectors (phase15_common.detector_block definitions, fit on ISIC train); pipeline check against the stored
  summary.json values; PREREGISTER bootstrap (ID by image as pre-registered for the image-level split, and by lesion
  group as sensitivity; pad_heldout by patient; Fitzpatrick by image / P13 cluster).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase15_1_single_dann as S  # noqa: E402  (installs the transformers stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import r5cpu_common as R  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

OUT = R.R5 / "p5"
FEAT = OUT / "features"
SRC = R.RES / "phase15" / "single_dann"
LAM = 4.0
OODS = ("pad_heldout", "fitzpatrick17k")


def all_runs():
    out = []
    for d in sorted(SRC.glob("single_dann_ladv*_s*")):
        m = re.fullmatch(r"single_dann_ladv([0-9p]+)_s(\d+)", d.name)
        if m and (d / "summary.json").exists():
            out.append((float(m.group(1).replace("p", ".")), int(m.group(2)), d.name))
    return sorted(out)


@torch.no_grad()
def extract(task, only_lambda=None):
    rr = [r for r in all_runs() if only_lambda is None or abs(r[0] - only_lambda) < 1e-9]
    if task >= len(rr):
        print("task", task, "beyond", len(rr), "runs; nothing to do")
        return
    lam, seed, run = rr[task]
    dest = FEAT / run
    if (dest / "fitzpatrick17k.npz").exists():
        print("skip", run)
        return
    summ = json.loads((SRC / run / "summary.json").read_text())
    lit = S.SingleDannLightning.load_from_checkpoint(summ["best_checkpoint"], strict=False, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()
    spec = json.loads(S.P2_SPLIT.read_text())
    df = S.p2.load_master()
    isic, pad = df[df.domain == "isic"].copy(), df[df.domain == "pad_ufes"].copy()
    train, _val, test = S.p2.isic_splits(isic)
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    tf = build_val_transform_robust()
    frames = {"isic_train": SkinDataset(train, transform=tf), "isic_test": SkinDataset(test, transform=tf),
              "pad_full": SkinDataset(pad, transform=tf), "pad_heldout": SkinDataset(hold, transform=tf),
              "fitzpatrick17k": FolderJpegDataset(sorted(R.T.p3.FITZ_DIR.glob("*.jpg")), tf)}
    dest.mkdir(parents=True, exist_ok=True)
    for name, ds in frames.items():
        pack = S.collect_all(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=8), device)
        np.savez_compressed(dest / "{}.npz".format(name), z=pack["z"], logits=pack["logits"], labels=pack["labels"])
        print("saved", run, name, pack["z"].shape, flush=True)


def image_level_test_lesion_groups():
    s = pd.read_csv(R.C.ISIC_SPLIT_CSV)
    te = s[s.split == "test"].sort_values("row")
    return te.lesion_id.fillna(te.image).to_numpy()


def score():
    pre = R.prereg()
    n = int(pre["n_resamples"])
    rows, sc = [], {}
    for lam, seed, run in all_runs():
        d = FEAT / run
        if not (d / "fitzpatrick17k.npz").exists():
            continue
        f = {s: dict(np.load(d / "{}.npz".format(s))) for s in ("isic_train", "isic_test") + OODS}
        f = {k: {"z_lesion_norm": v["z"], "logits": v["logits"], "labels": v["labels"]} for k, v in f.items()}
        s_, _m = R.detector_scores(f["isic_train"], f["isic_test"], {o: f[o] for o in OODS})
        sc[(lam, seed)] = s_
        summ = json.loads((SRC / run / "summary.json").read_text())
        rec = {"lambda": lam, "seed": seed, "run": run, "leakage_pad_full": summ["leakage"]["pad_full"]["bal_acc_mean"],
               "id_bal_acc": summ["id_balanced_acc"]}
        for det, (sid, so) in s_.items():
            for o in OODS:
                rec["{}|{}".format(det, o)] = R.auroc(sid, so[o])
            stored = summ["ood"]["pad_heldout"].get(det)
            if isinstance(stored, float):
                rec["check|{}".format(det)] = rec["{}|pad_heldout".format(det)] - stored
        rows.append(rec)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summary.csv", index=False)
    rng = np.random.default_rng(int(pre["rng_seed"]))
    n_id = len(next(iter(sc.values()))["MSP"][0]) if sc else 0
    id_units = {"image (pre-registered)": np.arange(n_id).astype(str), "lesion group": image_level_test_lesion_groups()}
    boot = {}
    for unit, g in id_units.items():
        for o in OODS:
            idd, odd = R.draws(g, n, rng), R.draws(R.ood_groups(o), n, rng)
            for det in R.BOOT_DETS + ("MSP", "Energy_T1"):
                per = {s: (sc[(LAM, s)][det][0], sc[(LAM, s)][det][1][o]) for (l, s) in sc if l == LAM}
                if len(per) >= 2:
                    b = R.pooled_bootstrap(per, idd, odd)
                    b["k_points_below_0.5"] = int(sum(v["point"] < 0.5 for v in b["per_seed"].values()))
                    boot["{}|{}|{}".format(unit, o, det)] = b
    R.write_json(OUT / "results.json", {"commit": R.git_head(), "lambda": LAM, "bootstrap": boot, "per_run": rows})
    L = ["# Paper B R5 P5 — single-encoder GRL at λ = 4", "",
         "Design: `decisions/paperB_r5_p5.md`. Commit `{}`. Phase 15.1 single-encoder EfficientNet-B3 DANN, image-level ISIC split "
         "(the split these runs were trained on). Detectors fit on ISIC train. Bootstrap: PREREGISTER draws.".format(R.git_head()), ""]
    lam4 = df[df["lambda"] == LAM] if len(df) else df
    L.append("n (seeds) at λ = 4: {} ({}).".format(len(lam4), "/".join(map(str, sorted(lam4.seed))) if len(lam4) else "—"))
    chk = [c for c in df.columns if c.startswith("check|")]
    if chk:
        L.append("Pipeline check vs stored summary.json (pad_heldout): max |Δ| = {:.1e}.".format(df[chk].abs().max().max()))
    tex = []
    for unit in id_units:
        for o in OODS:
            L += ["", "## ID resampled by {}, OOD = {}".format(unit, o), "", "| Detector | mean ± s.d. | pooled 95% CI | per-seed points | "
                  "seeds with CI < 0.5 | seeds with point < 0.5 |", "|---|---|---|---|---|---|"]
            for det in R.BOOT_DETS + ("MSP", "Energy_T1"):
                b = boot.get("{}|{}|{}".format(unit, o, det))
                if not b:
                    continue
                L.append("| {} | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] | {} | {}/{} | {}/{} |".format(
                    R.DET_LABEL[det], b["mean"], b["sd"], b["pooled_ci_lo"], b["pooled_ci_hi"],
                    ", ".join("{:.3f}".format(v["point"]) for v in b["per_seed"].values()), b["k_seeds_ci_below_0.5"], b["n_seeds"],
                    b["k_points_below_0.5"], b["n_seeds"]))
                if unit.startswith("image") and o == "pad_heldout":
                    tex.append("{} & {} & {:.3f}\\,$\\pm$\\,{:.3f} & [{:.3f}, {:.3f}] & {}/{} \\\\".format(
                        R.DET_LABEL[det], b["n_seeds"], b["mean"], b["sd"], b["pooled_ci_lo"], b["pooled_ci_hi"], b["k_seeds_ci_below_0.5"], b["n_seeds"]))
    if len(df):
        L += ["", "## All λ (context; point AUROC, mean ± s.d.)", "", "| λ | n | leakage | " + " | ".join(R.DET_LABEL[d] for d in R.DETS) + " |",
              "|---:|---:|---|" + "---|" * len(R.DETS)]
        for lam in sorted(df["lambda"].unique()):
            d = df[df["lambda"] == lam]
            L.append("| {:g} | {} | {} | {} |".format(lam, len(d), R.fmt(d.leakage_pad_full.mean(), d.leakage_pad_full.std(ddof=1)),
                                                      " | ".join(R.fmt(d["{}|pad_heldout".format(x)].mean(), d["{}|pad_heldout".format(x)].std(ddof=1)) for x in R.DETS)))
    b = boot.get("image (pre-registered)|pad_heldout|mahalanobis_classcond")
    if b:
        L += ["", "## Plain reading", "", "At λ = 4 (n = {}) the single-encoder Mahalanobis AUROC on pad_heldout is {:.3f} ± {:.3f}, pooled 95% CI "
              "[{:.3f}, {:.3f}]; {} of {} seeds have a CI entirely below 0.5 and {} of {} a point estimate below 0.5.".format(
                  b["n_seeds"], b["mean"], b["sd"], b["pooled_ci_lo"], b["pooled_ci_hi"], b["k_seeds_ci_below_0.5"], b["n_seeds"],
                  b["k_points_below_0.5"], b["n_seeds"])]
    L += ["", "## LaTeX rows (detector & n & mean ± s.d. & pooled CI & seeds CI < 0.5), pad_heldout", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", type=int, default=None)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--only_lambda", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, r in enumerate(all_runs()):
            print(i, *r)
    if a.extract is not None:
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        extract(a.extract, a.only_lambda)
    if a.score:
        score()


if __name__ == "__main__":
    main()
