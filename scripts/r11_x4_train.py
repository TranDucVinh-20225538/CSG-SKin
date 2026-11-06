#!/usr/bin/env python3
"""Paper B R11 X4: BCN → HAM dual-encoder (main recipe), shift 100/0.

Design: decisions/paperB_r11_x4.md
  Source BCN labelled, adversary HAM (ham_adv / ham_heldout ≥20% lesions).
  Split seed 42 fixed. λ∈{0,2}×5 seeds shift100; λ=2×5 shift0.
  Three-stream: bcn_cls + bcn_adv (±dose_mix to HAM) + ham_adv. last.ckpt.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402  # installs transformers stub before lightning

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

import phase16_bcn_ham_data as bdata  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_train_transform_robust, build_val_transform_robust  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402

OUT = R.RES / "r11" / "x4"
CKPT = R.CKPT / "r11_x4"
SPLIT_SEED = 42
SPLIT_PATH = OUT / "split_seed42.json"
OODS = ("ham_heldout", "pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
LABELLED_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")  # ham_heldout + SCIN reported not counted


def git_head():
    return subprocess.run(
        ["git", "-C", str(R.PAPERB), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def lam_tag(lam):
    return "{:g}".format(float(lam)).replace(".", "p")


def tag_for(lam, seed, shift):
    return "x4_ladv{}_shift{}_s{}".format(lam_tag(lam), int(round(100 * float(shift))), int(seed))


def jobs():
    out = []
    for lam in (0.0, 2.0):
        for seed in R.SEEDS:
            out.append({"lam": lam, "seed": seed, "shift": 1.0})
    for seed in R.SEEDS:
        out.append({"lam": 2.0, "seed": seed, "shift": 0.0})
    return out


def class_mix_y(y):
    y = np.asarray(y, int)
    return np.bincount(y, minlength=R.NCLS).astype(float) / max(len(y), 1)


def tv(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return 0.5 * float(np.abs(p - q).sum())


def load_or_create_splits():
    OUT.mkdir(parents=True, exist_ok=True)
    df = bdata.load_bcn_ham_frame()
    splits, meta = bdata.prepare_b1_splits(df, SPLIT_SEED, ham_heldout_frac=0.30)
    if not SPLIT_PATH.exists():
        doc = {"seed": SPLIT_SEED, "meta": meta,
               "ham_heldout_frac": 0.30,
               "n_ham_lesions_heldout": int(splits["ham_heldout"]["lesion_id"].nunique()),
               "n_ham_lesions_adv": int(splits["ham_adv"]["lesion_id"].nunique()),
               "tv_bcn_train_vs_ham_adv": tv(class_mix_y(splits["bcn_train"]["label_idx"]),
                                             class_mix_y(splits["ham_adv"]["label_idx"])),
               "tv_bcn_train_vs_ham_heldout": tv(class_mix_y(splits["bcn_train"]["label_idx"]),
                                                class_mix_y(splits["ham_heldout"]["label_idx"])),
               "pi_bcn_train": class_mix_y(splits["bcn_train"]["label_idx"]).tolist(),
               "pi_ham_adv": class_mix_y(splits["ham_adv"]["label_idx"]).tolist(),
               "pi_ham_heldout": class_mix_y(splits["ham_heldout"]["label_idx"]).tolist(),
               "commit": git_head()}
        SPLIT_PATH.write_text(json.dumps(doc, indent=2) + "\n")
        (OUT / "TV_PRIORS.json").write_text(json.dumps(doc, indent=2) + "\n")
    return splits, meta


def build_streams(splits, adv_mix):
    tf = build_train_transform_robust()
    adv_probs = None if adv_mix is None else R.class_weighted_probs(splits["bcn_train"], adv_mix)
    streams = [
        R.Stream("bcn_cls", splits["bcn_train"], None, cls=True),
        R.Stream("bcn_adv", splits["bcn_train"], adv_probs, adv=0, ctx=0, src_labels=True),
        R.Stream("ham_adv", splits["ham_adv"][["path"]], None, adv=1, ctx=1),
    ]
    return R.MultiStreamDataset(streams, tf), R.multistream_collate, [s.describe() for s in streams]


def v1_bcn_ham(Xi, Xh, gi, gh, pseeds=(42, 52, 62)):
    X = np.r_[Xi, Xh]
    d = np.r_[np.zeros(len(Xi)), np.ones(len(Xh))]
    g = np.r_[np.char.add("B", np.asarray(gi).astype(str)),
              np.char.add("H", np.asarray(gh).astype(str))]
    bals = []
    for s in pseeds:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(X[a], d[a])
        bals.append(float(balanced_accuracy_score(d[b], clf.predict(X[b]))))
    return {"bal_acc_mean": float(np.mean(bals)),
            "bal_acc_sd": float(np.std(bals, ddof=1)) if len(bals) > 1 else 0.0}


@torch.no_grad()
def extract_all(net, splits, data_isic, feat_dir, num_workers, device):
    feat_dir.mkdir(parents=True, exist_ok=True)
    tf = build_val_transform_robust()
    sets = {
        "bcn_train": SkinDataset(splits["bcn_train"], transform=tf),
        "bcn_val": SkinDataset(splits["bcn_val"], transform=tf),
        "bcn_test": SkinDataset(splits["bcn_test"], transform=tf),
        "ham_adv": SkinDataset(splits["ham_adv"], transform=tf),
        "ham_heldout": SkinDataset(splits["ham_heldout"], transform=tf),
        "pad_heldout": SkinDataset(data_isic["pad_heldout"], transform=tf),
        "fitzpatrick17k": FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf),
    }
    for n in R.P7A_NAMES:
        csv = R.P7A_SETS / "{}.csv".format(n)
        sets[n] = FolderJpegDataset(pd.read_csv(csv, keep_default_na=False).path.tolist(), tf)
    # Scorer layout aliases
    alias = {"isic_train": "bcn_train", "isic_test": "bcn_test", "isic_val": "bcn_val"}
    for name, ds in sets.items():
        out = feat_dir / "{}.npz".format(name)
        if out.exists():
            continue
        pack = x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers), device)
        np.savez_compressed(out, **{k: pack[k] for k in R.FEAT_KEYS})
        print("features", name, pack["labels"].shape, flush=True)
    # copy aliases for Scorer / run_metrics compatibility
    for a, src in alias.items():
        dst = feat_dir / "{}.npz".format(a)
        if not dst.exists():
            dst.symlink_to((feat_dir / "{}.npz".format(src)).resolve())


def train_one(lam, seed, shift, skip_done=True, batch_size=32, num_workers=4):
    tag = tag_for(lam, seed, shift)
    run_dir = OUT / "runs" / tag
    feat_dir = OUT / "features" / tag
    ckpt_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if skip_done and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("checkpoint_policy") == "last" and (feat_dir / "bcn_test.npz").exists():
            print("skip_done", summary_path, flush=True)
            return

    seed_everything(seed)
    pl.seed_everything(seed, workers=True)
    splits, meta = load_or_create_splits()
    recipe = R.CURRENT
    bcn_mix = class_mix_y(splits["bcn_train"]["label_idx"])
    ham_mix = class_mix_y(splits["ham_adv"]["label_idx"])
    adv_mix = None
    if float(shift) < 1.0:
        adv_mix = R.dose_mix(shift, bcn_mix, ham_mix)
        # support change: classes with zero in BCN but positive target weight
    support_note = None
    if adv_mix is not None:
        lost = [R.LABELS[c] for c in range(R.NCLS) if bcn_mix[c] == 0 and adv_mix[c] > 0]
        gained_zero = [R.LABELS[c] for c in range(R.NCLS) if ham_mix[c] > 0 and bcn_mix[c] == 0]
        support_note = {"bcn_zero_but_target_pos": lost, "ham_pos_bcn_zero": gained_zero,
                        "tv_bcn_vs_target": tv(bcn_mix, adv_mix), "tv_bcn_vs_ham": tv(bcn_mix, ham_mix)}

    ds, collate, streams = build_streams(splits, adv_mix)
    cfg = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "shift": float(shift),
        "split_seed": SPLIT_SEED, "recipe": recipe, "streams": streams,
        "bcn_mix": bcn_mix.tolist(), "ham_adv_mix": ham_mix.tolist(),
        "adv_mix": None if adv_mix is None else np.asarray(adv_mix).tolist(),
        "support_note": support_note, "split_meta": meta,
        "architecture": "dual_encoder_csg_effb3", "design": "decisions/paperB_r11_x4.md",
        "commit": git_head(),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=float) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last = ckpt_dir / "last.ckpt"
    if last.exists() and R.trained_to_end(ckpt_dir, recipe):
        print("ckpt exists, eval only", last, flush=True)
        best = R.find_best(ckpt_dir) or last
    else:
        model = R.make_model(recipe, lam)
        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc",
            mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
        trainer = pl.Trainer(
            max_epochs=int(recipe["max_epochs"]),
            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
            callbacks=[ckpt_cb],
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            default_root_dir=str(run_dir), log_every_n_steps=20)
        tl = p2.make_loader(ds, batch_size, num_workers, True, drop_last=True, collate=collate)
        vl = p2.make_loader(SkinDataset(splits["bcn_val"], transform=build_val_transform_robust()),
                            batch_size, num_workers, False)
        print("train", tag, "device", device, "n=", len(ds), flush=True)
        trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl)
        best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
        last = ckpt_dir / "last.ckpt"

    primary = last if last.exists() else best
    net = R.load_net(primary, device)
    data_isic = R.load_data()
    extract_all(net, splits, data_isic, feat_dir, num_workers, device)

    # metrics: treat bcn as isic via aliases
    Fs = R.load_feats(feat_dir)
    # ensure aliases loaded
    for a, src in (("isic_train", "bcn_train"), ("isic_test", "bcn_test"), ("isic_val", "bcn_val")):
        if a not in Fs and src in Fs:
            Fs[a] = Fs[src]
    m = R.run_metrics(Fs, ood_names=list(OODS))
    # V1: BCN test ∪ HAM heldout
    gi = splits["bcn_test"]["lesion_id"].astype(str).to_numpy()
    gh = splits["ham_heldout"]["lesion_id"].astype(str).to_numpy()
    v1 = v1_bcn_ham(Fs["bcn_test"]["z_lesion_norm"], Fs["ham_heldout"]["z_lesion_norm"], gi, gh)
    m["heldout_leakage_V1"] = v1["bal_acc_mean"]
    m["heldout_leakage_V1_sd"] = v1["bal_acc_sd"]

    summary = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "shift": float(shift),
        "checkpoint_policy": "last",
        "primary_checkpoint": str(primary),
        "id_balanced_acc": float(m["id"]["bal_acc"]),
        "heldout_leakage_V1": float(v1["bal_acc_mean"]),
        "metrics": m, "config": cfg, "commit": git_head(),
        "tv_bcn_vs_ham_adv": tv(bcn_mix, ham_mix),
        "support_note": support_note,
    }
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("saved", summary_path, "id_bal", summary["id_balanced_acc"],
          "V1", summary["heldout_leakage_V1"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()
    js = jobs()
    if a.list:
        for i, j in enumerate(js):
            print(i, tag_for(j["lam"], j["seed"], j["shift"]), j)
        print("n_jobs", len(js))
        return
    if a.task_id is None:
        raise SystemExit("need --task_id")
    j = js[a.task_id]
    train_one(j["lam"], j["seed"], j["shift"], skip_done=a.skip_done,
              batch_size=a.batch_size, num_workers=a.num_workers)


if __name__ == "__main__":
    main()
