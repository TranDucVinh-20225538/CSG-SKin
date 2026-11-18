#!/usr/bin/env python3
"""Paper B R14: ISIC 2019 → MILK10k dermoscopic dual CSG train (+ light stage-1 metrics).

Streams: isic_cls, isic_adv (resampled to prior), milk_adv.
Tags: r14_ladv0_prior100_s{seed}, r14_ladv2_prior{100,50,0}_s{seed}.
NaN guard → mark degenerate. last.ckpt primary; best.ckpt also scored later.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r14_pair3_common as C  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_train_transform_robust, build_val_transform_robust  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402


class NanGuardCallback(pl.Callback):
    def __init__(self):
        self.triggered = False
        self.reason = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.triggered:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        val = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)
        if math.isnan(val) or math.isinf(val):
            self.triggered = True
            self.reason = "loss_nan_inf_epoch{}_step{}".format(trainer.current_epoch, batch_idx)
            trainer.should_stop = True


def v1_group(Xi, Xh, gi, gh, pseeds=(42, 52, 62)):
    X = np.r_[Xi, Xh]
    d = np.r_[np.zeros(len(Xi)), np.ones(len(Xh))]
    g = np.r_[np.char.add("I", np.asarray(gi).astype(str)),
              np.char.add("M", np.asarray(gh).astype(str))]
    bals = []
    for s in pseeds:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(X[a], d[a])
        bals.append(float(balanced_accuracy_score(d[b], clf.predict(X[b]))))
    return {"bal_acc_mean": float(np.mean(bals)),
            "bal_acc_sd": float(np.std(bals, ddof=1)) if len(bals) > 1 else 0.0}


def build_streams(data, adv_mix):
    tf = build_train_transform_robust()
    adv_probs = None if adv_mix is None else R.class_weighted_probs(data["train"], adv_mix)
    streams = [
        R.Stream("isic_cls", data["train"], None, cls=True),
        R.Stream("isic_adv", data["train"], adv_probs, adv=0, ctx=0, src_labels=True),
        R.Stream("milk_adv", data["milk_adv"][["path"]], None, adv=1, ctx=1),
    ]
    return R.MultiStreamDataset(streams, tf), R.multistream_collate, [s.describe() for s in streams]


@torch.no_grad()
def extract_all(net, data, feat_dir, num_workers, device, light=False):
    feat_dir.mkdir(parents=True, exist_ok=True)
    tf = build_val_transform_robust()
    sets = {
        "isic_train": SkinDataset(data["train"], transform=tf),
        "isic_val": SkinDataset(data["val"], transform=tf),
        "isic_test": SkinDataset(data["test"], transform=tf),
        "milk_adv": SkinDataset(data["milk_adv"], transform=tf),
        "milk_heldout": SkinDataset(data["milk_heldout"], transform=tf),
        # alias for Scorer / run_metrics pad_* naming when needed
        "pad_heldout": SkinDataset(data["milk_heldout"], transform=tf),
        "pad_adv": SkinDataset(data["milk_adv"], transform=tf),
    }
    if not light:
        sets["pad_ufes20"] = SkinDataset(data["pad_ufes20"], transform=tf)
        sets["fitzpatrick17k"] = FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf)
        for n in R.P7A_NAMES:
            csv = R.P7A_SETS / "{}.csv".format(n)
            sets[n] = FolderJpegDataset(pd.read_csv(csv, keep_default_na=False).path.tolist(), tf)
    else:
        # stage-1 needs Fitz/MED/SD for headroom rule + milk_heldout for V1
        sets["fitzpatrick17k"] = FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf)
        for n in ("mednode", "sd198"):
            csv = R.P7A_SETS / "{}.csv".format(n)
            sets[n] = FolderJpegDataset(pd.read_csv(csv, keep_default_na=False).path.tolist(), tf)

    for name, ds in sets.items():
        out = feat_dir / "{}.npz".format(name)
        if out.exists():
            continue
        pack = x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers), device)
        # NaN guard on features
        for k in ("z_lesion_norm", "logits"):
            arr = pack[k]
            if not np.isfinite(arr).all():
                raise RuntimeError("non_finite_features:{}:{}".format(name, k))
        np.savez_compressed(out, **{k: pack[k] for k in R.FEAT_KEYS})
        print("features", name, pack["labels"].shape, flush=True)


def isic_test_lesion_ids(data):
    lid_map = C.isic_lesion_map()
    stems = data["test"]["path"].map(lambda p: Path(p).stem)
    return stems.map(lambda s: lid_map.get(s, s)).astype(str).to_numpy()


def train_one(lam, prior, seed, skip_done=True, batch_size=32, num_workers=4, light=False):
    tag = C.tag_for(lam, prior, seed)
    run_dir = C.OUT / "runs" / tag
    feat_dir = C.OUT / "features" / tag
    ckpt_dir = C.CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if skip_done and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("checkpoint_policy") == "last" and (feat_dir / "isic_test.npz").exists():
            print("skip_done", summary_path, flush=True)
            return

    seed_everything(seed)
    pl.seed_everything(seed, workers=True)
    data = C.load_data()
    recipe = R.CURRENT
    isic_mix = C.class_mix(data["train"].label_idx)
    milk_mix = C.class_mix(data["milk_adv"].label_idx)
    shift = C.PRIOR_SHIFT[prior]
    adv_mix = None
    if float(shift) < 1.0:
        adv_mix = C.dose_mix(shift, isic_mix, milk_mix)

    ds, collate, streams = build_streams(data, adv_mix)
    cfg = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "prior": prior, "shift": float(shift),
        "recipe": recipe, "streams": streams,
        "isic_mix": isic_mix.tolist(), "milk_adv_mix": milk_mix.tolist(),
        "adv_mix": None if adv_mix is None else np.asarray(adv_mix).tolist(),
        "tv_isic_milk_adv": C.tv(isic_mix, milk_mix),
        "architecture": "dual_encoder_csg_effb3",
        "design": "decisions/paperB_r14_pair3.md",
        "commit": C.git_head(), "light_extract": bool(light),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=float) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last = ckpt_dir / "last.ckpt"
    nan_cb = NanGuardCallback()
    degenerate = False
    degenerate_reason = None

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
            callbacks=[ckpt_cb, nan_cb],
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            default_root_dir=str(run_dir), log_every_n_steps=20)
        tl = p2.make_loader(ds, batch_size, num_workers, True, drop_last=True, collate=collate)
        vl = p2.make_loader(SkinDataset(data["val"], transform=build_val_transform_robust()),
                            batch_size, num_workers, False)
        print("train", tag, "device", device, "n=", len(ds), flush=True)
        try:
            trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl)
        except Exception as e:
            degenerate = True
            degenerate_reason = "train_exception:{}".format(e)
        if nan_cb.triggered:
            degenerate = True
            degenerate_reason = nan_cb.reason
        best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
        last = ckpt_dir / "last.ckpt"

    if degenerate or not last.exists():
        summary = {
            "tag": tag, "lam": float(lam), "seed": int(seed), "prior": prior,
            "degenerate": True, "degenerate_reason": degenerate_reason or "missing_ckpt",
            "checkpoint_policy": "last", "config": cfg, "commit": C.git_head(),
        }
        summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
        print("DEGENERATE", tag, degenerate_reason, flush=True)
        return

    primary = last if last.exists() else best
    net = R.load_net(primary, device)
    try:
        extract_all(net, data, feat_dir, num_workers, device, light=light)
    except RuntimeError as e:
        if "non_finite_features" in str(e):
            summary = {
                "tag": tag, "lam": float(lam), "seed": int(seed), "prior": prior,
                "degenerate": True, "degenerate_reason": str(e),
                "checkpoint_policy": "last", "primary_checkpoint": str(primary),
                "config": cfg, "commit": C.git_head(),
            }
            summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
            print("DEGENERATE", tag, e, flush=True)
            return
        raise

    Fs = R.load_feats(feat_dir)
    ood_names = ["milk_heldout", "fitzpatrick17k", "mednode", "sd198"]
    if not light:
        ood_names = list(C.OOD_ALL)
    # map milk_heldout into scorer via alias already saved as pad_heldout too
    m = R.run_metrics(Fs, ood_names=[n for n in ood_names if n in Fs or n == "milk_heldout"])
    # ensure milk_heldout ood block
    if "milk_heldout" in Fs and "milk_heldout" not in m.get("ood", {}):
        sc = R.Scorer(Fs["isic_train"])
        sid = sc.scores(Fs["isic_test"])
        so = sc.scores(Fs["milk_heldout"])
        blk = {d: R.auroc(sid[d], so[d]) for d in R.DETS}
        blk["n"] = int(len(so["MSP"]))
        nc = sc.nearest_class(Fs["milk_heldout"]["z_lesion_norm"])
        blk["nv_nearest_frac"] = float(np.mean(nc == R.NV))
        blk["nearest_class_share"] = {C.LABELS[c]: float(np.mean(nc == c)) for c in range(C.NCLS)}
        m.setdefault("ood", {})["milk_heldout"] = blk

    gi = isic_test_lesion_ids(data)
    gh = data["milk_heldout"]["lesion_id"].astype(str).to_numpy()
    v1 = v1_group(Fs["isic_test"]["z_lesion_norm"], Fs["milk_heldout"]["z_lesion_norm"], gi, gh)

    # predicted-majority share on milk_heldout
    pred = Fs["milk_heldout"]["logits"].argmax(1)
    maj = int(np.bincount(pred, minlength=C.NCLS).argmax())
    maj_share = float(np.mean(pred == maj))

    summary = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "prior": prior,
        "shift": float(shift), "degenerate": False,
        "checkpoint_policy": "last",
        "primary_checkpoint": str(primary),
        "best_checkpoint": str(best) if best and Path(best).exists() else None,
        "id_balanced_acc": float(m["id"]["bal_acc"]),
        "heldout_leakage_V1": float(v1["bal_acc_mean"]),
        "heldout_leakage_V1_sd": float(v1["bal_acc_sd"]),
        "milk_heldout_pred_majority_label": C.LABELS[maj],
        "milk_heldout_pred_majority_share": maj_share,
        "metrics": m, "config": cfg, "commit": C.git_head(),
        "tv_isic_vs_milk_adv": C.tv(isic_mix, milk_mix),
    }
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("saved", summary_path, "id_bal", summary["id_balanced_acc"],
          "V1", summary["heldout_leakage_V1"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--stage1", action="store_true", help="use stage-1 job list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--light", action="store_true", help="extract only sets needed for stage-1 gates")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tag", type=str, default=None)
    a = ap.parse_args()
    js = C.jobs_stage1() if a.stage1 else C.jobs_main()
    if a.list:
        for i, j in enumerate(js):
            print(i, j["tag"], j)
        print("n_jobs", len(js))
        return
    if a.tag:
        # find by tag
        matches = [j for j in (C.jobs_main() + C.jobs_stage1()) if j["tag"] == a.tag]
        if not matches:
            raise SystemExit("unknown tag {}".format(a.tag))
        j = matches[0]
    else:
        if a.task_id is None:
            raise SystemExit("need --task_id or --tag")
        j = js[a.task_id]
    train_one(j["lam"], j["prior"], j["seed"], skip_done=a.skip_done,
              batch_size=a.batch_size, num_workers=a.num_workers, light=a.light or a.stage1)


if __name__ == "__main__":
    main()
