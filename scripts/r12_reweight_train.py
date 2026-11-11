#!/usr/bin/env python3
"""Paper B R12 Block A/B training — fixed-image adv reweight (+ optional ERM init).

Decisions: decisions/paperB_r12_reweight.md
Does NOT call R.run() (that hardcodes oracle_w from pad for iwdan_oracle).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
# r6_common installs the transformers stub before Lightning/torchmetrics import.
import r6_common as R  # noqa: E402
import r12_reweight_common as C  # noqa: E402

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning import seed_everything as pl_seed  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

ERM_CK = R.CKPT / "r5_p8" / "erm_ref"


def all_jobs():
    """Train jobs only (no reference rows). Order: Block A baselines, A λ×prior, Block B."""
    out = []
    for j in C.jobs_block_a():
        out.append({**j, "init": "scratch"})
    for j in C.jobs_block_b():
        out.append(j)
    return out


def erm_ckpt(seed):
    return ERM_CK / "single_dann_ladv0_s{}".format(int(seed)) / "last.ckpt"


def load_erm_into_dual(model, seed):
    """Copy ERM EffB3 backbone into both dual backbones; leave heads random.
    Returns audit dict. Raises if shapes mismatch.
    """
    path = erm_ckpt(seed)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]
    dsd = model.state_dict()
    mapped, skipped = {}, []
    for k, v in raw.items():
        if not k.startswith("net.backbone."):
            skipped.append(k)
            continue
        suffix = k[len("net.backbone."):]  # features.*
        for branch in ("model.lesion_backbone.", "model.context_backbone."):
            dk = branch + suffix
            if dk not in dsd:
                raise KeyError("dual missing {}".format(dk))
            if tuple(dsd[dk].shape) != tuple(v.shape):
                raise ValueError("shape mismatch {} {} vs {}".format(dk, dsd[dk].shape, v.shape))
            mapped[dk] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # expected: heads / buffers not in mapped
    return {
        "erm_path": str(path),
        "n_erm_tensors": len(raw),
        "n_mapped": len(mapped),
        "n_skipped_erm_non_backbone": len(skipped),
        "load_missing_count": len(missing),
        "load_unexpected_count": len(unexpected),
        "note": "both lesion_backbone and context_backbone from same ERM; heads random",
    }


def requested_mix(data, prior):
    isic = R.class_mix(data["train"].label_idx)
    pad = R.class_mix(data["pad_adv"].label_idx)
    shift = C.PRIOR_SHIFT[prior]
    return R.dose_mix(shift, isic, pad), isic, pad


def run_one(job, args):
    tag = job["tag"]
    out_root = C.OUT
    run_dir = out_root / "runs" / tag
    feat_dir = out_root / "features" / tag
    ckpt_dir = C.CKPT / tag
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json", flush=True)
        return

    seed = int(job["seed"])
    lam = float(job["lam"])
    prior = job["prior"]
    init = job.get("init", "scratch")
    recipe = R.CURRENT

    seed_everything(seed)
    pl_seed(seed, workers=True)
    for d in (run_dir, ckpt_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = R.load_data()
    pi_req, pi_src, pi_pad = requested_mix(data, prior)
    w, invalid, mean_w = C.class_weights(pi_req, pi_src)
    if invalid:
        cfg = {"tag": tag, "status": "invalid", "invalid_classes": invalid, "prior": prior}
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (run_dir / "summary.json").write_text(json.dumps({**cfg, "metrics": None}, indent=2) + "\n")
        print("INVALID", tag, invalid, flush=True)
        return

    # λ=0: plain DANN path — no IW, no prior dependence.
    # λ>0: iwdan_oracle + iw_cls=False = DANN with fixed source-adv domain-loss weights only.
    if lam == 0.0:
        align, iw_cls, oracle_w = "dann", False, None
    else:
        align, iw_cls, oracle_w = "iwdan_oracle", False, w.tolist()

    # Fixed image schedules: adv_mix=None (no class-prior resampling).
    ds, collate, streams = R.build_train_set("three", recipe, data, adv_mix=None)
    cfg = {
        "tag": tag, "block": job["block"], "role": job.get("role"), "init": init,
        "recipe": recipe, "lambda_adv": lam, "seed": seed, "prior": prior,
        "prior_shift": C.PRIOR_SHIFT[prior],
        "pi_requested": pi_req.tolist(), "pi_source": pi_src.tolist(), "pi_pad_adv": pi_pad.tolist(),
        "oracle_w": oracle_w, "mean_stream_w": mean_w, "align": align, "iw_cls": iw_cls,
        "adv_mix": None, "mode": "three", "ckpt_policy": R.CKPT_POLICY,
        "streams": streams, "commit": C.git_head(),
        "disclosure": "π_pad_adv from pad_adv labels at start only; never in task loss; not label-free",
        "protocol": "fixed-image adversary reweight (≠ R1 resampling; ≠ R2 IWDAN)",
        "out_root": str(out_root), "ckpt_dir": str(ckpt_dir),
        "n": {k: int(len(v)) for k, v in data.items()},
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.dry_run:
        print("dry_run", tag, "n=", len(ds), "align=", align, "w=", None if oracle_w is None else np.round(w, 3).tolist(),
              flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    best = R.find_best(ckpt_dir) if R.trained_to_end(ckpt_dir, recipe) else None
    erm_audit = None
    if best is not None:
        print("checkpoint exists, evaluation only", best, flush=True)
    else:
        iw_log = str(run_dir / "iw_log.jsonl") if align.startswith("iw") else None
        if iw_log and not args.resume and Path(iw_log).exists():
            Path(iw_log).write_text("")
        model = R.make_model(recipe, lam, align=align, class_weights=None, iw_log=iw_log,
                             iw_cls=iw_cls, oracle_w=oracle_w,
                             adv_lr_multiplier=30.0, p_source=pi_src.tolist())
        if init == "erm":
            erm_audit = load_erm_into_dual(model, seed)
            (run_dir / "erm_init_audit.json").write_text(json.dumps(erm_audit, indent=2) + "\n")
            print("ERM_INIT", erm_audit["n_mapped"], "tensors", flush=True)
        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc",
            mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
        trainer = pl.Trainer(
            max_epochs=int(recipe["max_epochs"]),
            accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
            callbacks=[ckpt_cb],
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            default_root_dir=str(run_dir), log_every_n_steps=20,
            deterministic=False)
        last = ckpt_dir / "last.ckpt"
        tl = p2.make_loader(ds, args.batch_size, args.num_workers, True, drop_last=True, collate=collate)
        vl = p2.make_loader(SkinDataset(data["val"], transform=build_val_transform_robust()),
                            args.batch_size, args.num_workers, False)
        trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl,
                    ckpt_path=str(last) if args.resume and last.exists() else None)
        best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last

    last = ckpt_dir / "last.ckpt"
    primary = last if R.CKPT_POLICY == "last" else best
    if not primary.exists():
        raise FileNotFoundError(primary)
    print("primary", primary, "epoch", R.ckpt_epoch(primary), flush=True)
    net = R.load_net(primary, device)
    R.extract(net, data, feat_dir, "full", args.num_workers, device)
    m = R.run_metrics(R.load_feats(feat_dir))
    fb = out_root / "features_bestval" / tag
    net = R.load_net(best, device)
    R.extract(net, data, fb, "full", args.num_workers, device)
    secondary = {"checkpoint": str(best), "epoch": R.ckpt_epoch(best),
                 "metrics": R.run_metrics(R.load_feats(fb)), "features": str(fb)}
    wall = time.time() - t0
    summary = {
        "tag": tag, "block": job["block"], "status": "ok", "init": init, "prior": prior,
        "lambda_adv": lam, "seed": seed, "align": align, "iw_cls": iw_cls,
        "adv_mix": None, "oracle_w": oracle_w, "recipe": recipe,
        "checkpoint_policy": R.CKPT_POLICY, "primary_checkpoint": str(primary),
        "primary_epoch": R.ckpt_epoch(primary), "best_checkpoint": str(best),
        "best_epoch": R.ckpt_epoch(best), "secondary_bestval": secondary,
        "features": str(feat_dir), "metrics": m, "erm_init": erm_audit,
        "wall_seconds": wall, "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("saved", run_dir / "summary.json", "wall_s={:.1f}".format(wall), flush=True)


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--block", choices=("A", "B", "all"), default="all")
    a = p.parse_args()
    js = all_jobs()
    if a.block == "A":
        js = [j for j in js if j["block"] == "A"]
    elif a.block == "B":
        js = [j for j in js if j["block"] == "B"]
    if a.list:
        return R.print_jobs([{**j, "recipe": R.CURRENT, "mode": "three",
                              "align": "dann" if j["lam"] == 0 else "iwdan_oracle"} for j in js])
    if a.task_id is None:
        raise SystemExit("need --task_id or --list")
    run_one(js[a.task_id], a)


if __name__ == "__main__":
    main()
