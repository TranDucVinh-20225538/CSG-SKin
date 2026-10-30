#!/usr/bin/env python3
"""Paper B R5: dual-encoder CSG on the lesion-level ISIC split, any backbone (Q3) or a moment-matching penalty on
z_lesion_norm instead of / next to the GRL adversary (P11).

Everything not named by a flag is the R2 item-4 run: lesion-level split (StratifiedGroupKFold by lesion_id, seed 42),
PAD patient split pad_adv / pad_heldout, CombinedTrainDataset (one ISIC + one PAD image per index, so every batch is
domain-balanced), robust train transform, lesion branch = grey copy of the input, checkpoint = best val/acc
(MulticlassAccuracy, macro), 16-mixed precision. Recipe values come from --recipe (r5_recipe.py).

Outputs: results/paperB/r5/<item>/runs/<tag>/{config.json, adversary_log.jsonl, summary.json, scores.npz, features/}
Usage: train_r5_dual_isic.py --item q3 --backbone convnext_tiny --lambda_adv 2 --seed 42 [--recipe current]
       train_r5_dual_isic.py --item p11 --backbone effb3 --lambda_adv 0 --penalty mkmmd --penalty_weight 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402  (installs the transformers stub via train_phase3_sweep)
import r5_recipe  # noqa: E402

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402

import r5_models as M  # noqa: E402
from phase15_common import adversary_gate  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", required=True, help="output folder name under results base (q3, p11, t1, t2, ...)")
    ap.add_argument("--backbone", default="effb3", choices=M.BACKBONES)
    ap.add_argument("--lambda_adv", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--penalty", default="none", choices=("none", "mkmmd", "coral", "mmd_legacy"))
    ap.add_argument("--penalty_weight", type=float, default=0.0)
    ap.add_argument("--max_epochs", type=int, default=0, help="0 = recipe value; >0 overrides (P11 longer schedule)")
    ap.add_argument("--backbone_lr_mult", type=float, default=-1.0,
                    help="-1 = 0.1 for ViT backbones (DINOv2 / timm ViT), 1.0 otherwise")
    ap.add_argument("--pad_adv_frac", type=float, default=1.0,
                    help="patient-stratified fraction of pad_adv (1.0 = Table 1; feeds adv + context branches)")
    ap.add_argument("--pad_adv_subsample_seed", type=int, default=0,
                    help="fixed subsample seed (one draw per frac; default 0)")
    ap.add_argument("--results_dir", default="",
                    help="If set, write under this dir/runs/<tag> (e.g. results/paperB/r7/t1). Else r5/<item>/runs.")
    ap.add_argument("--tag_prefix", default="", help="e.g. r7_t1_")
    ap.add_argument("--ckpt_subdir", default="", help="checkpoints/<subdir>/<tag>; empty = r5_<item>")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    r5_recipe.add_args(ap)
    return ap.parse_args()


def run_tag(a, epochs_override):
    t = "{}dual_{}_ladv{}_s{}".format(a.tag_prefix, a.backbone, R.lam_tag(a.lambda_adv), a.seed)
    if abs(float(a.pad_adv_frac) - 1.0) > 1e-12:
        t += "_pad{}".format(R.lam_tag(a.pad_adv_frac))
    if a.penalty != "none":
        t += "_{}{}".format(a.penalty, R.lam_tag(a.penalty_weight))
    if epochs_override:
        t += "_ep{}".format(epochs_override)
    return t + r5_recipe.tag(a.recipe)


def main():
    a = parse()
    rec, src = r5_recipe.resolve(a.recipe)
    epochs = a.max_epochs or int(rec["max_epochs"])
    tag = run_tag(a, a.max_epochs)
    base = Path(a.results_dir) if a.results_dir else (R.R5 / a.item)
    if a.results_dir and not Path(a.results_dir).is_absolute():
        base = R.PAPERB / a.results_dir
    run_dir = base / "runs" / tag
    ck_name = a.ckpt_subdir or "r5_{}".format(a.item)
    ckpt_dir = R.PAPERB / "checkpoints" / ck_name / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if a.skip_done and (run_dir / "summary.json").exists() and not a.dry_run:
        print("skip_done", tag)
        return
    ok, wpath = M.weights_available(a.backbone)
    if not ok:
        raise SystemExit("pretrained weights for {} not in cache ({}); see decisions/paperB_r5_q3.md".format(
            a.backbone, wpath))
    seed_everything(a.seed)
    pl.seed_everything(a.seed, workers=True)
    D = R.isic_lesion_data()
    D["pad_adv"], pad_meta = R.subsample_pad_adv(D["pad_adv"], a.pad_adv_frac, a.pad_adv_subsample_seed)
    bb_mult = a.backbone_lr_mult if a.backbone_lr_mult > 0 else (0.1 if a.backbone in M.VIT_LIKE else 1.0)
    cw = R.class_weights(D["train"].label_idx, 8) if rec["class_balance"] == "weighted_ce" else None
    kw = dict(backbone=a.backbone, lesion_classes=8, domain_classes=2,
              lesion_latent_dim=int(rec["lesion_latent_dim"]), context_latent_dim=int(rec["context_latent_dim"]),
              learning_rate=float(rec["lr"]), weight_decay=float(rec["weight_decay"]),
              lambda_ctx=float(rec["lambda_ctx"]), lambda_adv=float(a.lambda_adv), lambda_orth=float(rec["lambda_orth"]),
              adv_lr_multiplier=float(rec["adv_lr_multiplier"]), lesion_cls_lr_multiplier=float(rec["cls_lr_multiplier"]),
              cls_weight=cw, backbone_lr_mult=bb_mult, penalty=a.penalty, penalty_weight=float(a.penalty_weight))
    cfg = {"tag": tag, "item": a.item, "architecture": "dual_encoder_csg", "split": "lesion-level (R2 item 4)",
           "seed": a.seed, "max_epochs": epochs, "model_kwargs": kw, "weights": wpath,
           "pad_adv_frac": float(a.pad_adv_frac), "pad_adv_subsample": pad_meta,
           "pad_adv_feeds": "adversarial and context branches; pad_heldout unchanged",
           "scoring_space": "z_lesion_norm", **r5_recipe.describe(rec, src)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    val_ds = SkinDataset(D["val"], transform=build_val_transform_robust())
    if a.dry_run:
        print("dry_run", tag, "train", len(D["train"]), "val", len(val_ds), "pad_adv", len(D["pad_adv"]),
              "pad_hold", len(D["pad_hold"]), "pad_meta", pad_meta.get("class_counts"),
              "extra", {k: len(v) for k, v in D["extra"].items()})
        return

    t0 = time.time()
    adv_log = run_dir / "adversary_log.jsonl"
    if adv_log.exists() and not a.resume:
        adv_log.write_text("")
    lit = M.CSGR5Lightning(adv_log_path=str(adv_log), **kw)
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc",
                                           mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                         callbacks=[ckpt_cb], precision="16-mixed" if torch.cuda.is_available() else "32-true",
                         default_root_dir=str(run_dir), log_every_n_steps=20)
    tl, _mode = R.isic_train_loader(D, rec["class_balance"], int(rec["batch_size"]), a.num_workers, a.seed)
    vl = R.p2.make_loader(val_ds, int(rec["batch_size"]), a.num_workers, False)
    last = ckpt_dir / "last.ckpt"
    trainer.fit(lit, train_dataloaders=tl, val_dataloaders=vl,
                ckpt_path=str(last) if a.resume and last.exists() else None)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
    t_train = time.time() - t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(ckpt, out_dir):
        net = M.load_dual_net(ckpt, kw, device)

        def forward(x):
            out = net(x, x_lesion=None, return_latents=True)
            return out[0], out[3]

        return R.eval_isic(forward, D, device, out_dir, workers=a.num_workers)

    # Primary = last.ckpt (decisions/paperB_r5_p8.md 21:57; P11 amendment 2026-10-16); best-val secondary.
    summ = evaluate(last, run_dir)
    sec = evaluate(best, run_dir / "bestval")
    rows = [json.loads(l) for l in adv_log.read_text().splitlines() if l.strip()] if adv_log.exists() else []
    ep_last = int(torch.load(str(last), map_location="cpu", weights_only=False)["epoch"])
    ep_best = int(torch.load(str(best), map_location="cpu", weights_only=False)["epoch"])
    summ.update({"tag": tag, "seed": a.seed, "lambda_adv": a.lambda_adv, "backbone": a.backbone, "penalty": a.penalty,
                 "penalty_weight": a.penalty_weight, "max_epochs": epochs, "checkpoint_policy": "last",
                 "pad_adv_frac": float(a.pad_adv_frac), "pad_adv_subsample": pad_meta,
                 "primary_checkpoint": str(last), "primary_epoch": ep_last, "best_checkpoint": str(best),
                 "best_epoch": ep_best,
                 "secondary_bestval": {"checkpoint": str(best), "epoch": ep_best, "out_dir": str(run_dir / "bestval"),
                                       **{k: sec[k] for k in ("id_balanced_acc", "leakage", "ood", "xfer_pad_heldout_6class",
                                                              "val_balanced_acc") if k in sec}},
                 "recipe": a.recipe, "gate": adversary_gate(rows, M.LN2, a.lambda_adv,
                                                            leakage_bal=summ["leakage"]["test_vs_pad_full"]["bal_acc_mean"],
                                                            floor=0.5),
                 "wall_seconds_train": t_train, "wall_seconds_total": time.time() - t0,
                 "gpu_hours": (time.time() - t0) / 3600.0, "commit": R.git_head()})
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "leak", summ["leakage"]["test_vs_pad_full"]["bal_acc_mean"],
          "maha pad_heldout", summ["ood"]["pad_heldout"]["mahalanobis_classcond"], flush=True)


if __name__ == "__main__":
    main()
