#!/usr/bin/env python3
"""Paper B R5: single-encoder DANN (ResNet-50 or EfficientNet-B3) on the lesion-level ISIC split, with the P9
adversary toolbox (r5_models.SingleEncToolboxLit) and optional P11 penalties.

config 'baseline' = the Phase 15.1 / R1 item-5 recipe (current GRL schedule, one derm head, adv LR ×30, 40 epochs)
on the lesion-level split. Ladder configs are cumulative (each adds to the previous one):
  T1_ganin_warmup  Ganin & Lempitsky α schedule (α = 0 at step 0, γ = 10, linear progress) + 2 adversary warm-up epochs
  T2_bank_reinit   + 3 heads (linear = probe family, derm 2-layer, 3-layer), 2 inner head steps / batch,
                     one head re-initialised every 500 steps (round robin)
  T3_strong        + 5 heads (2 linear, derm, wide derm, 3-layer), 5 inner steps, head LR ×10 instead of ×30
  T4_long          + twice the recipe's epochs
Domain-balanced batches are already the default (CombinedTrainDataset pairs one ISIC and one PAD image per index).

P9 ladder (--ladder, λ = 2, seed 42): rungs baseline, T1 … T4 in order, one job at a time per backbone. Stop rule,
checked before training: skip if an earlier rung of the same backbone reached the dual-encoder leakage level (Table 1
probe ≤ 0.60) or if the P9 ladder GPU-hours already used (both backbones) + this run's estimate exceed 20.
The ladder is superseded by the 2026-10-15 amendment (decisions/paperB_r5_p9.md; scripts/r5_p9v2.py): configs
gw_lr<L>_d<D> = Ganin schedule + 2 warm-up epochs, one head of depth D, adversary LR ×L; --cap_gpu_h skips a run if
the GPU-h of the summaries already in <item>/<stage>/ + this run's estimate exceed the cap.

Outputs: results/paperB/r5/<item>/<stage>/<tag>/{config.json, adversary_log.jsonl, summary.json, scores.npz, features/}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_recipe  # noqa: E402

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402

import r5_models as M  # noqa: E402
from phase15_common import adversary_gate  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

_T1 = {"grl_schedule": "ganin", "adv_warmup_epochs": 2}
_T2 = dict(_T1, head_specs=[[1, 1.0], [2, 1.0], [3, 1.0]], adv_inner_steps=2, reinit_every_steps=500)
_T3 = dict(_T2, head_specs=[[1, 1.0], [1, 1.0], [2, 1.0], [2, 2.0], [3, 1.0]], adv_inner_steps=5,
           adv_lr_multiplier=10.0)
_T4 = dict(_T3, epoch_mult=2)
CONFIGS = {"baseline": {}, "T1_ganin_warmup": _T1, "T2_bank_reinit": _T2, "T3_strong": _T3, "T4_long": _T4}
LADDER = ("baseline", "T1_ganin_warmup", "T2_bank_reinit", "T3_strong", "T4_long")
STOP_LEAKAGE = 0.60
BUDGET_GPU_H = 20.0
EST_GPU_H = {"resnet50": 1.1, "effb3": 1.2, "dinov2_vits14": 1.6, "dinov2_vitb14": 2.4}
P9_LADDER = R.R5 / "p9" / "ladder"
SELECTED = R.R5 / "p9" / "SELECTED_{}{}.json"
SINGLE_BACKBONES = ("resnet50", "effb3", "dinov2_vits14", "dinov2_vitb14")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="p9")
    ap.add_argument("--stage", default="ladder", help="subfolder: ladder | stage2 | cells | erm | runs")
    ap.add_argument("--backbone", default="resnet50", choices=SINGLE_BACKBONES)
    ap.add_argument("--config", default="baseline",
                    help="toolbox config name, or 'selected' = P9's pick for this backbone (else baseline)")
    ap.add_argument("--lambda_adv", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--penalty", default="none", choices=("none", "mkmmd", "coral", "mmd_legacy"))
    ap.add_argument("--penalty_weight", type=float, default=0.0)
    ap.add_argument("--max_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=-1.0,
                    help="-1 = 0.1 for ViT backbones (DINOv2), 1.0 otherwise")
    ap.add_argument("--cap_gpu_h", type=float, default=0.0, help="stage-wide GPU-h cap (0 = none)")
    ap.add_argument("--ladder", action="store_true", help="apply the P9 stop rule / budget cap before training")
    ap.add_argument("--results_dir", default="",
                    help="If set, write under this dir/<stage>/<tag>. Else r5/<item>/<stage>.")
    ap.add_argument("--tag_prefix", default="", help="e.g. r7_t1_")
    ap.add_argument("--ckpt_subdir", default="", help="checkpoints/<subdir>/<tag>; empty = r5_<item>/<stage>")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    r5_recipe.add_args(ap)
    return ap.parse_args()


def gw_config(name):
    """gw_lr<L>_d<D> -> Ganin schedule, 2 adversary warm-up epochs, one head of depth D, adversary LR ×L."""
    lr, d = name[len("gw_lr"):].split("_d")
    return {"grl_schedule": "ganin", "adv_warmup_epochs": 2, "head_specs": [[int(d), 1.0]],
            "adv_lr_multiplier": float(lr)}


def resolve_config(name, backbone, recipe):
    if name.startswith("gw_lr"):
        return name, gw_config(name), "P9 v2 grid"
    if name == "selected":
        p = Path(str(SELECTED).format(backbone, r5_recipe.tag(recipe)))
        if p.exists():
            sel = json.loads(p.read_text())
            return sel["config"], dict(CONFIGS[sel["config"]]), str(p)
        return "baseline", {}, "P9 selected none for {} -> baseline".format(backbone)
    return name, dict(CONFIGS[name]), "named"


def ladder_state(backbone, recipe):
    used, reached = 0.0, []
    suf = r5_recipe.tag(recipe)
    for f in P9_LADDER.glob("*/summary.json"):
        s = json.loads(f.read_text())
        if (suf and not s["tag"].endswith(suf)) or (not suf and "_rec" in s["tag"]):
            continue
        used += float(s.get("gpu_hours", 0.0))
        if s.get("backbone") == backbone and s["leakage"]["test_vs_pad_full"]["bal_acc_mean"] <= STOP_LEAKAGE:
            reached.append(s["tag"])
    return used, reached


def ckpt_epoch(ckpt):
    return int(torch.load(str(ckpt), map_location="cpu", weights_only=False)["epoch"])


def main():
    a = parse()
    rec, src = r5_recipe.resolve(a.recipe)
    cname, tb, csrc = resolve_config(a.config, a.backbone, a.recipe)
    epochs = a.max_epochs or int(rec["max_epochs"]) * int(tb.pop("epoch_mult", 1))
    tb["adv_lr_multiplier"] = float(tb.get("adv_lr_multiplier", rec["adv_lr_multiplier"]))
    tag = "{}single_{}_{}_ladv{}_s{}".format(a.tag_prefix, a.backbone, cname, R.lam_tag(a.lambda_adv), a.seed)
    if a.penalty != "none":
        tag += "_{}{}".format(a.penalty, R.lam_tag(a.penalty_weight))
    if a.max_epochs:
        tag += "_ep{}".format(a.max_epochs)
    tag += r5_recipe.tag(a.recipe)
    base = Path(a.results_dir) if a.results_dir else (R.R5 / a.item)
    if a.results_dir and not Path(a.results_dir).is_absolute():
        base = R.PAPERB / a.results_dir
    run_dir = base / a.stage / tag
    ck_name = a.ckpt_subdir or "r5_{}".format(a.item)
    ckpt_dir = R.PAPERB / "checkpoints" / ck_name / a.stage / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if a.skip_done and (run_dir / "summary.json").exists() and not a.dry_run:
        print("skip_done", tag)
        return
    if a.backbone in M.BACKBONES:
        ok, wpath = M.weights_available(a.backbone)
        if not ok:
            raise SystemExit("pretrained weights for {} not in cache ({})".format(a.backbone, wpath))
    else:
        wpath = None
    if a.cap_gpu_h:
        used = sum(float(json.loads(f.read_text()).get("gpu_hours", 0.0))
                   for f in (base / a.stage).glob("*/summary.json"))
        est = EST_GPU_H.get(a.backbone, 1.6) * epochs / 40.0
        if used + est > a.cap_gpu_h:
            why = {"tag": tag, "skipped": True, "stage_gpu_h_used": used, "this_run_est": est, "cap": a.cap_gpu_h}
            (run_dir / "skipped.json").write_text(json.dumps(why, indent=2) + "\n")
            print("cap stop", why, flush=True)
            return
    if a.ladder:
        used, reached = ladder_state(a.backbone, a.recipe)
        est = EST_GPU_H.get(a.backbone, 1.6) * epochs / 40.0
        if reached or used + est > BUDGET_GPU_H:
            why = {"tag": tag, "skipped": True, "reached_by": reached, "ladder_gpu_h_used": used, "this_run_est": est,
                   "cap": BUDGET_GPU_H, "stop_leakage": STOP_LEAKAGE}
            (run_dir / "skipped.json").write_text(json.dumps(why, indent=2) + "\n")
            print("ladder stop", why, flush=True)
            return
    seed_everything(a.seed)
    pl.seed_everything(a.seed, workers=True)
    D = R.isic_lesion_data()
    cw = R.class_weights(D["train"].label_idx, 8) if rec["class_balance"] == "weighted_ce" else None
    val_ds = SkinDataset(D["val"], transform=build_val_transform_robust())
    bs = int(rec["batch_size"])
    total = epochs * (len(D["train"]) // bs)
    bb_mult = a.backbone_lr_mult if a.backbone_lr_mult > 0 else (0.1 if a.backbone in M.VIT_LIKE else 1.0)
    kw = dict(backbone=a.backbone, n_classes=8, n_domains=2, lr=float(rec["lr"]), weight_decay=float(rec["weight_decay"]),
              cls_lr_multiplier=float(rec["cls_lr_multiplier"]), lambda_adv=float(a.lambda_adv), toolbox=tb,
              cls_weight=cw, penalty=a.penalty, penalty_weight=float(a.penalty_weight), total_steps=total,
              backbone_lr_mult=bb_mult)
    cfg = {"tag": tag, "item": a.item, "stage": a.stage, "architecture": "single_encoder_dann_" + a.backbone,
           "split": "lesion-level (R2 item 4)", "config": cname, "config_source": csrc,
           "toolbox": dict(M.TOOLBOX_DEFAULT, **tb), "max_epochs": epochs, "model_kwargs": kw,
           "weights": wpath, "backbone_lr_mult": bb_mult,
           "erm_note": ("λ_adv=0: no domain-head gradient reaches the encoder" if float(a.lambda_adv) == 0 else None),
           "scoring_space": "z_norm (BN of pooled backbone features)",
           **r5_recipe.describe(rec, src, not_applicable=r5_recipe.DUAL_ONLY)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if a.dry_run:
        print("dry_run", tag, "train", len(D["train"]), "steps", total, "bb_lr_mult", bb_mult)
        return

    t0 = time.time()
    adv_log = run_dir / "adversary_log.jsonl"
    if adv_log.exists() and not a.resume:
        adv_log.write_text("")
    lit = M.SingleEncToolboxLit(log_path=str(adv_log), **kw)
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc",
                                           mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                         callbacks=[ckpt_cb], precision="16-mixed" if torch.cuda.is_available() else "32-true",
                         default_root_dir=str(run_dir), log_every_n_steps=20)
    tl, _mode = R.isic_train_loader(D, rec["class_balance"], bs, a.num_workers, a.seed)
    vl = R.p2.make_loader(val_ds, bs, a.num_workers, False)
    last = ckpt_dir / "last.ckpt"
    trainer.fit(lit, train_dataloaders=tl, val_dataloaders=vl,
                ckpt_path=str(last) if a.resume and last.exists() else None)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(ckpt, out_dir):
        ev = M.SingleEncToolboxLit(pretrained=False, **kw)
        ev.load_state_dict(torch.load(str(ckpt), map_location="cpu")["state_dict"])
        net = ev.net.to(device).eval()
        return R.eval_isic(lambda x: net(x), D, device, out_dir, workers=a.num_workers)

    # Primary = last.ckpt (decisions/paperB_r5_p8.md, deviation 2026-10-15 21:57); best-val is secondary.
    summ = evaluate(last, run_dir)
    sec = evaluate(best, run_dir / "bestval")
    summ["checkpoint_policy"] = "last"
    summ["primary_checkpoint"], summ["primary_epoch"] = str(last), ckpt_epoch(last)
    summ["best_epoch"] = ckpt_epoch(best)
    summ["secondary_bestval"] = {"checkpoint": str(best), "epoch": summ["best_epoch"], "out_dir": str(run_dir / "bestval"),
                                 **{k: sec[k] for k in ("id_balanced_acc", "leakage", "ood", "xfer_pad_heldout_6class") if k in sec}}
    rows = [json.loads(l) for l in adv_log.read_text().splitlines() if l.strip()] if adv_log.exists() else []
    summ.update({"tag": tag, "seed": a.seed, "lambda_adv": a.lambda_adv, "backbone": a.backbone, "config": cname,
                 "penalty": a.penalty, "penalty_weight": a.penalty_weight, "max_epochs": epochs,
                 "best_checkpoint": str(best), "recipe": a.recipe,
                 "gate": adversary_gate(rows, M.LN2, a.lambda_adv,
                                        leakage_bal=summ["leakage"]["test_vs_pad_full"]["bal_acc_mean"], floor=0.5),
                 "gpu_hours": (time.time() - t0) / 3600.0, "commit": R.git_head()})
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "leak", summ["leakage"]["test_vs_pad_full"]["bal_acc_mean"],
          "maha pad_heldout", summ["ood"]["pad_heldout"]["mahalanobis_classcond"], flush=True)


if __name__ == "__main__":
    main()
