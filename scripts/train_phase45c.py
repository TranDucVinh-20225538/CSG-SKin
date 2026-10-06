#!/usr/bin/env python3
"""Phase 4.5c: hold {DF, VASC, SCC} together; score each held-out class separately.

3 models × 3 seeds = 9 GPU runs. Submit only after Phase 3 (job 60512) finishes.
Near/far index is computed from an 8-class baseline_soft on ISIC train only
(held-out classes have centroids there; 5-class 4.5c training cannot supply them).
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = object
    _tf.AutoTokenizer = object
    _cfg = types.ModuleType("transformers.configuration_utils")

    class _PC:
        def __eq__(self, other):
            return False

    _cfg.PretrainedConfig = _PC
    _tf.configuration_utils = _cfg
    sys.modules["transformers"] = _tf
    sys.modules["transformers.configuration_utils"] = _cfg

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase4_semantic_ood as p4
from eval_ood_dual_branch import find_policy_ckpt, method_ckpt_dir
from eval_phase45 import detectors, fpr95, score_block, spaces_of

from src.datasets.skin_dataset import SkinDataset, build_train_transform_robust, build_val_transform_robust
from src.models.baseline import BaselineResNet50
from src.models.csg_lightning import CSGLiteLightning
from src.models.effb3_single import EffB3SingleLightning
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
HOLD = ["DF", "VASC", "SCC"]
KEEP = ["MEL", "NV", "BCC", "AK", "BKL"]
SEEDS = (42, 52, 62)
METHODS = ("baseline", "effb3", "runB_orth1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_root", type=Path, default=PAPERB / "results" / "paperB" / "phase4_5" / "hold3")
    p.add_argument("--ckpt_root", type=Path, default=PAPERB / "checkpoints" / "phase45c")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def near_far_from_8class_baseline(isic_train, keep, hold, device):
    """Centroids on ISIC train, 8-class baseline_soft seed 42. Training data only."""
    ckpt = find_policy_ckpt(method_ckpt_dir("baseline_soft", 42))
    lit = BaselineResNet50.load_from_checkpoint(str(ckpt), strict=False)
    net = lit.net.to(device).eval()
    eval_tf = p4.baseline_eval_tf()
    ds = SkinDataset(isic_train, transform=eval_tf)
    loader = p4.make_loader(ds, 64, 4, False)
    pack = p4.collect("baseline", net, loader, device)
    z, y = pack["backbone_raw"], pack["labels"]
    # y is 8-class original label_idx
    cents = {}
    for c in range(8):
        m = y == c
        if m.any():
            cents[c] = z[m].mean(axis=0)
    names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    keep_i = [names.index(k) for k in keep]
    out = {}
    for h in hold:
        hi = names.index(h)
        if hi not in cents:
            continue
        ch = cents[hi]
        chn = ch / (np.linalg.norm(ch) + 1e-12)
        sims = []
        for k in keep_i:
            ck = cents[k]
            ckn = ck / (np.linalg.norm(ck) + 1e-12)
            sims.append(float(chn @ ckn))
        out[h] = {
            "nearest_keep": keep[int(np.argmax(sims))],
            "cosine_to_nearest_keep": float(max(sims)),
            "cosine_to_all_keep": {keep[i]: sims[i] for i in range(len(keep))},
            "note": "8-class baseline_soft s42, ISIC train only; higher cosine = nearer (harder near-OOD)",
        }
    return out, str(ckpt)


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "4c_{}_s{}".format(args.method, args.seed)
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json")
        return

    isic = p4.load_isic()
    train8, val8, test8 = p4.isic_splits(isic)
    train, mapping = p4.remap_kept(train8, KEEP)
    val, _ = p4.remap_kept(val8, KEEP)
    id_test, _ = p4.remap_kept(test8, KEEP)
    ood = isic[isic["label"].isin(HOLD)].copy().reset_index(drop=True)
    n_classes = 5

    split_note = {
        "config": "4c",
        "hold_out": HOLD,
        "keep": KEEP,
        "label_mapping": mapping,
        "n_train": int(len(train)),
        "n_id": int(len(id_test)),
        "n_ood": int(len(ood)),
        "pad_involved": False,
        "ood_scored_separately": HOLD,
    }
    (run_dir / "split.json").write_text(json.dumps(split_note, indent=2) + "\n")

    if args.method == "baseline":
        train_tf, eval_tf = p4.baseline_train_tf(), p4.baseline_eval_tf()
        batch_size, lr = 96, 2e-4
        train_ds = SkinDataset(train, transform=train_tf)
    elif args.method == "effb3":
        train_tf, eval_tf = build_train_transform_robust(), build_val_transform_robust()
        batch_size, lr = 32, 1e-4
        train_ds = SkinDataset(train, transform=train_tf)
    else:
        train_tf, eval_tf = build_train_transform_robust(), build_val_transform_robust()
        batch_size, lr = 32, 1e-4
        train_ds = p4.DomainConstantDataset(train, transform=train_tf)

    if args.dry_run:
        print("dry_run", run_name, "train", len(train_ds), "id", len(id_test), "ood", len(ood), "keep", KEEP)
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **split_note}, indent=2) + "\n")
        return

    if args.method == "baseline":
        lit = BaselineResNet50(
            num_classes=n_classes, learning_rate=lr, weight_decay=1e-4, label_smoothing=0.03, warmup_epochs=5, max_epochs=args.max_epochs
        )
    elif args.method == "effb3":
        lit = EffB3SingleLightning(num_classes=n_classes, latent_dim=16, learning_rate=lr, weight_decay=1e-4)
    else:
        lit = CSGLiteLightning(
            lesion_classes=n_classes,
            domain_classes=2,
            lesion_latent_dim=16,
            context_latent_dim=64,
            learning_rate=lr,
            weight_decay=1e-4,
            lambda_ctx=1.0,
            lambda_adv=2.0,
            lambda_orth=1.0,
            lambda_supcon=0.0,
            backbone_variant="b3",
            pretrained=True,
        )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    use_gpu = (not args.cpu) and torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        precision="16-mixed" if use_gpu else "32-true",
        default_root_dir=str(run_dir),
        log_every_n_steps=20,
    )
    ckpt_path = None
    last = ckpt_dir / "last.ckpt"
    if args.resume and last.exists():
        ckpt_path = str(last)
    val_ds = SkinDataset(val, transform=eval_tf)
    train_loader = p4.make_loader(train_ds, batch_size, args.num_workers, True, drop_last=True)
    val_loader = p4.make_loader(val_ds, batch_size, args.num_workers, False)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last

    if args.method == "baseline":
        lit = BaselineResNet50.load_from_checkpoint(str(best), strict=False)
        net = lit.net
    elif args.method == "effb3":
        lit = EffB3SingleLightning.load_from_checkpoint(str(best), strict=False)
        net = lit.net
    else:
        lit = CSGLiteLightning.load_from_checkpoint(str(best), strict=False)
        net = lit.model
    device = torch.device("cuda" if use_gpu else "cpu")
    net = net.to(device).eval()

    def loader(frame):
        return p4.make_loader(SkinDataset(frame, transform=eval_tf), 64, args.num_workers, False)

    ptr = p4.collect(args.method, net, loader(train), device)
    pid = p4.collect(args.method, net, loader(id_test), device)
    pood = p4.collect(args.method, net, loader(ood), device)
    ood_names = ood["label"].to_numpy()
    rng = np.random.default_rng(args.seed)

    spaces = {}
    for space, ztr in spaces_of(args.method, ptr).items():
        dets = detectors(
            ztr,
            ptr["labels"],
            spaces_of(args.method, pid)[space],
            spaces_of(args.method, pood)[space],
            pid["logits"],
            pood["logits"],
            n_classes,
        )
        block = {}
        for det, (sid, sood) in dets.items():
            entry = {"pooled_three": score_block(sid, sood, 1000, rng)}
            for cls in HOLD:
                m = ood_names == cls
                entry[cls] = score_block(sid, sood[m], 1000, rng)
            block[det] = entry
        spaces[space] = block

    nf, nf_ckpt = near_far_from_8class_baseline(train8, KEEP, HOLD, device)
    pred = pid["logits"].argmax(1)
    summary = {
        "run_name": run_name,
        "config": "4c",
        "method": args.method,
        "seed": args.seed,
        "best_checkpoint": str(best),
        "id_acc": float(accuracy_score(pid["labels"], pred)),
        "id_balanced_acc": float(balanced_accuracy_score(pid["labels"], pred)),
        "spaces": spaces,
        "near_far_index": nf,
        "near_far_source_ckpt": nf_ckpt,
        "n_id": int(len(id_test)),
        "n_ood": int(len(ood)),
        "n_ood_by_class": {c: int((ood_names == c).sum()) for c in HOLD},
        "pad_involved": False,
        "double_dissociation_claimed": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json")


if __name__ == "__main__":
    main()
