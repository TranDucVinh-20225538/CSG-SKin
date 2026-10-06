#!/usr/bin/env python3
"""Phase 12.1b — Camelyon17 DANN with derm-matched adversary.

Locate where leakage actually drops. Coarse checkpoints are not reused.
Outputs under results/paperB/phase12/camelyon17/matched/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, densenet121
from wilds import get_dataset

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase12_camelyon_dann as p12
from src.utils.seed import seed_everything

DATA_ROOT = p12.DATA_ROOT
OUT_ROOT = p12.OUT_ROOT / "matched"
CKPT_ROOT = p12.CKPT_ROOT.parent / "camelyon17_matched"
SPLIT_NPZ = p12.SPLIT_NPZ
TRAIN_HOSPITALS = p12.TRAIN_HOSPITALS
HOSP_TO_DOMAIN = p12.HOSP_TO_DOMAIN


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def set_lambd(self, lambd):
        self.lambd = float(lambd)

    def forward(self, x):
        return _GRLFn.apply(x, self.lambd)


def derm_domain_head(in_dim, n_domains):
    """Same template as CSGLite.domain_classifier_adv, scaled to in_dim."""
    h1 = max(in_dim, 32)
    h2 = max(in_dim // 2, 16)
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.BatchNorm1d(h1),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(h2, n_domains),
    )


class MatchedDannDenseNet121(nn.Module):
    def __init__(self, n_classes=2, n_domains=3, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet121(weights=weights)
        self.feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, n_classes)
        self.grl = GradientReversal(1.0)
        self.domain_head = derm_domain_head(self.feat_dim, n_domains)

    def forward(self, x):
        z = self.backbone(x)
        z_norm = self.feat_bn(z)
        logits = self.classifier(z_norm)
        dlogits = self.domain_head(self.grl(z_norm))
        return logits, dlogits, z_norm


class BalancedHospitalBatchSampler:
    def __init__(self, hospitals, n_per_domain, n_batches, seed, epoch=0):
        self.by_h = {h: np.where(np.asarray(hospitals) == h)[0] for h in TRAIN_HOSPITALS}
        for h, idx in self.by_h.items():
            if len(idx) == 0:
                raise RuntimeError("no train samples for hospital {}".format(h))
        self.n_per = int(n_per_domain)
        self.n_batches = int(n_batches)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1009 + self.epoch)
        for _ in range(self.n_batches):
            batch = []
            for h in TRAIN_HOSPITALS:
                batch.extend(rng.choice(self.by_h[h], size=self.n_per, replace=True).tolist())
            rng.shuffle(batch)
            yield batch


def grl_alpha(step, total_steps, gamma=20.0, power=0.5, alpha_min=0.2):
    denom = max(int(total_steps) - 1, 1)
    progress = max(0.0, min(1.0, float(step) / float(denom)))
    shaped = progress ** float(power)
    alpha = (2.0 / (1.0 + math.exp(-float(gamma) * shaped))) - 1.0
    return float(max(float(alpha_min), alpha)), float(progress)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=30, help="must be divisible by 3")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--adv_lr_multiplier", type=float, default=30.0)
    p.add_argument("--cls_lr_multiplier", type=float, default=0.2)
    p.add_argument("--grl_gamma", type=float, default=20.0)
    p.add_argument("--grl_progress_power", type=float, default=0.5)
    p.add_argument("--grl_alpha_min", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--knn_bank", type=int, default=p12.KNN_BANK)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    return p.parse_args()


def release_blob():
    p = DATA_ROOT / "camelyon17_v1.0" / "RELEASE_v1.0.txt"
    raw = p.read_bytes()
    return {"path": str(p.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}


def main():
    args = parse_args()
    if args.batch_size % 3 != 0:
        raise SystemExit("batch_size must be divisible by 3 for hospital-balanced batches")
    seed_everything(args.seed)
    run_name = "matched_densenet121_ladv{}_s{}".format(p12.lam_tag(args.lambda_adv), args.seed)
    run_dir = OUT_ROOT / run_name
    ckpt_dir = CKPT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if args.skip_done and summary_path.exists() and not args.dry_run:
        print("skip_done", summary_path)
        return
    if not SPLIT_NPZ.exists():
        raise RuntimeError("id_val split missing")

    split = np.load(SPLIT_NPZ)
    ds = get_dataset(dataset="camelyon17", root_dir=str(DATA_ROOT), download=False, split_scheme="official")
    train_idx = np.where(ds.split_array == ds.split_dict["train"])[0]
    val_idx = np.where(ds.split_array == ds.split_dict["val"])[0]
    test_idx = np.where(ds.split_array == ds.split_dict["test"])[0]
    select_idx = split["select_idx"]
    score_idx = split["score_idx"]
    train_h = ds.metadata_array[train_idx, 0].numpy().astype(int)
    n_per = args.batch_size // 3
    n_batches = max(len(train_idx) // args.batch_size, 1)

    cfg = {
        "run_name": run_name,
        "phase": "12.1b",
        "lambda_adv": float(args.lambda_adv),
        "seed": int(args.seed),
        "recipe": "derm-matched adversary on DenseNet-121",
        "adv_lr_multiplier": float(args.adv_lr_multiplier),
        "cls_lr_multiplier": float(args.cls_lr_multiplier),
        "grl_schedule": {
            "form": "2/(1+exp(-gamma * p^k))-1",
            "gamma": float(args.grl_gamma),
            "progress_power": float(args.grl_progress_power),
            "alpha_min": float(args.grl_alpha_min),
        },
        "optimizer": "AdamW, 3 param groups (encoder, adv*30, cls*0.2)",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch": "hospital-balanced, {} per hospital, batch {}".format(n_per, args.batch_size),
        "domain_head": "Linear-BN-ReLU-Drop0.5 x2 + Linear on BN-normalised 1024-d features",
        "grl_on": "feat_bn(z_backbone)",
        "data_root": str(DATA_ROOT.resolve()),
        "read_only_data_root": True,
        "release": release_blob(),
        "n_train": int(len(train_idx)),
        "n_id_val_select": int(len(select_idx)),
        "n_id_val_score": int(len(score_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_batches_per_epoch": int(n_batches),
        "leaderboard_comparable": False,
        "coarse_checkpoints_reused": False,
        "lambda0_is_matched_control": bool(args.lambda_adv == 0.0),
        "lambda0_note": (
            "When lambda_adv=0, L_adv is omitted so GRL contributes no encoder gradient; "
            "BN on features, hospital-balanced batches, AdamW 3 groups and classifier LR scaling still apply. "
            "This is the only valid λ=0 anchor for the matched recipe. Coarse λ=0 is not comparable."
        ),
        "audit": str(Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/PHASE12_1B_AUDIT.md")),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.dry_run:
        print("dry_run", json.dumps({k: cfg[k] for k in ("run_name", "lambda_adv", "adv_lr_multiplier", "batch", "n_batches_per_epoch")}))
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "run", run_name, flush=True)
    model = MatchedDannDenseNet121().to(device)
    opt = torch.optim.AdamW(
        [
            {
                "params": list(model.backbone.parameters()) + list(model.feat_bn.parameters()),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": list(model.domain_head.parameters()),
                "lr": args.lr * args.adv_lr_multiplier,
                "weight_decay": args.weight_decay,
            },
            {
                "params": list(model.classifier.parameters()),
                "lr": args.lr * args.cls_lr_multiplier,
                "weight_decay": args.weight_decay,
            },
        ]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_ds = p12.CamelyonIndexDataset(ds, train_idx, p12.train_tf())
    sel_loader = p12.loader_for(ds, select_idx, p12.eval_tf(), 32, args.num_workers, False)
    best_acc = -1.0
    best_path = ckpt_dir / "best.pt"
    history = []
    global_step = 0
    total_steps = args.epochs * n_batches

    if not args.skip_train:
        for epoch in range(1, args.epochs + 1):
            model.train()
            sampler = BalancedHospitalBatchSampler(train_h, n_per, n_batches, args.seed, epoch=epoch)
            tr_loader = DataLoader(
                train_ds,
                batch_sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
            )
            running, nseen, adv_correct = 0.0, 0, 0
            last_alpha = 0.2
            diverged_epoch = False
            for x, y, h in tr_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                d = torch.tensor([HOSP_TO_DOMAIN[int(v)] for v in h.tolist()], device=device)
                alpha, progress = grl_alpha(
                    global_step, total_steps, args.grl_gamma, args.grl_progress_power, args.grl_alpha_min
                )
                model.grl.set_lambd(alpha)
                last_alpha = alpha
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits, dlogits, _z = model(x)
                    loss = F.cross_entropy(logits, y)
                    if args.lambda_adv > 0:
                        loss = loss + float(args.lambda_adv) * F.cross_entropy(dlogits, d)
                loss_val = float(loss.detach())
                if not math.isfinite(loss_val):
                    diverged_epoch = True
                    print("NONFINITE_LOSS epoch", epoch, "step", global_step, flush=True)
                    break
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += loss_val * len(y)
                nseen += len(y)
                adv_correct += int((dlogits.argmax(1) == d).sum().item())
                global_step += 1
            if diverged_epoch:
                rec = {
                    "epoch": epoch,
                    "train_loss": None,
                    "id_val_select_acc": None,
                    "train_adv_acc": None,
                    "grl_alpha_end_epoch": last_alpha,
                    "diverged": True,
                }
                history.append(rec)
                print("epoch", epoch, rec, flush=True)
                break
            sel_acc = p12.eval_acc(model, sel_loader, device)
            rec = {
                "epoch": epoch,
                "train_loss": running / max(nseen, 1),
                "id_val_select_acc": sel_acc,
                "train_adv_acc": adv_correct / max(nseen, 1),
                "grl_alpha_end_epoch": last_alpha,
            }
            history.append(rec)
            print("epoch", epoch, rec, flush=True)
            if sel_acc > best_acc:
                best_acc = sel_acc
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "id_val_select_acc": sel_acc,
                        "lambda_adv": args.lambda_adv,
                        "seed": args.seed,
                        "matched_adversary": True,
                    },
                    best_path,
                )
                print("saved best", best_path, "acc", sel_acc, flush=True)

    def _is_diverged(hist):
        if not hist:
            return False
        for h in hist:
            for key in ("train_loss", "id_val_select_acc"):
                v = h.get(key)
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    return True
            if h.get("diverged"):
                return True
        accs = [h["id_val_select_acc"] for h in hist if isinstance(h.get("id_val_select_acc"), (int, float)) and math.isfinite(h["id_val_select_acc"])]
        return bool(accs) and max(accs) < 0.55

    diverged = _is_diverged(history)
    if not best_path.exists():
        summary = {
            "run_name": run_name,
            "lambda_adv": float(args.lambda_adv),
            "seed": int(args.seed),
            "matched_adversary": True,
            "diverged": True,
            "divergence_note": "No finite checkpoint. Training failure, not outcome (b).",
            "history": history,
            "config_path": str(run_dir / "config.json"),
            "leaderboard_comparable": False,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("saved diverged", summary_path, flush=True)
        return
    blob = torch.load(best_path, map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    packs = {}
    for name, idx in [
        ("train", train_idx),
        ("id_val_select", select_idx),
        ("id_val_score", score_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        print("extract", name, len(idx), flush=True)
        packs[name] = p12.collect(
            model, p12.loader_for(ds, idx, p12.eval_tf(), 32, args.num_workers, False), device
        )

    ztr, ytr = packs["train"]["z"], packs["train"]["y"]
    idp = packs["id_val_score"]
    leak = p12.leakage_3class(ztr, packs["train"]["hospital"], idp["z"], idp["hospital"])
    ood = {}
    for split_name in ("val", "test"):
        op = packs[split_name]
        ood[split_name] = {
            "hospital": int(np.unique(op["hospital"])[0]),
            "never_in_adversary": True,
            "accuracy": p12.acc_pack(op["logits"], op["y"]),
            "detectors": p12.detector_block(
                ztr, ytr, idp["z"], idp["logits"], op["z"], op["logits"], args.knn_bank, args.seed
            ),
            "n": int(len(op["y"])),
        }

    summary = {
        "run_name": run_name,
        "lambda_adv": float(args.lambda_adv),
        "seed": int(args.seed),
        "matched_adversary": True,
        "best_checkpoint": str(best_path),
        "best_epoch": int(blob["epoch"]),
        "id_val_select_acc": float(blob["id_val_select_acc"]),
        "id_val_score": p12.acc_pack(idp["logits"], idp["y"]),
        "leakage_train_hospitals_3class": leak,
        "ood": ood,
        "history": history,
        "diverged": bool(diverged),
        "divergence_note": (
            "Training failure: do not read collapsed ID as entanglement evidence (outcome b)."
            if diverged
            else ""
        ),
        "config_path": str(run_dir / "config.json"),
        "leaderboard_comparable": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", summary_path, flush=True)
    print(
        "LEAK",
        leak["balanced_acc"],
        "bal_floor",
        leak["balanced_acc_floor"],
        "plain_maj",
        leak["majority_baseline_acc"],
        "ID",
        summary["id_val_score"]["acc"],
        "Maha h2",
        ood["test"]["detectors"]["mahalanobis_classcond_sharedcov"],
        "kNN h2",
        ood["test"]["detectors"]["knn_k50"],
        "diverged",
        diverged,
        flush=True,
    )


if __name__ == "__main__":
    main()
