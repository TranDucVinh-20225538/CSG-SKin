#!/usr/bin/env python3
"""Phase 12.1 Camelyon17 — single-encoder DANN (DenseNet-121), coarse λ scan.

Read-only against DST-Skin WILDS copies. Outputs only under results/paperB/phase12/.
Checkpoint on id_val_select. OOD / ID metrics on id_val_score, hospital 1 (val), hospital 2 (test).
Never select on OOD val. Not leaderboard-comparable.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from wilds import get_dataset

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPTS))
from src.utils import ood_metrics
from src.utils.seed import seed_everything

DATA_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT_ROOT = PAPERB / "results" / "paperB" / "phase12" / "camelyon17"
CKPT_ROOT = PAPERB / "checkpoints" / "phase12" / "camelyon17"
SPLIT_NPZ = OUT_ROOT / "id_val_split.npz"

TRAIN_HOSPITALS = (0, 3, 4)
HOSP_TO_DOMAIN = {0: 0, 3: 1, 4: 2}
N_CLASSES = 2
N_DOMAINS = 3
KNN_BANK = 30000
KNN_K = 50
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

    def forward(self, x):
        return _GRLFn.apply(x, self.lambd)


class DannDenseNet121(nn.Module):
    """WILDS reference backbone + task head + GRL domain head over training hospitals."""

    def __init__(self, n_classes=2, n_domains=3, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet121(weights=weights)
        self.feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(self.feat_dim, n_classes)
        self.grl = GradientReversal(1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_domains),
        )

    def forward(self, x):
        z = self.backbone(x)
        logits = self.classifier(z)
        dlogits = self.domain_head(self.grl(z))
        return logits, dlogits, z


class CamelyonIndexDataset(Dataset):
    def __init__(self, wilds_ds, indices, transform):
        self.ds = wilds_ds
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.ds.get_input(idx)
        if self.transform is not None:
            x = self.transform(x)
        y = int(self.ds.y_array[idx])
        h = int(self.ds.metadata_array[idx, 0])
        return x, y, h


def train_tf():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_tf():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def release_blob():
    p = DATA_ROOT / "camelyon17_v1.0" / "RELEASE_v1.0.txt"
    raw = p.read_bytes()
    return {
        "path": str(p.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": raw.decode("utf-8"),
    }


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--knn_bank", type=int, default=KNN_BANK)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--skip_train", action="store_true", help="eval an existing best ckpt only")
    return p.parse_args()


def ece_from_logits(logits, labels, n_bins=15):
    probs = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=1).numpy()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(labels), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def auroc_pair(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return float(roc_auc_score(y, s))


def maha_min_vec(z, means, prec):
    z = z.astype(np.float64)
    p = prec.astype(np.float64)
    mins = np.full(len(z), np.inf, dtype=np.float64)
    for c in range(len(means)):
        delta = z - means[c].astype(np.float64)
        d2 = np.einsum("nd,dd,nd->n", delta, p, delta)
        mins = np.minimum(mins, d2)
    return mins


def cosine_ood(z, means):
    prot = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (zn @ prot.T).max(axis=1)


def loader_for(ds, idx, tf, batch, workers, shuffle):
    return DataLoader(
        CamelyonIndexDataset(ds, idx, tf),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    zs, ys, hs, lgs = [], [], [], []
    for x, y, h in loader:
        x = x.to(device, non_blocking=True)
        logits, _d, z = model(x)
        zs.append(z.cpu().numpy())
        lgs.append(logits.cpu().numpy())
        ys.append(y.numpy())
        hs.append(h.numpy())
    return {
        "z": np.concatenate(zs),
        "logits": np.concatenate(lgs),
        "y": np.concatenate(ys),
        "hospital": np.concatenate(hs),
    }


def acc_pack(logits, y):
    pred = logits.argmax(1)
    return {
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "ece": ece_from_logits(logits, y),
        "n": int(len(y)),
        "leaderboard_comparable": False,
        "checkpoint_split": "id_val_select",
        "note": "Not comparable to WILDS leaderboard (those select on OOD val / hospital 1).",
    }


def leakage_3class(z_train, h_train, z_eval, h_eval):
    dtr = np.array([HOSP_TO_DOMAIN[int(h)] for h in h_train], dtype=int)
    dve = np.array([HOSP_TO_DOMAIN[int(h)] for h in h_eval], dtype=int)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    clf.fit(z_train, dtr)
    pred = clf.predict(z_eval)
    proba = clf.predict_proba(z_eval)
    counts = np.bincount(dve, minlength=N_DOMAINS).astype(np.float64)
    maj = float(counts.max() / max(counts.sum(), 1))
    try:
        auc = float(roc_auc_score(dve, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "protocol": "logistic 3-class hospital probe, fit on train features, eval on id_val_score; hospitals {0,3,4}",
        "balanced_acc": float(balanced_accuracy_score(dve, pred)),
        "acc": float(accuracy_score(dve, pred)),
        "auroc_ovr_macro": auc,
        "n_classes": int(N_DOMAINS),
        "balanced_acc_floor": 1.0 / float(N_DOMAINS),
        "balanced_acc_floor_note": "1/K for K-class balanced accuracy; a constant predictor scores recall 1 on one class and 0 on the others",
        "majority_baseline_acc": maj,
        "majority_baseline_note": "plain-accuracy majority (mode class prior). Do not compare balanced_acc against this.",
        "n": int(len(dve)),
    }


def leakage_binary(z_id, z_ood, seeds=(42, 52, 62)):
    z = np.concatenate([z_id, z_ood])
    y = np.concatenate([np.zeros(len(z_id)), np.ones(len(z_ood))])
    bals, aucs = [], []
    for s in seeds:
        ztr, zte, ytr, yte = train_test_split(z, y, test_size=0.3, random_state=s, stratify=y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
        clf.fit(ztr, ytr)
        pred = clf.predict(zte)
        score = clf.predict_proba(zte)[:, 1]
        bals.append(float(balanced_accuracy_score(yte, pred)))
        aucs.append(float(roc_auc_score(yte, score)))
    return {
        "protocol": "logistic binary ID(id_val_score) vs OOD-hospital, 70/30, 3 seeds — analogue of derm leakage",
        "balanced_acc_mean": float(np.mean(bals)),
        "balanced_acc_std": float(np.std(bals, ddof=1)),
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs, ddof=1)),
        "n_id": int(len(z_id)),
        "n_ood": int(len(z_ood)),
    }


def detector_block(ztr, ytr, zid, lid, zood, lood, knn_bank, seed):
    """All statistics fit on train only. Lead with Maha class-cond and kNN."""
    n_classes = int(ytr.max()) + 1
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(
        ztr, ytr, num_classes=n_classes, reg_eps=1e-3
    )
    sid = maha_min_vec(zid, means, prec)
    sood = maha_min_vec(zood, means, prec)
    mu = ztr.mean(axis=0)
    centered = ztr - mu
    cov = (centered.T @ centered) / max(len(ztr) - 1, 1)
    cov = cov + 1e-3 * np.eye(ztr.shape[1])
    prec_ag = np.linalg.inv(cov)
    sid_ag = maha_min_vec(zid, mu[None, :], prec_ag)
    sood_ag = maha_min_vec(zood, mu[None, :], prec_ag)

    rng = np.random.default_rng(seed)
    strata = ytr.astype(int)
    # stratified subsample for kNN bank
    chosen = []
    per = max(knn_bank // n_classes, 1)
    for c in range(n_classes):
        idx = np.where(strata == c)[0]
        take = min(per, len(idx))
        chosen.append(rng.choice(idx, size=take, replace=False))
    bank_idx = np.concatenate(chosen)
    if len(bank_idx) > knn_bank:
        bank_idx = rng.choice(bank_idx, size=knn_bank, replace=False)
    bank = ztr[bank_idx]
    bank_n = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
    knn = NearestNeighbors(n_neighbors=min(KNN_K, len(bank_n)), algorithm="auto", metric="euclidean")
    knn.fit(bank_n)

    def knn_score(z):
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        d, _ = knn.kneighbors(zn)
        return d.mean(axis=1)

    p_id = torch.softmax(torch.from_numpy(lid), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(lood), dim=1).numpy()
    e_id = torch.logsumexp(torch.from_numpy(lid).float(), dim=1).numpy()
    e_ood = torch.logsumexp(torch.from_numpy(lood).float(), dim=1).numpy()
    present = [c for c in range(n_classes) if np.any(ytr == c)]
    cmeans = np.stack([ztr[ytr == c].mean(axis=0) for c in present])
    return {
        "mahalanobis_classcond_sharedcov": auroc_pair(sid, sood),
        "mahalanobis_agnostic": auroc_pair(sid_ag, sood_ag),
        "knn_k50": auroc_pair(knn_score(zid), knn_score(zood)),
        "cosine_max": auroc_pair(cosine_ood(zid, cmeans), cosine_ood(zood, cmeans)),
        "MSP": auroc_pair(-p_id.max(axis=1), -p_ood.max(axis=1)),
        "Energy_T1": auroc_pair(-e_id, -e_ood),
        "knn_bank_n": int(len(bank_idx)),
        "knn_bank_note": "stratified subsample of train (full 302k 1024-d kNN is too heavy); recorded",
        "binary_logit_note": (
            "MSP/Energy are weak on binary Camelyon17 (MSP in [0.5,1]). "
            "Lead with Maha class-cond and kNN. Do not read a flat MSP curve as a failed replicate."
        ),
        "fit": "train split only",
        "id_reference": "id_val_score",
    }


def eval_acc(model, loader, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y, _h in loader:
            x = x.to(device, non_blocking=True)
            logits, _, _ = model(x)
            correct += int((logits.argmax(1).cpu() == y).sum())
            n += len(y)
    return correct / max(n, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    run_name = "dann_densenet121_ladv{}_s{}".format(lam_tag(args.lambda_adv), args.seed)
    run_dir = OUT_ROOT / "coarse" / run_name
    ckpt_dir = CKPT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if args.skip_done and summary_path.exists() and not args.dry_run:
        print("skip_done", summary_path)
        return
    if not SPLIT_NPZ.exists():
        raise RuntimeError("id_val split missing: {}".format(SPLIT_NPZ))

    split = np.load(SPLIT_NPZ)
    ds = get_dataset(dataset="camelyon17", root_dir=str(DATA_ROOT), download=False, split_scheme="official")
    train_idx = np.where(ds.split_array == ds.split_dict["train"])[0]
    val_idx = np.where(ds.split_array == ds.split_dict["val"])[0]
    test_idx = np.where(ds.split_array == ds.split_dict["test"])[0]
    select_idx = split["select_idx"]
    score_idx = split["score_idx"]
    # hygiene: select/score are halves of id_val, disjoint from train/val/test
    if set(select_idx) & set(score_idx):
        raise RuntimeError("id_val_select overlaps id_val_score")
    if set(train_idx) & set(select_idx) or set(train_idx) & set(score_idx):
        raise RuntimeError("train overlaps id_val halves")

    cfg = {
        "run_name": run_name,
        "lambda_adv": float(args.lambda_adv),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optimizer": "SGD momentum=0.9",
        "backbone": "densenet121",
        "pretrained": True,
        "input": "native 96x96, ImageNet normalize, train RandomHorizontalFlip",
        "architecture": "single-encoder DANN, not dual-encoder CSG",
        "domain_head": "3-class over training hospitals {0,3,4} with GRL",
        "data_root": str(DATA_ROOT.resolve()),
        "read_only_data_root": True,
        "release": release_blob(),
        "split_scheme": "official",
        "id_eval": "id_val_score (half of official id_val; no id_test exists)",
        "checkpoint_on": "id_val_select",
        "never_checkpoint_on": "val (hospital 1)",
        "ood_columns": ["val_hospital_1", "test_hospital_2"],
        "n_train": int(len(train_idx)),
        "n_id_val_select": int(len(select_idx)),
        "n_id_val_score": int(len(score_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "leaderboard_comparable": False,
        "binary_task_note": (
            "Camelyon17 is binary. MSP is confined to [0.5, 1]. Lead with Mahalanobis "
            "class-conditional (shared covariance) and kNN k=50."
        ),
        "class_composition": "train / id_val / test are all balanced ~50/50; composition matched by construction. No 2.5b restriction needed.",
        "knn_bank": int(args.knn_bank),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.dry_run:
        print("dry_run", json.dumps({k: cfg[k] for k in ("run_name", "n_train", "n_id_val_select", "n_id_val_score", "n_val", "n_test", "lambda_adv")}))
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "run", run_name, flush=True)
    model = DannDenseNet121().to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    tr_loader = loader_for(ds, train_idx, train_tf(), args.batch_size, args.num_workers, True)
    sel_loader = loader_for(ds, select_idx, eval_tf(), args.batch_size, args.num_workers, False)
    best_acc = -1.0
    best_path = ckpt_dir / "best.pt"
    history = []

    if not args.skip_train:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running, nseen = 0.0, 0
            for x, y, h in tr_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                d = torch.tensor([HOSP_TO_DOMAIN[int(v)] for v in h.tolist()], device=device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits, dlogits, _z = model(x)
                    loss = F.cross_entropy(logits, y)
                    if args.lambda_adv > 0:
                        loss = loss + float(args.lambda_adv) * F.cross_entropy(dlogits, d)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running += float(loss.detach()) * len(y)
                nseen += len(y)
            sel_acc = eval_acc(model, sel_loader, device)
            rec = {"epoch": epoch, "train_loss": running / max(nseen, 1), "id_val_select_acc": sel_acc}
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
                    },
                    best_path,
                )
                print("saved best", best_path, "acc", sel_acc, flush=True)
    if not best_path.exists():
        raise RuntimeError("no best checkpoint")
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
        packs[name] = collect(
            model,
            loader_for(ds, idx, eval_tf(), args.batch_size, args.num_workers, False),
            device,
        )

    # Fit-set assertion
    if packs["train"]["z"].shape[0] != len(train_idx):
        raise RuntimeError("train pack size mismatch")
    tr_h = set(int(v) for v in packs["train"]["hospital"])
    if tr_h != set(TRAIN_HOSPITALS):
        raise RuntimeError("train hospitals {} != {}".format(tr_h, TRAIN_HOSPITALS))

    ztr, ytr = packs["train"]["z"], packs["train"]["y"]
    idp = packs["id_val_score"]
    ood = {}
    for split_name in ("val", "test"):
        op = packs[split_name]
        ood[split_name] = {
            "hospital": int(np.unique(op["hospital"])[0]),
            "never_in_adversary": True,
            "accuracy": acc_pack(op["logits"], op["y"]),
            "detectors": detector_block(
                ztr, ytr, idp["z"], idp["logits"], op["z"], op["logits"], args.knn_bank, args.seed
            ),
            "leakage_binary_id_vs_this": leakage_binary(idp["z"], op["z"]),
            "n": int(len(op["y"])),
        }

    summary = {
        "run_name": run_name,
        "lambda_adv": float(args.lambda_adv),
        "seed": int(args.seed),
        "best_checkpoint": str(best_path),
        "best_epoch": int(blob["epoch"]),
        "id_val_select_acc": float(blob["id_val_select_acc"]),
        "id_val_score": acc_pack(idp["logits"], idp["y"]),
        "id_val_select_eval": acc_pack(packs["id_val_select"]["logits"], packs["id_val_select"]["y"]),
        "leakage_train_hospitals_3class": leakage_3class(
            ztr, packs["train"]["hospital"], idp["z"], idp["hospital"]
        ),
        "ood": ood,
        "history": history,
        "config_path": str(run_dir / "config.json"),
        "primary_detectors": ["mahalanobis_classcond_sharedcov", "knn_k50"],
        "binary_msp_note": cfg["binary_task_note"],
        "leaderboard_comparable": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", summary_path, flush=True)
    print(
        "ID acc",
        summary["id_val_score"]["acc"],
        "test Maha",
        ood["test"]["detectors"]["mahalanobis_classcond_sharedcov"],
        "val Maha",
        ood["val"]["detectors"]["mahalanobis_classcond_sharedcov"],
        flush=True,
    )


if __name__ == "__main__":
    main()
