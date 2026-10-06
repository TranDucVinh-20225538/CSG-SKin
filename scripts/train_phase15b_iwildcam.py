#!/usr/bin/env python3
"""Phase 15b Item B — iWildCam coarse DANN scan.

Single-encoder DANN, ResNet-50, full 243-way domain head on train locations.
Coarse λ grid, one seed. Dense sampling and the domain-count ablation wait
until this scan shows where leakage moves.

Logs from epoch 1: adversary accuracy and loss, GRL gradient norm into the
features, and the classification gradient norm at the same features.
OOD AUROC is reported unrestricted and restricted to classes present in both
id_test and the OOD test split. Two leakage numbers: 243-way (floor 1/243)
and binary train-vs-OOD (floor 0.5).
"""

from __future__ import annotations

import argparse
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
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from wilds import get_dataset

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
from eval_ood_dual_branch import cosine_ood_scores, fit_class_means, fit_knn, knn_scores
from phase15_common import (
    GradientReversal,
    adversary_gate,
    auroc_pair,
    derm_domain_head,
    ece_from_logits,
    grl_alpha,
    leakage_probe,
)
from src.utils import ood_metrics
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
DATA_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
OUT_ROOT = PAPERB / "results" / "paperB" / "phase15b" / "iwildcam"
CKPT_ROOT = PAPERB / "checkpoints" / "phase15b" / "iwildcam"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
KNN_GALLERY_CAP = 20000


class ResNet50Dann(nn.Module):
    def __init__(self, n_classes, n_domains, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
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

    def set_grl_lambda(self, lambd):
        self.grl.set_lambd(lambd)


class IWildCamLightning(pl.LightningModule):
    def __init__(self, n_classes, n_domains, loc_to_domain, learning_rate, weight_decay, lambda_adv, log_path, ln_k):
        super().__init__()
        self.save_hyperparameters(ignore=["log_path", "loc_to_domain"])
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lambda_adv = float(lambda_adv)
        self.ln_k = float(ln_k)
        self.net = ResNet50Dann(n_classes=n_classes, n_domains=n_domains, pretrained=True)
        table = torch.full((int(max(loc_to_domain)) + 1,), -1, dtype=torch.long)
        for loc, dom in loc_to_domain.items():
            table[int(loc)] = int(dom)
        self.register_buffer("loc_table", table, persistent=False)
        self._last_alpha = 0.0
        self._g_grl = []
        self._g_cls = []
        self.log_path = Path(log_path) if log_path else None

    def _set_alpha(self):
        total = 0
        try:
            total = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        except RuntimeError:
            total = 0
        alpha, progress = grl_alpha(self.global_step, total)
        self.net.set_grl_lambda(alpha)
        self._last_alpha = alpha
        return alpha, progress

    def training_step(self, batch, batch_idx):
        images, y, meta = batch
        loc = meta[:, 0].long()
        domain = self.loc_table[loc]
        if (domain < 0).any():
            raise RuntimeError("training batch contains a location outside the 243-way head")
        alpha, progress = self._set_alpha()
        logits, dlog, z = self.net(images)
        loss_cls = F.cross_entropy(logits, y)
        loss_adv = F.cross_entropy(dlog, domain)
        g_adv = torch.autograd.grad(loss_adv, z, retain_graph=True, allow_unused=True)[0]
        g_cls = torch.autograd.grad(loss_cls, z, retain_graph=True, allow_unused=True)[0]
        g_adv_n = float(g_adv.detach().float().norm()) if g_adv is not None else 0.0
        g_cls_n = float(g_cls.detach().float().norm()) if g_cls is not None else 0.0
        self._g_grl.append(g_adv_n)
        self._g_cls.append(g_cls_n)
        loss = loss_cls + self.lambda_adv * loss_adv
        with torch.no_grad():
            acc_adv = (dlog.argmax(1) == domain).float().mean()
            acc_cls = (logits.argmax(1) == y).float().mean()
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train/loss_adv", loss_adv, on_step=False, on_epoch=True)
        self.log("train/acc_adv", acc_adv, on_step=False, on_epoch=True)
        self.log("train/acc_cls", acc_cls, on_step=False, on_epoch=True)
        self.log("train/grl_a", alpha, on_step=False, on_epoch=True)
        self.log("train/g_enc_grl", g_adv_n, on_step=False, on_epoch=True)
        self.log("train/g_enc_cls", g_cls_n, on_step=False, on_epoch=True)
        self.log("train/grl_progress", progress, on_step=True, on_epoch=False)
        if self.current_epoch == 0 and batch_idx == 0 and self.log_path:
            rec = {
                "event": "first_batch_epoch1",
                "loss_adv": float(loss_adv.detach()),
                "acc_adv": float(acc_adv.detach()),
                "g_enc_grl": g_adv_n,
                "g_enc_cls": g_cls_n,
                "lnK": self.ln_k,
                "grl_alpha": alpha,
            }
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("GATE first_batch", rec, flush=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, y, _meta = batch
        logits, _, _ = self.net(images)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        def _f(key, default=0.0):
            if key in metrics:
                v = metrics[key]
                return float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            return float(default)

        row = {
            "epoch": int(self.current_epoch) + 1,
            "acc_adv": _f("train/acc_adv"),
            "loss_adv": _f("train/loss_adv"),
            "acc_cls": _f("train/acc_cls"),
            "grl_alpha": float(self._last_alpha),
            "g_enc_grl_mean": float(np.mean(self._g_grl)) if self._g_grl else 0.0,
            "g_enc_cls_mean": float(np.mean(self._g_cls)) if self._g_cls else 0.0,
            "lnK": self.ln_k,
        }
        self._g_grl = []
        self._g_cls = []
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(
            "[epoch {:02d}] acc_adv={:.4f} loss_adv={:.4f} acc_cls={:.4f} grl_a={:.4f} g_grl={:.4g} g_cls={:.4g}".format(
                row["epoch"], row["acc_adv"], row["loss_adv"], row["acc_cls"], row["grl_alpha"],
                row["g_enc_grl_mean"], row["g_enc_cls_mean"],
            ),
            flush=True,
        )

    def configure_optimizers(self):
        adv_lr_multiplier = 30.0
        cls_lr_multiplier = 0.2
        adv = list(self.net.domain_head.parameters())
        cls = list(self.net.classifier.parameters())
        adv_ids = {id(p) for p in adv}
        cls_ids = {id(p) for p in cls}
        main = [p for p in self.net.parameters() if id(p) not in adv_ids and id(p) not in cls_ids]
        return torch.optim.AdamW(
            [
                {"params": main, "lr": self.learning_rate, "weight_decay": self.weight_decay},
                {"params": adv, "lr": self.learning_rate * adv_lr_multiplier, "weight_decay": self.weight_decay},
                {"params": cls, "lr": self.learning_rate * cls_lr_multiplier, "weight_decay": self.weight_decay},
            ]
        )


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def make_tf(train, image_size):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def maha_min(features, means, precision):
    p = np.asarray(precision, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    pm = p @ means.T
    mu_quad = np.einsum("cd,dc->c", means, pm)
    mins = np.empty(len(features), dtype=np.float64)
    for i in range(0, len(features), 512):
        x = features[i:i + 512]
        xpx = np.einsum("bd,bd->b", x @ p, x)
        d2 = xpx[:, None] - 2.0 * (x @ pm) + mu_quad[None, :]
        mins[i:i + 512] = d2.min(axis=1)
    return mins.astype(np.float32)


def gallery_index(y, cap, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(y) <= cap:
        return np.arange(len(y))
    quota = np.maximum(1, (counts / counts.sum() * cap).astype(int))
    while int(quota.sum()) > cap:
        quota[int(np.argmax(quota))] -= 1
    while int(quota.sum()) < cap:
        room = counts - quota
        j = int(np.argmax(room))
        if room[j] <= 0:
            break
        quota[j] += 1
    chosen = []
    for c, q in zip(classes, quota):
        idx = np.where(y == c)[0]
        take = idx if len(idx) <= int(q) else rng.choice(idx, int(q), replace=False)
        chosen.append(np.atleast_1d(take))
    return np.concatenate(chosen)


def detector_suite(ztr, ytr, zid, zood, logits_id, logits_ood, n_classes, knn_cap, seed):
    present = np.unique(ytr.astype(int))
    remap = {int(c): i for i, c in enumerate(present)}
    y_fit = np.array([remap[int(c)] for c in ytr], dtype=np.int64)
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(
        ztr, y_fit, num_classes=len(present), reg_eps=1e-3
    )
    gidx = gallery_index(ytr, knn_cap, seed)
    knn = fit_knn(ztr[gidx], k=50)
    cmeans, present_means = fit_class_means(ztr, ytr, n_classes)
    p_id = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
    e_id = torch.logsumexp(torch.from_numpy(logits_id).float(), dim=1).numpy()
    e_ood = torch.logsumexp(torch.from_numpy(logits_ood).float(), dim=1).numpy()
    return {
        "mahalanobis_classcond": auroc_pair(maha_min(zid, means, prec), maha_min(zood, means, prec)),
        "cosine_max": auroc_pair(
            cosine_ood_scores(zid, cmeans, present_means),
            cosine_ood_scores(zood, cmeans, present_means),
        ),
        "knn_k50": auroc_pair(knn_scores(knn, zid), knn_scores(knn, zood)),
        "MSP": auroc_pair(-p_id.max(axis=1), -p_ood.max(axis=1)),
        "Energy_T1": auroc_pair(-e_id, -e_ood),
        "fit": "iWildCam train split only",
        "n_classes_in_maha": int(len(present)),
        "knn_gallery": int(len(gidx)),
        "knn_gallery_cap": int(knn_cap),
    }


def subsample_domains(z, domain, cap, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for d in np.unique(domain):
        idx = np.where(domain == d)[0]
        if len(idx) < 2:
            continue
        take = idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)
        keep.append(np.atleast_1d(take))
    if not keep:
        raise RuntimeError("domain probe subsample is empty")
    sel = np.concatenate(keep)
    return z[sel], domain[sel]


def safe_leakage(z, domain):
    try:
        return leakage_probe(z, domain)
    except ValueError:
        domain = np.asarray(domain)
        k = int(len(np.unique(domain)))
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        bals = []
        for s in (42, 52, 62):
            ztr, zte, ytr, yte = train_test_split(z, domain, test_size=0.3, random_state=s)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
            clf.fit(ztr, ytr)
            bals.append(float(balanced_accuracy_score(yte, clf.predict(zte))))
        return {
            "bal_acc_mean": float(np.mean(bals)),
            "bal_acc_std": float(np.std(bals, ddof=1)) if len(bals) > 1 else 0.0,
            "n_domains": k,
            "chance_floor_balanced": 1.0 / k,
            "metric": "balanced_accuracy",
            "stratify": False,
        }


@torch.no_grad()
def collect(net, loader, device):
    net.eval()
    z, lg, y, loc = [], [], [], []
    for images, labels, meta in loader:
        images = images.to(device)
        logits, _, z_norm = net(images)
        z.append(z_norm.cpu().numpy())
        lg.append(logits.cpu().numpy())
        y.append(labels.numpy())
        loc.append(meta[:, 0].numpy())
    return {
        "z": np.concatenate(z),
        "logits": np.concatenate(lg),
        "labels": np.concatenate(y),
        "location": np.concatenate(loc),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=12)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_root", type=Path, default=OUT_ROOT)
    p.add_argument("--ckpt_root", type=Path, default=CKPT_ROOT)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load last/best checkpoint and run OOD/leakage eval only.",
    )
    return p.parse_args()


def resolve_ckpt(ckpt_dir: Path) -> str:
    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return str(last)
    best = sorted(ckpt_dir.glob("best*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return str(best[0])
    raise FileNotFoundError("no checkpoint in {}".format(ckpt_dir))


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "iwildcam_dann_ladv{}_s{}".format(lam_tag(args.lambda_adv), args.seed)
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run and not args.eval_only:
        print("skip_done", run_dir / "summary.json")
        return

    dataset = get_dataset(dataset="iwildcam", download=False, root_dir=str(DATA_ROOT))
    loc_idx = dataset.metadata_fields.index("location")
    if loc_idx != 0:
        raise RuntimeError("location is not metadata column 0; the trainer indexes column 0")

    def subset_columns(subset):
        idx = subset.indices
        meta = dataset.metadata_array[idx]
        labels = dataset.y_array[idx]
        return meta, labels

    train_raw = dataset.get_subset("train", transform=make_tf(True, args.image_size))
    id_val = dataset.get_subset("id_val", transform=make_tf(False, args.image_size))
    id_test = dataset.get_subset("id_test", transform=make_tf(False, args.image_size))
    ood_test = dataset.get_subset("test", transform=make_tf(False, args.image_size))
    train_meta, _train_y = subset_columns(train_raw)
    train_loc = train_meta[:, loc_idx].long().numpy()
    locations = sorted(int(v) for v in np.unique(train_loc))
    if len(locations) != 243:
        raise RuntimeError("expected 243 train locations, found {}".format(len(locations)))
    loc_to_domain = {loc: i for i, loc in enumerate(locations)}
    _id_meta, id_y_t = subset_columns(id_test)
    _ood_meta, ood_y_t = subset_columns(ood_test)
    id_y = id_y_t.numpy().astype(int)
    ood_y = ood_y_t.numpy().astype(int)
    train_locs = set(int(v) for v in np.unique(train_loc))
    ood_locs = set(int(v) for v in np.unique(_ood_meta[:, loc_idx].long().numpy()))
    id_locs = set(int(v) for v in np.unique(_id_meta[:, loc_idx].long().numpy()))
    if train_locs & ood_locs:
        raise RuntimeError("train and OOD test share locations")
    if not id_locs <= train_locs:
        raise RuntimeError("id_test has a location absent from train")
    shared = sorted(set(id_y.tolist()) & set(ood_y.tolist()))
    ln_k = float(math.log(len(locations)))
    cfg = {
        "run_name": run_name,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "architecture": "single_encoder_dann_resnet50",
        "domain_head": "243-way on official train locations. Not clustered.",
        "n_train_locations": len(locations),
        "n_classes": int(dataset.n_classes),
        "n_train": int(len(train_raw)),
        "n_id_val": int(len(id_val)),
        "n_id_test": int(len(id_test)),
        "n_ood_test": int(len(ood_test)),
        "n_id_test_classes": int(len(np.unique(id_y))),
        "n_ood_test_classes": int(len(np.unique(ood_y))),
        "n_shared_classes": int(len(shared)),
        "image_size": args.image_size,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "adv_lr_multiplier": 30.0,
        "cls_lr_multiplier": 0.2,
        "optimizer": "AdamW three groups, derm-matched",
        "grl": "ramp gamma 20, power 0.5, alpha_min 0.2",
        "epoch_budget": "12 epochs, the WILDS ResNet-50 reference length, used to locate the cliff. Not the dermatology 40-epoch budget.",
        "knn_gallery_cap": KNN_GALLERY_CAP,
        "knn_note": "kNN gallery is a class-stratified subsample of at most 20000 train embeddings. Mahalanobis, cosine, MSP and Energy use the full train set.",
        "leakage_kway_floor": 1.0 / len(locations),
        "leakage_binary_floor": 0.5,
        "fit_ood_on": "iWildCam train only",
        "checkpoint_monitor": "id_val accuracy",
        "download": False,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("dry counts", json.dumps({k: cfg[k] for k in (
        "n_train_locations", "n_classes", "n_train", "n_id_test", "n_ood_test",
        "n_id_test_classes", "n_ood_test_classes", "n_shared_classes",
    )}), flush=True)
    if args.dry_run:
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    adv_log = run_dir / "adversary_log.jsonl"
    if adv_log.exists() and not args.resume and not args.eval_only:
        adv_log.write_text("")
    if args.eval_only:
        best = resolve_ckpt(ckpt_dir)
        cfg["eval_only"] = True
        cfg["note"] = "Training skipped; metrics from existing checkpoint (epoch-11 resume abandoned)."
        print("eval_only checkpoint", best, flush=True)
    else:
        model = IWildCamLightning(
            n_classes=int(dataset.n_classes),
            n_domains=len(locations),
            loc_to_domain=loc_to_domain,
            learning_rate=args.lr,
            weight_decay=1e-4,
            lambda_adv=args.lambda_adv,
            log_path=adv_log,
            ln_k=ln_k,
        )
        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            callbacks=[ckpt_cb],
            default_root_dir=str(run_dir),
            enable_progress_bar=False,
            log_every_n_steps=50,
        )
        train_loader = torch.utils.data.DataLoader(
            train_raw, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            id_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        ckpt_path = str(ckpt_dir / "last.ckpt") if args.resume and (ckpt_dir / "last.ckpt").exists() else None
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        best = ckpt_cb.best_model_path or str(ckpt_dir / "last.ckpt")
    loaded = IWildCamLightning.load_from_checkpoint(
        best, loc_to_domain=loc_to_domain, log_path=None, ln_k=ln_k, map_location="cpu", strict=False,
    )
    device = torch.device("cuda")
    net = loaded.net.to(device).eval()
    eval_loader_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # Train features use the eval transform, not the training augmentation.
    train_eval = dataset.get_subset("train", transform=make_tf(False, args.image_size))
    tr = collect(net, torch.utils.data.DataLoader(train_eval, **eval_loader_kw), device)
    idp = collect(net, torch.utils.data.DataLoader(id_test, **eval_loader_kw), device)
    odp = collect(net, torch.utils.data.DataLoader(ood_test, **eval_loader_kw), device)
    yid = idp["labels"].astype(int)
    yood = odp["labels"].astype(int)
    shared_set = set(shared)
    id_mask = np.isin(yid, list(shared_set))
    ood_mask = np.isin(yood, list(shared_set))
    suite_full = detector_suite(
        tr["z"], tr["labels"].astype(int), idp["z"], odp["z"], idp["logits"], odp["logits"],
        int(dataset.n_classes), KNN_GALLERY_CAP, args.seed,
    )
    suite_shared = detector_suite(
        tr["z"], tr["labels"].astype(int), idp["z"][id_mask], odp["z"][ood_mask],
        idp["logits"][id_mask], odp["logits"][ood_mask],
        int(dataset.n_classes), KNN_GALLERY_CAP, args.seed,
    )
    z_k, d_k = subsample_domains(tr["z"], np.array([loc_to_domain[int(v)] for v in tr["location"]]), cap=12, seed=args.seed)
    leak_k = safe_leakage(z_k, d_k)
    rng = np.random.default_rng(args.seed)
    n_bin = 4000
    i_tr = rng.choice(len(tr["z"]), n_bin, replace=False)
    i_ood = rng.choice(len(odp["z"]), n_bin, replace=False)
    z_bin = np.concatenate([tr["z"][i_tr], odp["z"][i_ood]])
    d_bin = np.concatenate([np.zeros(n_bin, dtype=int), np.ones(n_bin, dtype=int)])
    leak_bin = safe_leakage(z_bin, d_bin)
    pred = idp["logits"].argmax(1)
    id_ece, id_conf = ece_from_logits(idp["logits"], yid)
    ood_ece, ood_conf = ece_from_logits(odp["logits"], yood)
    rows = []
    if adv_log.exists():
        rows = [json.loads(line) for line in adv_log.read_text().splitlines() if line.strip()]
    gate = adversary_gate(
        rows, ln_k, args.lambda_adv,
        leakage_bal=leak_bin["bal_acc_mean"], floor=0.5,
    )
    summary = {
        "run_name": run_name,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "checkpoint": best,
        "id_acc": float(accuracy_score(yid, pred)),
        "id_balanced_acc": float(balanced_accuracy_score(yid, pred)),
        "id_ece": id_ece,
        "id_mean_confidence": id_conf,
        "ood_ece": ood_ece,
        "ood_mean_confidence": ood_conf,
        "ood_unrestricted": suite_full,
        "ood_shared_classes": suite_shared,
        "n_shared_scored_id": int(id_mask.sum()),
        "n_shared_scored_ood": int(ood_mask.sum()),
        "leakage_kway": leak_k,
        "leakage_binary_train_vs_ood": leak_bin,
        "gate": gate,
        "config": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", run_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
