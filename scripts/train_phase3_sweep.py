#!/usr/bin/env python3
"""Phase 3: λ_adv sweep on runB_orth1 with the Phase 2 PAD patient split.

Original grid: λ_adv ∈ {0, 0.25, 0.5, 1, 2, 4, 8}, λ_orth=1, 3 seeds;
endpoints {0, 2, 8} to 5 seeds → 27 runs.

Additions: Fitzpatrick17k OOD and 6-class-restricted AUROC at every λ.
λ_adv=2 reuses Phase 2 checkpoints (same recipe, same split) and only re-evals.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase2_pad_holdout as p2
from eval_fitz_domain_axis import FolderJpegDataset
from eval_ood_dual_branch import cosine_ood_scores, fit_class_means, fit_knn, knn_scores

from src.datasets.skin_dataset import (
    CombinedTrainDataset,
    SkinDataset,
    build_lesion_branch_transform_gray,
    build_train_transform_robust,
    build_val_transform_robust,
    csg_lite_paired_collate,
)
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.seed import seed_everything

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
FITZ_DIR = PAPERB / "data" / "fitzpatrick17k" / "images"
PAD_CLASSES = (0, 1, 2, 3, 4, 7)
JOBS = []
for _lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    _seeds = (42, 52, 62, 72, 82) if _lam in (0.0, 2.0, 8.0) else (42, 52, 62)
    for _s in _seeds:
        JOBS.append((_lam, _s))
assert len(JOBS) == 27


def lam_tag(x):
    return "{:g}".format(x).replace(".", "p")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lambda_orth", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_root", type=Path, default=PAPERB / "results" / "paperB" / "phase3_sweep")
    p.add_argument("--ckpt_root", type=Path, default=PAPERB / "checkpoints" / "phase3")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--adv_lr_multiplier", type=float, default=30.0)
    return p.parse_args()


def ece_from_logits(logits, labels, n_bins=15):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
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


def maha_scores(ztr, ytr, zid, zood, n_classes=8):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(
        ztr, ytr, num_classes=n_classes, reg_eps=1e-3
    )
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    return sid, sood


def auroc_pair(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return float(roc_auc_score(y, s))


def leakage_probe(z, domain, seeds=(42, 52, 62)):
    accs, bals, aurocs = [], [], []
    for s in seeds:
        ztr, zte, ytr, yte = train_test_split(z, domain, test_size=0.3, random_state=s, stratify=domain)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(ztr, ytr)
        pred = clf.predict(zte)
        score = clf.predict_proba(zte)[:, 1]
        accs.append(float(accuracy_score(yte, pred)))
        bals.append(float(balanced_accuracy_score(yte, pred)))
        aurocs.append(float(roc_auc_score(yte, score)))
    return {
        "acc_mean": float(np.mean(accs)),
        "bal_acc_mean": float(np.mean(bals)),
        "auroc_mean": float(np.mean(aurocs)),
        "acc_std": float(np.std(accs, ddof=1)),
        "bal_acc_std": float(np.std(bals, ddof=1)),
        "auroc_std": float(np.std(aurocs, ddof=1)),
        "n_probe_seeds": len(seeds),
    }


def find_best_ckpt(ckpt_dir: Path):
    best = sorted(ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = ckpt_dir / "last.ckpt"
    return last if last.exists() else None


def detector_block(ztr, ytr, zid, yid, zood, logits_id, logits_ood):
    sid, sood = maha_scores(ztr, ytr, zid, zood)
    six = np.isin(yid, PAD_CLASSES)
    out = {
        "mahalanobis_classcond": {
            "unrestricted": auroc_pair(sid, sood),
            "id_6class_restricted": auroc_pair(sid[six], sood) if six.any() else None,
        }
    }
    means, present = fit_class_means(ztr, ytr, 8)
    out["cosine_max"] = auroc_pair(cosine_ood_scores(zid, means, present), cosine_ood_scores(zood, means, present))
    knn = fit_knn(ztr, k=50)
    out["knn_k50"] = auroc_pair(knn_scores(knn, zid), knn_scores(knn, zood))
    p_id = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
    out["MSP"] = auroc_pair(-p_id.max(axis=1), -p_ood.max(axis=1))
    e_id = torch.logsumexp(torch.from_numpy(logits_id).float(), dim=1).numpy()
    e_ood = torch.logsumexp(torch.from_numpy(logits_ood).float(), dim=1).numpy()
    out["Energy_T1"] = auroc_pair(-e_id, -e_ood)
    return out, sid, sood


@torch.no_grad()
def collect_all(net, loader, device, max_batches=None):
    net.eval()
    zl, zc, lg, y = [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = batch[0], batch[1]
        images = images.to(device)
        out = net(images, x_lesion=None, return_latents=True)
        logits, _dctx, _dadv, z_l, z_ctx = out
        zl.append(z_l.cpu().numpy())
        zc.append(z_ctx.cpu().numpy())
        lg.append(logits.cpu().numpy())
        y.append(labels.numpy() if torch.is_tensor(labels) else np.asarray(labels))
    return {
        "z_lesion": np.concatenate(zl),
        "z_context": np.concatenate(zc),
        "logits": np.concatenate(lg),
        "labels": np.concatenate(y),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "runB_orth1_ladv{}_s{}".format(lam_tag(args.lambda_adv), args.seed)
    if abs(args.adv_lr_multiplier - 30.0) > 1e-9:
        run_name += "_advlr{}".format(lam_tag(args.adv_lr_multiplier))
    run_dir = args.output_root / run_name
    ckpt_dir = args.ckpt_root / run_name
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        print("skip_done", run_dir / "summary.json")
        return

    if not P2_SPLIT.exists():
        raise RuntimeError("Phase 2 PAD split missing: {}".format(P2_SPLIT))
    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, val, test = p2.isic_splits(isic)
    pad_adv = pad[pad.path.isin(set(spec["pad_adv_paths"]))].reset_index(drop=True)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    if pad_adv.empty or pad_hold.empty:
        raise RuntimeError("Phase 2 PAD split loaded empty")

    train_tf = build_train_transform_robust()
    les_tf = build_lesion_branch_transform_gray()
    eval_tf = build_val_transform_robust()
    train_ds = CombinedTrainDataset(train, pad_adv, transform=train_tf, lesion_transform=les_tf)
    val_ds = SkinDataset(val, transform=eval_tf)

    cfg = vars(args).copy()
    cfg["output_root"] = str(args.output_root)
    cfg["ckpt_root"] = str(args.ckpt_root)
    cfg["run_name"] = run_name
    cfg["n_pad_adv"] = int(len(pad_adv))
    cfg["n_pad_heldout"] = int(len(pad_hold))
    cfg["phase2_split"] = str(P2_SPLIT)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.dry_run:
        print(
            "dry_run",
            run_name,
            "train",
            len(train_ds),
            "val",
            len(val_ds),
            "pad_adv",
            len(pad_adv),
            "pad_hold",
            len(pad_hold),
            "fitz",
            len(list(FITZ_DIR.glob("*.jpg"))),
        )
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    reuse_p2 = abs(args.lambda_adv - 2.0) < 1e-12
    p2_ckpt_dir = PAPERB / "checkpoints" / "phase2" / "runB_orth1_padhold_s{}".format(args.seed)
    best = None
    if reuse_p2:
        best = find_best_ckpt(p2_ckpt_dir)
        if best is not None:
            print("reusing Phase 2 ckpt", best, flush=True)

    if best is None:
        model = CSGLiteLightning(
            lesion_classes=8,
            domain_classes=2,
            lesion_latent_dim=16,
            context_latent_dim=64,
            learning_rate=args.lr,
            weight_decay=1e-4,
            lambda_ctx=1.0,
            lambda_adv=args.lambda_adv,
            lambda_orth=args.lambda_orth,
            lambda_supcon=0.0,
            adv_lr_multiplier=args.adv_lr_multiplier,
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
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[ckpt_cb],
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            default_root_dir=str(run_dir),
            log_every_n_steps=20,
        )
        ckpt_path = None
        last = ckpt_dir / "last.ckpt"
        if args.resume and last.exists():
            ckpt_path = str(last)
        train_loader = p2.make_loader(
            train_ds, args.batch_size, args.num_workers, True, drop_last=True, collate=csg_lite_paired_collate
        )
        val_loader = p2.make_loader(val_ds, args.batch_size, args.num_workers, False)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
        print("best", best, flush=True)

    lit = CSGLiteLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.model.to(device).eval()

    def loader(frame, shuffle=False):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, args.num_workers, shuffle)

    train_pack = collect_all(net, loader(train), device)
    id_pack = collect_all(net, loader(test), device)
    pad_full = collect_all(net, loader(pad), device)
    pad_adv_p = collect_all(net, loader(pad_adv), device)
    pad_hold_p = collect_all(net, loader(pad_hold), device)
    fitz_paths = sorted(FITZ_DIR.glob("*.jpg"))
    fitz_ds = FolderJpegDataset(fitz_paths, eval_tf)
    fitz_loader = DataLoader(fitz_ds, batch_size=64, shuffle=False, num_workers=args.num_workers)
    fitz_pack = collect_all(net, fitz_loader, device)

    pred = id_pack["logits"].argmax(1)
    yid = id_pack["labels"]
    id_acc = float(accuracy_score(yid, pred))
    id_bal = float(balanced_accuracy_score(yid, pred))
    id_ece = ece_from_logits(id_pack["logits"], yid)
    pad_pred = pad_full["logits"].argmax(1)
    pad_acc = float(accuracy_score(pad_full["labels"], pad_pred))
    pad_bal = float(balanced_accuracy_score(pad_full["labels"], pad_pred))

    domain = np.concatenate([np.zeros(len(id_pack["z_lesion"])), np.ones(len(pad_full["z_lesion"]))])
    z_l_all = np.concatenate([id_pack["z_lesion"], pad_full["z_lesion"]])
    z_c_all = np.concatenate([id_pack["z_context"], pad_full["z_context"]])
    leak = {
        "z_lesion": leakage_probe(z_l_all, domain),
        "z_context": leakage_probe(z_c_all, domain),
        "protocol": "logistic domain probe, 70/30 on ISIC-test+PAD-full, 3 seeds; not used as an OOD detector",
    }

    ood = {}
    for name, pack in [
        ("pad_adv", pad_adv_p),
        ("pad_heldout", pad_hold_p),
        ("pad_full", pad_full),
        ("fitzpatrick17k", fitz_pack),
    ]:
        les, _, _ = detector_block(
            train_pack["z_lesion"],
            train_pack["labels"],
            id_pack["z_lesion"],
            id_pack["labels"],
            pack["z_lesion"],
            id_pack["logits"],
            pack["logits"],
        )
        ctx_sid, ctx_sood = maha_scores(
            train_pack["z_context"], train_pack["labels"], id_pack["z_context"], pack["z_context"]
        )
        ood[name] = {
            "z_lesion": les,
            "z_context_maha_unrestricted": auroc_pair(ctx_sid, ctx_sood),
            "n_ood": int(len(pack["z_lesion"])),
        }
        print(name, "z_lesion_maha", les["mahalanobis_classcond"], "z_ctx", ood[name]["z_context_maha_unrestricted"])

    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "lambda_adv": args.lambda_adv,
        "lambda_orth": args.lambda_orth,
        "best_checkpoint": str(best),
        "reused_phase2_ckpt": bool(reuse_p2 and str(best).find("phase2") >= 0),
        "id_acc": id_acc,
        "id_balanced_acc": id_bal,
        "id_ece": id_ece,
        "pad_acc": pad_acc,
        "pad_balanced_acc": pad_bal,
        "leakage": leak,
        "ood": ood,
        "n_pad_adv": int(len(pad_adv)),
        "n_pad_heldout": int(len(pad_hold)),
        "n_fitz": int(len(fitz_pack["z_lesion"])),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json")


if __name__ == "__main__":
    main()
