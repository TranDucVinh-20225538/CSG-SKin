#!/usr/bin/env python3
"""Phase 1.6a: linear domain-head OOD scores on frozen representations. CPU by default."""

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
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B3_Weights, ResNet50_Weights, efficientnet_b3, resnet50

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_ood_dual_branch as p1

from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
from src.models.baseline import BaselineNet
from src.models.effb3_single import EffB3SingleNet

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_6")
SEEDS = (42, 52, 62, 72, 82)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=OUT)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed42_only", action="store_true", help="Trained-model checkpoints: seed 42 only.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is visible.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_extract", action="store_true", help="Reuse cached npz under output_dir/features/")
    return p.parse_args()


def fpr95(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) if idx.size else 1.0


def ood_from_scores(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return {
        "AUROC": float(roc_auc_score(y, s)),
        "AUPR_IN": float(average_precision_score(1 - y, -s)),
        "AUPR_OUT": float(average_precision_score(y, s)),
        "FPR95": fpr95(y, s),
        "n_id": int(len(sid)),
        "n_ood": int(len(sood)),
    }


def logistic_ood(zid, zood, seeds=SEEDS):
    x = np.concatenate([zid, zood], axis=0)
    y = np.concatenate([np.zeros(len(zid), np.int64), np.ones(len(zood), np.int64)])
    rows = []
    for s in seeds:
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=s, stratify=y)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, random_state=s),
        )
        clf.fit(xtr, ytr)
        score = clf.predict_proba(xte)[:, 1]
        sid = score[yte == 0]
        sood = score[yte == 1]
        mets = ood_from_scores(sid, sood)
        mets["seed"] = int(s)
        rows.append(mets)
    keys = ["AUROC", "AUPR_IN", "AUPR_OUT", "FPR95"]
    agg = {k + "_mean": float(np.mean([r[k] for r in rows])) for k in keys}
    agg.update({k + "_std": float(np.std([r[k] for r in rows], ddof=1)) for k in keys})
    agg["n_seeds"] = len(rows)
    agg["protocol"] = "logistic domain head, 70/30 stratified on ISIC-test+PAD, 5 seeds; score=P(PAD)"
    return agg, rows


def pca_k1(ztr, zid, zood):
    pca = PCA(n_components=1, random_state=42)
    pca.fit(ztr)
    idp = pca.transform(zid)[:, 0]
    oodp = pca.transform(zood)[:, 0]
    if oodp.mean() < idp.mean():
        idp, oodp = -idp, -oodp
    mets = ood_from_scores(idp, oodp)
    mets["var_explained"] = float(pca.explained_variance_ratio_[0])
    mets["fit"] = "PCA k=1 fit on ISIC train only"
    return mets


@torch.no_grad()
def extract_loader(model, forward_fn, loader, device, max_batches=None):
    model.eval()
    chunks, ys = [], []
    for i, (images, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        z = forward_fn(model, images)
        chunks.append(z.detach().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(chunks), np.concatenate(ys)


def fwd_imagenet_r50(model, x):
    return model(x)


def fwd_imagenet_effb3(model, x):
    return model(x)


def fwd_baseline(model, x):
    _, feat = model(x, return_features=True)
    return feat


def fwd_effb3_control(model, x):
    x_in = model._rgb_to_gray3(x)
    return model.backbone(x_in)


def fwd_z_context(model, x):
    feat_ctx = model.extract_context_features(x)
    return model.context_projector(feat_ctx)


def build_imagenet_r50():
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    return net.eval()


def build_imagenet_effb3():
    net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    return net.eval()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = args.output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        torch.set_num_threads(max(int(args.num_workers) * 2, 8))
    else:
        device = torch.device("cuda")
    print("device", device, flush=True)

    splits = p1.build_split_frames(p1.METADATA)
    tf = build_val_transform_robust()
    max_b = 2 if args.dry_run else None

    def loader(frame):
        ds = SkinDataset(frame, transform=tf)
        if args.dry_run:
            ds.data = ds.data.iloc[: max_b * args.batch_size].reset_index(drop=True)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0 if args.dry_run else args.num_workers,
            pin_memory=device.type == "cuda",
        )

    specs = [
        ("imagenet_resnet50_raw", "imagenet", None, build_imagenet_r50, fwd_imagenet_r50),
        ("imagenet_effb3_raw", "imagenet", None, build_imagenet_effb3, fwd_imagenet_effb3),
        ("baseline_backbone_raw", "baseline_soft", 42, None, fwd_baseline),
        ("effb3_control_backbone_raw", "effb3_control", 42, None, fwd_effb3_control),
        ("z_context", "runB_orth1", 42, None, fwd_z_context),
    ]

    results = []
    for name, method, seed, builder, fwd in specs:
        npz = feat_dir / "{}.npz".format(name)
        if args.skip_extract and npz.exists() and not args.dry_run:
            pack = np.load(npz)
            ztr, zid, zood = pack["train"], pack["id"], pack["ood"]
            print("loaded cache", name, ztr.shape, flush=True)
        else:
            print("extract", name, flush=True)
            if builder is not None:
                model = builder().to(device).eval()
            else:
                ckpt = p1.find_policy_ckpt(p1.method_ckpt_dir(method, seed))
                print("  ckpt", ckpt, flush=True)
                model, _kind = p1.load_model(method, ckpt, device)
            packs = {}
            for split_name, frame in [
                ("train", splits["isic_train"]),
                ("id", splits["isic_test"]),
                ("ood", splits["pad"]),
            ]:
                z, y = extract_loader(model, fwd, loader(frame), device, max_batches=max_b)
                packs[split_name] = {"z": z, "y": y}
                print("  ", split_name, z.shape, flush=True)
            ztr, zid, zood = packs["train"]["z"], packs["id"]["z"], packs["ood"]["z"]
            if not args.dry_run:
                np.savez_compressed(npz, train=ztr, id=zid, ood=zood, y_train=packs["train"]["y"], y_id=packs["id"]["y"])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        log_agg, log_rows = logistic_ood(zid, zood)
        pca = pca_k1(ztr, zid, zood)
        rec = {
            "representation": name,
            "dim": int(ztr.shape[1]),
            "n_train": int(len(ztr)),
            "n_id": int(len(zid)),
            "n_ood": int(len(zood)),
            "supervision_of_representation": (
                "domain-supervised" if name == "z_context" else "unsupervised_or_imagenet_or_class_supervised"
            ),
            "linear_domain_head": log_agg,
            "linear_per_seed": log_rows,
            "pca_k1": pca,
        }
        results.append(rec)
        print(
            name,
            "dim",
            rec["dim"],
            "linAUROC",
            "{:.4f}".format(log_agg["AUROC_mean"]),
            "pca1",
            "{:.4f}".format(pca["AUROC"]),
            "var",
            "{:.3f}".format(pca["var_explained"]),
            flush=True,
        )

    dest = args.output_dir / ("supervised_head_dryrun.json" if args.dry_run else "supervised_head.json")
    dest.write_text(json.dumps({"device": str(device), "results": results}, indent=2) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
