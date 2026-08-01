#!/usr/bin/env python3
# Check latent domain leakage via linear probing on CSG-lite checkpoint.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule
from src.datasets.splits import build_id_ood_test_dataloaders
from src.models.baseline import BaselineResNet50
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import seed_everything

# ================= CONFIG =================
SEED = 42
BATCH_SIZE = 96
NUM_WORKERS = 12
METADATA_CSV = PROJECT_ROOT / "data" / "master_metadata.csv"
DEFAULT_CSG_CKPT = PROJECT_ROOT / "checkpoints" / "csg_lite"


def parse_args():
    p = argparse.ArgumentParser(description="Check domain leakage via linear probing (CSG-lite)")
    p.add_argument("--metadata", type=Path, default=METADATA_CSV, help="master_metadata.csv")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CSG_CKPT,
        help=".ckpt file or directory with CSG-lite checkpoints",
    )
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--test_size", type=float, default=0.3, help="Probe split ratio for held-out eval")
    p.add_argument("--max_iter", type=int, default=2000, help="LogisticRegression max_iter")
    p.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "csg", "baseline"],
        help="Checkpoint model type. auto tries to infer from checkpoint keys.",
    )
    p.add_argument(
        "--probe_seeds",
        type=str,
        default="42,52,62",
        help="Comma-separated seeds for multi-seed probe stability check (e.g. 42,52,62).",
    )
    p.add_argument(
        "--shuffle_test_seed",
        type=int,
        default=123,
        help="Seed for domain-label shuffle sanity test.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    return p.parse_args()


@torch.no_grad()
def collect_test_features_csg(model, loader):
    """
    Collect all test features in dataloader order.
    Returns numpy arrays: z_lesion, z_context, backbone_raw, lesion logits.
    """
    model.eval()
    z_lesion_all = []
    z_context_all = []
    backbone_all = []
    logits_all = []
    labels_all = []

    for images, labels in loader:
        images = images.to(next(model.parameters()).device, non_blocking=True)
        labels = labels.to(images.device, non_blocking=True)

        features = model.extract_context_features(images)
        z_lesion = model.encode_z_lesion(images)
        z_context = model.context_projector(features)
        logits, _, _ = model(images, x_lesion=None)

        z_lesion_all.append(z_lesion.cpu())
        z_context_all.append(z_context.cpu())
        backbone_all.append(features.cpu())
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())

    return (
        torch.cat(z_lesion_all, dim=0).numpy(),
        torch.cat(z_context_all, dim=0).numpy(),
        torch.cat(backbone_all, dim=0).numpy(),
        torch.cat(logits_all, dim=0).numpy(),
        torch.cat(labels_all, dim=0).numpy(),
    )


@torch.no_grad()
def collect_test_features_baseline(model, loader):
    """
    Baseline checkpoint has only one feature space (backbone pre-FC).
    For compatibility with leakage table, map:
      z_lesion <- backbone_raw
      z_context <- backbone_raw
    """
    model.eval()
    backbone_all = []
    logits_all = []
    labels_all = []
    for images, labels in loader:
        images = images.to(next(model.parameters()).device, non_blocking=True)
        labels = labels.to(images.device, non_blocking=True)
        logits, features = model(images, return_features=True)
        backbone_all.append(features.cpu())
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    backbone_np = torch.cat(backbone_all, dim=0).numpy()
    logits_np = torch.cat(logits_all, dim=0).numpy()
    labels_np = torch.cat(labels_all, dim=0).numpy()
    return backbone_np, backbone_np, backbone_np, logits_np, labels_np


def infer_model_type(ckpt_path):
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw.get("state_dict", raw if isinstance(raw, dict) else {})
    keys = list(state.keys()) if isinstance(state, dict) else []
    if any(k.startswith("model.lesion_backbone.") or k.startswith("model.context_backbone.") for k in keys):
        return "csg"
    if any(k.startswith("net.backbone.") for k in keys):
        return "baseline"
    # Fallback to csg, matching previous script behavior.
    return "csg"


def domain_to_index(values):
    out = []
    for v in values:
        s = str(v)
        if s == "isic":
            out.append(0)
        elif s == "pad_ufes":
            out.append(1)
        else:
            raise ValueError("Unknown domain value: {}".format(s))
    return np.asarray(out, dtype=np.int64)


def linear_probe_accuracy(features, domains, seed, test_size, max_iter):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        domains,
        test_size=test_size,
        random_state=seed,
        stratify=domains,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return float(accuracy_score(y_test, preds))


def parse_seed_list(text):
    vals = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("probe_seeds is empty. Provide e.g. --probe_seeds 42,52,62")
    return vals


def multi_seed_probe(features, domains, seeds, test_size, max_iter):
    accs = []
    for s in seeds:
        accs.append(linear_probe_accuracy(features, domains, seed=s, test_size=test_size, max_iter=max_iter))
    arr = np.asarray(accs, dtype=np.float64)
    return accs, float(arr.mean()), float(arr.std(ddof=0))


def compute_ece_from_logits(logits, labels, n_bins=15):
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


def main():
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError("Metadata CSV not found: {}".format(args.metadata))

    probe_seeds = parse_seed_list(args.probe_seeds)

    ckpt = ood_metrics.find_checkpoint(args.ckpt)
    if ckpt is None:
        raise FileNotFoundError("CSG-lite checkpoint not found from: {}".format(args.ckpt))

    seed_everything(SEED)
    pl.seed_everything(SEED, workers=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model_type = args.model_type
    if model_type == "auto":
        model_type = infer_model_type(ckpt)

    print("Loading checkpoint: {} (model_type={})".format(ckpt, model_type))
    if model_type == "csg":
        lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), strict=False)
        model = lit.model
    elif model_type == "baseline":
        lit = BaselineResNet50.load_from_checkpoint(str(ckpt))
        model = lit
    else:
        raise ValueError("Unsupported model_type: {}".format(model_type))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    datamodule = SkinDataModule(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    test_loader = datamodule.test_dataloader()
    test_df = datamodule.ood_test_dataset.data.reset_index(drop=True)
    domain_targets = domain_to_index(test_df["domain"].tolist())

    n_isic = int((domain_targets == 0).sum())
    n_pad = int((domain_targets == 1).sum())
    n_all = int(domain_targets.shape[0])
    maj_acc = max(n_isic, n_pad) / max(n_all, 1)

    if model_type == "csg":
        z_lesion, z_context, backbone_raw, logits_test, y_test = collect_test_features_csg(model, test_loader)
    else:
        z_lesion, z_context, backbone_raw, logits_test, y_test = collect_test_features_baseline(model, test_loader)
    if len(domain_targets) != z_lesion.shape[0]:
        raise RuntimeError(
            "Domain target size mismatch with collected features: {} vs {}".format(
                len(domain_targets), z_lesion.shape[0]
            )
        )

    accs_z_lesion, mean_z_lesion, std_z_lesion = multi_seed_probe(
        z_lesion, domain_targets, seeds=probe_seeds, test_size=args.test_size, max_iter=args.max_iter
    )
    accs_z_context, mean_z_context, std_z_context = multi_seed_probe(
        z_context, domain_targets, seeds=probe_seeds, test_size=args.test_size, max_iter=args.max_iter
    )
    accs_backbone, mean_backbone, std_backbone = multi_seed_probe(
        backbone_raw, domain_targets, seeds=probe_seeds, test_size=args.test_size, max_iter=args.max_iter
    )

    rng = np.random.default_rng(args.shuffle_test_seed)
    shuffled_domains = domain_targets.copy()
    rng.shuffle(shuffled_domains)
    acc_shuf_z_lesion = linear_probe_accuracy(
        z_lesion, shuffled_domains, seed=probe_seeds[0], test_size=args.test_size, max_iter=args.max_iter
    )

    print("\n=== Probe Balance Check ===")
    print("Probe samples total: {} | ISIC(domain=0): {} | PAD(domain=1): {}".format(n_all, n_isic, n_pad))
    print("Majority-class baseline accuracy: {:.4f}".format(maj_acc))

    print("\n=== Domain Leakage Check (Linear Probe, ISIC=0 vs PAD=1) ===")
    print("Probe split: train={:.0f}% / test={:.0f}%".format((1.0 - args.test_size) * 100, args.test_size * 100))
    print("Seeds: {}".format(probe_seeds))
    print("{:<16} {:>12} {:>12}".format("Feature", "Acc(mean)", "Acc(std)"))
    print("{:<16} {:>12.4f} {:>12.4f}".format("z_context", mean_z_context, std_z_context))
    print("{:<16} {:>12.4f} {:>12.4f}".format("backbone_raw", mean_backbone, std_backbone))
    print("{:<16} {:>12.4f} {:>12.4f}".format("z_lesion", mean_z_lesion, std_z_lesion))
    print("z_lesion per-seed acc: {}".format(["{:.4f}".format(v) for v in accs_z_lesion]))

    print("\n=== Shuffle Label Test (Sanity) ===")
    print("Shuffle seed: {}".format(args.shuffle_test_seed))
    print("z_lesion probe acc with shuffled domain labels: {:.4f}".format(acc_shuf_z_lesion))

    print("\nExpected direction:")
    print("- z_context: high domain accuracy (often > 0.90)")
    print("- z_lesion: as low as possible (toward 0.50 random)")
    if mean_z_lesion > 0.80:
        print("[WARN] z_lesion domain accuracy > 0.80 -> possible context leakage / GRL underperforming.")

    # ID lesion accuracy check on ISIC test only (8-class).
    id_loader, _ood_loader = build_id_ood_test_dataloaders(datamodule)
    logits_id, y_id = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in id_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if model_type == "csg":
                out, _, _ = model(images)
            else:
                out = model(images)
            logits_id.append(out.cpu())
            y_id.append(labels.cpu())
    logits_id = torch.cat(logits_id, dim=0).numpy()
    y_id = torch.cat(y_id, dim=0).numpy()
    pred_id = logits_id.argmax(axis=1)
    id_acc = float((pred_id == y_id).mean())
    id_bal_acc = float(balanced_accuracy_score(y_id, pred_id))
    id_ece = compute_ece_from_logits(logits_id, y_id)
    print("\n=== ID Accuracy Check (ISIC test, 8-class) ===")
    print("ID Test Accuracy: {:.4f}".format(id_acc))
    print("ID Balanced Accuracy: {:.4f}".format(id_bal_acc))
    print("ID ECE: {:.4f}".format(id_ece))
    if args.output_json is not None:
        payload = {
            "checkpoint": str(ckpt),
            "model_type": model_type,
            "probe_seeds": probe_seeds,
            "probe_total": int(n_all),
            "probe_isic": int(n_isic),
            "probe_pad": int(n_pad),
            "probe_majority_baseline_acc": float(maj_acc),
            "z_context_acc_mean": float(mean_z_context),
            "z_context_acc_std": float(std_z_context),
            "backbone_raw_acc_mean": float(mean_backbone),
            "backbone_raw_acc_std": float(std_backbone),
            "z_lesion_acc_mean": float(mean_z_lesion),
            "z_lesion_acc_std": float(std_z_lesion),
            "z_lesion_acc_per_seed": [float(v) for v in accs_z_lesion],
            "z_lesion_shuffle_acc": float(acc_shuf_z_lesion),
            "id_acc": float(id_acc),
            "id_balanced_acc": float(id_bal_acc),
            "id_ece": float(id_ece),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Saved leakage report: {}".format(args.output_json))


if __name__ == "__main__":
    main()
