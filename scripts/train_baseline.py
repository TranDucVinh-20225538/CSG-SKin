# scripts/train_baseline.py
# Train ResNet50 baseline + MSP / Energy / Mahalanobis OOD evaluation (PAD-UFES OOD).

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule, audit_label_consistency_and_print_head
from src.models.baseline import BaselineResNet50
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import seed_everything
from src.utils import ood_metrics
from src.datasets.splits import build_id_ood_test_dataloaders

# ================= CONFIG =================
# Aligned with Ban_sao_datn/scripts/train_resnet50_robust.py (AdamW, aug, warmup+cosine).
SEED = 42
BATCH_SIZE = 96
NUM_WORKERS = 12
MAX_EPOCHS = 40
WARMUP_EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
NUM_CLASSES = 8
METADATA_CSV = PROJECT_ROOT / "data" / "master_metadata.csv"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet50 baseline (CSG-Skin)")
    p.add_argument("--metadata", type=Path, default=METADATA_CSV, help="master_metadata.csv")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--lr", type=float, default=LR, help="AdamW lr (try 1e-4 or 3e-4)")
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        help="CrossEntropy label smoothing (0 disables)",
    )
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    p.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS, help="LR warmup epochs")
    p.add_argument(
        "--no_robust_transforms",
        action="store_true",
        help="Use lighter Resize(224) aug instead of robust Resize256/RandomResizedCrop/RandomErasing",
    )
    p.add_argument("--seed", type=int, default=SEED, help="Global seed for reproducibility.")
    p.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name. If empty, an automatic name is used.",
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to store run artifacts (config/metrics/logs).",
    )
    return p.parse_args()


def _compute_ece(logits, labels, n_bins=15):
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
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (float(mask.sum()) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def _write_simple_yaml(path, data):
    lines = []
    for k in sorted(data.keys()):
        v = data[k]
        if isinstance(v, Path):
            v = str(v)
        lines.append("{}: {}".format(k, json.dumps(v)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError("Metadata CSV not found: {}".format(args.metadata))

    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    run_name = args.run_name.strip() or "baseline_e{}_s{}".format(args.max_epochs, args.seed)
    run_dir = args.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_simple_yaml(
        run_dir / "config.yaml",
        {
            "script": "train_baseline.py",
            "run_name": run_name,
            "seed": args.seed,
            "metadata": args.metadata,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing,
            "use_robust_transforms": not args.no_robust_transforms,
        },
    )

    datamodule = SkinDataModule(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_robust_transforms=not args.no_robust_transforms,
    )
    datamodule.setup()
    audit_label_consistency_and_print_head(datamodule, n_rows=5)

    model = BaselineResNet50(
        num_classes=NUM_CLASSES,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        sanity_isic_eval_items=getattr(datamodule, "sanity_isic_eval_items", None),
    )

    ckpt_dir = CKPT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    loggers = [pl.loggers.CSVLogger(save_dir=str(run_dir / "csv_logs"), name="", version="")]
    if importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"):
        loggers.append(pl.loggers.TensorBoardLogger(save_dir=str(run_dir / "tb_logs"), name="", version=""))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_cb],
        log_every_n_steps=20,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=loggers,
    )

    trainer.fit(model, datamodule=datamodule)

    if checkpoint_cb.best_model_path:
        print("Loading best checkpoint: {}".format(checkpoint_cb.best_model_path))
        model = BaselineResNet50.load_from_checkpoint(checkpoint_cb.best_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = datamodule.train_dataloader()
    means, precision = ood_metrics.compute_mahalanobis_params_baseline(
        model, train_loader, device, num_classes=NUM_CLASSES
    )

    id_loader, ood_loader = build_id_ood_test_dataloaders(datamodule)
    ood_metrics.print_baseline_id_ood_table(model, id_loader, ood_loader, device, means, precision)
    logits_id, y_id, _features_id = ood_metrics.collect_logits_labels_features_baseline(model, id_loader, device)
    pred_id = logits_id.argmax(axis=1)
    id_acc = float((pred_id == y_id).mean())
    id_bal_acc = float(balanced_accuracy_score(y_id, pred_id))
    id_ece = _compute_ece(logits_id, y_id)
    id_maha = ood_metrics.mahalanobis_min_squared_distances(_features_id, means, precision)
    logits_ood, _y_ood, features_ood = ood_metrics.collect_logits_labels_features_baseline(model, ood_loader, device)
    ood_maha = ood_metrics.mahalanobis_min_squared_distances(features_ood, means, precision)
    y_true = np.concatenate([np.zeros_like(id_maha), np.ones_like(ood_maha)]).astype(int)
    maha_scores = np.concatenate([id_maha, ood_maha])
    maha_auroc = float(ood_metrics.roc_auc_score(y_true, maha_scores))
    maha_fpr95 = float(ood_metrics.fpr_at_95_tpr(y_true, maha_scores))
    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "best_checkpoint": str(checkpoint_cb.best_model_path or ""),
        "id_acc": id_acc,
        "id_balanced_acc": id_bal_acc,
        "id_ece": id_ece,
        "ood_maha_auroc": maha_auroc,
        "ood_maha_fpr95": maha_fpr95,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved run artifacts: {}".format(run_dir))


if __name__ == "__main__":
    main()
