# scripts/train_csg.py
# Train CSG-lite (EffNet-B3) + Mahalanobis OOD on z_lesion.

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

from src.datasets.skin_dataset import SkinDataModule
from src.datasets.splits import build_id_ood_test_dataloaders
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import seed_everything

# ================= CONFIG =================
# Paired ISIC+PAD loader: each step stacks batch_size ISIC + batch_size PAD -> 2*batch_size images.
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 8
MAX_EPOCHS = 20
LR = 2e-4
WEIGHT_DECAY = 1e-4
LAMBDA_CTX = 1.0
# Domain leakage is currently very strong; start with stronger adversarial weight.
LAMBDA_ADV = 5.0
LAMBDA_ORTH = 10.0
LAMBDA_SUPCON = 1.0
SUPCON_TEMPERATURE = 0.1
ADV_LR_MULTIPLIER = 30.0
LESION_CLS_LR_MULTIPLIER = 0.2
GRL_LAMBDA = 1.0
GRL_GAMMA = 20.0
GRL_PROGRESS_POWER = 0.5
GRL_ALPHA_MIN = 0.2
PRINT_EVERY_N_STEPS = 0
LESION_LATENT_DIM = 16
CONTEXT_LATENT_DIM = 64
METADATA_CSV = PROJECT_ROOT / "data" / "master_metadata.csv"
CKPT_CSG_DIR = PROJECT_ROOT / "checkpoints" / "csg_lite"
RESULTS_DIR = PROJECT_ROOT / "results" / "csg_lite"


def parse_args():
    p = argparse.ArgumentParser(description="Train CSG-lite (EffNet-B3) + Mahalanobis OOD")
    p.add_argument("--metadata", type=Path, default=METADATA_CSV, help="master_metadata.csv")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--lr", type=float, default=LR, help="AdamW lr")
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    p.add_argument(
        "--backbone_variant",
        type=str,
        default="b3",
        choices=["b0", "b3"],
        help="Dual-encoder backbone variant (b0 lighter VRAM, b3 stronger).",
    )
    p.add_argument("--lesion_latent_dim", type=int, default=LESION_LATENT_DIM)
    p.add_argument("--context_latent_dim", type=int, default=CONTEXT_LATENT_DIM)
    p.add_argument("--lambda_ctx", type=float, default=LAMBDA_CTX)
    p.add_argument("--lambda_adv", type=float, default=LAMBDA_ADV)
    p.add_argument("--lambda_orth", type=float, default=LAMBDA_ORTH, help="Orthogonal loss weight between z_lesion and z_context.")
    p.add_argument(
        "--lambda_supcon",
        type=float,
        default=LAMBDA_SUPCON,
        help="Cross-domain supervised contrastive loss weight on z_lesion.",
    )
    p.add_argument("--supcon_temperature", type=float, default=SUPCON_TEMPERATURE)
    p.add_argument(
        "--adv_lr_multiplier",
        type=float,
        default=ADV_LR_MULTIPLIER,
        help="Multiplier for adversarial head learning rate (domain_classifier_adv).",
    )
    p.add_argument(
        "--lesion_cls_lr_multiplier",
        type=float,
        default=LESION_CLS_LR_MULTIPLIER,
        help="Multiplier for lesion classifier learning rate (to reduce classifier dominance).",
    )
    p.add_argument("--grl_lambda", type=float, default=GRL_LAMBDA)
    p.add_argument("--grl_gamma", type=float, default=GRL_GAMMA, help="Steepness for dynamic GRL alpha schedule.")
    p.add_argument(
        "--grl_progress_power",
        type=float,
        default=GRL_PROGRESS_POWER,
        help="Progress shaping power (<1 rises faster early, >1 rises slower).",
    )
    p.add_argument(
        "--grl_alpha_min",
        type=float,
        default=GRL_ALPHA_MIN,
        help="Minimum GRL alpha floor to enforce adversarial pressure from early steps.",
    )
    p.add_argument(
        "--print_every_n_steps",
        type=int,
        default=PRINT_EVERY_N_STEPS,
        help="Deprecated: step-wise printing (set 0). Epoch summary is printed automatically.",
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

    run_name = args.run_name.strip() or "csg_b{}_e{}_s{}".format(args.backbone_variant, args.max_epochs, args.seed)
    run_dir = args.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_simple_yaml(
        run_dir / "config.yaml",
        {
            "script": "train_csg.py",
            "run_name": run_name,
            "seed": args.seed,
            "metadata": args.metadata,
            "backbone_variant": args.backbone_variant,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lesion_latent_dim": args.lesion_latent_dim,
            "context_latent_dim": args.context_latent_dim,
            "lambda_ctx": args.lambda_ctx,
            "lambda_adv": args.lambda_adv,
            "lambda_orth": args.lambda_orth,
            "lambda_supcon": args.lambda_supcon,
            "supcon_temperature": args.supcon_temperature,
            "adv_lr_multiplier": args.adv_lr_multiplier,
            "lesion_cls_lr_multiplier": args.lesion_cls_lr_multiplier,
            "grl_lambda": args.grl_lambda,
            "grl_gamma": args.grl_gamma,
            "grl_progress_power": args.grl_progress_power,
            "grl_alpha_min": args.grl_alpha_min,
        },
    )

    datamodule = SkinDataModule(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        csg_lite_train=True,
    )
    datamodule.setup()
    csg_train_loader = datamodule.train_dataloader()

    model = CSGLiteLightning(
        lesion_latent_dim=args.lesion_latent_dim,
        context_latent_dim=args.context_latent_dim,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lambda_ctx=args.lambda_ctx,
        lambda_adv=args.lambda_adv,
        lambda_orth=args.lambda_orth,
        lambda_supcon=args.lambda_supcon,
        supcon_temperature=args.supcon_temperature,
        adv_lr_multiplier=args.adv_lr_multiplier,
        lesion_cls_lr_multiplier=args.lesion_cls_lr_multiplier,
        grl_lambda=args.grl_lambda,
        grl_gamma=args.grl_gamma,
        grl_progress_power=args.grl_progress_power,
        grl_alpha_min=args.grl_alpha_min,
        print_every_n_steps=args.print_every_n_steps,
        backbone_variant=args.backbone_variant,
    )

    ckpt_dir = CKPT_CSG_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    loggers = [pl.loggers.CSVLogger(save_dir=str(run_dir / "csv_logs"), name="", version="")]
    if importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"):
        loggers.append(pl.loggers.TensorBoardLogger(save_dir=str(run_dir / "tb_logs"), name="", version=""))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[ckpt_cb],
        log_every_n_steps=20,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=loggers,
    )

    val_loader = datamodule.val_dataloader()
    trainer.fit(model, train_dataloaders=csg_train_loader, val_dataloaders=val_loader)

    best_csg = ood_metrics.find_checkpoint(Path(ckpt_cb.best_model_path)) if ckpt_cb.best_model_path else None
    if best_csg:
        print("Loading best checkpoint: {}".format(best_csg))
        model = CSGLiteLightning.load_from_checkpoint(str(best_csg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    csg_net = model.model

    train_loader = datamodule.train_dataloader()
    z_train, y_train = ood_metrics.collect_z_lesion_labels_csg(csg_net, train_loader, device)
    means_csg, prec_csg = ood_metrics.compute_mahalanobis_params_from_arrays(z_train, y_train)

    id_loader, ood_loader = build_id_ood_test_dataloaders(datamodule)
    z_id, _ = ood_metrics.collect_z_lesion_labels_csg(csg_net, id_loader, device)
    z_ood, _ = ood_metrics.collect_z_lesion_labels_csg(csg_net, ood_loader, device)
    csg_auroc, csg_fpr95 = ood_metrics.mahalanobis_auroc_fpr95(z_id, z_ood, means_csg, prec_csg)
    logits_id, y_id = [], []
    csg_net.eval()
    with torch.no_grad():
        for images, labels in id_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out, _, _ = csg_net(images)
            logits_id.append(out.cpu())
            y_id.append(labels.cpu())
    logits_id = torch.cat(logits_id, dim=0).numpy()
    y_id = torch.cat(y_id, dim=0).numpy()
    pred_id = logits_id.argmax(axis=1)
    id_acc = float((pred_id == y_id).mean())
    id_bal_acc = float(balanced_accuracy_score(y_id, pred_id))
    id_ece = _compute_ece(logits_id, y_id)

    print("\n=== Final Evaluation ===")
    print("ID Test Accuracy: {:.4f}".format(id_acc))
    print("ID Balanced Accuracy: {:.4f}".format(id_bal_acc))
    print("ID ECE: {:.4f}".format(id_ece))
    print("Mahalanobis OOD (ID=ISIC test, OOD=PAD-UFES)")
    print("Feature space: z_lesion (EffNet-B3)")
    print("{:<14} {:>8} {:>8}".format("Method", "AUROC", "FPR@95"))
    print("{:<14} {:>8.4f} {:>8.4f}".format("CSG-lite", csg_auroc, csg_fpr95))
    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "best_checkpoint": str(best_csg) if best_csg else "",
        "id_acc": id_acc,
        "id_balanced_acc": id_bal_acc,
        "id_ece": id_ece,
        "ood_maha_auroc": float(csg_auroc),
        "ood_maha_fpr95": float(csg_fpr95),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved run artifacts: {}".format(run_dir))


if __name__ == "__main__":
    main()
