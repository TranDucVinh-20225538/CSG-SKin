#!/usr/bin/env python3
"""Evaluate OOD scores on existing CSG checkpoints (no retraining).

Scores:
- Energy (from lesion logits)
- MSP (maximum softmax probability)
- Cosine prototype similarity on z_lesion
- Mahalanobis (cleaned): fit means/cov on ISIC train only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule, SkinDataset
from src.datasets.splits import build_id_ood_test_dataloaders, load_filtered_master
from src.models.csg_lightning import CSGLiteLightning
from src.utils import ood_metrics
from src.utils.paths import PROJECT_ROOT
from src.utils.seed import seed_everything


DEFAULT_METADATA = PROJECT_ROOT / "data" / "master_metadata_lesion_only_soft.csv"
DEFAULT_CKPTS = "checkpoints/csg_lite/runB_s42/best-31.ckpt,checkpoints/csg_lite/runB_s52/best-34.ckpt,checkpoints/csg_lite/runB_s62/best-28.ckpt"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Energy/MSP/Cosine/Mahalanobis on CSG checkpoints.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--ckpts", type=str, default=DEFAULT_CKPTS, help="Comma-separated checkpoint paths.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--temperatures",
        type=str,
        default="1,10,100",
        help="Comma-separated temperatures for Energy score (e.g. 1,10,100).",
    )
    p.add_argument("--reg_eps", type=float, default=1e-3, help="Diagonal regularization for cleaned Mahalanobis.")
    p.add_argument(
        "--debug_maha",
        action="store_true",
        help="Print min-Mahalanobis d^2 stats and a few concrete ID vs OOD samples (sanity: not label leak).",
    )
    p.add_argument(
        "--debug_n_samples",
        type=int,
        default=5,
        help="How many ID and OOD rows to print when --debug_maha.",
    )
    return p.parse_args()


@torch.no_grad()
def collect_logits_z_labels(model, loader, device):
    model.eval()
    logits_all, z_les_all, z_ctx_all, y_all = [], [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _dctx, _dadv, z_les, z_ctx = model(images, x_lesion=None, return_latents=True)
        logits_all.append(logits.cpu())
        z_les_all.append(z_les.cpu())
        z_ctx_all.append(z_ctx.cpu())
        y_all.append(labels.cpu())
    return (
        torch.cat(logits_all, dim=0).numpy(),
        torch.cat(z_les_all, dim=0).numpy(),
        torch.cat(z_ctx_all, dim=0).numpy(),
        torch.cat(y_all, dim=0).numpy(),
    )


def build_isic_train_loader_eval(metadata_csv, batch_size, num_workers, eval_transform):
    df = load_filtered_master(metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    if isic_df.empty:
        raise ValueError("No ISIC rows in metadata.")
    split_state = 42
    isic_train_val, _isic_test = train_test_split(
        isic_df,
        test_size=0.2,
        stratify=isic_df["label_idx"],
        random_state=split_state,
    )
    isic_train, _isic_val = train_test_split(
        isic_train_val,
        test_size=0.2,
        stratify=isic_train_val["label_idx"],
        random_state=split_state,
    )
    ds = SkinDataset(isic_train, transform=eval_transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def auroc_from_id_ood_scores(id_scores, ood_scores):
    y = np.concatenate([np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))


def _stats_line(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    return "{}: n={} min={:.4g} p50={:.4g} mean={:.4g} max={:.4g}".format(
        name, a.size, float(np.min(a)), float(np.median(a)), float(np.mean(a)), float(np.max(a))
    )


def print_maha_debug(ckpt_name, id_les, ood_les, id_ctx, ood_ctx, n_print, seed):
    """Show raw min squared Mahalanobis distances: ID = ISIC test, OOD = PAD (separate sets, no train leak)."""
    print("\n--- debug_maha: {} ---".format(ckpt_name))
    print("Min d^2 = distance to nearest class-conditional Gaussian (ISIC-train fit).")
    print(_stats_line("z_les  ID (ISIC test)", id_les))
    print(_stats_line("z_les  OOD (PAD)", ood_les))
    print(_stats_line("z_ctx  ID (ISIC test)", id_ctx))
    print(_stats_line("z_ctx  OOD (PAD)", ood_ctx))
    print(
        "Interpretation: OOD is never used to fit (mu, Sigma); ID test is held-out ISIC. "
        "If z_ctx OOD min d^2 >> z_ctx ID max d^2, AUROC ~1 is a real separation, not train-time OOD leak."
    )
    rng = np.random.default_rng(seed)
    n_id, n_ood = len(id_les), len(ood_les)
    id_ix = rng.choice(n_id, size=min(n_print, n_id), replace=False) if n_id else np.array([], dtype=int)
    ood_ix = rng.choice(n_ood, size=min(n_print, n_ood), replace=False) if n_ood else np.array([], dtype=int)
    print("Sample z_les min d^2 (idx -> value):")
    for i in id_ix:
        print("  ID  [{}]  {:.6g}".format(int(i), float(id_les[i])))
    for i in ood_ix:
        print("  OOD [{}]  {:.6g}".format(int(i), float(ood_les[i])))
    print("Sample z_ctx min d^2 (idx -> value):")
    for i in id_ix:
        print("  ID  [{}]  {:.6g}".format(int(i), float(id_ctx[i])))
    for i in ood_ix:
        print("  OOD [{}]  {:.6g}".format(int(i), float(ood_ctx[i])))


def main():
    args = parse_args()
    if not args.metadata.is_file():
        raise FileNotFoundError("Metadata not found: {}".format(args.metadata))
    temperatures = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    if not temperatures:
        raise ValueError("No valid temperatures parsed from --temperatures.")
    ckpts = [Path(x.strip()) for x in args.ckpts.split(",") if x.strip()]
    if not ckpts:
        raise ValueError("No checkpoints provided.")

    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = SkinDataModule(metadata_csv=args.metadata, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()
    id_loader, ood_loader = build_id_ood_test_dataloaders(dm)
    isic_train_loader = build_isic_train_loader_eval(
        args.metadata, args.batch_size, args.num_workers, dm.eval_transform
    )

    t_headers = ["E@T={}".format(int(t) if t.is_integer() else t) for t in temperatures]
    print(
        "Checkpoint | {} | MSP | Cos(z_les) | Cos(z_ctx) | Maha(z_les) | Maha(z_ctx)".format(
            " | ".join(t_headers)
        )
    )
    for ckpt in ckpts:
        if not ckpt.is_absolute():
            ckpt = (PROJECT_ROOT / ckpt).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError("Missing checkpoint: {}".format(ckpt))

        lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), strict=False)
        model = lit.model.to(device)

        # Collect logits / latents.
        logits_tr, zles_tr, zctx_tr, y_tr = collect_logits_z_labels(model, isic_train_loader, device)
        logits_id, zles_id, zctx_id, _y_id = collect_logits_z_labels(model, id_loader, device)
        logits_ood, zles_ood, zctx_ood, _y_ood = collect_logits_z_labels(model, ood_loader, device)

        # 1) Energy score (higher is more OOD-like), tested at multiple temperatures.
        energy_aurocs = []
        for t in temperatures:
            id_energy = (-t * torch.logsumexp(torch.from_numpy(logits_id) / t, dim=1)).numpy()
            ood_energy = (-t * torch.logsumexp(torch.from_numpy(logits_ood) / t, dim=1)).numpy()
            energy_aurocs.append(auroc_from_id_ood_scores(id_energy, ood_energy))

        # 2) MSP score (higher is more OOD-like when negated).
        id_probs = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
        ood_probs = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
        id_msp = -id_probs.max(axis=1)
        ood_msp = -ood_probs.max(axis=1)
        msp_auroc = auroc_from_id_ood_scores(id_msp, ood_msp)

        # 3) Cosine prototype score on z_lesion / z_context (higher -> OOD-like via negated max cosine).
        n_classes = int(logits_id.shape[1])
        prot_les = np.zeros((n_classes, zles_tr.shape[1]), dtype=np.float64)
        prot_ctx = np.zeros((n_classes, zctx_tr.shape[1]), dtype=np.float64)
        for c in range(n_classes):
            m = y_tr == c
            if not np.any(m):
                raise ValueError("Class {} absent in ISIC train; cannot build prototype.".format(c))
            prot_les[c] = zles_tr[m].mean(axis=0)
            prot_ctx[c] = zctx_tr[m].mean(axis=0)
        p_les = prot_les / (np.linalg.norm(prot_les, axis=1, keepdims=True) + 1e-12)
        p_ctx = prot_ctx / (np.linalg.norm(prot_ctx, axis=1, keepdims=True) + 1e-12)
        zles_id_n = zles_id / (np.linalg.norm(zles_id, axis=1, keepdims=True) + 1e-12)
        zles_ood_n = zles_ood / (np.linalg.norm(zles_ood, axis=1, keepdims=True) + 1e-12)
        zctx_id_n = zctx_id / (np.linalg.norm(zctx_id, axis=1, keepdims=True) + 1e-12)
        zctx_ood_n = zctx_ood / (np.linalg.norm(zctx_ood, axis=1, keepdims=True) + 1e-12)
        id_cos_les = -(zles_id_n @ p_les.T).max(axis=1)
        ood_cos_les = -(zles_ood_n @ p_les.T).max(axis=1)
        id_cos_ctx = -(zctx_id_n @ p_ctx.T).max(axis=1)
        ood_cos_ctx = -(zctx_ood_n @ p_ctx.T).max(axis=1)
        cos_les_auroc = auroc_from_id_ood_scores(id_cos_les, ood_cos_les)
        cos_ctx_auroc = auroc_from_id_ood_scores(id_cos_ctx, ood_cos_ctx)

        # 4) Mahalanobis cleaned: fit on ISIC train only + stronger diagonal regularization.
        means_les, precision_les = ood_metrics.compute_mahalanobis_params_from_arrays(
            zles_tr, y_tr, num_classes=n_classes, reg_eps=args.reg_eps
        )
        means_ctx, precision_ctx = ood_metrics.compute_mahalanobis_params_from_arrays(
            zctx_tr, y_tr, num_classes=n_classes, reg_eps=args.reg_eps
        )
        id_maha_les = ood_metrics.mahalanobis_min_squared_distances(zles_id, means_les, precision_les)
        ood_maha_les = ood_metrics.mahalanobis_min_squared_distances(zles_ood, means_les, precision_les)
        id_maha_ctx = ood_metrics.mahalanobis_min_squared_distances(zctx_id, means_ctx, precision_ctx)
        ood_maha_ctx = ood_metrics.mahalanobis_min_squared_distances(zctx_ood, means_ctx, precision_ctx)
        maha_les_auroc = auroc_from_id_ood_scores(id_maha_les, ood_maha_les)
        maha_ctx_auroc = auroc_from_id_ood_scores(id_maha_ctx, ood_maha_ctx)

        energy_str = " | ".join("{:.4f}".format(v) for v in energy_aurocs)
        print(
            "{} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
                ckpt.name, energy_str, msp_auroc, cos_les_auroc, cos_ctx_auroc, maha_les_auroc, maha_ctx_auroc
            )
        )
        if args.debug_maha:
            print_maha_debug(
                ckpt.name,
                id_maha_les,
                ood_maha_les,
                id_maha_ctx,
                ood_maha_ctx,
                n_print=args.debug_n_samples,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()

