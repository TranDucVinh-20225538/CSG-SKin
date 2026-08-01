#!/usr/bin/env python3
"""Run EffNet-B3 single-encoder control experiment with 5-seed evaluation."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.skin_dataset import SkinDataModule
from src.datasets.splits import build_id_ood_test_dataloaders
from src.models.effb3_single import EffB3SingleLightning
from src.utils import ood_metrics
from src.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="EffNet-B3 single-encoder control (5 seeds).")
    p.add_argument("--metadata", type=Path, default=ROOT / "data" / "master_metadata_lesion_only_soft.csv")
    p.add_argument("--seeds", type=str, default="42,52,62,72,82")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--run_train", action="store_true", help="Train missing seeds if summary absent.")
    p.add_argument("--output_dir", type=Path, default=ROOT / "results" / "effb3_control")
    return p.parse_args()


def _parse_seeds(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _ece(logits, labels, n_bins=15):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(labels), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece += (float(m.sum()) / n) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def _find_ckpt(ckpt_dir):
    if not ckpt_dir.exists():
        return None
    best = sorted(ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = sorted(ckpt_dir.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return last[0] if last else None


@torch.no_grad()
def _collect_logits_features(model, loader, device):
    logits, feats, labels = [], [], []
    model.eval()
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        out, z = model(images, return_features=True)
        logits.append(out.cpu())
        feats.append(z.cpu())
        labels.append(y.cpu())
    return (
        torch.cat(logits, dim=0).numpy(),
        torch.cat(feats, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
    )


def _domain_labels_for_test(dm):
    df = dm.ood_test_dataset.data.reset_index(drop=True)
    return np.asarray((df["domain"].astype(str) == "pad_ufes").astype(np.int64), dtype=np.int64)


def _probe_acc(features, domains, seeds):
    vals = []
    for s in seeds:
        xtr, xte, ytr, yte = train_test_split(features, domains, test_size=0.3, random_state=s, stratify=domains)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
        clf.fit(xtr, ytr)
        vals.append(float(accuracy_score(yte, clf.predict(xte))))
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _ood_scores(logits_tr, z_tr, y_tr, logits_id, z_id, logits_ood, z_ood):
    n_classes = int(logits_id.shape[1])
    y_bin = np.concatenate([np.zeros(len(z_id), dtype=np.int64), np.ones(len(z_ood), dtype=np.int64)])
    out = {}

    # MSP
    p_id = torch.softmax(torch.from_numpy(logits_id), dim=1).numpy()
    p_ood = torch.softmax(torch.from_numpy(logits_ood), dim=1).numpy()
    s_id = -p_id.max(axis=1)
    s_ood = -p_ood.max(axis=1)
    out["MSP"] = {"AUROC": float(roc_auc_score(y_bin, np.concatenate([s_id, s_ood])))}

    # Energy
    s_id = (-torch.logsumexp(torch.from_numpy(logits_id), dim=1)).numpy()
    s_ood = (-torch.logsumexp(torch.from_numpy(logits_ood), dim=1)).numpy()
    out["Energy"] = {"AUROC": float(roc_auc_score(y_bin, np.concatenate([s_id, s_ood])))}

    # Cosine
    prot = np.zeros((n_classes, z_tr.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        m = y_tr == c
        if np.any(m):
            prot[c] = z_tr[m].mean(axis=0)
    prot = prot / (np.linalg.norm(prot, axis=1, keepdims=True) + 1e-12)
    zid = z_id / (np.linalg.norm(z_id, axis=1, keepdims=True) + 1e-12)
    zood = z_ood / (np.linalg.norm(z_ood, axis=1, keepdims=True) + 1e-12)
    s_id = -(zid @ prot.T).max(axis=1)
    s_ood = -(zood @ prot.T).max(axis=1)
    out["Cosine"] = {"AUROC": float(roc_auc_score(y_bin, np.concatenate([s_id, s_ood])))}

    # Mahalanobis
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(z_tr, y_tr, num_classes=n_classes, reg_eps=1e-3)
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
    out["Mahalanobis"] = {"AUROC": float(roc_auc_score(y_bin, np.concatenate([s_id, s_ood])))}
    return out


def _train_if_needed(args, seed, run_name, run_dir):
    summary_fp = run_dir / "summary.json"
    if summary_fp.exists():
        return
    if not args.run_train:
        raise FileNotFoundError(
            f"Missing {summary_fp}. Re-run with --run_train to train this seed."
        )

    seed_everything(seed)
    pl.seed_everything(seed, workers=True)
    dm = SkinDataModule(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        csg_lite_train=False,
        use_robust_transforms=True,
    )
    dm.setup()
    model = EffB3SingleLightning(
        num_classes=8,
        latent_dim=args.latent_dim,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pretrained=True,
    )
    ckpt_dir = ROOT / "checkpoints" / "effb3_single" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cb = pl.callbacks.ModelCheckpoint(
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
        callbacks=[cb],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=20,
    )
    trainer.fit(model, datamodule=dm)


def main():
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_seed").mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    leakage_rows = []
    ood_rows = []

    for seed in seeds:
        run_name = f"effb3_single_s{seed}"
        run_dir = out / "per_seed" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        _train_if_needed(args, seed, run_name, run_dir)

        ckpt = _find_ckpt(ROOT / "checkpoints" / "effb3_single" / run_name)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found for {run_name}")

        seed_everything(seed)
        pl.seed_everything(seed, workers=True)
        dm = SkinDataModule(
            metadata_csv=args.metadata,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            csg_lite_train=False,
            use_robust_transforms=True,
        )
        dm.setup()
        id_loader, ood_loader = build_id_ood_test_dataloaders(dm)
        train_loader = dm.train_dataloader()

        model = EffB3SingleLightning.load_from_checkpoint(str(ckpt))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        logits_tr, z_tr, y_tr = _collect_logits_features(model, train_loader, device)
        logits_id, z_id, y_id = _collect_logits_features(model, id_loader, device)
        logits_ood, z_ood, _y_ood = _collect_logits_features(model, ood_loader, device)

        pred_id = logits_id.argmax(axis=1)
        acc = float((pred_id == y_id).mean())
        bacc = float(balanced_accuracy_score(y_id, pred_id))
        ece = _ece(logits_id, y_id)
        metrics_rows.append({"seed": seed, "acc": acc, "balanced_acc": bacc, "ece": ece})

        domains = _domain_labels_for_test(dm)
        z_test = np.concatenate([z_id, z_ood], axis=0)
        leak_m, leak_s = _probe_acc(z_test, domains, seeds)
        leakage_rows.append({"seed": seed, "leakage_probe_mean": leak_m, "leakage_probe_std": leak_s})

        scores = _ood_scores(logits_tr, z_tr, y_tr, logits_id, z_id, logits_ood, z_ood)
        for st, vals in scores.items():
            ood_rows.append({"seed": seed, "score_type": st, "auroc": float(vals["AUROC"])})

        payload = {
            "seed": seed,
            "checkpoint": str(ckpt),
            "id_metrics": {"acc": acc, "balanced_acc": bacc, "ece": ece},
            "leakage_probe": {"mean": leak_m, "std": leak_s},
            "ood": scores,
        }
        (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Save required files
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "acc", "balanced_acc", "ece"])
        w.writeheader()
        for r in metrics_rows:
            w.writerow(r)

    with (out / "leakage_probe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "leakage_probe_mean", "leakage_probe_std"])
        w.writeheader()
        for r in leakage_rows:
            w.writerow(r)

    with (out / "ood_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "score_type", "auroc"])
        w.writeheader()
        for r in ood_rows:
            w.writerow(r)

    # Comparison table
    cbm_dir = ROOT / "results" / "cbm_revision"
    b = json.loads((cbm_dir / "baseline_soft_n5.json").read_text(encoding="utf-8"))["aggregate"]
    a = json.loads((cbm_dir / "runA_grl_n5.json").read_text(encoding="utf-8"))["aggregate"]
    r = json.loads((cbm_dir / "runB_orth1_n5.json").read_text(encoding="utf-8"))["aggregate"]
    m_arr = np.asarray([r["balanced_acc"] for r in metrics_rows], dtype=np.float64)
    e_arr = np.asarray([r["ece"] for r in metrics_rows], dtype=np.float64)
    l_arr = np.asarray([r["leakage_probe_mean"] for r in leakage_rows], dtype=np.float64)
    ood_df = {}
    for st in ["Mahalanobis", "Energy", "MSP", "Cosine"]:
        vals = [x["auroc"] for x in ood_rows if x["score_type"] == st]
        if vals:
            ood_df[st] = float(np.mean(vals))
    ood_best = max(ood_df.items(), key=lambda x: x[1])[0] if ood_df else "N/A"
    ood_best_val = ood_df.get(ood_best, float("nan"))

    comparison = [
        ["ResNet50 Baseline", b["id_balanced_acc"]["mean"], b["id_ece"]["mean"], b["z_lesion_acc_mean"]["mean"], b["ood_maha_auroc"]["mean"]],
        ["EffNet-B3 Single Encoder", float(m_arr.mean()), float(e_arr.mean()), float(l_arr.mean()), float(ood_best_val)],
        ["Run A / GRL", a["id_balanced_acc"]["mean"], a["id_ece"]["mean"], a["z_lesion_acc_mean"]["mean"], a["ood_maha_auroc"]["mean"]],
        ["Run B (orth=1.0)", r["id_balanced_acc"]["mean"], r["id_ece"]["mean"], r["z_lesion_acc_mean"]["mean"], r["ood_maha_auroc"]["mean"]],
    ]

    comp_fp = out / "comparison_table.csv"
    with comp_fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Bal Acc", "ECE", "Leakage", "OOD AUROC"])
        for row in comparison:
            w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}"])

    text_lines = []
    text_lines.append("EffNet-B3 Single Encoder Control Summary")
    text_lines.append(f"Seeds: {seeds}")
    text_lines.append("")
    text_lines.append("Aggregate (EffNet-B3 single encoder):")
    text_lines.append(f"- Accuracy: {np.mean([r['acc'] for r in metrics_rows]):.4f} +- {np.std([r['acc'] for r in metrics_rows], ddof=0):.4f}")
    text_lines.append(f"- Balanced Accuracy: {m_arr.mean():.4f} +- {m_arr.std(ddof=0):.4f}")
    text_lines.append(f"- ECE: {e_arr.mean():.4f} +- {e_arr.std(ddof=0):.4f}")
    text_lines.append(f"- Leakage probe: {l_arr.mean():.4f} +- {l_arr.std(ddof=0):.4f}")
    text_lines.append(f"- Best OOD score type by AUROC: {ood_best} ({ood_best_val:.4f})")
    text_lines.append("")
    text_lines.append("Interpretation hint:")
    text_lines.append(
        "If EffNet-B3 improves utility but still retains high leakage vs CSG, "
        "this supports that backbone capacity improves utility while disentanglement improves representation quality."
    )
    (out / "summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print(f"Saved: {out / 'metrics.csv'}")
    print(f"Saved: {out / 'leakage_probe.csv'}")
    print(f"Saved: {out / 'ood_scores.csv'}")
    print(f"Saved: {comp_fp}")
    print(f"Saved: {out / 'summary.txt'}")


if __name__ == "__main__":
    main()

