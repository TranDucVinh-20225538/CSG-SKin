#!/usr/bin/env python3
"""Phase 16 B1 — coarse λ_adv scan, BCN (ID) vs HAM (OOD), unmasked 448px."""

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
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import Dataset

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
for p in (REPO, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import phase16_bcn_ham_data as bdata
import train_phase2_pad_holdout as p2
from phase15_common import auroc_pair, detector_block, ece_from_logits, leakage_probe
from src.utils import ood_metrics
from phase16_b0 import build_eval_tf, build_train_tf
from src.datasets.skin_dataset import SkinDataset
from src.utils.seed import seed_everything
from train_phase15_1_single_dann import SingleDannLightning, collect_all, gate_from_log, lam_tag

OUT_ROOT = ROOT / "results" / "paperB" / "phase16" / "b1"
CKPT_ROOT = ROOT / "checkpoints" / "phase16" / "b1"
SPLIT_PATH = OUT_ROOT / "b1_split_seed42.json"
SPLIT_SEED = 42  # lesion partitions fixed; --seed is training/init (+ reweighted bootstrap) only
PREREG = ROOT / "results" / "paperB" / "phase16" / "PREREGISTER_B.json"
COARSE_GRID = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8]


class BcnHamPairedDataset(Dataset):
    def __init__(self, bcn_df: pd.DataFrame, ham_adv_df: pd.DataFrame, transform):
        self.bcn = bcn_df.reset_index(drop=True)
        self.ham = ham_adv_df.reset_index(drop=True)
        self.transform = transform
        self.n_bcn = len(self.bcn)
        self.n_ham = len(self.ham)
        if self.n_bcn == 0 or self.n_ham == 0:
            raise ValueError("empty BCN train or HAM adv")

    def __len__(self):
        return max(self.n_bcn, self.n_ham)

    def _item(self, row, domain_id: int):
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        d = torch.tensor(domain_id, dtype=torch.long)
        return img, y, d

    def __getitem__(self, idx):
        bcn_row = self.bcn.iloc[idx % self.n_bcn]
        ham_row = self.ham.iloc[idx % self.n_ham]
        return self._item(bcn_row, 0), self._item(ham_row, 1)


def bcn_ham_paired_collate(batch):
    x_b = torch.stack([b[0][0] for b in batch])
    y_b = torch.stack([b[0][1] for b in batch])
    x_h = torch.stack([b[1][0] for b in batch])
    y_h = torch.stack([b[1][1] for b in batch])
    images_ctx = torch.cat([x_b, x_h], dim=0)
    y_cls = torch.cat([y_b, y_h], dim=0)
    domain = torch.cat(
        [torch.zeros(len(batch), dtype=torch.long), torch.ones(len(batch), dtype=torch.long)]
    )
    return images_ctx, images_ctx, y_cls, y_cls, domain


def assert_b1_hygiene(bcn_train, bcn_val, bcn_test, ham_adv, ham_hold):
    sets = {
        "bcn_train": set(bcn_train["path"].astype(str)),
        "bcn_val": set(bcn_val["path"].astype(str)),
        "bcn_test": set(bcn_test["path"].astype(str)),
        "ham_adv": set(ham_adv["path"].astype(str)),
        "ham_heldout": set(ham_hold["path"].astype(str)),
    }
    for a, b in (
        ("bcn_train", "bcn_val"),
        ("bcn_train", "bcn_test"),
        ("bcn_val", "bcn_test"),
        ("ham_adv", "ham_heldout"),
    ):
        if sets[a] & sets[b]:
            raise RuntimeError(f"path overlap {a}/{b}")
    if sets["ham_adv"] & sets["bcn_train"]:
        pass  # different sites, paths unique by image
    return {k: len(v) for k, v in sets.items()}


def maha_ood_variants_full(z_tr, y_tr, z_id, y_id, z_ood, y_ood, n_classes=8, n_rep=4000, seed=0):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(
        z_tr, y_tr, num_classes=n_classes, reg_eps=1e-3
    )
    sid = ood_metrics.mahalanobis_min_squared_distances(z_id, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, prec)
    unrestricted = auroc_pair(sid, sood)
    shared = sorted(set(np.unique(y_id).tolist()) & set(np.unique(y_ood).tolist()))
    m_shared = np.isin(y_id, shared)
    shared_auc = auroc_pair(sid[m_shared], sood) if m_shared.any() else float("nan")
    reweighted = float("nan")
    if shared:
        counts = np.array([(y_ood == c).sum() for c in shared], dtype=np.float64)
        probs = counts / max(counts.sum(), 1.0)
        rng = np.random.default_rng(seed)
        idx_by = {c: np.where(y_id == c)[0] for c in shared}
        chosen = []
        for _ in range(min(n_rep, len(y_id))):
            c = int(rng.choice(shared, p=probs))
            if len(idx_by[c]) == 0:
                continue
            chosen.append(int(rng.choice(idx_by[c])))
        if chosen:
            reweighted = auroc_pair(sid[np.asarray(chosen)], sood)
    return {
        "unrestricted": unrestricted,
        "shared_classes": shared_auc,
        "reweighted_to_ood_class_mix": reweighted,
        "n_shared_classes": len(shared),
        "shared_class_indices": shared,
    }


def load_or_create_splits(split_seed: int = SPLIT_SEED):
    df = bdata.load_bcn_ham_frame()
    splits, meta = bdata.prepare_b1_splits(df, split_seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SPLIT_PATH.exists():
        doc = {
            "seed": split_seed,
            "meta": meta,
            "ham_heldout_lesions": sorted(splits["ham_heldout"]["lesion_id"].unique().tolist()),
            "bcn_test_lesions": sorted(splits["bcn_test"]["lesion_id"].unique().tolist()),
        }
        SPLIT_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    hygiene = assert_b1_hygiene(
        splits["bcn_train"],
        splits["bcn_val"],
        splits["bcn_test"],
        splits["ham_adv"],
        splits["ham_heldout"],
    )
    return df, splits, meta, hygiene


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_adv", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    run_name = "bcn_ham_ladv{}_s{}".format(lam_tag(args.lambda_adv), args.seed)
    run_dir = OUT_ROOT / run_name
    ckpt_dir = CKPT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if args.skip_done and summary_path.exists() and not args.dry_run:
        print("skip_done", summary_path)
        return

    _df, splits, meta, hygiene = load_or_create_splits(SPLIT_SEED)
    train_tf = build_train_tf(args.image_size)
    eval_tf = build_eval_tf(args.image_size)
    train_ds = BcnHamPairedDataset(splits["bcn_train"], splits["ham_adv"], train_tf)
    val_ds = SkinDataset(splits["bcn_val"], transform=eval_tf)

    cfg = {
        "phase": "16B1_coarse",
        "run_name": run_name,
        "lambda_adv": args.lambda_adv,
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "architecture": "single_encoder_dann_effb3",
        "preprocessing": "unmasked 448 (same as B0)",
        "image_size": args.image_size,
        "id_site": "BCN",
        "ood_site": "HAM_heldout",
        "split_meta": meta,
        "hygiene": hygiene,
        "split_json": str(SPLIT_PATH),
        "preregister": str(PREREG),
        "coarse_grid": COARSE_GRID,
        "fit_ood_on": "BCN train only",
        "leakage_protocol": "BCN test + HAM heldout embeddings, binary site probe",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    if args.dry_run:
        print("dry_run", run_name, "train_steps", len(train_ds), "bcn_val", len(val_ds))
        (run_dir / "dry_run_ok.json").write_text(json.dumps({"ok": True, **cfg}, indent=2) + "\n")
        return

    adv_log = run_dir / "adversary_log.jsonl"
    if adv_log.exists() and not args.resume:
        adv_log.write_text("")
    model = SingleDannLightning(lambda_adv=args.lambda_adv, learning_rate=args.lr, log_path=adv_log)
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
    train_loader = p2.make_loader(
        train_ds, args.batch_size, args.num_workers, True, drop_last=True, collate=bcn_ham_paired_collate
    )
    val_loader = p2.make_loader(val_ds, args.batch_size, args.num_workers, False)
    last = ckpt_dir / "last.ckpt"
    ckpt_path = str(last) if args.resume and last.exists() else None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last

    lit = SingleDannLightning.load_from_checkpoint(str(best), strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lit.net.to(device).eval()

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, args.num_workers, False)

    train_pack = collect_all(net, loader(splits["bcn_train"]), device)
    id_pack = collect_all(net, loader(splits["bcn_test"]), device)
    hold_pack = collect_all(net, loader(splits["ham_heldout"]), device)

    yid = id_pack["labels"]
    pred_id = id_pack["logits"].argmax(1)
    yhold = hold_pack["labels"]
    pred_hold = hold_pack["logits"].argmax(1)
    id_ece, id_conf = ece_from_logits(id_pack["logits"], yid)
    hold_ece, hold_conf = ece_from_logits(hold_pack["logits"], yhold)

    domain_leak = np.concatenate([np.zeros(len(id_pack["z"])), np.ones(len(hold_pack["z"]))])
    leak = leakage_probe(np.concatenate([id_pack["z"], hold_pack["z"]]), domain_leak)

    detectors = detector_block(
        train_pack["z"],
        train_pack["labels"],
        id_pack["z"],
        hold_pack["z"],
        id_pack["logits"],
        hold_pack["logits"],
        n_classes=8,
    )
    maha_var = maha_ood_variants_full(
        train_pack["z"],
        train_pack["labels"],
        id_pack["z"],
        yid,
        hold_pack["z"],
        yhold,
        seed=args.seed,
    )

    rows = []
    if adv_log.exists():
        for line in adv_log.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    gate = gate_from_log(rows, leak["bal_acc_mean"], floor=0.5, lambda_adv=args.lambda_adv)

    summary = {
        "run_name": run_name,
        "seed": args.seed,
        "lambda_adv": args.lambda_adv,
        "best_checkpoint": str(best),
        "leakage_ham_heldout": leak,
        "id_bcn_test_balanced_acc": float(balanced_accuracy_score(yid, pred_id)),
        "id_bcn_test_plain_acc": float(accuracy_score(yid, pred_id)),
        "ood_ham_heldout_balanced_acc": float(balanced_accuracy_score(yhold, pred_hold)),
        "ood_ham_heldout_plain_acc": float(accuracy_score(yhold, pred_hold)),
        "id_ece": id_ece,
        "id_mean_confidence": id_conf,
        "ood_ece_ham_heldout": hold_ece,
        "ood_mean_confidence_ham_heldout": hold_conf,
        "ood_detectors_ham_heldout": detectors,
        "ood_auroc_maha": maha_var,
        "gate": gate,
        "hygiene": hygiene,
        "config": cfg,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "saved",
        summary_path,
        "leakage_bal",
        leak["bal_acc_mean"],
        "maha",
        maha_var["unrestricted"],
        flush=True,
    )


if __name__ == "__main__":
    main()
