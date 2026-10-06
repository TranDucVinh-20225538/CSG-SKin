#!/usr/bin/env python3
"""CORAL anomaly: mean shift vs covariance match, from existing checkpoints.

No new training. ISIC train vs pad_adv, the pair CORAL saw.
"""

from __future__ import annotations

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

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase2_pad_holdout as p2
from train_phase15_2_objectives import InvariantLightning
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
CKPT = PAPERB / "checkpoints" / "phase15" / "objectives"
OUT = PAPERB / "results" / "paperB" / "phase15b" / "coral_anomaly"


def latest_ckpt(run_dir: Path):
    best = sorted(run_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = run_dir / "last.ckpt"
    return last if last.exists() else None


@torch.no_grad()
def embed(net, frame, eval_tf, device, workers):
    loader = torch.utils.data.DataLoader(
        SkinDataset(frame, transform=eval_tf), batch_size=64, shuffle=False, num_workers=workers
    )
    zs = []
    net.eval()
    for batch in loader:
        images = batch[0].to(device)
        _logits, _d, z = net(images)
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs)


def stats(z_s, z_t):
    mu_s = z_s.mean(0)
    mu_t = z_t.mean(0)
    mean_l2 = float(np.linalg.norm(mu_s - mu_t))
    def cov(z):
        zc = z - z.mean(0, keepdims=True)
        return (zc.T @ zc) / max(len(z) - 1, 1)
    delta = cov(z_s) - cov(z_t)
    frob = float(np.linalg.norm(delta))
    d = z_s.shape[1]
    coral = float((delta * delta).sum() / (4.0 * d * d))
    return {"mean_l2": mean_l2, "cov_frobenius": frob, "coral_penalty": coral, "dim": int(d), "n_src": int(len(z_s)), "n_tgt": int(len(z_t))}


def main():
    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, _val, _test = p2.isic_splits(isic)
    pad_adv = pad[pad.path.isin(set(spec["pad_adv_paths"]))].reset_index(drop=True)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    eval_tf = build_val_transform_robust()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for run_dir in sorted(CKPT.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir.name.startswith("coral_") or run_dir.name.startswith("erm_")):
            continue
        ckpt = latest_ckpt(run_dir)
        if ckpt is None:
            print("missing ckpt", run_dir.name)
            continue
        print("embed", run_dir.name, ckpt.name, flush=True)
        lit = InvariantLightning.load_from_checkpoint(str(ckpt), map_location="cpu", strict=False)
        net = lit.net.to(device).eval()
        z_tr = embed(net, train, eval_tf, device, 4)
        z_adv = embed(net, pad_adv, eval_tf, device, 4)
        z_hold = embed(net, pad_hold, eval_tf, device, 4)
        rec = {
            "run": run_dir.name,
            "objective": lit.objective,
            "weight": float(lit.weight),
            "seed": int(lit.hparams.get("seed", -1)) if hasattr(lit, "hparams") else None,
            "checkpoint": str(ckpt),
            "train_vs_pad_adv": stats(z_tr, z_adv),
            "train_vs_pad_heldout": stats(z_tr, z_hold),
        }
        # seed is in the run name
        rec["seed"] = int(run_dir.name.rsplit("_s", 1)[-1])
        runs.append(rec)
        del lit, net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (OUT / "per_run.json").write_text(json.dumps(runs, indent=2) + "\n")

    # aggregate by weight
    from collections import defaultdict
    by = defaultdict(list)
    for r in runs:
        by[(r["objective"], r["weight"])].append(r)
    lines = [
        "# CORAL anomaly — mean distance vs covariance discrepancy",
        "",
        "No new training. Features from existing Phase 15.2 checkpoints.",
        "Pair CORAL was trained on: ISIC train vs pad_adv. pad_heldout is reported beside it and was not in the penalty.",
        "",
        "Hypothesis: classification can satisfy CORAL by matching covariances while pushing domain means apart.",
        "",
        "| objective | weight | n | mean L2 (adv) | cov Frobenius (adv) | CORAL term (adv) | mean L2 (heldout) |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for key in sorted(by, key=lambda k: (k[0], k[1])):
        recs = by[key]
        def col(path):
            xs = [path(r) for r in recs]
            return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        mean_a = col(lambda r: r["train_vs_pad_adv"]["mean_l2"])
        fro_a = col(lambda r: r["train_vs_pad_adv"]["cov_frobenius"])
        cor_a = col(lambda r: r["train_vs_pad_adv"]["coral_penalty"])
        mean_h = col(lambda r: r["train_vs_pad_heldout"]["mean_l2"])
        lines.append(
            "| {} | {:g} | {} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.6f} ± {:.6f} | {:.3f} ± {:.3f} |".format(
                key[0], key[1], len(recs), mean_a[0], mean_a[1], fro_a[0], fro_a[1], cor_a[0], cor_a[1], mean_h[0], mean_h[1]
            )
        )
    lines.append("")
    lines.append("If mean L2 grows with weight while the CORAL term shrinks, the anomaly is the mean escaping a covariance-only penalty.")
    lines.append("")
    (OUT / "CORAL_ANOMALY.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "CORAL_ANOMALY.md")


if __name__ == "__main__":
    main()
