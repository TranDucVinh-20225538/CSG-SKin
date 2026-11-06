#!/usr/bin/env python3
"""Paper B R11 X3: train classifier+adversary head on frozen ERM features.

Design: decisions/paperB_r11_x3.md
  - Frozen EffB3 features (1536-d) from X3 cache
  - 2-layer MLP → latent 16; Linear classifier; derm_domain_head + GRL on latent
  - Same GRL schedule / adv LR ×30 / cls LR ×0.2 / AdamW 2e-4 / 40 epochs as main recipe
  - Three-stream: isic_cls + isic_adv (+ optional dose_mix) + pad_adv
  - last.ckpt primary; write projected features for all sets; summary + V1

Usage:
  r11_x3_train.py --task_id N [--skip_done] [--list]
  r11_x3_train.py --lam 2 --seed 42 --shift 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import Dataset

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from phase15_common import derm_domain_head, grl_alpha, GradientReversal  # noqa: E402
import r5_p9v2 as V1  # noqa: E402

OUT = R.RES / "r11" / "x3"
CACHE = OUT / "cache"
CKPT = R.CKPT / "r11_x3"
LATENT = 16
HIDDEN = 512
NCLS = R.NCLS
MAX_EPOCHS = 40
LR = 2e-4
WD = 1e-4
ADV_LR_MULT = 30.0
CLS_LR_MULT = 0.2
BATCH = 64
OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
ALL_SETS = ("isic_train", "isic_val", "isic_test", "pad_adv", "pad_heldout", "pad_full",
            "fitzpatrick17k", "mednode", "sd198", "scin")


def git_head():
    return subprocess.run(
        ["git", "-C", str(R.PAPERB), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def lam_tag(lam: float) -> str:
    s = "{:g}".format(float(lam))
    return s.replace(".", "p")


def tag_for(lam, seed, shift):
    return "x3_ladv{}_shift{}_s{}".format(lam_tag(lam), int(round(100 * float(shift))), int(seed))


def jobs():
    out = []
    for lam in (0.0, 0.5, 2.0, 8.0):
        for seed in R.SEEDS:
            out.append({"lam": lam, "seed": seed, "shift": 1.0})
    for seed in R.SEEDS:
        out.append({"lam": 2.0, "seed": seed, "shift": 0.0})
    return out


class HeadNet(nn.Module):
    """2-layer MLP → 16-d + classifier + GRL adversary (derm head)."""

    def __init__(self, in_dim=1536, latent=LATENT, hidden=HIDDEN, n_classes=NCLS, n_domains=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent),
        )
        self.bn = nn.BatchNorm1d(latent)
        self.classifier = nn.Linear(latent, n_classes)
        self.grl = GradientReversal(1.0)
        self.domain_head = derm_domain_head(latent, n_domains)

    def set_grl_lambda(self, a):
        self.grl.set_lambd(a)

    def forward(self, z, return_latent=False):
        h = self.bn(self.mlp(z))
        logits = self.classifier(h)
        dlog = self.domain_head(self.grl(h))
        if return_latent:
            return logits, dlog, h
        return logits, dlog


class FeatStream(Dataset):
    """Indexable feature pack with optional multinomial sampling weights."""

    def __init__(self, z, y, probs=None):
        self.z = torch.as_tensor(np.asarray(z, np.float32))
        self.y = torch.as_tensor(np.asarray(y, np.int64))
        self.probs = None if probs is None else torch.as_tensor(np.asarray(probs, np.float64))

    def __len__(self):
        return len(self.z)

    def sample(self, n, rng: torch.Generator):
        if self.probs is None:
            idx = torch.randint(0, len(self), (n,), generator=rng)
        else:
            idx = torch.multinomial(self.probs, n, replacement=True, generator=rng)
        return self.z[idx], self.y[idx]


def load_cache(seed):
    d = CACHE / "s{}".format(seed)
    out = {}
    for n in ALL_SETS:
        p = d / "{}.npz".format(n)
        if not p.exists():
            raise FileNotFoundError(p)
        pack = dict(np.load(p, allow_pickle=True))
        out[n] = pack
    return out


def class_mix_y(y):
    y = np.asarray(y, int)
    return np.bincount(y, minlength=NCLS).astype(float) / max(len(y), 1)


def weighted_probs(y, target_mix):
    y = np.asarray(y, int)
    n = np.bincount(y, minlength=NCLS)
    p = np.asarray(target_mix, float)[y] / np.maximum(n[y], 1)
    return p / p.sum()


@torch.no_grad()
def forward_pack(net, z, device):
    net.eval()
    zs, lgs = [], []
    for i in range(0, len(z), 256):
        zb = torch.as_tensor(z[i:i + 256], dtype=torch.float32, device=device)
        logits, _, h = net(zb, return_latent=True)
        zs.append(h.cpu().numpy())
        lgs.append(logits.float().cpu().numpy())
    return np.concatenate(zs).astype(np.float32), np.concatenate(lgs).astype(np.float32)


def evaluate_run(net, cache, device):
    feats = {}
    for name, pack in cache.items():
        z = pack["z_lesion_norm"]
        h, lg = forward_pack(net, z, device)
        feats[name] = {
            "z_lesion": h, "z_lesion_norm": h,
            "z_context": np.zeros_like(h),
            "logits": lg, "labels": pack["labels"].astype(int),
        }
    m = R.run_metrics(feats, ood_names=list(OODS))
    G = V1.groups()
    assert np.array_equal(feats["isic_test"]["labels"].astype(int), G["yi"])
    v1 = V1.v1_probe(feats["isic_test"]["z_lesion_norm"], feats["pad_heldout"]["z_lesion_norm"], G["gi"], G["gp"])
    m["heldout_leakage_V1"] = v1["bal_acc_mean"]
    m["heldout_leakage_V1_sd"] = v1["bal_acc_sd"]
    return m, feats


def train_one(lam, seed, shift, skip_done=True, num_workers=0):
    tag = tag_for(lam, seed, shift)
    run_dir = OUT / "runs" / tag
    feat_dir = OUT / "features" / tag
    ckpt_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if skip_done and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("checkpoint_policy") == "last" and (feat_dir / "isic_test.npz").exists():
            print("skip_done", summary_path, flush=True)
            return

    torch.manual_seed(seed)
    np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_cache(seed)
    z_tr, y_tr = cache["isic_train"]["z_lesion_norm"], cache["isic_train"]["labels"].astype(int)
    z_pad, y_pad = cache["pad_adv"]["z_lesion_norm"], cache["pad_adv"]["labels"].astype(int)
    z_va, y_va = cache["isic_val"]["z_lesion_norm"], cache["isic_val"]["labels"].astype(int)

    isic_mix = class_mix_y(y_tr)
    pad_mix = class_mix_y(y_pad)
    adv_mix = R.dose_mix(shift, isic_mix, pad_mix) if float(shift) < 1.0 else None
    adv_probs = None if adv_mix is None else weighted_probs(y_tr, adv_mix)

    cls_stream = FeatStream(z_tr, y_tr)
    adv_isic = FeatStream(z_tr, y_tr, adv_probs)
    adv_pad = FeatStream(z_pad, y_pad)

    net = HeadNet(in_dim=z_tr.shape[1]).to(device)
    opt = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()) + list(net.bn.parameters()), "lr": LR},
        {"params": net.classifier.parameters(), "lr": LR * CLS_LR_MULT},
        {"params": net.domain_head.parameters(), "lr": LR * ADV_LR_MULT},
    ], weight_decay=WD)

    n_steps = int(math.ceil(len(z_tr) / BATCH)) * MAX_EPOCHS
    step = 0
    rng = torch.Generator()
    rng.manual_seed(seed)
    epoch_rows = []
    best_val, best_state = -1.0, None

    cfg = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "shift": float(shift),
        "latent": LATENT, "hidden": HIDDEN, "max_epochs": MAX_EPOCHS, "batch": BATCH,
        "lr": LR, "adv_lr_mult": ADV_LR_MULT, "cls_lr_mult": CLS_LR_MULT,
        "isic_train_mix": isic_mix.tolist(),
        "pad_adv_mix": pad_mix.tolist(),
        "adv_mix": None if adv_mix is None else np.asarray(adv_mix).tolist(),
        "no_augmentation": True,
        "cache": str(CACHE / "s{}".format(seed)),
        "commit": git_head(),
        "design": "decisions/paperB_r11_x3.md",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("train", tag, "device", device, "n_steps", n_steps, flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        sum_cls = sum_adv = sum_acc_adv = 0.0
        n_b = 0
        for _ in range(int(math.ceil(len(z_tr) / BATCH))):
            zc, yc = cls_stream.sample(BATCH, rng)
            za0, _ = adv_isic.sample(BATCH, rng)
            za1, _ = adv_pad.sample(BATCH, rng)
            zc, yc = zc.to(device), yc.to(device)
            za0, za1 = za0.to(device), za1.to(device)
            z_adv = torch.cat([za0, za1], 0)
            d_adv = torch.cat([torch.zeros(BATCH, dtype=torch.long, device=device),
                               torch.ones(BATCH, dtype=torch.long, device=device)])

            alpha, _ = grl_alpha(step, n_steps)
            net.set_grl_lambda(alpha)
            logits, _ = net(zc)
            _, dlog = net(z_adv)
            loss_cls = F.cross_entropy(logits, yc)
            loss_adv = F.cross_entropy(dlog, d_adv)
            loss = loss_cls + float(lam) * loss_adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sum_cls += float(loss_cls)
                sum_adv += float(loss_adv)
                sum_acc_adv += float((dlog.argmax(1) == d_adv).float().mean())
            step += 1
            n_b += 1

        # val bal_acc
        net.eval()
        _h_va, lg_va = forward_pack(net, z_va, device)
        val_bal = float(balanced_accuracy_score(y_va, lg_va.argmax(1)))
        row = {
            "epoch": epoch,
            "loss_cls": sum_cls / max(n_b, 1),
            "loss_adv": sum_adv / max(n_b, 1),
            "acc_adv": sum_acc_adv / max(n_b, 1),
            "val_bal_acc": val_bal,
            "grl_alpha": float(grl_alpha(step - 1, n_steps)[0]),
        }
        epoch_rows.append(row)
        print("epoch", epoch, "val_bal={:.3f}".format(val_bal),
              "loss_cls={:.3f}".format(row["loss_cls"]),
              "loss_adv={:.3f}".format(row["loss_adv"]),
              "acc_adv={:.3f}".format(row["acc_adv"]), flush=True)
        if val_bal >= best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    # last.ckpt = end of training; also save best-val
    last_path = ckpt_dir / "last.ckpt"
    best_path = ckpt_dir / "best.ckpt"
    torch.save({"state_dict": net.state_dict(), "epoch": MAX_EPOCHS, "tag": tag}, last_path)
    if best_state is not None:
        torch.save({"state_dict": best_state, "epoch": "best_val", "tag": tag, "val_bal": best_val}, best_path)

    # Primary = last
    net.load_state_dict(torch.load(last_path, map_location=device)["state_dict"])
    m, feats = evaluate_run(net, cache, device)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in feats.items():
        np.savez_compressed(feat_dir / "{}.npz".format(name), **pack)

    id_bal = float(m["id"]["bal_acc"])
    summary = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "shift": float(shift),
        "checkpoint_policy": "last",
        "primary_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "id_balanced_acc": id_bal,
        "heldout_leakage_V1": float(m["heldout_leakage_V1"]),
        "metrics": m,
        "epoch_log": epoch_rows,
        "config": cfg,
        "commit": git_head(),
    }
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (run_dir / "epoch_log.json").write_text(json.dumps(epoch_rows, indent=1) + "\n")
    print("saved", summary_path, "id_bal", id_bal, "V1", m["heldout_leakage_V1"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--shift", type=float)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    a = ap.parse_args()
    js = jobs()
    if a.list:
        for i, j in enumerate(js):
            print(i, tag_for(j["lam"], j["seed"], j["shift"]), j)
        print("n_jobs", len(js))
        return
    if a.task_id is not None:
        j = js[a.task_id]
        train_one(j["lam"], j["seed"], j["shift"], skip_done=a.skip_done)
        return
    if a.lam is None or a.seed is None or a.shift is None:
        raise SystemExit("need --task_id or --lam/--seed/--shift")
    train_one(a.lam, a.seed, a.shift, skip_done=a.skip_done)


if __name__ == "__main__":
    main()
