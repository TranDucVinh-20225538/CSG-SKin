#!/usr/bin/env python3
"""Paper B R11b Y2: X3 head+adversary rerun with working-adversary configs.

Screen/main write only under results/paperB/r11b/y2/. Reads R11 X3 feature cache.
Never modifies results/paperB/r11/.

Configs (locked in decisions/paperB_r11b.md):
  A: LeakyReLU(0.2) adversary, no dropout, adv LR ×30
  B: as A, adv LR ×1
  C: as B, 5 adv steps / head step, DANN α=2/(1+exp(-10p))-1 (no α_min)
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

MAIN = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
WT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b")
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(MAIN / "scripts"))

import r6_common as R  # noqa: E402
from phase15_common import GradientReversal  # noqa: E402
import r5_p9v2 as V1  # noqa: E402
from r11_x3_train import (  # noqa: E402
    FeatStream, forward_pack, class_mix_y, weighted_probs,
    HIDDEN, LR, WD, CLS_LR_MULT, BATCH, MAX_EPOCHS, NCLS, ALL_SETS, OODS,
)

CACHE = MAIN / "results" / "paperB" / "r11" / "x3" / "cache"
OUT = WT / "results" / "paperB" / "r11b" / "y2"
CKPT = MAIN / "checkpoints" / "r11b_y2"
LN2 = float(math.log(2.0))
SCREEN_SEEDS = (42, 52)
MAIN_SEEDS = R.SEEDS  # 42..82


def git_head():
    return subprocess.run(
        ["git", "-C", str(WT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def lam_tag(lam: float) -> str:
    return "{:g}".format(float(lam)).replace(".", "p")


def dann_alpha(step, total_steps):
    """Standard DANN: α = 2/(1+exp(-10p)) - 1, no α_min."""
    denom = max(int(total_steps) - 1, 1)
    p = max(0.0, min(1.0, float(step) / float(denom)))
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0), float(p)


def recipe_alpha(step, total_steps):
    """Main derm recipe GRL (alpha_min=0.2, gamma=20, power=0.5)."""
    from phase15_common import grl_alpha
    return grl_alpha(step, total_steps)


def derm_domain_head_leaky(in_dim, n_domains):
    """Y2 A/B/C: LeakyReLU(0.2), no dropout (else same widths as derm_domain_head)."""
    h1 = max(int(in_dim), 32)
    h2 = max(int(in_dim) // 2, 16)
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.BatchNorm1d(h1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(h1, h2),
        nn.BatchNorm1d(h2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(h2, n_domains),
    )


class HeadNetY2(nn.Module):
    def __init__(self, in_dim=1536, latent=16, hidden=HIDDEN, n_classes=NCLS, n_domains=2):
        super().__init__()
        self.latent = int(latent)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent),
        )
        self.bn = nn.BatchNorm1d(latent)
        self.classifier = nn.Linear(latent, n_classes)
        self.grl = GradientReversal(1.0)
        self.domain_head = derm_domain_head_leaky(latent, n_domains)

    def set_grl_lambda(self, a):
        self.grl.set_lambd(a)

    def forward(self, z, return_latent=False):
        h = self.bn(self.mlp(z))
        logits = self.classifier(h)
        dlog = self.domain_head(self.grl(h))
        if return_latent:
            return logits, dlog, h
        return logits, dlog


def tag_for(cfg, latent, lam, seed, shift):
    return "y2{}_lat{}_ladv{}_shift{}_s{}".format(
        cfg, int(latent), lam_tag(lam), int(round(100 * float(shift))), int(seed))


def screen_jobs():
    out = []
    for latent in (16, 128):
        for cfg in ("A", "B", "C"):
            for lam in (0.5, 2.0, 8.0):
                for seed in SCREEN_SEEDS:
                    out.append({"phase": "screen", "cfg": cfg, "latent": latent,
                                "lam": lam, "seed": seed, "shift": 1.0})
        for seed in SCREEN_SEEDS:
            out.append({"phase": "screen", "cfg": "A", "latent": latent,
                        "lam": 0.0, "seed": seed, "shift": 1.0})
    return out


def main_jobs(cfg, latent):
    out = []
    for lam in (0.0, 0.5, 2.0, 8.0):
        for seed in MAIN_SEEDS:
            out.append({"phase": "main", "cfg": cfg, "latent": latent,
                        "lam": lam, "seed": seed, "shift": 1.0})
    for seed in MAIN_SEEDS:
        out.append({"phase": "main", "cfg": cfg, "latent": latent,
                    "lam": 2.0, "seed": seed, "shift": 0.0})
    return out


def load_cache(seed):
    d = CACHE / "s{}".format(seed)
    return {n: dict(np.load(d / "{}.npz".format(n), allow_pickle=True)) for n in ALL_SETS}


def evaluate_run(net, cache, device):
    feats = {}
    for name, pack in cache.items():
        h, lg = forward_pack(net, pack["z_lesion_norm"], device)
        feats[name] = {
            "z_lesion": h, "z_lesion_norm": h,
            "z_context": np.zeros_like(h),
            "logits": lg, "labels": pack["labels"].astype(int),
        }
    m = R.run_metrics(feats, ood_names=list(OODS))
    G = V1.groups()
    assert np.array_equal(feats["isic_test"]["labels"].astype(int), G["yi"])
    v1 = V1.v1_probe(feats["isic_test"]["z_lesion_norm"], feats["pad_heldout"]["z_lesion_norm"],
                     G["gi"], G["gp"])
    m["heldout_leakage_V1"] = v1["bal_acc_mean"]
    m["heldout_leakage_V1_sd"] = v1["bal_acc_sd"]
    return m, feats


def cfg_hparams(cfg):
    if cfg == "A":
        return {"adv_lr_mult": 30.0, "adv_inner_steps": 1, "alpha_fn": "recipe"}
    if cfg == "B":
        return {"adv_lr_mult": 1.0, "adv_inner_steps": 1, "alpha_fn": "recipe"}
    if cfg == "C":
        return {"adv_lr_mult": 1.0, "adv_inner_steps": 5, "alpha_fn": "dann"}
    raise ValueError(cfg)


def train_one(cfg, latent, lam, seed, shift, phase="screen", skip_done=True):
    hp = cfg_hparams(cfg)
    tag = tag_for(cfg, latent, lam, seed, shift)
    run_dir = OUT / phase / "runs" / tag
    feat_dir = OUT / phase / "features" / tag
    ckpt_dir = CKPT / phase / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if skip_done and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("checkpoint_policy") == "last" and (feat_dir / "isic_test.npz").exists():
            print("skip_done", summary_path, flush=True)
            return

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_cache(seed)
    z_tr = cache["isic_train"]["z_lesion_norm"]
    y_tr = cache["isic_train"]["labels"].astype(int)
    z_pad = cache["pad_adv"]["z_lesion_norm"]
    y_pad = cache["pad_adv"]["labels"].astype(int)
    z_va = cache["isic_val"]["z_lesion_norm"]
    y_va = cache["isic_val"]["labels"].astype(int)

    isic_mix = class_mix_y(y_tr)
    pad_mix = class_mix_y(y_pad)
    adv_mix = R.dose_mix(shift, isic_mix, pad_mix) if float(shift) < 1.0 else None
    adv_probs = None if adv_mix is None else weighted_probs(y_tr, adv_mix)

    cls_stream = FeatStream(z_tr, y_tr)
    adv_isic = FeatStream(z_tr, y_tr, adv_probs)
    adv_pad = FeatStream(z_pad, y_pad)

    net = HeadNetY2(in_dim=z_tr.shape[1], latent=int(latent)).to(device)
    adv_lr_mult = float(hp["adv_lr_mult"])
    # A/B: single joint optimizer (as R11 X3). C: separate head/adv opts + inner steps.
    opt_joint = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()) + list(net.bn.parameters()), "lr": LR},
        {"params": net.classifier.parameters(), "lr": LR * CLS_LR_MULT},
        {"params": net.domain_head.parameters(), "lr": LR * adv_lr_mult},
    ], weight_decay=WD)
    opt_head = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()) + list(net.bn.parameters()), "lr": LR},
        {"params": net.classifier.parameters(), "lr": LR * CLS_LR_MULT},
    ], weight_decay=WD)
    opt_adv = torch.optim.AdamW(
        net.domain_head.parameters(), lr=LR * adv_lr_mult, weight_decay=WD)

    n_steps = int(math.ceil(len(z_tr) / BATCH)) * MAX_EPOCHS
    step = 0
    rng = torch.Generator()
    rng.manual_seed(seed)
    epoch_rows = []
    alpha_fn = dann_alpha if hp["alpha_fn"] == "dann" else recipe_alpha
    inner = int(hp["adv_inner_steps"])

    cfg_doc = {
        "tag": tag, "phase": phase, "cfg": cfg, "latent": int(latent),
        "lam": float(lam), "seed": int(seed), "shift": float(shift),
        "adv_lr_mult": adv_lr_mult, "adv_inner_steps": inner,
        "alpha_fn": hp["alpha_fn"], "max_epochs": MAX_EPOCHS, "batch": BATCH,
        "lr": LR, "cls_lr_mult": CLS_LR_MULT,
        "adversary": "LeakyReLU(0.2), no dropout",
        "cache": str(CACHE / "s{}".format(seed)),
        "commit": git_head(),
        "design": "decisions/paperB_r11b.md",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_doc, indent=2) + "\n")
    print("train", tag, "device", device, flush=True)

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
            d_adv = torch.cat([
                torch.zeros(BATCH, dtype=torch.long, device=device),
                torch.ones(BATCH, dtype=torch.long, device=device),
            ])

            alpha, _ = alpha_fn(step, n_steps)
            net.set_grl_lambda(alpha)

            if cfg == "C":
                # five adversary steps on frozen head features, then one head step via GRL
                for _k in range(inner):
                    if _k > 0:
                        za0, _ = adv_isic.sample(BATCH, rng)
                        za1, _ = adv_pad.sample(BATCH, rng)
                        za0, za1 = za0.to(device), za1.to(device)
                        z_adv = torch.cat([za0, za1], 0)
                        d_adv = torch.cat([
                            torch.zeros(BATCH, dtype=torch.long, device=device),
                            torch.ones(BATCH, dtype=torch.long, device=device),
                        ])
                    with torch.no_grad():
                        h_adv = net.bn(net.mlp(z_adv))
                    loss_a = F.cross_entropy(net.domain_head(h_adv), d_adv)
                    opt_adv.zero_grad(set_to_none=True)
                    loss_a.backward()
                    opt_adv.step()

                logits, _ = net(zc)
                _, dlog = net(z_adv)
                loss_cls = F.cross_entropy(logits, yc)
                loss_adv = F.cross_entropy(dlog, d_adv)
                loss = loss_cls + float(lam) * loss_adv
                opt_head.zero_grad(set_to_none=True)
                loss.backward()
                opt_head.step()
            else:
                # A/B: joint step identical to R11 X3 (leaky head / LR mult differ)
                logits, _ = net(zc)
                _, dlog = net(z_adv)
                loss_cls = F.cross_entropy(logits, yc)
                loss_adv = F.cross_entropy(dlog, d_adv)
                loss = loss_cls + float(lam) * loss_adv
                opt_joint.zero_grad(set_to_none=True)
                loss.backward()
                opt_joint.step()

            with torch.no_grad():
                sum_cls += float(loss_cls)
                sum_adv += float(loss_adv)
                sum_acc_adv += float((dlog.argmax(1) == d_adv).float().mean())
            step += 1
            n_b += 1

        net.eval()
        _h_va, lg_va = forward_pack(net, z_va, device)
        val_bal = float(balanced_accuracy_score(y_va, lg_va.argmax(1)))
        row = {
            "epoch": epoch,
            "loss_cls": sum_cls / max(n_b, 1),
            "loss_adv": sum_adv / max(n_b, 1),
            "acc_adv": sum_acc_adv / max(n_b, 1),
            "val_bal_acc": val_bal,
            "grl_alpha": float(alpha_fn(step - 1, n_steps)[0]),
        }
        epoch_rows.append(row)
        print("epoch", epoch, "val_bal={:.3f}".format(val_bal),
              "loss_adv={:.3f}".format(row["loss_adv"]),
              "acc_adv={:.3f}".format(row["acc_adv"]), flush=True)

    last_path = ckpt_dir / "last.ckpt"
    torch.save({"state_dict": net.state_dict(), "epoch": MAX_EPOCHS, "tag": tag}, last_path)
    net.load_state_dict(torch.load(last_path, map_location=device)["state_dict"])
    m, feats = evaluate_run(net, cache, device)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in feats.items():
        np.savez_compressed(feat_dir / "{}.npz".format(name), **pack)

    id_bal = float(m["id"]["bal_acc"])
    summary = {
        "tag": tag, "phase": phase, "cfg": cfg, "latent": int(latent),
        "lam": float(lam), "seed": int(seed), "shift": float(shift),
        "checkpoint_policy": "last",
        "primary_checkpoint": str(last_path),
        "id_balanced_acc": id_bal,
        "id_val_balanced_acc": float(epoch_rows[-1]["val_bal_acc"]),
        "heldout_leakage_V1": float(m["heldout_leakage_V1"]),
        "final_acc_adv": float(epoch_rows[-1]["acc_adv"]),
        "final_loss_adv": float(epoch_rows[-1]["loss_adv"]),
        "metrics": m,
        "epoch_log": epoch_rows,
        "config": cfg_doc,
        "commit": git_head(),
    }
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (run_dir / "epoch_log.json").write_text(json.dumps(epoch_rows, indent=1) + "\n")
    print("saved", summary_path, "id_bal", id_bal, "V1", m["heldout_leakage_V1"],
          "acc_adv", summary["final_acc_adv"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--phase", choices=("screen", "main"), default="screen")
    ap.add_argument("--cfg", choices=("A", "B", "C"))
    ap.add_argument("--latent", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--shift", type=float)
    ap.add_argument("--selected_cfg", type=str, help="main: cfg letter")
    ap.add_argument("--selected_latent", type=int, help="main: latent")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    a = ap.parse_args()

    if a.phase == "screen":
        js = screen_jobs()
    else:
        if not a.selected_cfg or not a.selected_latent:
            # allow reading selection file
            sel = OUT / "SELECTION.json"
            if sel.exists():
                doc = json.loads(sel.read_text())
                a.selected_cfg = doc["cfg"]
                a.selected_latent = int(doc["latent"])
            else:
                raise SystemExit("main needs --selected_cfg/--selected_latent or SELECTION.json")
        js = main_jobs(a.selected_cfg, a.selected_latent)

    if a.list:
        for i, j in enumerate(js):
            print(i, tag_for(j["cfg"], j["latent"], j["lam"], j["seed"], j["shift"]), j)
        print("n_jobs", len(js))
        return

    if a.task_id is not None:
        j = js[a.task_id]
        train_one(j["cfg"], j["latent"], j["lam"], j["seed"], j["shift"],
                  phase=j["phase"], skip_done=a.skip_done)
        return

    if a.cfg is None or a.latent is None or a.lam is None or a.seed is None or a.shift is None:
        raise SystemExit("need --task_id or full --cfg/--latent/--lam/--seed/--shift")
    train_one(a.cfg, a.latent, a.lam, a.seed, a.shift, phase=a.phase, skip_done=a.skip_done)


if __name__ == "__main__":
    main()
