#!/usr/bin/env python3
"""Paper B R11 X5: head+adversary on frozen ERM with synthetic smartphone target.

Design: decisions/paperB_r11_x5.md
  Source = natural ISIC (X3 cache); target = smartphone ISIC synth features.
  Target class mix t∈{0,0.25,0.5,0.75,1}: (1-t)·ISIC + t·PAD.
  Head as X3; λ∈{0,2}; seeds 42/52/62; 40 ep; last.ckpt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from phase15_common import grl_alpha  # noqa: E402
import r5_p9v2 as V1  # noqa: E402
from r11_x3_train import (  # noqa: E402
    HeadNet, FeatStream, forward_pack, git_head, lam_tag,
    LATENT, HIDDEN, LR, WD, ADV_LR_MULT, CLS_LR_MULT, BATCH, MAX_EPOCHS, NCLS,
)

OUT = R.RES / "r11" / "x5"
CACHE = R.RES / "r11" / "x3" / "cache"
SYNTH = OUT / "synth"
CKPT = R.CKPT / "r11_x5"
SEEDS = (42, 52, 62)
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMS = (0.0, 2.0)
OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin", "synth_isic_test")
NAT_SETS = ("isic_train", "isic_val", "isic_test", "pad_adv", "pad_heldout", "pad_full",
            "fitzpatrick17k", "mednode", "sd198", "scin")


def tag_for(lam, seed, t):
    return "x5_ladv{}_t{}_s{}".format(lam_tag(lam), lam_tag(t), int(seed))


def jobs():
    out = []
    for lam in LAMS:
        for t in TS:
            for seed in SEEDS:
                out.append({"lam": lam, "seed": seed, "t": t})
    return out


def class_mix_y(y):
    y = np.asarray(y, int)
    return np.bincount(y, minlength=NCLS).astype(float) / max(len(y), 1)


def weighted_probs(y, target_mix):
    y = np.asarray(y, int)
    n = np.bincount(y, minlength=NCLS)
    p = np.asarray(target_mix, float)[y] / np.maximum(n[y], 1)
    return p / p.sum()


def load_nat(seed):
    d = CACHE / "s{}".format(seed)
    return {n: dict(np.load(d / "{}.npz".format(n), allow_pickle=True)) for n in NAT_SETS}


def load_synth(seed):
    d = SYNTH / "s{}".format(seed)
    return {n: dict(np.load(d / "{}.npz".format(n), allow_pickle=True))
            for n in ("isic_train", "isic_val", "isic_test")}


@torch.no_grad()
def evaluate_run(net, nat, synth, device):
    feats = {}
    for name, pack in nat.items():
        h, lg = forward_pack(net, pack["z_lesion_norm"], device)
        feats[name] = {
            "z_lesion": h, "z_lesion_norm": h, "z_context": np.zeros_like(h),
            "logits": lg, "labels": pack["labels"].astype(int),
        }
    # transformed ISIC test as extra eval set
    sp = synth["isic_test"]
    h, lg = forward_pack(net, sp["z_lesion_norm"], device)
    feats["synth_isic_test"] = {
        "z_lesion": h, "z_lesion_norm": h, "z_context": np.zeros_like(h),
        "logits": lg, "labels": sp["labels"].astype(int),
    }
    m = R.run_metrics(feats, ood_names=[o for o in OODS if o != "synth_isic_test"] + ["synth_isic_test"])
    G = V1.groups()
    assert np.array_equal(feats["isic_test"]["labels"].astype(int), G["yi"])
    v1 = V1.v1_probe(feats["isic_test"]["z_lesion_norm"], feats["pad_heldout"]["z_lesion_norm"],
                     G["gi"], G["gp"])
    m["heldout_leakage_V1"] = v1["bal_acc_mean"]
    m["heldout_leakage_V1_sd"] = v1["bal_acc_sd"]
    return m, feats


def train_one(lam, seed, t, skip_done=True):
    tag = tag_for(lam, seed, t)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nat, synth = load_nat(seed), load_synth(seed)
    z_tr = nat["isic_train"]["z_lesion_norm"]
    y_tr = nat["isic_train"]["labels"].astype(int)
    z_va = nat["isic_val"]["z_lesion_norm"]
    y_va = nat["isic_val"]["labels"].astype(int)
    z_syn = synth["isic_train"]["z_lesion_norm"]
    y_syn = synth["isic_train"]["labels"].astype(int)

    isic_mix = class_mix_y(y_tr)
    # PAD mix from natural pad_adv labels (resampling target only)
    pad_mix = class_mix_y(nat["pad_adv"]["labels"].astype(int))
    # t=0 → ISIC mix; t=1 → PAD mix
    tgt_mix = R.dose_mix(1.0 - float(t), isic_mix, pad_mix)
    syn_probs = weighted_probs(y_syn, tgt_mix)

    cls_stream = FeatStream(z_tr, y_tr)
    adv_isic = FeatStream(z_tr, y_tr)  # domain 0, natural ISIC mix
    adv_syn = FeatStream(z_syn, y_syn, syn_probs)  # domain 1, mix t

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
        "tag": tag, "lam": float(lam), "seed": int(seed), "t": float(t),
        "latent": LATENT, "hidden": HIDDEN, "max_epochs": MAX_EPOCHS,
        "tgt_mix": np.asarray(tgt_mix).tolist(),
        "isic_mix": isic_mix.tolist(), "pad_mix": pad_mix.tolist(),
        "design": "decisions/paperB_r11_x5.md", "commit": git_head(),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("train", tag, "device", device, flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        sum_cls = sum_adv = sum_acc = 0.0
        n_b = 0
        for _ in range(int(math.ceil(len(z_tr) / BATCH))):
            zc, yc = cls_stream.sample(BATCH, rng)
            za0, _ = adv_isic.sample(BATCH, rng)
            za1, _ = adv_syn.sample(BATCH, rng)
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
                sum_acc += float((dlog.argmax(1) == d_adv).float().mean())
            step += 1
            n_b += 1
        _h, lg_va = forward_pack(net, z_va, device)
        val_bal = float(balanced_accuracy_score(y_va, lg_va.argmax(1)))
        row = {"epoch": epoch, "loss_cls": sum_cls / max(n_b, 1), "loss_adv": sum_adv / max(n_b, 1),
               "acc_adv": sum_acc / max(n_b, 1), "val_bal_acc": val_bal}
        epoch_rows.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print("epoch", epoch, "val_bal={:.3f}".format(val_bal), flush=True)
        if val_bal >= best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    last_path = ckpt_dir / "last.ckpt"
    best_path = ckpt_dir / "best.ckpt"
    torch.save({"state_dict": net.state_dict(), "epoch": MAX_EPOCHS, "tag": tag}, last_path)
    if best_state is not None:
        torch.save({"state_dict": best_state, "tag": tag, "val_bal": best_val}, best_path)

    net.load_state_dict(torch.load(last_path, map_location=device)["state_dict"])
    m, feats = evaluate_run(net, nat, synth, device)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in feats.items():
        np.savez_compressed(feat_dir / "{}.npz".format(name), **pack)

    summary = {
        "tag": tag, "lam": float(lam), "seed": int(seed), "t": float(t),
        "checkpoint_policy": "last",
        "primary_checkpoint": str(last_path),
        "id_balanced_acc": float(m["id"]["bal_acc"]),
        "heldout_leakage_V1": float(m["heldout_leakage_V1"]),
        "metrics": m, "epoch_log": epoch_rows, "config": cfg, "commit": git_head(),
    }
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("saved", summary_path, "id_bal", summary["id_balanced_acc"],
          "V1", summary["heldout_leakage_V1"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    a = ap.parse_args()
    js = jobs()
    if a.list:
        for i, j in enumerate(js):
            print(i, tag_for(j["lam"], j["seed"], j["t"]), j)
        print("n_jobs", len(js))
        return
    if a.task_id is None:
        raise SystemExit("need --task_id")
    j = js[a.task_id]
    train_one(j["lam"], j["seed"], j["t"], skip_done=a.skip_done)


if __name__ == "__main__":
    main()
