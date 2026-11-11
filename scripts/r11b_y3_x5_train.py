#!/usr/bin/env python3
"""Paper B R11b Y3: X5 rerun with Y2-selected head/adversary config.

Writes only results/paperB/r11b/y3/. Reads R11 X3 cache + X5 synth. Never touches r11/.
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

MAIN = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
WT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b")
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(MAIN / "scripts"))

import r6_common as R  # noqa: E402
import r5_p9v2 as V1  # noqa: E402
from r11_x3_train import FeatStream, forward_pack, HIDDEN, LR, WD, CLS_LR_MULT, BATCH, MAX_EPOCHS, NCLS  # noqa: E402
from r11b_y2_x3_train import (  # noqa: E402
    HeadNetY2, cfg_hparams, dann_alpha, recipe_alpha, git_head, lam_tag,
)

CACHE = MAIN / "results" / "paperB" / "r11" / "x3" / "cache"
SYNTH = MAIN / "results" / "paperB" / "r11" / "x5" / "synth"
OUT = WT / "results" / "paperB" / "r11b" / "y3"
CKPT = MAIN / "checkpoints" / "r11b_y3"
Y2_SEL = WT / "results" / "paperB" / "r11b" / "y2" / "SELECTION.json"
SEEDS = (42, 52, 62)
TS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMS = (0.0, 2.0)
OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin", "synth_isic_test")
NAT_SETS = ("isic_train", "isic_val", "isic_test", "pad_adv", "pad_heldout", "pad_full",
            "fitzpatrick17k", "mednode", "sd198", "scin")


def tag_for(cfg, latent, lam, seed, t):
    return "y3{}_lat{}_ladv{}_t{}_s{}".format(
        cfg, int(latent), lam_tag(lam), lam_tag(t), int(seed))


def jobs(cfg, latent):
    out = []
    for lam in LAMS:
        for t in TS:
            for seed in SEEDS:
                out.append({"cfg": cfg, "latent": latent, "lam": lam, "seed": seed, "t": t})
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
    sp = synth["isic_test"]
    h, lg = forward_pack(net, sp["z_lesion_norm"], device)
    feats["synth_isic_test"] = {
        "z_lesion": h, "z_lesion_norm": h, "z_context": np.zeros_like(h),
        "logits": lg, "labels": sp["labels"].astype(int),
    }
    m = R.run_metrics(feats, ood_names=list(OODS))
    G = V1.groups()
    assert np.array_equal(feats["isic_test"]["labels"].astype(int), G["yi"])
    v1 = V1.v1_probe(feats["isic_test"]["z_lesion_norm"], feats["pad_heldout"]["z_lesion_norm"],
                     G["gi"], G["gp"])
    m["heldout_leakage_V1"] = v1["bal_acc_mean"]
    m["heldout_leakage_V1_sd"] = v1["bal_acc_sd"]
    return m, feats


def train_one(cfg, latent, lam, seed, t, skip_done=True):
    hp = cfg_hparams(cfg)
    tag = tag_for(cfg, latent, lam, seed, t)
    run_dir = OUT / "runs" / tag
    feat_dir = OUT / "features" / tag
    ckpt_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if skip_done and summary_path.exists() and (feat_dir / "isic_test.npz").exists():
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
    pad_mix = class_mix_y(nat["pad_adv"]["labels"].astype(int))
    tgt_mix = R.dose_mix(1.0 - float(t), isic_mix, pad_mix)
    syn_probs = weighted_probs(y_syn, tgt_mix)

    cls_stream = FeatStream(z_tr, y_tr)
    adv_isic = FeatStream(z_tr, y_tr)
    adv_syn = FeatStream(z_syn, y_syn, syn_probs)

    net = HeadNetY2(in_dim=z_tr.shape[1], latent=int(latent)).to(device)
    adv_lr_mult = float(hp["adv_lr_mult"])
    opt_joint = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()) + list(net.bn.parameters()), "lr": LR},
        {"params": net.classifier.parameters(), "lr": LR * CLS_LR_MULT},
        {"params": net.domain_head.parameters(), "lr": LR * adv_lr_mult},
    ], weight_decay=WD)
    opt_head = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()) + list(net.bn.parameters()), "lr": LR},
        {"params": net.classifier.parameters(), "lr": LR * CLS_LR_MULT},
    ], weight_decay=WD)
    opt_adv = torch.optim.AdamW(net.domain_head.parameters(), lr=LR * adv_lr_mult, weight_decay=WD)

    n_steps = int(math.ceil(len(z_tr) / BATCH)) * MAX_EPOCHS
    step = 0
    rng = torch.Generator(); rng.manual_seed(seed)
    epoch_rows = []
    alpha_fn = dann_alpha if hp["alpha_fn"] == "dann" else recipe_alpha
    inner = int(hp["adv_inner_steps"])

    cfg_doc = {
        "tag": tag, "cfg": cfg, "latent": int(latent), "lam": float(lam),
        "seed": int(seed), "t": float(t), "adv_lr_mult": adv_lr_mult,
        "adv_inner_steps": inner, "alpha_fn": hp["alpha_fn"],
        "tgt_mix": np.asarray(tgt_mix).tolist(),
        "design": "decisions/paperB_r11b.md", "commit": git_head(),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_doc, indent=2) + "\n")
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
            alpha, _ = alpha_fn(step, n_steps)
            net.set_grl_lambda(alpha)

            if cfg == "C":
                for _k in range(inner):
                    if _k > 0:
                        za0, _ = adv_isic.sample(BATCH, rng)
                        za1, _ = adv_syn.sample(BATCH, rng)
                        za0, za1 = za0.to(device), za1.to(device)
                        z_adv = torch.cat([za0, za1], 0)
                        d_adv = torch.cat([torch.zeros(BATCH, dtype=torch.long, device=device),
                                           torch.ones(BATCH, dtype=torch.long, device=device)])
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
                sum_acc += float((dlog.argmax(1) == d_adv).float().mean())
            step += 1
            n_b += 1

        _h, lg_va = forward_pack(net, z_va, device)
        val_bal = float(balanced_accuracy_score(y_va, lg_va.argmax(1)))
        row = {"epoch": epoch, "loss_cls": sum_cls / max(n_b, 1), "loss_adv": sum_adv / max(n_b, 1),
               "acc_adv": sum_acc / max(n_b, 1), "val_bal_acc": val_bal,
               "grl_alpha": float(alpha_fn(step - 1, n_steps)[0])}
        epoch_rows.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print("epoch", epoch, "val_bal={:.3f}".format(val_bal),
                  "acc_adv={:.3f}".format(row["acc_adv"]), flush=True)

    last_path = ckpt_dir / "last.ckpt"
    torch.save({"state_dict": net.state_dict(), "epoch": MAX_EPOCHS, "tag": tag}, last_path)
    net.load_state_dict(torch.load(last_path, map_location=device)["state_dict"])
    m, feats = evaluate_run(net, nat, synth, device)
    feat_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in feats.items():
        np.savez_compressed(feat_dir / "{}.npz".format(name), **pack)

    summary = {
        "tag": tag, "cfg": cfg, "latent": int(latent),
        "lam": float(lam), "seed": int(seed), "t": float(t),
        "checkpoint_policy": "last",
        "primary_checkpoint": str(last_path),
        "id_balanced_acc": float(m["id"]["bal_acc"]),
        "heldout_leakage_V1": float(m["heldout_leakage_V1"]),
        "final_acc_adv": float(epoch_rows[-1]["acc_adv"]),
        "final_loss_adv": float(epoch_rows[-1]["loss_adv"]),
        "metrics": m, "epoch_log": epoch_rows, "config": cfg_doc, "commit": git_head(),
    }
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (run_dir / "epoch_log.json").write_text(json.dumps(epoch_rows, indent=1) + "\n")
    print("saved", summary_path, "V1", summary["heldout_leakage_V1"],
          "acc_adv", summary["final_acc_adv"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--cfg", type=str)
    ap.add_argument("--latent", type=int)
    a = ap.parse_args()
    if a.cfg is None or a.latent is None:
        sel = json.loads(Y2_SEL.read_text())
        a.cfg, a.latent = sel["cfg"], int(sel["latent"])
    js = jobs(a.cfg, a.latent)
    if a.list:
        for i, j in enumerate(js):
            print(i, tag_for(j["cfg"], j["latent"], j["lam"], j["seed"], j["t"]), j)
        print("n_jobs", len(js))
        return
    if a.task_id is None:
        raise SystemExit("need --task_id")
    j = js[a.task_id]
    train_one(j["cfg"], j["latent"], j["lam"], j["seed"], j["t"], skip_done=a.skip_done)


if __name__ == "__main__":
    main()
