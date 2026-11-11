#!/usr/bin/env python3
"""Paper B R11b Y1: diagnose X3/X5 adversary collapse (CPU).

For every R11 X3 and X5 run, from saved last.ckpt + cache:
  (1) final in-training adversary acc/loss (epoch_log)
  (2) dead ReLU fraction per adversary hidden layer (always-zero on isic_train∪pad_adv)
  (3) fresh derm_domain_head on frozen latents; held-out ISIC-test vs pad_heldout acc
  (4) GRL unit-test cosine (grad_h through GRL vs direct)

Writes results/paperB/r11b/y1/ — never touches results/paperB/r11/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Y1 is CPU-only; hide GPUs before importing torch (login node has broken driver).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

MAIN = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
WT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b")
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(MAIN / "scripts"))

from phase15_common import derm_domain_head, GradientReversal  # noqa: E402
from r11_x3_train import HeadNet, LATENT, HIDDEN  # noqa: E402

R11_X3 = MAIN / "results" / "paperB" / "r11" / "x3"
R11_X5 = MAIN / "results" / "paperB" / "r11" / "x5"
CKPT_X3 = MAIN / "checkpoints" / "r11_x3"
CKPT_X5 = MAIN / "checkpoints" / "r11_x5"
CACHE = R11_X3 / "cache"
OUT = WT / "results" / "paperB" / "r11b" / "y1"
LN2 = float(math.log(2.0))
FRESH_EPOCHS = 20
FRESH_LR = 2e-4
BATCH = 512


def load_npz(path: Path):
    return dict(np.load(path, allow_pickle=True))


def list_runs():
    rows = []
    for d in sorted((R11_X3 / "runs").iterdir()):
        if not d.is_dir():
            continue
        tag = d.name
        rows.append({
            "item": "x3", "tag": tag,
            "run_dir": d,
            "ckpt": CKPT_X3 / tag / "last.ckpt",
            "cache_seed": int(tag.rsplit("_s", 1)[-1]),
        })
    for d in sorted((R11_X5 / "runs").iterdir()):
        if not d.is_dir():
            continue
        tag = d.name
        rows.append({
            "item": "x5", "tag": tag,
            "run_dir": d,
            "ckpt": CKPT_X5 / tag / "last.ckpt",
            "cache_seed": int(tag.rsplit("_s", 1)[-1]),
        })
    return rows


def parse_meta(tag: str, item: str):
    # x3_ladv2_shift100_s42  or  x5_ladv2_t0p5_s42
    parts = tag.split("_")
    lam_s = parts[1].replace("ladv", "").replace("p", ".")
    lam = float(lam_s)
    seed = int(parts[-1][1:]) if parts[-1].startswith("s") else int(parts[-1].split("s")[-1])
    meta = {"lam": lam, "seed": seed}
    if item == "x3":
        sh = parts[2].replace("shift", "")
        meta["shift"] = float(sh) / 100.0
    else:
        t_s = parts[2].replace("t", "").replace("p", ".")
        meta["t"] = float(t_s)
    return meta


@torch.no_grad()
def latent_of(net: HeadNet, z: np.ndarray, device) -> np.ndarray:
    net.eval()
    outs = []
    for i in range(0, len(z), BATCH):
        zb = torch.as_tensor(z[i:i + BATCH], dtype=torch.float32, device=device)
        _, _, h = net(zb, return_latent=True)
        outs.append(h.cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def dead_relu_fractions(domain_head: nn.Sequential, h: np.ndarray, device):
    """Fraction of units that are exactly 0 after each ReLU on *all* inputs."""
    domain_head.eval()
    # Collect pre/post activations by walking modules
    x = torch.as_tensor(h, dtype=torch.float32, device=device)
    fracs = {}
    layer_i = 0
    with torch.no_grad():
        for m in domain_head:
            x = m(x)
            if isinstance(m, nn.ReLU):
                # x is post-ReLU: dead if == 0 for every sample
                always_zero = (x == 0).all(dim=0)
                fracs["relu{}".format(layer_i)] = float(always_zero.float().mean().cpu())
                layer_i += 1
    return fracs


def train_fresh_adv(h_isic, h_pad, h_isic_te, h_pad_ho, in_dim, device, seed):
    torch.manual_seed(seed)
    adv = derm_domain_head(in_dim, 2).to(device)
    opt = torch.optim.AdamW(adv.parameters(), lr=FRESH_LR, weight_decay=1e-4)
    z0 = torch.as_tensor(h_isic, dtype=torch.float32, device=device)
    z1 = torch.as_tensor(h_pad, dtype=torch.float32, device=device)
    n0, n1 = len(z0), len(z1)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    steps_per = max(1, int(math.ceil(max(n0, n1) / BATCH)))
    for _ep in range(FRESH_EPOCHS):
        adv.train()
        for _ in range(steps_per):
            i0 = torch.randint(0, n0, (BATCH,), generator=rng)
            i1 = torch.randint(0, n1, (BATCH,), generator=rng)
            zb = torch.cat([z0[i0], z1[i1]], 0)
            yb = torch.cat([
                torch.zeros(BATCH, dtype=torch.long, device=device),
                torch.ones(BATCH, dtype=torch.long, device=device),
            ])
            loss = F.cross_entropy(adv(zb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    adv.eval()
    with torch.no_grad():
        te = torch.as_tensor(h_isic_te, dtype=torch.float32, device=device)
        ho = torch.as_tensor(h_pad_ho, dtype=torch.float32, device=device)
        pred0 = adv(te).argmax(1).cpu().numpy()
        pred1 = adv(ho).argmax(1).cpu().numpy()
        y = np.concatenate([np.zeros(len(pred0), dtype=int), np.ones(len(pred1), dtype=int)])
        p = np.concatenate([pred0, pred1])
        acc = float(accuracy_score(y, p))
    return acc


def grl_cosine(net: HeadNet, z_batch: torch.Tensor, d_batch: torch.Tensor):
    """Cosine between ∇_h L_adv through GRL and ∇_h L_adv without GRL (adversary direction)."""
    net.eval()
    h = net.bn(net.mlp(z_batch)).detach().requires_grad_(True)
    # through GRL
    net.grl.set_lambd(1.0)
    dlog_grl = net.domain_head(net.grl(h))
    loss_grl = F.cross_entropy(dlog_grl, d_batch)
    g_grl = torch.autograd.grad(loss_grl, h, retain_graph=False, create_graph=False)[0]

    h2 = h.detach().requires_grad_(True)
    dlog_dir = net.domain_head(h2)
    loss_dir = F.cross_entropy(dlog_dir, d_batch)
    g_dir = torch.autograd.grad(loss_dir, h2, retain_graph=False, create_graph=False)[0]

    a = g_grl.reshape(-1)
    b = g_dir.reshape(-1)
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return cos


def diagnose_one(rec, device):
    import time
    t0 = time.time()
    tag = rec["tag"]
    item = rec["item"]
    meta = parse_meta(tag, item)
    run_dir = rec["run_dir"]
    elog_path = run_dir / "epoch_log.json"
    if not elog_path.exists():
        # fall back to summary
        summary = json.loads((run_dir / "summary.json").read_text())
        epochs = summary.get("epoch_log") or []
    else:
        epochs = json.loads(elog_path.read_text())
    last = epochs[-1] if epochs else {}
    acc_adv = float(last.get("acc_adv", float("nan")))
    loss_adv = float(last.get("loss_adv", float("nan")))

    ckpt = torch.load(rec["ckpt"], map_location=device)
    sd = ckpt["state_dict"]
    # infer latent from classifier weight
    latent = int(sd["classifier.weight"].shape[1])
    net = HeadNet(in_dim=1536, latent=latent, hidden=HIDDEN).to(device)
    net.load_state_dict(sd)
    net.eval()

    seed = rec["cache_seed"]
    cdir = CACHE / "s{}".format(seed)
    isic_tr = load_npz(cdir / "isic_train.npz")
    pad_adv = load_npz(cdir / "pad_adv.npz")
    isic_te = load_npz(cdir / "isic_test.npz")
    pad_ho = load_npz(cdir / "pad_heldout.npz")

    h_tr = latent_of(net, isic_tr["z_lesion_norm"], device)
    h_pad = latent_of(net, pad_adv["z_lesion_norm"], device)
    h_te = latent_of(net, isic_te["z_lesion_norm"], device)
    h_ho = latent_of(net, pad_ho["z_lesion_norm"], device)
    h_all = np.concatenate([h_tr, h_pad], 0)
    print("  latents {:.1f}s".format(time.time() - t0), flush=True)

    dead = dead_relu_fractions(net.domain_head, h_all, device)
    print("  dead_relu {:.1f}s".format(time.time() - t0), flush=True)
    fresh_acc = train_fresh_adv(h_tr, h_pad, h_te, h_ho, latent, device, seed=seed + 17)
    print("  fresh_adv {:.1f}s".format(time.time() - t0), flush=True)

    # one batch for GRL test: half isic_train, half pad_adv
    b = 64
    z0 = torch.as_tensor(isic_tr["z_lesion_norm"][:b], dtype=torch.float32, device=device)
    z1 = torch.as_tensor(pad_adv["z_lesion_norm"][:b], dtype=torch.float32, device=device)
    zb = torch.cat([z0, z1], 0)
    db = torch.cat([
        torch.zeros(b, dtype=torch.long, device=device),
        torch.ones(b, dtype=torch.long, device=device),
    ])
    cos = grl_cosine(net, zb, db)
    print("  grl {:.1f}s".format(time.time() - t0), flush=True)

    row = {
        "item": item,
        "tag": tag,
        **meta,
        "final_acc_adv": acc_adv,
        "final_loss_adv": loss_adv,
        "dead_relu": dead,
        "fresh_adv_heldout_acc": fresh_acc,
        "grl_grad_cosine": cos,
        "latent": latent,
        "n_epochs": len(epochs),
    }
    return row


def collapse_verdict(rows):
    """X3 λ=2 shift 100%: collapse if (1)≈0.5 and (3)≥0.70 in ≥4/5 seeds."""
    subset = [
        r for r in rows
        if r["item"] == "x3" and abs(r.get("lam", -1) - 2.0) < 1e-9
        and abs(r.get("shift", -1) - 1.0) < 1e-9
    ]
    n_hit = 0
    detail = []
    for r in subset:
        c1 = abs(r["final_acc_adv"] - 0.5) < 0.01 and abs(r["final_loss_adv"] - LN2) < 0.01
        c3 = r["fresh_adv_heldout_acc"] >= 0.70
        hit = c1 and c3
        n_hit += int(hit)
        detail.append({"tag": r["tag"], "c1": c1, "c3": c3, "hit": hit,
                       "acc_adv": r["final_acc_adv"], "fresh": r["fresh_adv_heldout_acc"]})
    confirmed = n_hit >= 4
    return {
        "n_seeds": len(subset),
        "n_hit": n_hit,
        "collapse_confirmed": confirmed,
        "rule": "(1)≈0.5 and (3)≥0.70 in ≥4 of 5 seeds at X3 λ=2 shift100",
        "detail": detail,
    }


def write_table(rows):
    table_path = OUT / "table.json"
    table_path.write_text(json.dumps(rows, indent=1) + "\n")
    verd = collapse_verdict(rows)
    (OUT / "verdict.json").write_text(json.dumps(verd, indent=1) + "\n")
    lines = [
        "# Y1 adversary collapse diagnose",
        "",
        "| item | tag | λ | final acc_adv | final loss_adv | dead relu0 | dead relu1 | fresh heldout acc | GRL cosine |",
        "|------|-----|---:|--------------:|---------------:|-----------:|-----------:|------------------:|-----------:|",
    ]
    for r in rows:
        d = r["dead_relu"]
        lines.append(
            "| {item} | {tag} | {lam:g} | {acc:.4f} | {loss:.4f} | {d0:.3f} | {d1:.3f} | {fresh:.4f} | {cos:.4f} |".format(
                item=r["item"], tag=r["tag"], lam=r["lam"],
                acc=r["final_acc_adv"], loss=r["final_loss_adv"],
                d0=d.get("relu0", float("nan")), d1=d.get("relu1", float("nan")),
                fresh=r["fresh_adv_heldout_acc"], cos=r["grl_grad_cosine"],
            )
        )
    lines += [
        "",
        "## Verdict",
        "",
        "Collapse confirmed: **{}** ({}/{} seeds at X3 λ=2 shift 100 %).".format(
            verd["collapse_confirmed"], verd["n_hit"], verd["n_seeds"]),
        "",
        "Rule: {}".format(verd["rule"]),
        "",
    ]
    (OUT / "TABLE.md").write_text("\n".join(lines) + "\n")
    print("wrote", table_path, "verdict", verd["collapse_confirmed"], flush=True)
    return verd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="0-based shard index")
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--finalize", action="store_true",
                    help="rebuild TABLE.md from all per-run json only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.finalize:
        rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("x*.json"))]
        write_table(rows)
        return

    device = torch.device(a.device)
    runs = list_runs()
    if a.limit:
        runs = runs[: a.limit]
    runs = [r for i, r in enumerate(runs) if i % a.n_shards == a.shard]
    print("shard {}/{} n={}".format(a.shard, a.n_shards, len(runs)), flush=True)

    for i, rec in enumerate(runs):
        out_p = OUT / "{}.json".format(rec["tag"])
        if a.skip_done and out_p.exists():
            print("[{}/{}] skip {}".format(i + 1, len(runs), rec["tag"]), flush=True)
            continue
        print("[{}/{}] {}".format(i + 1, len(runs), rec["tag"]), flush=True)
        if not rec["ckpt"].exists():
            print("  MISSING ckpt", rec["ckpt"], flush=True)
            continue
        row = diagnose_one(rec, device)
        out_p.write_text(json.dumps(row, indent=1) + "\n")
        print("  acc_adv={:.4f} loss={:.4f} fresh={:.4f} cos={:.4f} dead={}".format(
            row["final_acc_adv"], row["final_loss_adv"], row["fresh_adv_heldout_acc"],
            row["grl_grad_cosine"], row["dead_relu"]), flush=True)

    if a.n_shards == 1:
        rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("x*.json"))]
        write_table(rows)


if __name__ == "__main__":
    main()
