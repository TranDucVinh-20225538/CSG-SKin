#!/usr/bin/env python3
"""Paper B R5 P10: Camelyon17 two-hospital design (R2 item 5, primary assignment A = 3, B = 0, C = 2) with the
adversary toolbox, to test whether the "Leakage moved" gate can be met.

Everything not named here is R2 item 5: the same split file (results/paperB/r2/item5/split_camelyon2.npz), DenseNet-121
+ BN + linear classifier, Balanced A/B batches (16 + 16), 10 epochs, AdamW (lr 1e-4, wd 1e-4, head ×30, classifier
×0.2), AMP, 'grl' placement (λ scales the reversed gradient into the encoder; heads minimise the unscaled CE), best
epoch on A-select accuracy, the same evaluation (p12.detector_block, leakage_binary on ≤ 10,000 patches per side).

Ladder (λ = 1 unless stated, seed 42; one job at a time; each rung is cumulative):
  L1_bank      3 heads (linear, derm 2-layer, 3-layer), 2 inner head steps per batch, one head re-initialised every
               500 steps (round robin)
  L2_ganin     + Ganin α schedule (starts at 0) + 1 adversary warm-up epoch
  L3_strong    + 5 heads (2 linear, derm, wide derm, 3-layer), 5 inner steps, head LR ×10
  L4_lam3      L3 at λ = 3
  L5_lam10     L3 at λ = 10
Gate (decisions/paperB_r5_p10.md): held-out hospital site probe drops ≥ 0.10 vs λ = 0 mean (item5 seeds 42–44);
else inconclusive / "invariance not reached". Cap 40 GPU-h. Stage 2 (--stage2): first gate-meeting rung at seeds
43, 44 (λ = 0 = R2 item-5 runs).

--recipe: only the classifier LR multiplier applies (current = 0.2 = R2 item 5); latent size, class balance, batch size
and epochs are derm-task settings and are listed as not applicable (the Camelyon schedule stays at 10 epochs).

Outputs: results/paperB/r5/p10/{ladder,stage2}/<tag>/{config.json, adversary_log.jsonl, summary.json}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_r2_item5_camelyon2 as c2  # noqa: E402  (imports p12, which installs the transformers stub)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.models import DenseNet121_Weights, densenet121  # noqa: E402
from wilds import get_dataset  # noqa: E402

import r5_models as M  # noqa: E402
import r5_recipe  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

p12 = c2.p12
OUT = c2.PAPERB / "results" / "paperB" / "r5" / "p10"
CKPT = c2.PAPERB / "checkpoints" / "r5_p10"
GATE_DROP = 0.10  # probe_λ0 − probe_run ≥ this
CAP_GPU_H, EST_GPU_H = 40.0, 1.5
ITEM5_L0 = c2.PAPERB / "results" / "paperB" / "r2" / "item5" / "runs"


def lambda0_probe_mean():
    vals = []
    for s in (42, 43, 44):
        p = ITEM5_L0 / "camelyon2_ladv0_s{}".format(s) / "summary.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"])
    if not vals:
        raise SystemExit("P10 gate needs R2 item5 λ = 0 summaries (seeds 42–44)")
    return float(np.mean(vals))
_L1 = {"head_specs": [[1, 1.0], [2, 1.0], [3, 1.0]], "adv_inner_steps": 2, "reinit_every_steps": 500}
_L2 = dict(_L1, grl_schedule="ganin", adv_warmup_epochs=1)
_L3 = dict(_L2, head_specs=[[1, 1.0], [1, 1.0], [2, 1.0], [2, 2.0], [3, 1.0]], adv_inner_steps=5, adv_lr_multiplier=10.0)
RUNGS = {"L1_bank": (_L1, 1.0), "L2_ganin": (_L2, 1.0), "L3_strong": (_L3, 1.0), "L4_lam3": (_L3, 3.0),
         "L5_lam10": (_L3, 10.0),
         # 2026-10-17 label-shift arms (decisions amendment): L3 toolbox @ λ=2
         "natural_l2": (_L3, 2.0), "induced_l2": (_L3, 2.0)}
ORDER = tuple(RUNGS)
SHIFT_SEEDS = (42, 52, 62, 72, 82)
INDUCED_TUMOUR_A, INDUCED_TUMOUR_B = 0.80, 0.20


class Net(nn.Module):
    def __init__(self, head_specs):
        super().__init__()
        net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, 2)
        self.adv = M.AdversaryBank(self.feat_dim, 2, [tuple(h) for h in head_specs])
        self.coef = 0.0

    def forward(self, x):
        z = self.feat_bn(self.backbone(x))
        return self.classifier(z), z


class _EvalWrap(nn.Module):
    """p12.collect expects model(x) -> (logits, domain_logits, z)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits, z = self.net(x)
        return logits, logits, z


def gate_met(s, probe0=None):
    """Held-out hospital probe drop ≥ GATE_DROP vs λ = 0 mean; else invariance not reached."""
    probe0 = lambda0_probe_mean() if probe0 is None else probe0
    probe = s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"]
    return float(probe0 - probe) >= GATE_DROP


def p10_state(suffix):
    used, hits = 0.0, []
    for st in ("ladder", "stage2"):
        for f in sorted((OUT / st).glob("*/summary.json")):
            s = json.loads(f.read_text())
            used += float(s.get("gpu_hours", 0.0))
            same = s["tag"].endswith(suffix) if suffix else "_rec" not in s["tag"]
            if st == "ladder" and same and gate_met(s):
                hits.append(s)
    hits.sort(key=lambda s: ORDER.index(s["rung"]))
    return used, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=ORDER, default=None)
    ap.add_argument("--stage2", action="store_true", help="run the first gate-meeting ladder rung at --seed")
    ap.add_argument("--shift_arm", action="store_true",
                    help="2026-10-17 arms: write under p10/shift_arms/; no ladder stop rule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_done", action="store_true")
    r5_recipe.add_args(ap)
    args = ap.parse_args()
    rec, rsrc = r5_recipe.resolve(args.recipe)
    args.cls_lr_multiplier = float(rec["cls_lr_multiplier"])
    probe0 = lambda0_probe_mean()
    used, hits = p10_state(r5_recipe.tag(args.recipe))
    stage = "shift_arms" if args.shift_arm else ("stage2" if args.stage2 else "ladder")
    if args.stage2:
        if not hits:
            print("P10 stage 2: no ladder rung met the gate; nothing to run")
            return
        rung = hits[0]["rung"]
    else:
        if args.rung is None:
            raise SystemExit("--rung required for the ladder / shift arm")
        rung = args.rung
    tb_over, lam = RUNGS[rung]
    tb = dict(M.TOOLBOX_DEFAULT)
    tb.update(tb_over)
    tag = "camelyon2_{}_ladv{}_s{}".format(rung, p12.lam_tag(lam), args.seed) + r5_recipe.tag(args.recipe)
    run_dir, ckpt_dir = OUT / stage / tag, CKPT / stage / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and (run_dir / "summary.json").exists():
        print("skip_done", tag)
        return
    if not args.shift_arm and not args.stage2 and (hits or used + EST_GPU_H > CAP_GPU_H):
        why = {"tag": tag, "skipped": True, "gate_met_by": [h["tag"] for h in hits], "p10_gpu_h_used": used,
               "cap": CAP_GPU_H}
        (run_dir / "skipped.json").write_text(json.dumps(why, indent=2) + "\n")
        print("P10 ladder stop", why)
        return
    if args.stage2 and used + EST_GPU_H > CAP_GPU_H:
        (run_dir / "skipped.json").write_text(json.dumps({"tag": tag, "skipped": True, "p10_gpu_h_used": used,
                                                          "cap": CAP_GPU_H}, indent=2) + "\n")
        print("P10 stage 2 over cap")
        return
    if args.shift_arm and used + EST_GPU_H > CAP_GPU_H:
        (run_dir / "skipped.json").write_text(json.dumps({"tag": tag, "skipped": True, "p10_gpu_h_used": used,
                                                          "cap": CAP_GPU_H}, indent=2) + "\n")
        print("P10 shift arm over cap")
        return

    t0 = time.time()
    ha, hb, hc = c2.ASSIGN["primary"]
    seed_everything(args.seed)
    sp = dict(np.load(c2.split_path("primary")))
    ds = get_dataset(dataset="camelyon17", root_dir=str(p12.DATA_ROOT), download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(tb["head_specs"]).to(device)
    adv_lr = args.lr * float(tb["adv_lr_multiplier"])
    opt = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()) + list(model.feat_bn.parameters()), "lr": args.lr},
        {"params": model.adv.parameters(), "lr": adv_lr},
        {"params": model.classifier.parameters(), "lr": args.lr * args.cls_lr_multiplier},
    ], weight_decay=args.weight_decay)
    inner = torch.optim.AdamW(model.adv.parameters(), lr=adv_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_idx = np.r_[sp["a_train"], sp["b_adv"]]
    is_a = np.r_[np.ones(len(sp["a_train"]), bool), np.zeros(len(sp["b_adv"]), bool)]
    # labels for induced resampling (y from dataset metadata)
    y_all = np.asarray(ds.y[train_idx], int).reshape(-1)
    train_ds = p12.CamelyonIndexDataset(ds, train_idx, p12.train_tf())
    a_pos, b_pos = np.where(is_a)[0], np.where(~is_a)[0]
    induced = rung.startswith("induced")
    if induced:
        # Resample adversary-stream indices only: A side 80% tumour, B side 20% tumour.
        def _resample(pos, tumour_frac, rng):
            y = y_all[pos]
            tum, nor = pos[y == 1], pos[y == 0]
            n = len(pos)
            n_t = int(round(tumour_frac * n))
            n_n = n - n_t
            take_t = tum[rng.integers(0, len(tum), n_t)] if len(tum) else tum
            take_n = nor[rng.integers(0, len(nor), n_n)] if len(nor) else nor
            out = np.r_[take_t, take_n]
            rng.shuffle(out)
            return out
        rng0 = np.random.default_rng(args.seed + 17)
        a_pos = _resample(a_pos, INDUCED_TUMOUR_A, rng0)
        b_pos = _resample(b_pos, INDUCED_TUMOUR_B, rng0)
    n_per = args.batch_size // 2
    n_batches = max(1, min(len(a_pos), len(b_pos)) // n_per)
    total = args.epochs * n_batches
    sel_loader = p12.loader_for(ds, sp["a_sel"], p12.eval_tf(), 128, args.num_workers, False)
    cfg = dict(vars(args), tag=tag, stage=stage, rung=rung, lambda_adv=lam, toolbox=tb, hospitals=[ha, hb, hc],
               split=str(c2.split_path("primary")), n_batches_per_epoch=n_batches,
               induced_shift=induced,
               induced_tumour_frac={"A_adv": INDUCED_TUMOUR_A, "B_adv": INDUCED_TUMOUR_B} if induced else None,
               gate={"drop_ge": GATE_DROP, "lambda0_probe_mean": probe0,
                     "rule": "probe_λ0 − probe ≥ 0.10 else invariance not reached"},
               placement="grl (R2 item 5)",
               semantic_ood={"name": "NCT-CRC-HE-100K", "licence": "CC-BY-4.0",
                             "zenodo": "10.5281/zenodo.1214456",
                             "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
                             "note": "semantic OOD only; AUROC needs no label map; see decisions/paperB_r5_p10.md"},
               **r5_recipe.describe(rec, rsrc, not_applicable=[k for k in rec if k != "cls_lr_multiplier"]))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    log = run_dir / "adversary_log.jsonl"
    log.write_text("")
    step, best, reinits = 0, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        warm = epoch <= int(tb["adv_warmup_epochs"])
        loader = DataLoader(train_ds, batch_sampler=c2.Balanced(a_pos, b_pos, n_per, n_batches, args.seed, epoch),
                            num_workers=args.num_workers, pin_memory=True)
        ce, ac, inn = [], [], []
        for x, y, h in loader:
            x, y, h = x.to(device, non_blocking=True), y.to(device), h.to(device)
            d = (h == hb).long()
            alpha, _ = M.grl_alpha_schedule(step, total, tb["grl_schedule"])
            coef = 0.0 if warm else float(lam) * alpha
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, z = model(x)
                # Classifier on natural A labels only (d==0). Induced arm still uses true y on those rows.
                l_cls = F.cross_entropy(logits[d == 0], y[d == 0])
                outs = model.adv(z.detach() if warm else z, coef)
                l_adv = torch.stack([F.cross_entropy(o.float(), d) for o in outs]).mean()
                loss = l_cls + l_adv
            if not math.isfinite(float(loss)):
                raise RuntimeError("non-finite loss at step {}".format(step))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            zd = z.detach().float()
            for _ in range(int(tb["adv_inner_steps"])):
                inner.zero_grad(set_to_none=True)
                l_in = torch.stack([F.cross_entropy(o, d) for o in model.adv(zd, 0.0)]).mean()
                l_in.backward()
                inner.step()
                inn.append(float(l_in))
            n = int(tb["reinit_every_steps"])
            if n > 0 and step > 0 and step % n == 0:
                k = reinits % len(model.adv.heads)
                model.adv.reinit(k)
                for p in model.adv.heads[k].parameters():
                    opt.state.pop(p, None)
                    inner.state.pop(p, None)
                reinits += 1
            ce.append(float(l_adv))
            ac.append(float(torch.stack([(o.argmax(1) == d).float().mean() for o in outs]).mean()))
            step += 1
        sel = p12.eval_acc(_EvalWrap(model), sel_loader, device)
        rec = {"epoch": epoch, "loss_adv": float(np.mean(ce)), "acc_adv": float(np.mean(ac)),
               "inner_ce": float(np.mean(inn)) if inn else None, "id_select_acc": sel, "coef_end": coef,
               "n_reinit": reinits, "warmup": warm}
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("epoch", rec, flush=True)
        if sel > best:
            best = sel
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "sel": sel}, ckpt_dir / "best.pt")

    blob = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    ev = _EvalWrap(model)
    packs = {k: p12.collect(ev, p12.loader_for(ds, sp[k], p12.eval_tf(), 256, args.num_workers, False), device)
             for k in ("a_train", "a_test", "a_clean", "b_adv", "b_hold", "c")}
    tr = packs["a_train"]
    rng = np.random.default_rng(0)

    def sub(z, n=c2.PROBE_MAX):
        return z[rng.choice(len(z), min(n, len(z)), replace=False)]

    ood = {}
    for k in ("b_hold", "b_adv", "c"):
        op = packs[k]
        ood[k] = {"accuracy": p12.acc_pack(op["logits"], op["y"]), "n": int(len(op["y"]))}
        for idk, lab in (("a_test", "detectors"), ("a_clean", "detectors_slide_disjoint_id")):
            idp = packs[idk]
            ood[k][lab] = p12.detector_block(tr["z"], tr["y"], idp["z"], idp["logits"], op["z"], op["logits"],
                                             p12.KNN_BANK, args.seed)
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    summary = {"tag": tag, "stage": stage, "rung": rung, "assign": "primary", "hospitals": [ha, hb, hc],
               "lambda_adv": lam, "seed": args.seed, "toolbox": tb, "best_epoch": int(blob["epoch"]),
               "id_test": p12.acc_pack(packs["a_test"]["logits"], packs["a_test"]["y"]),
               "id_test_slide_disjoint": p12.acc_pack(packs["a_clean"]["logits"], packs["a_clean"]["y"]),
               "leakage_site_probe_a_vs_bhold": p12.leakage_binary(sub(packs["a_test"]["z"]), sub(packs["b_hold"]["z"])),
               "leakage_site_probe_aclean_vs_bhold": p12.leakage_binary(sub(packs["a_clean"]["z"]), sub(packs["b_hold"]["z"])),
               "ood": ood, "history": rows,
               "adversary": {"min_epoch_ce": min(r["loss_adv"] for r in rows), "last": rows[-1]},
               "gpu_hours": (time.time() - t0) / 3600.0}
    probe = summary["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"]
    summary["lambda0_probe_mean"] = probe0
    summary["probe_drop_vs_lambda0"] = float(probe0 - probe)
    summary["gate_met"] = bool(gate_met(summary, probe0))
    summary["invariance_not_reached"] = not summary["gate_met"]
    summary["b_heldout_accuracy"] = ood["b_hold"]["accuracy"]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "probe", probe, "drop", summary["probe_drop_vs_lambda0"],
          "ID acc", summary["id_test"]["acc"], "gate", summary["gate_met"],
          "invariance_not_reached", summary["invariance_not_reached"], flush=True)


if __name__ == "__main__":
    main()
