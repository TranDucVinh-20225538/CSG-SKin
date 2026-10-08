#!/usr/bin/env python3
"""R2 Item 5: Camelyon17 restricted to two hospitals, dermatology-mirrored design.

Hospital A: labelled ID. Two A slides are withheld entirely (slide-disjoint ID set); the rest is split
patch-level into train / select / test (primary ID test, shares slides with train).
Hospital B: split by slide into B_adv (adversary sees it, labels never used) and B_heldout (OOD).
Hospital C: never adversarially seen (Fitzpatrick analogue).
Assignments (primary 3/0/2, fallback 3/2/1) fixed in results/paperB/r2/PRECOMMIT_CAMELYON2.json.

--adv_placement grl  : λ scales the reversed gradient into the encoder only; head minimises unscaled CE.
--adv_placement loss : Phase 12.1b / iWildCam form, L = L_cls + λ·CE(head(GRL_α(z))) (diagnostic only).
Adversary CE / accuracy and the reversed-gradient norm are logged in 200-step windows from step 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase12_camelyon_dann as p12  # noqa: E402  (installs the transformers stub)
import train_phase12_1b as p12b  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.models import DenseNet121_Weights, densenet121  # noqa: E402
from wilds import get_dataset  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = PAPERB / "results" / "paperB" / "r2" / "item5"
CKPT = PAPERB / "checkpoints" / "r2_item5"
ASSIGN = {"primary": (3, 0, 2), "fallback": (3, 2, 1)}
A_HELDOUT_TUMOUR_SLIDE = 32
A_TUMOUR_FREE_CANDIDATES = (30, 31, 33, 34, 35, 36, 37)
LOG_WINDOW = 200
PROBE_MAX = 10000


def a_heldout_slides():
    return [A_HELDOUT_TUMOUR_SLIDE, int(np.random.default_rng(42).choice(A_TUMOUR_FREE_CANDIDATES))]


def split_path(assign):
    return OUT / ("split_camelyon2.npz" if assign == "primary" else "split_camelyon2_fallback.npz")


def run_prefix(assign):
    return "camelyon2" if assign == "primary" else "camelyon2fb"


class GRL(torch.autograd.Function):
    store = []

    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        out = -ctx.coef * g
        GRL.store.append(float(out.detach().float().norm(dim=1).mean()))
        return out, None


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.feat_dim = net.classifier.in_features
        net.classifier = nn.Identity()
        self.backbone = net
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, 2)
        self.domain_head = p12b.derm_domain_head(self.feat_dim, 2)
        self.coef = 0.0

    def forward(self, x):
        z = self.feat_bn(self.backbone(x))
        return self.classifier(z), self.domain_head(GRL.apply(z, self.coef)), z


def make_split(assign):
    ha, hb, hc = ASSIGN[assign]
    ds = get_dataset(dataset="camelyon17", root_dir=str(p12.DATA_ROOT), download=False)
    meta = ds.metadata_array.numpy()
    hosp, slide, y = meta[:, 0], meta[:, 1], meta[:, 2]
    held = a_heldout_slides()
    a_all = np.where(hosp == ha)[0]
    a_clean = a_all[np.isin(slide[a_all], held)]
    a = a_all[~np.isin(slide[a_all], held)]
    a_tv, a_test = train_test_split(a, test_size=0.2, stratify=slide[a] * 2 + y[a], random_state=42)
    a_train, a_sel = train_test_split(a_tv, test_size=0.2, stratify=slide[a_tv] * 2 + y[a_tv], random_state=42)
    b = np.where(hosp == hb)[0]
    adv_i, hold_i = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42).split(b, groups=slide[b]))
    b_adv, b_hold = b[adv_i], b[hold_i]
    c = np.where(hosp == hc)[0]
    assert not set(slide[b_adv]) & set(slide[b_hold])
    assert not set(slide[a_clean]) & set(slide[a_train])
    parts = dict(a_train=a_train, a_sel=a_sel, a_test=a_test, a_clean=a_clean, b_adv=b_adv, b_hold=b_hold, c=c)
    np.savez(split_path(assign), **parts)
    rec = {"assignment": assign, "hospitals": {"A": ha, "B": hb, "C": hc}, "a_withheld_slides": held}
    for k, v in parts.items():
        rec[k] = {"n": int(len(v)), "tumour_frac": float(y[v].mean()), "slides": sorted(int(s) for s in set(slide[v]))}
    split_path(assign).with_suffix(".json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


class Balanced:
    def __init__(self, a_pos, b_pos, n_per, n_batches, seed, epoch):
        self.a, self.b, self.n, self.nb = a_pos, b_pos, n_per, n_batches
        self.rng = np.random.default_rng(seed * 1009 + epoch)

    def __len__(self):
        return self.nb

    def __iter__(self):
        for _ in range(self.nb):
            batch = np.r_[self.rng.choice(self.a, self.n), self.rng.choice(self.b, self.n)]
            self.rng.shuffle(batch)
            yield batch.tolist()


def windows_summary(win):
    return {
        "min_window_adv_ce": float(min(w["adv_ce"] for w in win)) if win else None,
        "first_window": win[0] if win else None,
        "last_window": win[-1] if win else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_adv", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--assign", choices=tuple(ASSIGN), default="primary")
    ap.add_argument("--adv_placement", choices=("grl", "loss"), default="grl")
    ap.add_argument("--max_steps", type=int, default=0, help="diagnostic: stop after N steps, log only, no eval")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--adv_lr_multiplier", type=float, default=30.0)
    ap.add_argument("--cls_lr_multiplier", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--split_only", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ha, hb, hc = ASSIGN[args.assign]
    if args.split_only or not split_path(args.assign).exists():
        print(json.dumps(make_split(args.assign), indent=1))
        if args.split_only:
            return

    tag = "{}_ladv{}_s{}".format(run_prefix(args.assign), p12.lam_tag(args.lambda_adv), args.seed)
    if args.adv_placement == "loss":
        tag += "_placeloss"
    if args.max_steps:
        tag += "_diag{}".format(args.max_steps)
    run_dir, ckpt_dir = OUT / ("diag" if args.max_steps else "runs") / tag, CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / ("adversary_windows.json" if args.max_steps else "summary.json")
    if args.skip_done and done_file.exists():
        print("skip_done", tag)
        return
    seed_everything(args.seed)
    sp = dict(np.load(split_path(args.assign)))
    ds = get_dataset(dataset="camelyon17", root_dir=str(p12.DATA_ROOT), download=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    opt = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()) + list(model.feat_bn.parameters()), "lr": args.lr},
        {"params": model.domain_head.parameters(), "lr": args.lr * args.adv_lr_multiplier},
        {"params": model.classifier.parameters(), "lr": args.lr * args.cls_lr_multiplier},
    ], weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_idx = np.r_[sp["a_train"], sp["b_adv"]]
    is_a = np.r_[np.ones(len(sp["a_train"]), bool), np.zeros(len(sp["b_adv"]), bool)]
    train_ds = p12.CamelyonIndexDataset(ds, train_idx, p12.train_tf())
    a_pos, b_pos = np.where(is_a)[0], np.where(~is_a)[0]
    n_per = args.batch_size // 2
    n_batches = len(a_pos) // n_per
    total = args.epochs * n_batches
    sel_loader = p12.loader_for(ds, sp["a_sel"], p12.eval_tf(), 128, args.num_workers, False)
    cfg = dict(vars(args), tag=tag, n_batches_per_epoch=n_batches, hospitals=[ha, hb, hc],
               a_withheld_slides=a_heldout_slides(),
               adversary=("lambda on reversed gradient only; head minimises unscaled CE" if args.adv_placement == "grl"
                          else "Phase 12.1b form: L = L_cls + lambda * CE(head(GRL_alpha(z)))"),
               grl_schedule="alpha = max(0.2, 2/(1+exp(-20 p^0.5))-1)")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    hist, win, step, best = [], [], 0, -1.0
    ce_w, ac_w, gn_w = [], [], []
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = DataLoader(train_ds, batch_sampler=Balanced(a_pos, b_pos, n_per, n_batches, args.seed, epoch),
                            num_workers=args.num_workers, pin_memory=True)
        ep_ce, ep_ac, ep_n = 0.0, 0, 0
        for x, y, h in loader:
            x, y, h = x.to(device, non_blocking=True), y.to(device), h.to(device)
            d = (h == hb).long()
            alpha, _ = p12b.grl_alpha(step, total)
            if args.adv_placement == "grl":
                model.coef, w_adv = float(args.lambda_adv) * alpha, 1.0
            else:
                model.coef, w_adv = alpha, float(args.lambda_adv)
            GRL.store.clear()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, dlog, _ = model(x)
                l_cls = F.cross_entropy(logits[d == 0], y[d == 0])
                l_adv = F.cross_entropy(dlog.float(), d)
                loss = l_cls + w_adv * l_adv
            if not math.isfinite(float(loss)):
                raise RuntimeError("non-finite loss at step {}".format(step))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ce_w.append(float(l_adv))
            ac_w.append(float((dlog.argmax(1) == d).float().mean()))
            gn_w.append(GRL.store[-1] / scaler.get_scale() if GRL.store else 0.0)
            ep_ce += float(l_adv) * len(d)
            ep_ac += int((dlog.argmax(1) == d).sum())
            ep_n += len(d)
            step += 1
            if step % LOG_WINDOW == 0:
                win.append({"step": step, "epoch": epoch, "adv_ce": float(np.mean(ce_w)), "adv_acc": float(np.mean(ac_w)),
                            "grl_grad_norm": float(np.mean(gn_w)), "coef": model.coef, "w_adv": w_adv})
                ce_w, ac_w, gn_w = [], [], []
                print("win", win[-1], flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break
        sel = p12.eval_acc(model, sel_loader, device)
        rec = {"epoch": epoch, "adv_ce": ep_ce / ep_n, "adv_acc": ep_ac / ep_n, "id_select_acc": sel, "coef_end": model.coef}
        hist.append(rec)
        print("epoch", rec, flush=True)
        if sel > best:
            best = sel
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "sel": sel}, ckpt_dir / "best.pt")
    (run_dir / "adversary_windows.json").write_text(json.dumps(win) + "\n")
    if args.max_steps:
        print("diagnostic done", tag, windows_summary(win), flush=True)
        return

    blob = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    packs = {k: p12.collect(model, p12.loader_for(ds, sp[k], p12.eval_tf(), 256, args.num_workers, False), device)
             for k in ("a_train", "a_test", "a_clean", "b_adv", "b_hold", "c")}
    tr = packs["a_train"]
    rng = np.random.default_rng(0)

    def sub(z, n=PROBE_MAX):
        return z[rng.choice(len(z), min(n, len(z)), replace=False)]

    ood = {}
    for k in ("b_hold", "b_adv", "c"):
        op = packs[k]
        ood[k] = {"accuracy": p12.acc_pack(op["logits"], op["y"]), "n": int(len(op["y"]))}
        for idk, lab in (("a_test", "detectors"), ("a_clean", "detectors_slide_disjoint_id")):
            idp = packs[idk]
            ood[k][lab] = p12.detector_block(tr["z"], tr["y"], idp["z"], idp["logits"], op["z"], op["logits"], p12.KNN_BANK, args.seed)
    summary = {"tag": tag, "assign": args.assign, "hospitals": [ha, hb, hc], "lambda_adv": args.lambda_adv, "seed": args.seed,
               "adv_placement": args.adv_placement, "best_epoch": int(blob["epoch"]),
               "id_test": p12.acc_pack(packs["a_test"]["logits"], packs["a_test"]["y"]),
               "id_test_slide_disjoint": p12.acc_pack(packs["a_clean"]["logits"], packs["a_clean"]["y"]),
               "leakage_site_probe_a_vs_bhold": p12.leakage_binary(sub(packs["a_test"]["z"]), sub(packs["b_hold"]["z"])),
               "leakage_site_probe_aclean_vs_bhold": p12.leakage_binary(sub(packs["a_clean"]["z"]), sub(packs["b_hold"]["z"])),
               "ood": ood, "history": hist, "adversary": windows_summary(win), "n_windows": len(win)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "ID acc", summary["id_test"]["acc"],
          "ID acc slide-disjoint", summary["id_test_slide_disjoint"]["acc"],
          "Maha b_hold", ood["b_hold"]["detectors"]["mahalanobis_classcond_sharedcov"], flush=True)


if __name__ == "__main__":
    main()
