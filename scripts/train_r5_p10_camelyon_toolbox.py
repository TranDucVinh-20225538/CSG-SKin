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
from torch.utils.data import DataLoader, Dataset  # noqa: E402
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
NCT_PATHS = OUT / "nct_crc_he_eval_paths.txt"
NCT_KEY = "nct_crc_he"


def lambda0_probe_mean():
    vals = []
    for s in (42, 43, 44):
        p = ITEM5_L0 / "camelyon2_ladv0_s{}".format(s) / "summary.json"
        if p.exists():
            vals.append(json.loads(p.read_text())["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"])
    if not vals:
        raise SystemExit("P10 gate needs R2 item5 λ = 0 summaries (seeds 42–44)")
    return float(np.mean(vals))


def lambda0_id_bal_mean():
    vals = []
    for s in (42, 43, 44):
        p = ITEM5_L0 / "camelyon2_ladv0_s{}".format(s) / "summary.json"
        if p.exists():
            d = json.loads(p.read_text())["id_test"]
            vals.append(float(d.get("balanced_acc", d["acc"])))
    if not vals:
        raise SystemExit("P10 pilot needs R2 item5 λ = 0 ID bal (seeds 42–44)")
    return float(np.mean(vals))


def pilot_level_passes(summary, probe0=None, id0=None):
    """Probe drop ≥ 0.10 and ID bal within 0.05 of λ = 0 (pilot decisions)."""
    probe0 = lambda0_probe_mean() if probe0 is None else probe0
    id0 = lambda0_id_bal_mean() if id0 is None else id0
    probe = summary["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"]
    id_bal = float(summary["id_test"].get("balanced_acc", summary["id_test"]["acc"]))
    drop = float(probe0 - probe)
    return drop >= GATE_DROP and abs(id_bal - id0) <= ID_BAL_TOL, {
        "probe_drop": drop, "id_bal": id_bal, "id0": id0, "id_delta": float(id_bal - id0),
    }
_L1 = {"head_specs": [[1, 1.0], [2, 1.0], [3, 1.0]], "adv_inner_steps": 2, "reinit_every_steps": 500}
_L2 = dict(_L1, grl_schedule="ganin", adv_warmup_epochs=1)
_L3 = dict(_L2, head_specs=[[1, 1.0], [1, 1.0], [2, 1.0], [2, 2.0], [3, 1.0]], adv_inner_steps=5, adv_lr_multiplier=10.0)
# Camelyon adversary ladder pilot (decisions/paperB_r5_p10_ladder_pilot.md).
# L1–L4: full current P10 toolbox (_L3), vary only λ and adv_lr_multiplier.
# L5: stronger than toolbox — λ=20, deepest heads, re-init every 100 steps.
_P_DEEP = [[1, 1.0], [1, 1.0], [2, 2.0], [3, 1.0], [3, 2.0]]
RUNGS = {"L1_bank": (_L1, 1.0), "L2_ganin": (_L2, 1.0), "L3_strong": (_L3, 1.0), "L4_lam3": (_L3, 3.0),
         "L5_lam10": (_L3, 10.0),
         # 2026-10-17 label-shift arms (decisions amendment): L3 toolbox @ λ=2
         "natural_l2": (_L3, 2.0), "induced_l2": (_L3, 2.0),
         # pilot ladder (weak → strong), natural arm seed 42
         "P_L1": (dict(_L3, adv_lr_multiplier=30.0), 5.0),
         "P_L2": (dict(_L3, adv_lr_multiplier=100.0), 5.0),
         "P_L3": (dict(_L3, adv_lr_multiplier=30.0), 10.0),
         "P_L4": (dict(_L3, adv_lr_multiplier=100.0), 10.0),
         "P_L5": (dict(_L3, adv_lr_multiplier=100.0, head_specs=_P_DEEP, reinit_every_steps=100), 20.0)}
PILOT_ORDER = ("P_L1", "P_L2", "P_L3", "P_L4", "P_L5")
# Multi-seed arms at the pilot-selected level (same toolbox/λ; natural vs induced label shift).
for _pl in PILOT_ORDER:
    RUNGS["natural_{}".format(_pl)] = RUNGS[_pl]
    RUNGS["induced_{}".format(_pl)] = RUNGS[_pl]
ORDER = tuple(RUNGS)
SHIFT_SEEDS = (42, 52, 62, 72, 82)
INDUCED_TUMOUR_A, INDUCED_TUMOUR_B = 0.80, 0.20
ID_BAL_TOL = 0.05  # pilot gate: ID bal within this of λ=0 mean


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


class _PathImageDataset(Dataset):
    """RGB paths (NCT-CRC-HE patches) with dummy y/h for p12.collect."""

    def __init__(self, paths, tf):
        from PIL import Image
        self.paths = list(paths)
        self.tf = tf
        self._Image = Image

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = self._Image.open(self.paths[i]).convert("RGB")
        x = self.tf(im)
        # y unused for AUROC ID-vs-OOD; h unused
        return x, 0, 0


def nct_path_list():
    if not NCT_PATHS.exists():
        raise FileNotFoundError("NCT eval manifest missing: {}".format(NCT_PATHS))
    paths = [ln.strip() for ln in NCT_PATHS.read_text().splitlines() if ln.strip()]
    missing = [p for p in paths[:20] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("NCT paths missing on disk (e.g. {})".format(missing[0]))
    return paths


def collect_nct(ev, device, num_workers=4, batch=256):
    """Features/logits on the fixed NCT-CRC-HE eval path list (semantic OOD)."""
    paths = nct_path_list()
    ds = _PathImageDataset(paths, p12.eval_tf())
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    return p12.collect(ev, loader, device)


def score_nct_into_ood(ood, packs, ev, device, seed, num_workers=4):
    """Bug-fix helper: attach NCT detectors (A-test vs NCT) into ood dict."""
    nct = collect_nct(ev, device, num_workers=num_workers)
    tr, idp = packs["a_train"], packs["a_test"]
    block = p12.detector_block(tr["z"], tr["y"], idp["z"], idp["logits"], nct["z"], nct["logits"],
                               p12.KNN_BANK, seed)
    ood[NCT_KEY] = {"n": int(len(nct["y"])), "detectors": block,
                    "note": "semantic OOD NCT-CRC-HE-100K; AUROC ID(a_test) vs OOD; no label map"}
    return ood


def gate_met(s, probe0=None):
    """Held-out hospital probe drop ≥ GATE_DROP vs λ = 0 mean; else invariance not reached."""
    probe0 = lambda0_probe_mean() if probe0 is None else probe0
    probe = s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"]
    return float(probe0 - probe) >= GATE_DROP


FAST_PROBE_N = 2000  # per-epoch trajectory; full gate still uses PROBE_MAX at end


def score_site_and_id(model, ds, sp, device, seed, num_workers, probe0, score_nct=True):
    """Collect packs; return id_test, site probe, drop vs λ0, optional NCT ood (no gate labels)."""
    model.eval()
    ev = _EvalWrap(model)
    packs = {k: p12.collect(ev, p12.loader_for(ds, sp[k], p12.eval_tf(), 256, num_workers, False), device)
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
                                             p12.KNN_BANK, seed)
    if score_nct:
        score_nct_into_ood(ood, packs, ev, device, seed, num_workers=num_workers)
    id_test = p12.acc_pack(packs["a_test"]["logits"], packs["a_test"]["y"])
    probe_blk = p12.leakage_binary(sub(packs["a_test"]["z"]), sub(packs["b_hold"]["z"]))
    probe = float(probe_blk["balanced_acc_mean"])
    return {
        "id_test": id_test,
        "id_test_slide_disjoint": p12.acc_pack(packs["a_clean"]["logits"], packs["a_clean"]["y"]),
        "leakage_site_probe_a_vs_bhold": probe_blk,
        "leakage_site_probe_aclean_vs_bhold": p12.leakage_binary(sub(packs["a_clean"]["z"]),
                                                                sub(packs["b_hold"]["z"])),
        "ood": ood,
        "b_heldout_accuracy": ood["b_hold"]["accuracy"],
        "site_probe": probe,
        "probe_drop_vs_lambda0": float(probe0 - probe),
        "id_bal": float(id_test.get("balanced_acc", id_test["acc"])),
    }


def fast_epoch_site_id(model, ds, sp, device, num_workers, probe0, id0, n_probe=FAST_PROBE_N):
    """Cheap per-epoch site probe + ID bal (subsampled). Descriptive; not locked."""
    model.eval()
    ev = _EvalWrap(model)
    rng = np.random.default_rng(0)
    # Subsample indices before collect to keep the epoch loop affordable.
    def take(key, n=4000):
        idx = np.asarray(sp[key])
        if len(idx) > n:
            idx = idx[rng.choice(len(idx), n, replace=False)]
        return idx

    packs = {
        k: p12.collect(ev, p12.loader_for(ds, take(k), p12.eval_tf(), 256, num_workers, False), device)
        for k in ("a_test", "b_hold")
    }

    def sub(z, n=n_probe):
        return z[rng.choice(len(z), min(n, len(z)), replace=False)]

    id_test = p12.acc_pack(packs["a_test"]["logits"], packs["a_test"]["y"])
    id_bal = float(id_test.get("balanced_acc", id_test["acc"]))
    probe = float(p12.leakage_binary(sub(packs["a_test"]["z"]), sub(packs["b_hold"]["z"]))["balanced_acc_mean"])
    drop = float(probe0 - probe)
    return {
        "site_probe_fast": probe,
        "probe_drop_vs_lambda0_fast": drop,
        "id_bal_fast": id_bal,
        "id_acc_fast": float(id_test["acc"]),
        "fast_probe_n": int(n_probe),
        "fast_would_pass_p10_gate": bool(drop >= GATE_DROP),
        "fast_would_pass_pilot": bool(drop >= GATE_DROP and abs(id_bal - id0) <= ID_BAL_TOL),
    }


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
    ap.add_argument("--pilot", action="store_true",
                    help="Camelyon adversary ladder pilot (decisions/paperB_r5_p10_ladder_pilot.md)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if summary.json exists (for posthoc last.pt scoring)")
    ap.add_argument("--epoch_traj", action="store_true",
                    help="each epoch: fast site probe + ID bal into history (descriptive)")
    r5_recipe.add_args(ap)
    args = ap.parse_args()
    rec, rsrc = r5_recipe.resolve(args.recipe)
    args.cls_lr_multiplier = float(rec["cls_lr_multiplier"])
    probe0 = lambda0_probe_mean()
    id0 = lambda0_id_bal_mean()
    used, hits = p10_state(r5_recipe.tag(args.recipe))
    if args.stage2:
        if not hits:
            print("P10 stage 2: no ladder rung met the gate; nothing to run")
            return
        rung = hits[0]["rung"]
    else:
        if args.rung is None:
            raise SystemExit("--rung required for the ladder / shift arm / pilot")
        rung = args.rung
    if args.pilot and rung not in PILOT_ORDER:
        raise SystemExit("pilot rung must be one of {}".format(PILOT_ORDER))
    # Pilot follow-up multi-seed → shift_arms_pilot/ (outside the 40 GPU-h ladder cap).
    if args.pilot:
        stage = "pilot"
    elif args.shift_arm:
        stage = "shift_arms_pilot" if "_P_L" in rung else "shift_arms"
    elif args.stage2:
        stage = "stage2"
    else:
        stage = "ladder"
    tb_over, lam = RUNGS[rung]
    tb = dict(M.TOOLBOX_DEFAULT)
    tb.update(tb_over)
    tag = "camelyon2_{}_ladv{}_s{}".format(rung, p12.lam_tag(lam), args.seed) + r5_recipe.tag(args.recipe)
    run_dir, ckpt_dir = OUT / stage / tag, CKPT / stage / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_done and not args.force and (run_dir / "summary.json").exists():
        print("skip_done", tag)
        return
    # Pilot: stop if an earlier level already passed, or GPU-h used by pilot ≥ 10.
    if args.pilot:
        pilot_used = 0.0
        selected = None
        for pr in PILOT_ORDER:
            for f in sorted((OUT / "pilot").glob("camelyon2_{}_*/summary.json".format(pr))):
                s = json.loads(f.read_text())
                pilot_used += float(s.get("gpu_hours") or 0.0)
                ok, _ = pilot_level_passes(s, probe0, id0)
                if ok and selected is None:
                    selected = s
        if selected is not None:
            why = {"tag": tag, "skipped": True, "reason": "earlier_pilot_level_passed",
                   "passed_tag": selected["tag"], "pilot_gpu_h_used": pilot_used, "cap": 10.0}
            (run_dir / "skipped.json").write_text(json.dumps(why, indent=2) + "\n")
            print("P10 pilot stop", why, flush=True)
            return
        if pilot_used + EST_GPU_H > 10.0:
            why = {"tag": tag, "skipped": True, "reason": "pilot_cap", "pilot_gpu_h_used": pilot_used, "cap": 10.0}
            (run_dir / "skipped.json").write_text(json.dumps(why, indent=2) + "\n")
            print("P10 pilot over cap", why, flush=True)
            return
    if not args.shift_arm and not args.stage2 and not args.pilot and (hits or used + EST_GPU_H > CAP_GPU_H):
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
    # Cap applies only to the old L3@λ=2 shift_arms wave; pilot multi-seed is outside the cap.
    if args.shift_arm and stage == "shift_arms" and used + EST_GPU_H > CAP_GPU_H:
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
    y_all = np.asarray(ds.y_array[train_idx], int).reshape(-1)
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
            n = int(tb.get("reinit_every_steps") or 0)
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
        # Pilot L5: re-init all heads every N epochs (decisions/paperB_r5_p10_ladder_pilot.md).
        n_ep = int(tb.get("reinit_every_epochs") or 0)
        if n_ep > 0 and epoch % n_ep == 0:
            for k in range(len(model.adv.heads)):
                model.adv.reinit(k)
                for p in model.adv.heads[k].parameters():
                    opt.state.pop(p, None)
                    inner.state.pop(p, None)
                reinits += 1
        sel = p12.eval_acc(_EvalWrap(model), sel_loader, device)
        rec = {"epoch": epoch, "loss_adv": float(np.mean(ce)), "acc_adv": float(np.mean(ac)),
               "inner_ce": float(np.mean(inn)) if inn else None, "id_select_acc": sel, "coef_end": coef,
               "n_reinit": reinits, "warmup": warm}
        if args.epoch_traj:
            traj = fast_epoch_site_id(model, ds, sp, device, args.num_workers, probe0, id0)
            rec.update(traj)
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("epoch", rec, flush=True)
        if sel > best:
            best = sel
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "sel": sel}, ckpt_dir / "best.pt")

    # Last-epoch weights (derm R1/R2 use last.ckpt). Official Camelyon gate still uses best/ID-select.
    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs, "sel": sel}, ckpt_dir / "last.pt")
    print("scoring posthoc last.pt (descriptive; does not enter gate)", flush=True)
    last_metrics = score_site_and_id(model, ds, sp, device, args.seed, args.num_workers, probe0, score_nct=False)
    posthoc_last = {
        "note": "post hoc; not locked; derm R1/R2 analogue (last epoch). Gate/best unchanged.",
        "epoch": int(args.epochs),
        "site_probe": last_metrics["site_probe"],
        "probe_drop_vs_lambda0": last_metrics["probe_drop_vs_lambda0"],
        "id_bal": last_metrics["id_bal"],
        "id_test": last_metrics["id_test"],
        "b_heldout_accuracy": last_metrics["b_heldout_accuracy"],
        "leakage_site_probe_a_vs_bhold": last_metrics["leakage_site_probe_a_vs_bhold"],
    }
    print("posthoc_last", {k: posthoc_last[k] for k in ("site_probe", "probe_drop_vs_lambda0", "id_bal")},
          flush=True)

    blob = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    best_metrics = score_site_and_id(model, ds, sp, device, args.seed, args.num_workers, probe0, score_nct=True)
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    pass_pilot = [int(r["epoch"]) for r in rows if r.get("fast_would_pass_pilot")]
    pass_p10 = [int(r["epoch"]) for r in rows if r.get("fast_would_pass_p10_gate")]
    epoch_traj = {
        "note": "per-epoch fast site probe (n={}) + ID bal; descriptive; does not change best/gate".format(
            FAST_PROBE_N),
        "enabled": bool(args.epoch_traj),
        "any_fast_would_pass_pilot": bool(pass_pilot),
        "any_fast_would_pass_p10_gate": bool(pass_p10),
        "epochs_fast_pass_pilot": pass_pilot,
        "epochs_fast_pass_p10_gate": pass_p10,
    }
    summary = {"tag": tag, "stage": stage, "rung": rung, "assign": "primary", "hospitals": [ha, hb, hc],
               "lambda_adv": lam, "seed": args.seed, "toolbox": tb, "best_epoch": int(blob["epoch"]),
               "id_test": best_metrics["id_test"],
               "id_test_slide_disjoint": best_metrics["id_test_slide_disjoint"],
               "leakage_site_probe_a_vs_bhold": best_metrics["leakage_site_probe_a_vs_bhold"],
               "leakage_site_probe_aclean_vs_bhold": best_metrics["leakage_site_probe_aclean_vs_bhold"],
               "ood": best_metrics["ood"], "history": rows,
               "adversary": {"min_epoch_ce": min(r["loss_adv"] for r in rows), "last": rows[-1]},
               "gpu_hours": (time.time() - t0) / 3600.0,
               "posthoc_last": posthoc_last,
               "epoch_traj": epoch_traj}
    probe = summary["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"]
    summary["lambda0_probe_mean"] = probe0
    summary["probe_drop_vs_lambda0"] = float(probe0 - probe)
    summary["gate_met"] = bool(gate_met(summary, probe0))
    summary["invariance_not_reached"] = not summary["gate_met"]
    summary["b_heldout_accuracy"] = best_metrics["b_heldout_accuracy"]
    if args.pilot:
        ok, info = pilot_level_passes(summary, probe0, id0)
        summary["pilot_level_passed"] = bool(ok)
        summary["pilot_gate"] = info
        if ok:
            (OUT / "pilot" / "SELECTED.json").write_text(json.dumps({
                "tag": tag, "rung": rung, "lambda_adv": lam, "seed": args.seed,
                "toolbox": tb, **info,
            }, indent=2) + "\n")
            print("P10 pilot SELECTED", tag, info, flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "probe", probe, "drop", summary["probe_drop_vs_lambda0"],
          "ID acc", summary["id_test"]["acc"], "gate", summary["gate_met"],
          "invariance_not_reached", summary["invariance_not_reached"],
          "posthoc_last_drop", posthoc_last["probe_drop_vs_lambda0"],
          "posthoc_last_id_bal", posthoc_last["id_bal"], flush=True)


if __name__ == "__main__":
    main()
