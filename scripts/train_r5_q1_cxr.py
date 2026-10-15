#!/usr/bin/env python3
"""Paper B R5 Q1: the dual-encoder CSG on a second task, chest X-ray normal vs pneumonia.

Task      NIH ChestX-ray14 (domain 0) is the labelled domain; Kermany pediatric CXR (domain 1) is the adversary's
          target domain (the PAD-UFES role). Kermany labels never enter a loss; they are used for X-dom accuracy,
          class mix and the decomposition only. Label definitions and splits: scripts/r5_q1_cxr_data.py.
Model     CSGR5Lightning, EfficientNet-B3 ×2 (the main-task backbone), lesion_classes = 2, domain_classes = 2,
          same heads / latents / GRL schedule / optimiser groups as the main task (recipe via --recipe).
          Inputs are grey PNGs loaded as 3-channel RGB; the lesion branch's internal grey conversion is a no-op.
Adapted   class-weighted CE (NIH train is 2.3 % pneumonia; macro val/acc selection), CXR augmentation (random
          resized crop 0.7-1, rotation ±10°, brightness/contrast 0.3, random erasing; no flips, no hue/saturation),
          epochs set so the optimiser step count matches the main task (40 × 506 steps -> 16 × 1,235).
Modes     paired (default): batch = 32 NIH (classifier + adversary rows) + 32 kerm_adv (adversary rows).
          dose (--adv_mix_shift f, conditional R1 repeat): batch = 32 NIH classifier rows (natural mix, no adversary)
          + 32 NIH adversary rows resampled so the NIH-vs-Kermany pneumonia gap keeps fraction f
          (f = 1: 2.3 %, f = 0.5: 37.6 %, f = 0: 73.0 % = Kermany) + 32 kerm_adv rows.
Evaluation NIH test (ID); Shenzhen (primary OOD; TB / normal subsets descriptive); kerm_heldout (secondary OOD,
          held-out patients of the adversary's domain; per-class + decomposition); leakage probe NIH test vs Kermany
          full (Table 1 analogue), NIH test vs kerm_heldout, NIH val vs kerm_adv (selection side).

Images are read from $R5_CXR_ROOT (node-local copy made by r5_q1_cxr_data.py --stage_to at job start).
Outputs: results/paperB/r5/q1/<stage>/<tag>/{config.json, adversary_log.jsonl, summary.json, scores.npz, features/}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_recipe  # noqa: E402

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

import r5_models as M  # noqa: E402
import r5_q1_cxr_data as Q  # noqa: E402
from phase15_common import adversary_gate, ece_from_logits, leakage_probe  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAIN_STEPS_PER_EPOCH = 16211 // 32
CLASSES = Q.CLASSES


def train_tf():
    return transforms.Compose([
        transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random")])


def eval_tf():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class CXR(Dataset):
    def __init__(self, frame, root, tf):
        self.paths = [str(Path(root) / p) for p in frame.path]
        self.y = frame.label_idx.astype(int).to_numpy()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def load(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))

    def __getitem__(self, i):
        return self.load(i), torch.tensor(int(self.y[i]), dtype=torch.long)


class PairedCXR(Dataset):
    """Two-stream (default): index i -> NIH row i (classifier + adversary, domain 0), kerm_adv row i mod n (domain 1).
    Three-stream (adv_rows or cls_rows given): NIH classifier row cls_rows[i] (or i), a separate NIH adversary row
    adv_rows[i] (or i), and the kerm_adv row. Dose mode resamples adv_rows; class_balance = sampler resamples
    cls_rows (class-balanced) and keeps the adversary's NIH stream at the natural mix."""

    def __init__(self, nih, kerm, adv_rows=None, cls_rows=None):
        self.nih, self.kerm, self.adv_rows, self.cls_rows = nih, kerm, adv_rows, cls_rows
        self.three = adv_rows is not None or cls_rows is not None

    def __len__(self):
        return len(self.nih)

    def __getitem__(self, i):
        c = int(self.cls_rows[i]) if self.cls_rows is not None else i
        k = i % len(self.kerm)
        out = [(self.nih.load(c), int(self.nih.y[c]), 0), (self.kerm.load(k), int(self.kerm.y[k]), 1)]
        if self.three:
            j = int(self.adv_rows[i]) if self.adv_rows is not None else i
            out.append((self.nih.load(j), int(self.nih.y[j]), 0))
        return out


def make_collate(dose):
    def collate(batch):
        nih = [b[0] for b in batch]
        ker = [b[1] for b in batch]
        streams = [nih] + ([[b[2] for b in batch]] if dose else []) + [ker]
        x = torch.stack([s[0] for st in streams for s in st])
        y_true = torch.tensor([s[1] for st in streams for s in st], dtype=torch.long)
        dom = torch.tensor([s[2] for st in streams for s in st], dtype=torch.long)
        n = len(batch)
        y_cls = torch.full_like(y_true, -1)
        y_cls[:n] = y_true[:n]
        out = (x, x, y_cls, y_true, dom)
        if dose:
            m = torch.ones_like(dom, dtype=torch.bool)
            m[:n] = False
            out = out + (m,)
        return out
    return collate


def dose_rows(y, f, seed):
    """NIH adversary-stream indices (with replacement) with pneumonia prevalence p(f) = f·p_NIH + (1-f)·p_Kermany.
    f = 1 returns a plain permutation (the 3-stream control)."""
    rng = np.random.default_rng(seed + 7919)
    n = len(y)
    if f >= 1.0:
        return rng.permutation(n), float(np.mean(y))
    p_n, p_k = float(np.mean(y)), KERM_PNEU_FRAC
    p = f * p_n + (1.0 - f) * p_k
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(p * n))
    rows = np.r_[rng.choice(pos, n_pos, replace=True), rng.choice(neg, n - n_pos, replace=True)]
    return rng.permutation(rows), p


KERM_PNEU_FRAC = 4273 / 5856


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="gate", help="gate | sweep | dose (output subfolder)")
    ap.add_argument("--lambda_adv", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adv_mix_shift", type=float, default=-1.0,
                    help="dose mode: fraction of the NIH-vs-Kermany class-mix gap kept in the adversary's NIH stream")
    ap.add_argument("--cls_balance", default="auto", choices=("auto", "weighted_ce", "none", "sampler"),
                    help="auto = weighted_ce unless the recipe sets class_balance != none")
    ap.add_argument("--max_epochs", type=int, default=0, help="0 = step-matched to the recipe's main-task epochs")
    ap.add_argument("--root", default=os.environ.get("R5_CXR_ROOT", ""))
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_done", action="store_true")
    r5_recipe.add_args(ap)
    return ap.parse_args()


@torch.no_grad()
def collect(net, ds, device, workers):
    def fwd(x):
        out = net(x, x_lesion=None, return_latents=True)
        return out[0], out[3]

    return R.collect(fwd, DataLoader(ds, batch_size=128, shuffle=False, num_workers=workers, pin_memory=True), device)


def eval_cxr(net, S, root, device, out_dir, workers):
    tf = eval_tf()
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    P = {}
    for n in Q.SETS:
        P[n] = collect(net, CXR(S[n], root, tf), device, workers)
        np.savez_compressed(out_dir / "features" / "{}.npz".format(n), **P[n])
        print("features", n, P[n]["z"].shape, flush=True)
    tr, te = P["nih_train"], P["nih_test"]
    det = R.Detectors(tr["z"], tr["labels"], 2)
    sc_id, _ = det.scores(te["z"], te["logits"])
    y_id = te["labels"].astype(int)

    def cls_block(p):
        y = p["labels"].astype(int)
        prob = torch.softmax(torch.from_numpy(p["logits"]).float(), 1).numpy()[:, 1]
        e, c = ece_from_logits(p["logits"], y)
        return {"acc": float(accuracy_score(y, prob > 0.5)), "balanced_acc": float(balanced_accuracy_score(y, prob > 0.5)),
                "auroc_pneumonia": float(roc_auc_score(y, prob)), "ece": e, "mean_confidence": c}

    def probe(a, b):
        return leakage_probe(np.concatenate([a["z"], b["z"]]), np.r_[np.zeros(len(a["z"])), np.ones(len(b["z"]))])

    summ = {"id": cls_block(te), "xdom_kerm_heldout": cls_block(P["kerm_heldout"]),
            "leakage": {"test_vs_kerm_full": probe(te, P["kerm_full"]), "test_vs_kerm_heldout": probe(te, P["kerm_heldout"]),
                        "val_vs_kerm_adv_selection": probe(P["nih_val"], P["kerm_adv"]),
                        "primary": "test_vs_kerm_full (Table 1 analogue)"},
            "id_class_mix": {c: int((y_id == k).sum()) for k, c in enumerate(CLASSES)}, "ood": {}}
    scores = {"id__{}".format(k): v for k, v in sc_id.items()}
    blk, sc = R.ood_block(det, sc_id, y_id, P["shenzhen"], None, CLASSES, 0)
    tb = S["shenzhen"].tb.to_numpy()
    for name, m in (("tb", tb == 1), ("normal", tb == 0)):
        blk["subset_" + name] = {"n": int(m.sum()), **{d: R.auroc(sc_id[d], sc[d][m]) for d in R.DETS}}
    summ["ood"]["shenzhen"] = blk
    scores.update({"shenzhen__{}".format(k): v for k, v in sc.items()})
    for name in ("kerm_heldout", "kerm_full"):
        blk, sc = R.ood_block(det, sc_id, y_id, P[name], P[name]["labels"].astype(int), CLASSES, 0)
        summ["ood"][name] = blk
        scores.update({"{}__{}".format(name, k): v for k, v in sc.items()})
    for name in ("shenzhen", "kerm_heldout"):
        b = summ["ood"][name]
        print(name, "maha {:.3f} knn {:.3f} cos {:.3f}".format(b["mahalanobis_classcond"], b["knn_k50"], b["cosine_max"]),
              flush=True)
    np.savez_compressed(out_dir / "scores.npz", **scores)
    return summ


def main():
    a = parse()
    rec, src = r5_recipe.resolve(a.recipe)
    S = Q.load_index()
    dose = a.adv_mix_shift >= 0.0
    bal = a.cls_balance if a.cls_balance != "auto" else (
        rec["class_balance"] if rec["class_balance"] != "none" else "weighted_ce")
    bs = int(rec["batch_size"])
    spe = len(S["nih_train"]) // bs
    epochs = a.max_epochs or max(1, int(round(int(rec["max_epochs"]) * MAIN_STEPS_PER_EPOCH / spe)))
    tag = "cxr_effb3_ladv{}_s{}".format(R.lam_tag(a.lambda_adv), a.seed)
    if dose:
        tag += "_shift{}".format(R.lam_tag(a.adv_mix_shift))
    if a.cls_balance != "auto":
        tag += "_" + a.cls_balance
    if a.max_epochs:
        tag += "_ep{}".format(a.max_epochs)
    tag += r5_recipe.tag(a.recipe)
    run_dir = R.R5 / "q1" / a.stage / tag
    ckpt_dir = R.PAPERB / "checkpoints" / "r5_q1" / a.stage / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if a.skip_done and (run_dir / "summary.json").exists() and not a.dry_run:
        print("skip_done", tag)
        return
    if not a.root and not a.dry_run:
        raise SystemExit("set --root or R5_CXR_ROOT to the node-local staged copy")
    seed_everything(a.seed)
    pl.seed_everything(a.seed, workers=True)
    y_tr = S["nih_train"].label_idx.to_numpy()
    cw = R.class_weights(y_tr, 2) if bal == "weighted_ce" else None
    adv_rows, adv_p = (dose_rows(y_tr, a.adv_mix_shift, a.seed) if dose else (None, None))
    cls_rows = None
    if bal == "sampler":
        w = 1.0 / np.bincount(y_tr)[y_tr]
        cls_rows = np.random.default_rng(a.seed + 104729).choice(len(y_tr), len(y_tr), replace=True, p=w / w.sum())
    kw = dict(backbone="effb3", lesion_classes=2, domain_classes=2,
              lesion_latent_dim=int(rec["lesion_latent_dim"]), context_latent_dim=int(rec["context_latent_dim"]),
              learning_rate=float(rec["lr"]), weight_decay=float(rec["weight_decay"]),
              lambda_ctx=float(rec["lambda_ctx"]), lambda_adv=float(a.lambda_adv), lambda_orth=float(rec["lambda_orth"]),
              adv_lr_multiplier=float(rec["adv_lr_multiplier"]), lesion_cls_lr_multiplier=float(rec["cls_lr_multiplier"]),
              cls_weight=cw, backbone_lr_mult=1.0, penalty="none", penalty_weight=0.0)
    cfg = {"tag": tag, "stage": a.stage, "task": "CXR normal vs pneumonia; NIH (domain 0) -> Kermany (domain 1)",
           "mode": "dose" if dose else "paired",
           "streams": "three" if (dose or bal == "sampler") else "two", "adv_mix_shift": a.adv_mix_shift if dose else None,
           "adv_stream_pneumonia_frac": adv_p, "class_balance": bal, "max_epochs": epochs, "steps_per_epoch": spe,
           "seed": a.seed, "model_kwargs": kw, "index": str(Q.OUT),
           "class_mix": json.loads((Q.OUT / "CLASS_MIX.json").read_text()) if (Q.OUT / "CLASS_MIX.json").exists() else None,
           **r5_recipe.describe(rec, src)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if a.dry_run:
        print("dry_run", tag, "epochs", epochs, "steps/epoch", spe, "class_balance", bal, "adv_p", adv_p)
        return

    t0 = time.time()
    nih_tr = CXR(S["nih_train"], a.root, train_tf())
    ker = CXR(S["kerm_adv"], a.root, train_tf())
    ds = PairedCXR(nih_tr, ker, adv_rows, cls_rows)
    tl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=a.num_workers, pin_memory=True,
                    persistent_workers=a.num_workers > 0, drop_last=True, collate_fn=make_collate(ds.three))
    vl = DataLoader(CXR(S["nih_val"], a.root, eval_tf()), batch_size=64, shuffle=False, num_workers=a.num_workers,
                    pin_memory=True, persistent_workers=a.num_workers > 0)
    adv_log = run_dir / "adversary_log.jsonl"
    adv_log.write_text("")
    lit = M.CSGR5Lightning(adv_log_path=str(adv_log), **kw)
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc",
                                           mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                         callbacks=[ckpt_cb], precision="16-mixed" if torch.cuda.is_available() else "32-true",
                         default_root_dir=str(run_dir), log_every_n_steps=50)
    trainer.fit(lit, train_dataloaders=tl, val_dataloaders=vl)
    best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else ckpt_dir / "last.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = M.load_dual_net(best, kw, device)
    summ = eval_cxr(net, S, a.root, device, run_dir, a.num_workers)
    rows = [json.loads(l) for l in adv_log.read_text().splitlines() if l.strip()]
    summ.update({"tag": tag, "stage": a.stage, "seed": a.seed, "lambda_adv": a.lambda_adv,
                 "adv_mix_shift": a.adv_mix_shift if dose else None, "adv_stream_pneumonia_frac": adv_p,
                 "max_epochs": epochs, "best_checkpoint": str(best), "recipe": a.recipe,
                 "gate": adversary_gate(rows, M.LN2, a.lambda_adv,
                                        leakage_bal=summ["leakage"]["test_vs_kerm_full"]["bal_acc_mean"], floor=0.5),
                 "gpu_hours": (time.time() - t0) / 3600.0, "commit": R.git_head()})
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "leak", summ["leakage"]["test_vs_kerm_full"]["bal_acc_mean"],
          "maha shenzhen", summ["ood"]["shenzhen"]["mahalanobis_classcond"], flush=True)


if __name__ == "__main__":
    main()
