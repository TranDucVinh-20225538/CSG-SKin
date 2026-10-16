#!/usr/bin/env python3
"""Shared training / evaluation path for Paper B R5 P8, R5 P7(b), R6 R1 and R6 R2.

Wraps the lesion-level Table 1 path (train_r2_item4_lesion_split.py -> train_phase3_sweep.py) without changing it:
same CSGLite model (subclassed), lesion-level ISIC split, Phase 2 PAD patient split, optimiser groups, adversary LR x30,
GRL ramp, val/acc (macro recall) checkpointing, eval transform, detectors fit on ISIC train only.

New behaviour lives only here, behind explicit arguments:
  recipe        latent size, classifier LR multiplier, epochs, class balancing (none / loss / sampling)
  stream mode   "two"   = legacy CombinedTrainDataset (ISIC rows feed classifier and adversary; current setting)
                "three" = ISIC classifier stream + separate ISIC adversary stream + PAD adversary stream
                "p7b"   = ISIC classifier stream + BCN adversary stream + HAM adversary stream (no PAD anywhere)
  dose_mix      class mix of the ISIC adversary stream (R6 R1)
  align         dann / cdan / iwdan / iwcdan / iwdan_oracle (R6 R2)

PAD labels never enter a loss: PAD streams are built from the path column only, the legacy collate's y_supcon tensor
(which carries PAD labels) is discarded, and the supervised-contrastive term is not computed.
Decisions: decisions/paperB_r5_p8.md, paperB_r6_r1.md, paperB_r6_r2.md, paperB_r5_p7b.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
import train_phase3_sweep as p3  # noqa: E402  (installs the transformers stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import extract_phase13_features as x13  # noqa: E402
import r2_common as C  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402
from analyze_phase13 import conf_pack  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from eval_ood_dual_branch import cosine_ood_scores, fit_class_means, fit_knn, knn_scores  # noqa: E402
from phase15_common import derm_domain_head  # noqa: E402
from phase16_bcn_ham_data import site_from_lesion  # noqa: E402
from src.datasets.skin_dataset import (  # noqa: E402
    CombinedTrainDataset,
    SkinDataset,
    build_lesion_branch_transform_gray,
    build_train_transform_robust,
    build_val_transform_robust,
    csg_lite_paired_collate,
)
from src.models.csg_lightning import CSGLiteLightning  # noqa: E402
from src.utils import ood_metrics  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

PAPERB = C.PAPERB
RES = PAPERB / "results" / "paperB"
CKPT = PAPERB / "checkpoints"
P8_DIR = RES / "r5" / "p8"
P7A_SETS = RES / "r5" / "p7a" / "sets"
P7A_NAMES = ("mednode", "sd198", "scin")
ITEM4_FEAT = C.R2 / "item4" / "features"
PREREG = RES / "PREREGISTER_BOOTSTRAP.json"
SEEDS = (42, 52, 62, 72, 82)
NCLS = 8
LABELS = C.LABELS
NV, AK = 1, 3
PAD_CLASSES = C.PAD_CLASSES
FEAT_KEYS = ("z_lesion", "z_lesion_norm", "z_context", "logits", "labels")
COMPETENCE_TARGET = 0.70
ERM_SINGLE_ENCODER_IMAGE_LEVEL = 0.779
# R1 degraded-run rule (decisions/paperB_r6_r1.md, 2026-10-16 10:05): chance = 1/8; degraded < 2× chance.
CHANCE_ID_BAL = 1.0 / NCLS
DEGRADED_ID_BAL = 2.0 * CHANCE_ID_BAL  # 0.25
ID_DROP_FLAG = 0.10  # flag (keep in verdict) if ID bal is > this below matched λ = 0
REPLACEMENT_SEEDS = (92, 102)  # only if > 1 of 5 seeds degrade in a cell
BOOT_DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")
DETS = BOOT_DETS + ("MSP", "Energy_T1")
DET_LABEL = {"mahalanobis_classcond": "Mahalanobis", "knn_k50": "kNN k=50", "cosine_max": "Cosine", "MSP": "MSP",
             "Energy_T1": "Energy"}

# ----------------------------------------------------------------------------------------------------------------------
# Recipes
# ----------------------------------------------------------------------------------------------------------------------
CURRENT = {"name": "current", "latent": 16, "cls_lr_mult": 0.2, "max_epochs": 40, "balance": "none"}
# P8 stage A grid (λ = 0, 40 epochs). A0 is the current recipe. Fixed in decisions/paperB_r5_p8.md.
P8_GRID = {
    "A0": {"latent": 16, "cls_lr_mult": 0.2, "balance": "none"},
    "A1": {"latent": 64, "cls_lr_mult": 1.0, "balance": "none"},
    "A2": {"latent": 128, "cls_lr_mult": 1.0, "balance": "none"},
    "A3": {"latent": 64, "cls_lr_mult": 5.0, "balance": "none"},
    "A4": {"latent": 64, "cls_lr_mult": 1.0, "balance": "loss"},
    "A5": {"latent": 64, "cls_lr_mult": 1.0, "balance": "sampling"},
    "A6": {"latent": 16, "cls_lr_mult": 1.0, "balance": "none"},
    "A7": {"latent": 128, "cls_lr_mult": 1.0, "balance": "loss"},
}


def grid_recipe(cid, epochs=40):
    r = dict(P8_GRID[cid])
    r.update(name="{}_e{}".format(cid, int(epochs)), max_epochs=int(epochs), grid_id=cid)
    return r


def resolve_recipe(name):
    """current | p8 | auto | A<k>_e<epochs>. 'auto' = main-recipe rule outcome (MAIN_RECIPE_RULE.json)."""
    if name == "current":
        return dict(CURRENT)
    if name == "p8":
        p = P8_DIR / "selection_stageB.json"
        if not p.exists():
            raise SystemExit("P8 recipe not selected yet ({} missing)".format(p))
        r = dict(json.loads(p.read_text())["recipe"])
        r["name"] = "p8"
        return r
    if name == "auto":
        p = P8_DIR / "MAIN_RECIPE_RULE.json"
        if not p.exists():
            raise SystemExit("main-recipe rule outcome missing ({}); run scripts/r5_main_recipe_rule.py first".format(p))
        return resolve_recipe(json.loads(p.read_text())["main_recipe"])
    m = re.fullmatch(r"(A\d)_e(\d+)", name)
    if m and m.group(1) in P8_GRID:
        return grid_recipe(m.group(1), int(m.group(2)))
    raise SystemExit("unknown recipe {}".format(name))


def recipe_tag(r):
    return "cur" if r["name"] == "current" else r["name"]


def lambda0_ckpt_dir(r, seed):
    """λ = 0 checkpoint of the same recipe and seed (user decision 2026-10-15: reuse, evaluation only).
    current: the Table 1 lesion-level runs (R2 item 4); P8 recipe: the P8 main λ = 0 runs."""
    if r["name"] == "current":
        return CKPT / "r2_item4" / "runB_orth1_ladv0_s{}".format(seed)
    s = P8_DIR / "main" / "runs" / "main_ladv0_s{}".format(seed) / "summary.json"
    if not s.exists():
        raise SystemExit("P8 main λ = 0 run missing (needed for reuse): {}".format(s))
    return Path(json.loads(s.read_text())["best_checkpoint"]).parent


def default_mode(recipe):
    """Class-balanced sampling needs a separate natural-mix ISIC adversary stream (otherwise the adversary would see a
    near-uniform ISIC mix, i.e. a different label shift)."""
    return "three" if recipe["balance"] == "sampling" else "two"


def lam_tag(x):
    return C.lam_tag(x)


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
def load_data():
    """Exactly the frames of the lesion-level Table 1 runs (item 4)."""
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy().reset_index(drop=True)
    train, val, test = T.isic_splits_by_lesion(isic)
    spec = json.loads(p3.P2_SPLIT.read_text())
    pad_adv = pad[pad.path.isin(set(spec["pad_adv_paths"]))].reset_index(drop=True)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    return {"train": train, "val": val, "test": test, "pad_full": pad, "pad_adv": pad_adv, "pad_heldout": pad_hold}


def add_site(frame):
    """ISIC acquisition site from the lesion_id prefix (BCN_/HAM_/MSK*; null lesion_id -> 'empty')."""
    meta = pd.read_csv(C.ISIC_META, usecols=["image", "lesion_id"])
    stem = frame.path.str.extract(r"([^/]+)\.\w+$")[0]
    lid = stem.map(dict(zip(meta.image, meta.lesion_id)))
    out = frame.copy()
    out["site"] = [site_from_lesion(x) for x in lid]
    return out


def class_mix(labels):
    y = np.asarray(labels, int)
    return np.bincount(y, minlength=NCLS) / max(len(y), 1)


def pad_patients(frame):
    return np.array([C.PAT.search(str(p)).group(1) for p in frame.path])


def dose_mix(shift, isic_mix, pad_mix):
    """shift = 1 -> ISIC train mix (current setting); 0 -> PAD adversary-set mix; linear in between."""
    m = float(shift) * np.asarray(isic_mix) + (1.0 - float(shift)) * np.asarray(pad_mix)
    return m / m.sum()


def class_weighted_probs(frame, target_mix):
    y = frame.label_idx.to_numpy().astype(int)
    n = np.bincount(y, minlength=NCLS)
    p = np.asarray(target_mix, float)[y] / np.maximum(n[y], 1)
    return p / p.sum()


def balanced_loss_weights(train):
    """Inverse-frequency weights normalised so that sum_c pi_c w_c = 1 under the ISIC train mix."""
    pi = class_mix(train.label_idx)
    return (1.0 / (NCLS * pi)).tolist()


class Stream:
    def __init__(self, name, frame, probs=None, cls=False, adv=-1, ctx=-1, src_labels=False):
        self.name = name
        self.paths = frame.path.tolist()
        has_lab = "label_idx" in frame.columns
        self.labels = frame.label_idx.to_numpy().astype(int) if has_lab else np.full(len(frame), -1, dtype=int)
        if (cls or src_labels) and not has_lab:
            raise ValueError("stream {} needs labels".format(name))
        self.probs = None if probs is None else torch.as_tensor(np.asarray(probs, float), dtype=torch.double)
        self.cls, self.adv, self.ctx, self.src_labels = bool(cls), int(adv), int(ctx), bool(src_labels)

    def describe(self):
        lab = self.labels[self.labels >= 0]
        return {"name": self.name, "n": len(self.paths), "sampling": "weighted" if self.probs is not None else "uniform",
                "classifier": self.cls, "adv_domain": self.adv, "ctx_domain": self.ctx,
                "labels_loaded": bool(len(lab)), "expected_mix": (
                    np.bincount(self.labels, weights=self.probs.numpy(), minlength=NCLS).tolist()
                    if self.probs is not None and len(lab) else (class_mix(lab).tolist() if len(lab) else None))}


class MultiStreamDataset(Dataset):
    """Item idx = one image per stream. Stream 0 without weights is walked by idx (shuffled by the loader, like
    CombinedTrainDataset's ISIC side); every other stream is drawn with replacement (torch RNG, seeded per worker by
    pl.seed_everything(workers=True)). Length = stream 0 size, so steps/epoch equal the current setting."""

    def __init__(self, streams, transform):
        self.streams, self.transform = streams, transform

    def __len__(self):
        return len(self.streams[0].paths)

    def _pick(self, k, s, idx):
        if s.probs is None:
            return idx % len(s.paths) if k == 0 else int(torch.randint(len(s.paths), (1,)).item())
        return int(torch.multinomial(s.probs, 1).item())

    def __getitem__(self, idx):
        out = []
        for k, s in enumerate(self.streams):
            j = self._pick(k, s, idx)
            x = self.transform(Image.open(s.paths[j]).convert("RGB"))
            y = int(s.labels[j])
            out.append((x, y if s.cls else -1, y if s.src_labels else -1, s.adv, s.ctx))
        return out


def multistream_collate(batch):
    cols = {"x": [], "y_cls": [], "y_src": [], "d_adv": [], "d_ctx": []}
    for k in range(len(batch[0])):
        cols["x"].append(torch.stack([b[k][0] for b in batch]))
        for j, key in enumerate(("y_cls", "y_src", "d_adv", "d_ctx"), start=1):
            cols[key].append(torch.tensor([b[k][j] for b in batch], dtype=torch.long))
    return {k: torch.cat(v) for k, v in cols.items()}


def build_train_set(mode, recipe, data, adv_mix=None):
    """Returns (dataset, collate, stream description)."""
    tf = build_train_transform_robust()
    cls_probs = (class_weighted_probs(data["train"], np.full(NCLS, 1.0 / NCLS))
                 if recipe["balance"] == "sampling" else None)
    if mode == "two":
        if cls_probs is not None or adv_mix is not None:
            raise ValueError("two-stream mode cannot resample the classifier or adversary stream")
        ds = CombinedTrainDataset(data["train"], data["pad_adv"], transform=tf,
                                  lesion_transform=build_lesion_branch_transform_gray())
        return ds, csg_lite_paired_collate, [{"name": "legacy CombinedTrainDataset", "isic": len(data["train"]),
                                               "pad_adv": len(data["pad_adv"])}]
    if mode == "three":
        adv_probs = None if adv_mix is None else class_weighted_probs(data["train"], adv_mix)
        streams = [Stream("isic_cls", data["train"], cls_probs, cls=True),
                   Stream("isic_adv", data["train"], adv_probs, adv=0, ctx=0, src_labels=True),
                   Stream("pad_adv", data["pad_adv"][["path"]], None, adv=1, ctx=1)]
    elif mode == "p7b":
        tr = add_site(data["train"])
        streams = [Stream("isic_cls", data["train"], cls_probs, cls=True),
                   Stream("isic_bcn", tr[tr.site == "BCN"].reset_index(drop=True), None, adv=0, ctx=0, src_labels=True),
                   Stream("isic_ham", tr[tr.site == "HAM"].reset_index(drop=True), None, adv=1, ctx=1, src_labels=True)]
    else:
        raise ValueError(mode)
    return MultiStreamDataset(streams, tf), multistream_collate, [s.describe() for s in streams]


# ----------------------------------------------------------------------------------------------------------------------
# Label-shift estimation (Lipton et al. 2018 BBSE; Tachet des Combes et al. 2020 QP form)
# ----------------------------------------------------------------------------------------------------------------------
def solve_label_shift_qp(conf, mu, w_max=None):
    """conf[i, j] = P_S(yhat = i, y = j), mu[i] = P_T(yhat = i).
    min_w ||conf w - mu||^2  s.t.  w >= 0,  sum_j w_j P_S(y = j) = 1."""
    conf, mu = np.asarray(conf, float), np.asarray(mu, float)
    p_s = conf.sum(0)
    k = conf.shape[1]

    def fun(w):
        r = conf @ w - mu
        return float(r @ r)

    res = minimize(fun, np.ones(k), jac=lambda w: 2.0 * conf.T @ (conf @ w - mu), method="SLSQP",
                   bounds=[(0.0, w_max)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: float(p_s @ w - 1.0), "jac": lambda w: p_s}],
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    return w, {"qp_success": bool(res.success), "qp_residual": fun(w), "p_source": p_s.tolist(),
               "est_target_mix": (w * p_s / max(float(w @ p_s), 1e-12)).tolist()}


def bbse_estimate(y_src, pred_src, pred_tgt):
    conf = np.zeros((NCLS, NCLS))
    np.add.at(conf, (np.asarray(pred_src, int), np.asarray(y_src, int)), 1.0)
    conf /= conf.sum()
    mu = np.bincount(np.asarray(pred_tgt, int), minlength=NCLS) / max(len(pred_tgt), 1)
    return solve_label_shift_qp(conf, mu)


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class R6Lightning(CSGLiteLightning):
    """CSGLiteLightning with (i) row-masked losses for multi-stream batches, (ii) optional class-weighted CE,
    (iii) CDAN / IWDAN / IWCDAN domain losses. With align='dann', class_weights=None and a legacy two-stream batch the
    loss equals the base class's (λ_supcon = 0 there, so its term is dropped here)."""

    def __init__(self, align="dann", class_weights=None, iw_cls=True, iw_ma=0.5, iw_max=None, iw_log=None,
                 oracle_w=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if align not in ("dann", "cdan", "iwdan", "iwcdan", "iwdan_oracle"):
            raise ValueError(align)
        self.align = align
        self.oracle = align == "iwdan_oracle"
        self.iw = align in ("iwdan", "iwcdan", "iwdan_oracle")
        self.cond = align in ("cdan", "iwcdan")
        nc = int(self.hparams.lesion_classes)
        cw = torch.ones(nc) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("cls_w", cw.float())
        ow = torch.ones(nc) if oracle_w is None else torch.as_tensor(oracle_w, dtype=torch.float32)
        self.register_buffer("im_w", ow.float())
        self.register_buffer("acc_conf", torch.zeros(nc, nc, dtype=torch.float64))
        self.register_buffer("acc_tgt", torch.zeros(nc, dtype=torch.float64))
        if self.cond:
            self.cdan_head = derm_domain_head(int(self.hparams.lesion_latent_dim) * nc, int(self.hparams.domain_classes))

    @staticmethod
    def _unify(batch):
        if isinstance(batch, dict):
            return batch
        images_ctx, _les, y, _y_supcon_discarded, domain = batch
        return {"x": images_ctx, "y_cls": y, "y_src": y, "d_adv": domain, "d_ctx": domain}

    def training_step(self, batch, batch_idx):
        b = self._unify(batch)
        alpha, progress = self._compute_grl_alpha()
        self.model.set_grl_lambda(alpha)
        self._last_alpha = float(alpha)
        y_hat, d_ctx, d_adv, z_l, z_c = self.model(b["x"], x_lesion=None, return_latents=True)
        y_hat32 = y_hat.float()
        zero = y_hat32.new_tensor(0.0)

        w_cls = self.cls_w * (self.im_w.float() if (self.iw and self.hparams.iw_cls) else 1.0)
        has_cls = b["y_cls"] >= 0
        loss_cls = F.cross_entropy(y_hat32, b["y_cls"], weight=w_cls, ignore_index=-1) if has_cls.any() else zero

        mc = b["d_ctx"] >= 0
        loss_ctx = F.cross_entropy(d_ctx[mc].float(), b["d_ctx"][mc]) if mc.any() else zero

        ma = b["d_adv"] >= 0
        if self.cond:
            p = torch.softmax(y_hat32, dim=1).detach()
            h = torch.bmm(p.unsqueeze(2), self.model.grl(z_l).float().unsqueeze(1)).flatten(1)
            d_logits = self.cdan_head(h)
        else:
            d_logits = d_adv
        if ma.any():
            per = F.cross_entropy(d_logits[ma].float(), b["d_adv"][ma], reduction="none")
            w_row = torch.ones_like(per)
            if self.iw:
                ys = b["y_src"][ma]
                src = (b["d_adv"][ma] == 0) & (ys >= 0)
                w_row = torch.where(src, self.im_w.float()[ys.clamp(min=0)], w_row)
            loss_adv = (w_row * per).mean()
        else:
            loss_adv = zero

        z_c_proj = self.model.project_context_for_orth(z_c)
        loss_orth = torch.mean(F.cosine_similarity(z_l, z_c_proj, dim=1).pow(2))
        loss = (loss_cls + self.lambda_ctx * loss_ctx + self.lambda_adv * loss_adv + self.lambda_orth * loss_orth)

        with torch.no_grad():
            pred = y_hat32.argmax(1)
            if self.iw and not self.oracle:
                nc = self.acc_tgt.numel()
                src = (b["d_adv"] == 0) & (b["y_src"] >= 0)
                tgt = b["d_adv"] == 1
                idx = pred[src] * nc + b["y_src"][src]
                self.acc_conf.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
                self.acc_tgt.index_add_(0, pred[tgt], torch.ones_like(pred[tgt], dtype=torch.float64))
            acc_adv = (d_logits[ma].argmax(1) == b["d_adv"][ma]).float().mean() if ma.any() else zero
            acc_ctx = (d_ctx[mc].argmax(1) == b["d_ctx"][mc]).float().mean() if mc.any() else zero
            acc_cls = (pred[has_cls] == b["y_cls"][has_cls]).float().mean() if has_cls.any() else zero
        n = b["x"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=n)
        for k, v in (("loss_cls", loss_cls), ("loss_ctx", loss_ctx), ("loss_adv", loss_adv), ("loss_orth", loss_orth),
                     ("acc_adv", acc_adv), ("acc_ctx", acc_ctx), ("acc_cls", acc_cls)):
            self.log("train/" + k, v, on_step=False, on_epoch=True, batch_size=n)
        self.log("train/grl_a", alpha, on_step=False, on_epoch=True, batch_size=n)
        self.log("train/grl_progress", progress, on_step=True, on_epoch=False, batch_size=n)
        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if not self.iw:
            return
        if self.oracle:
            w = self.im_w.detach().cpu().numpy()
            row = {"epoch": int(self.current_epoch) + 1, "oracle": True, "w": w.tolist(),
                   "note": "oracle, not deployable; w fixed from true pad_adv mix"}
            print("[iw-oracle] epoch {} w={}".format(row["epoch"], np.round(w, 3).tolist()), flush=True)
            if self.hparams.iw_log:
                with open(self.hparams.iw_log, "a") as f:
                    f.write(json.dumps(row) + "\n")
            return
        if self.acc_conf.sum() <= 0 or self.acc_tgt.sum() <= 0:
            return
        conf = (self.acc_conf / self.acc_conf.sum()).cpu().numpy()
        mu = (self.acc_tgt / self.acc_tgt.sum()).cpu().numpy()
        w_new, info = solve_label_shift_qp(conf, mu, self.hparams.iw_max)
        ma = float(self.hparams.iw_ma)
        w = ma * self.im_w.detach().cpu().numpy() + (1.0 - ma) * w_new
        self.im_w.copy_(torch.as_tensor(w, dtype=self.im_w.dtype))
        row = {"epoch": int(self.current_epoch) + 1, "w_new": w_new.tolist(), "w": w.tolist(), "mu_target_pred": mu.tolist(),
               "est_target_mix_from_w": (w * conf.sum(0) / max(float(w @ conf.sum(0)), 1e-12)).tolist(), **info}
        print("[iw] epoch {} w={}".format(row["epoch"], np.round(w, 3).tolist()), flush=True)
        if self.hparams.iw_log:
            with open(self.hparams.iw_log, "a") as f:
                f.write(json.dumps(row) + "\n")
        self.acc_conf.zero_()
        self.acc_tgt.zero_()

    def configure_optimizers(self):
        adv = list(self.model.domain_classifier_adv.parameters()) + (list(self.cdan_head.parameters()) if self.cond else [])
        cls = list(self.model.lesion_classifier.parameters())
        ids = {id(p) for p in adv + cls}
        main = [p for p in self.parameters() if id(p) not in ids]
        return torch.optim.AdamW([
            {"params": main, "lr": self.learning_rate, "weight_decay": self.weight_decay},
            {"params": adv, "lr": self.learning_rate * self.adv_lr_multiplier, "weight_decay": self.weight_decay},
            {"params": cls, "lr": self.learning_rate * self.lesion_cls_lr_multiplier, "weight_decay": self.weight_decay},
        ])


def oracle_weights_from_mix(p_src, p_tgt, eps=1e-8):
    """w_j = π_T(j)/P_S(j) so Σ_j w_j P_S(j) = 1 when π_T sums to 1."""
    p_src, p_tgt = np.asarray(p_src, float), np.asarray(p_tgt, float)
    w = np.zeros_like(p_src)
    m = p_src > eps
    w[m] = p_tgt[m] / p_src[m]
    return w


def make_model(recipe, lam, align="dann", class_weights=None, iw_log=None, lr=1e-4, iw_cls=True, oracle_w=None):
    """Same constructor arguments as train_phase3_sweep.main except the recipe fields."""
    return R6Lightning(
        align=align, class_weights=class_weights, iw_cls=iw_cls, iw_log=iw_log, oracle_w=oracle_w,
        lesion_classes=8, domain_classes=2, lesion_latent_dim=int(recipe["latent"]), context_latent_dim=64,
        learning_rate=lr, weight_decay=1e-4, lambda_ctx=1.0, lambda_adv=float(lam), lambda_orth=1.0, lambda_supcon=0.0,
        adv_lr_multiplier=30.0, lesion_cls_lr_multiplier=float(recipe["cls_lr_mult"]), backbone_variant="b3",
        pretrained=True)


# Primary checkpoint policy (decisions/paperB_r5_p8.md, deviation 2026-10-15 21:57): last.ckpt (end of training) for
# P8 and everything that inherits its recipe; the best-val checkpoint is evaluated as secondary.
CKPT_POLICY = "last"


def ckpt_epoch(ckpt):
    m = re.search(r"best-(\d+)\.ckpt$", str(ckpt))
    if m:
        return int(m.group(1))
    return int(torch.load(str(ckpt), map_location="cpu", weights_only=False).get("epoch", -1))


def trained_to_end(ckpt_dir, recipe):
    last = Path(ckpt_dir) / "last.ckpt"
    return last.exists() and ckpt_epoch(last) >= int(recipe["max_epochs"]) - 1


def find_best(ckpt_dir):
    best = sorted(Path(ckpt_dir).glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    last = Path(ckpt_dir) / "last.ckpt"
    return last if last.exists() else None


def load_net(ckpt, device):
    try:
        lit = R6Lightning.load_from_checkpoint(str(ckpt), map_location="cpu")
    except Exception as e:  # legacy CSGLiteLightning checkpoint (current-recipe Table 1 runs)
        print("R6Lightning load failed ({}); loading as CSGLiteLightning".format(type(e).__name__), flush=True)
        lit = CSGLiteLightning.load_from_checkpoint(str(ckpt), map_location="cpu")
    return lit.model.to(device).eval()


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation (features -> metrics; identical code for new runs and reused reference runs)
# ----------------------------------------------------------------------------------------------------------------------
def eval_datasets(data, mode):
    tf = build_val_transform_robust()
    sets = {"isic_val": SkinDataset(data["val"], transform=tf)}
    if mode == "val":
        return sets
    for name, key in (("isic_train", "train"), ("isic_test", "test"), ("pad_full", "pad_full"), ("pad_adv", "pad_adv"),
                      ("pad_heldout", "pad_heldout")):
        sets[name] = SkinDataset(data[key], transform=tf)
    sets["fitzpatrick17k"] = FolderJpegDataset(sorted(p3.FITZ_DIR.glob("*.jpg")), tf)
    for n in P7A_NAMES:
        csv = P7A_SETS / "{}.csv".format(n)
        if csv.exists():
            sets[n] = FolderJpegDataset(pd.read_csv(csv, keep_default_na=False).path.tolist(), tf)
    return sets


def extract(net, data, feat_dir, mode, num_workers, device, save_backbone=False):
    feat_dir.mkdir(parents=True, exist_ok=True)
    keys = FEAT_KEYS + (("backbone_raw_lesion",) if save_backbone else ())
    for name, ds in eval_datasets(data, mode).items():
        out = feat_dir / "{}.npz".format(name)
        if out.exists():
            continue
        pack = x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers), device)
        np.savez_compressed(out, **{k: pack[k] for k in keys})
        print("features", feat_dir.name, name, pack["labels"].shape, flush=True)


def load_feats(feat_dir, names=None):
    feat_dir = Path(feat_dir)
    out = {}
    for p in sorted(feat_dir.glob("*.npz")):
        if names is None or p.stem in names:
            out[p.stem] = dict(np.load(p))
    return out


class Scorer:
    """Table 1 detectors on z_lesion_norm / logits, statistics fit on ISIC train only. Higher = more OOD."""

    def __init__(self, tr):
        z, y = tr["z_lesion_norm"], tr["labels"].astype(int)
        self.means, self.prec = ood_metrics.compute_mahalanobis_params_from_arrays(z, y, num_classes=NCLS, reg_eps=1e-3)
        self.cmeans, self.present = fit_class_means(z, y, NCLS)
        self.knn = fit_knn(z, k=50)

    def scores(self, pack):
        z, lg = pack["z_lesion_norm"], torch.from_numpy(pack["logits"]).float()
        return {
            "mahalanobis_classcond": ood_metrics.mahalanobis_min_squared_distances(z, self.means, self.prec).astype(np.float64),
            "cosine_max": cosine_ood_scores(z, self.cmeans, self.present).astype(np.float64),
            "knn_k50": knn_scores(self.knn, z).astype(np.float64),
            "MSP": (-torch.softmax(lg, 1).max(1).values.numpy()).astype(np.float64),
            "Energy_T1": (-torch.logsumexp(lg, 1).numpy()).astype(np.float64),
        }

    def nearest_class(self, z):
        z = np.asarray(z, np.float64)
        P, M = self.prec.astype(np.float64), self.means.astype(np.float64)
        zp = z @ P
        d2 = np.einsum("nd,nd->n", zp, z)[:, None] + np.einsum("cd,cd->c", M @ P, M)[None] - 2.0 * zp @ M.T
        return d2.argmin(1)

    def nearest_class_share(self, z):
        nc = self.nearest_class(z)
        return {LABELS[c]: float(np.mean(nc == c)) for c in range(NCLS)}


def auroc(sid, sood, w_id=None):
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    w = None if w_id is None else np.r_[w_id, np.ones(len(sood))]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=w))


def xdom(logits, y):
    y = np.asarray(y, int)
    pred = PAD_CLASSES[np.asarray(logits)[:, PAD_CLASSES].argmax(1)]
    rec = recall_score(y, pred, labels=PAD_CLASSES, average=None, zero_division=0)
    return {"bal_acc_6cls": float(balanced_accuracy_score(y, pred)),
            "recall_6cls": {LABELS[c]: float(r) for c, r in zip(PAD_CLASSES, rec)},
            "pred_share_6cls": {LABELS[c]: float(np.mean(pred == c)) for c in PAD_CLASSES}}


def decomposition(sid, yid, sood, yood):
    """R3 A4 / R4 check on a labelled OOD set (labels used for evaluation only). ID-side per-class AUROC_c, ID reweighted
    to the OOD class mix, the identity sum_c pi_ood(c) AUROC_c, and OOD-side per-class AUROC (ID all vs OOD class c)."""
    yid, yood = np.asarray(yid, int), np.asarray(yood, int)
    pi_id, pi_ood = class_mix(yid), class_mix(yood)
    id_cls = {LABELS[c]: auroc(sid[yid == c], sood) for c in range(NCLS) if (yid == c).any()}
    w = np.where(pi_id > 0, pi_ood / np.maximum(pi_id, 1e-12), 0.0)[yid]
    rw = auroc(sid, sood, w)
    ident = float(sum(pi_ood[c] * id_cls[LABELS[c]] for c in range(NCLS) if pi_ood[c] > 0))
    ood_cls = {LABELS[c]: auroc(sid, sood[yood == c]) for c in PAD_CLASSES if (yood == c).sum() >= 1}
    return {"id_class_auroc": id_cls, "reweighted_to_ood_mix": rw, "identity_sum": ident,
            "identity_abs_diff": abs(rw - ident), "ood_class_auroc": ood_cls,
            "ood_nv_vs_rest": {"nv": auroc(sid, sood[yood == NV]) if (yood == NV).any() else None,
                               "non_nv": auroc(sid, sood[yood != NV]) if (yood != NV).any() else None}}


def classmatched_leakage(z_isic, y_isic, z_pad, y_pad, seed=0):
    """ISIC test subsample (without replacement) at the PAD class mix vs PAD; labels used only to build the subsample."""
    rng = np.random.default_rng(seed)
    y_isic, y_pad = np.asarray(y_isic, int), np.asarray(y_pad, int)
    pi = class_mix(y_pad)
    n_c = np.bincount(y_isic, minlength=NCLS)
    m = int(min(n_c[c] / pi[c] for c in range(NCLS) if pi[c] > 0))
    idx = np.concatenate([rng.choice(np.flatnonzero(y_isic == c), int(round(m * pi[c])), replace=False)
                          for c in range(NCLS) if pi[c] > 0 and round(m * pi[c]) > 0])
    z = np.concatenate([z_isic[idx], z_pad])
    d = np.r_[np.zeros(len(idx)), np.ones(len(z_pad))]
    out = p3.leakage_probe(z, d)
    out["n_isic_subsample"] = int(len(idx))
    return out


def run_metrics(Fs, ood_names=None, site_test=None):
    """Point metrics of one run from its feature dict. Fs keys: isic_train, isic_test, pad_full, pad_heldout, ... ."""
    out = {}
    if "isic_val" in Fs:
        v = Fs["isic_val"]
        pv = v["logits"].argmax(1)
        out["val"] = {"bal_acc": float(balanced_accuracy_score(v["labels"], pv)), "acc": float(accuracy_score(v["labels"], pv)),
                      "ece": p3.ece_from_logits(v["logits"], v["labels"]),
                      "recall": dict(zip(LABELS, recall_score(v["labels"], pv, labels=range(NCLS), average=None,
                                                              zero_division=0).tolist()))}
    if "isic_test" not in Fs or "isic_train" not in Fs:
        return out
    te, tr = Fs["isic_test"], Fs["isic_train"]
    pt = te["logits"].argmax(1)
    out["id"] = {"bal_acc": float(balanced_accuracy_score(te["labels"], pt)), "acc": float(accuracy_score(te["labels"], pt)),
                 "ece": p3.ece_from_logits(te["logits"], te["labels"]),
                 "recall": dict(zip(LABELS, recall_score(te["labels"], pt, labels=range(NCLS), average=None,
                                                         zero_division=0).tolist())),
                 "mean_conf": float(torch.softmax(torch.from_numpy(te["logits"]).float(), 1).max(1).values.mean())}
    if "pad_full" in Fs:
        pf = Fs["pad_full"]
        z = np.concatenate([te["z_lesion_norm"], pf["z_lesion_norm"]])
        d = np.r_[np.zeros(len(te["z_lesion_norm"])), np.ones(len(pf["z_lesion_norm"]))]
        out["leakage"] = p3.leakage_probe(z, d)
        out["leakage_classmatched"] = classmatched_leakage(te["z_lesion_norm"], te["labels"], pf["z_lesion_norm"], pf["labels"])
    if site_test is not None:
        m = np.isin(site_test, ["BCN", "HAM"])
        out["leakage_bcn_ham"] = p3.leakage_probe(te["z_lesion_norm"][m], (site_test[m] == "HAM").astype(int))
    if "pad_heldout" in Fs:
        h = Fs["pad_heldout"]
        out["ood_ece_pad_heldout"] = conf_pack(h["logits"], h["labels"], restrict=PAD_CLASSES)[0]["ece"]
        out["ood_mean_conf_pad_heldout"] = float(torch.softmax(torch.from_numpy(h["logits"]).float(), 1).max(1).values.mean())
        out["xdom_pad_heldout"] = xdom(h["logits"], h["labels"])
        if "isic_val" in Fs:
            v = Fs["isic_val"]
            for tgt in ("pad_adv", "pad_heldout", "pad_full"):
                if tgt in Fs:
                    _w, info = bbse_estimate(v["labels"], v["logits"].argmax(1), Fs[tgt]["logits"].argmax(1))
                    out.setdefault("bbse", {})[tgt] = {"est_mix": info["est_target_mix"], "true_mix": class_mix(Fs[tgt]["labels"]).tolist()}
    if "pad_full" in Fs:
        out["xdom_pad_full"] = xdom(Fs["pad_full"]["logits"], Fs["pad_full"]["labels"])
    sc = Scorer(tr)
    sid = sc.scores(te)
    names = ood_names or [n for n in ("pad_heldout", "pad_full", "pad_adv", "fitzpatrick17k") + P7A_NAMES if n in Fs]
    out["ood"] = {}
    for n in names:
        if n not in Fs:
            continue
        so = sc.scores(Fs[n])
        blk = {d: auroc(sid[d], so[d]) for d in DETS}
        blk["n"] = int(len(so["MSP"]))
        nc = sc.nearest_class(Fs[n]["z_lesion_norm"])
        blk["nv_nearest_frac"] = float(np.mean(nc == NV))
        blk["nearest_class_share"] = {LABELS[c]: float(np.mean(nc == c)) for c in range(NCLS)}
        if n.startswith("pad_"):
            blk["decomposition_maha"] = decomposition(sid["mahalanobis_classcond"], te["labels"],
                                                      so["mahalanobis_classcond"], Fs[n]["labels"])
        out["ood"][n] = blk
    id_nc = sc.nearest_class(te["z_lesion_norm"])
    out["id_nv_nearest_frac"] = float(np.mean(id_nc == NV))
    out["id_nearest_class_share"] = {LABELS[c]: float(np.mean(id_nc == c)) for c in range(NCLS)}
    out["degraded"] = bool(out["id"]["bal_acc"] < DEGRADED_ID_BAL)
    return out


def table1_cols(m, ood_primary="pad_heldout"):
    """Table 1 columns (r4_report LES_COLS order) from run_metrics output."""
    return {"leakage": m["leakage"]["bal_acc_mean"], "id_bal": m["id"]["bal_acc"], "id_ece": m["id"]["ece"],
            "ood_ece": m["ood_ece_pad_heldout"], "maha": m["ood"][ood_primary]["mahalanobis_classcond"],
            "fitz": m["ood"]["fitzpatrick17k"]["mahalanobis_classcond"], "xdom": m["xdom_pad_heldout"]["bal_acc_6cls"]}


# ----------------------------------------------------------------------------------------------------------------------
# Train + evaluate one run
# ----------------------------------------------------------------------------------------------------------------------
def run(spec, args):
    """spec: item, out_root, ckpt_root, tag, recipe, lam, seed, mode, align, adv_mix (or None), eval ('full'|'val'),
    reuse_ckpt (dir or None), extra (dict written to config)."""
    out_root, tag = Path(spec["out_root"]), spec["tag"]
    run_dir, feat_dir = out_root / "runs" / tag, out_root / "features" / tag
    ckpt_dir = Path(spec["ckpt_root"]) / tag
    if args.skip_done and (run_dir / "summary.json").exists() and not args.dry_run:
        if json.loads((run_dir / "summary.json").read_text()).get("checkpoint_policy") == CKPT_POLICY:
            print("skip_done", run_dir / "summary.json", flush=True)
            return
        print("re-evaluating under checkpoint policy", CKPT_POLICY, run_dir, flush=True)
        old = out_root / "features_bestval" / tag
        if feat_dir.exists() and not old.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            feat_dir.rename(old)
    seed = int(spec["seed"])
    seed_everything(seed)
    pl.seed_everything(seed, workers=True)
    for d in (run_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    data = load_data()
    recipe = spec["recipe"]
    cw = balanced_loss_weights(data["train"]) if recipe["balance"] == "loss" else None
    adv_mix = spec.get("adv_mix")
    if spec.get("adv_shift") is not None and float(spec["adv_shift"]) < 1.0:
        # PAD labels enter here only as class proportions of the ISIC adversary stream; never in a loss.
        adv_mix = dose_mix(spec["adv_shift"], class_mix(data["train"].label_idx), class_mix(data["pad_adv"].label_idx))
    spec = dict(spec, adv_mix=None if adv_mix is None else np.asarray(adv_mix).tolist())
    ds, collate, streams = build_train_set(spec["mode"], recipe, data, adv_mix)
    cfg = {k: v for k, v in spec.items() if k not in ("out_root", "ckpt_root")}
    cfg.update(out_root=str(out_root), ckpt_dir=str(ckpt_dir), streams=streams, class_weights=cw,
               isic_train_mix=class_mix(data["train"].label_idx).tolist(),
               pad_adv_mix_for_resampling_only=class_mix(data["pad_adv"].label_idx).tolist(),
               n={k: int(len(v)) for k, v in data.items()}, commit=C.git_head())
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.dry_run:
        print("dry_run", tag, json.dumps({"n_train_items": len(ds), "streams": streams}, default=str)[:2000], flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_dir = Path(spec["reuse_ckpt"]) if spec.get("reuse_ckpt") else ckpt_dir
    best = find_best(src_dir) if (spec.get("reuse_ckpt") or trained_to_end(ckpt_dir, recipe)) else None
    if best is not None:
        print("checkpoint exists, evaluation only (reuse = {})".format(bool(spec.get("reuse_ckpt"))), best, flush=True)
    else:
        align = spec.get("align", "dann")
        iw_log = str(run_dir / "iw_log.jsonl") if align.startswith("iw") else None
        if iw_log and not args.resume and Path(iw_log).exists():
            Path(iw_log).write_text("")
        oracle_w = None
        if align == "iwdan_oracle":
            oracle_w = oracle_weights_from_mix(class_mix(data["train"].label_idx),
                                               class_mix(data["pad_adv"].label_idx)).tolist()
            cfg["oracle_w"] = oracle_w
            cfg["oracle_note"] = "oracle, not deployable; PAD labels set w only"
            (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        model = make_model(recipe, spec["lam"], align=align, class_weights=cw, iw_log=iw_log,
                           iw_cls=spec.get("iw_cls", True), oracle_w=oracle_w)
        ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_dir), filename="best-{epoch:02d}", monitor="val/acc",
                                               mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False)
        cbs = [ckpt_cb]
        if spec.get("save_every_epoch"):
            cbs.append(pl.callbacks.ModelCheckpoint(dirpath=str(ckpt_dir / "epochs"), filename="ep-{epoch:02d}", save_top_k=-1,
                                                    every_n_epochs=1, save_weights_only=True, auto_insert_metric_name=False))
        trainer = pl.Trainer(max_epochs=int(recipe["max_epochs"]), accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1, callbacks=cbs, precision="16-mixed" if torch.cuda.is_available() else "32-true",
                             default_root_dir=str(run_dir), log_every_n_steps=20,
                             deterministic=bool(getattr(args, "deterministic", False)))
        last = ckpt_dir / "last.ckpt"
        tl = p2.make_loader(ds, args.batch_size, args.num_workers, True, drop_last=True, collate=collate)
        vl = p2.make_loader(SkinDataset(data["val"], transform=build_val_transform_robust()), args.batch_size,
                            args.num_workers, False)
        trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl,
                    ckpt_path=str(last) if args.resume and last.exists() else None)
        best = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else last
    last = src_dir / "last.ckpt"
    primary = last if CKPT_POLICY == "last" else best
    if not primary.exists():
        raise FileNotFoundError(primary)
    print("primary ({})".format(CKPT_POLICY), primary, "epoch", ckpt_epoch(primary), "| best-val", best,
          "epoch", ckpt_epoch(best), flush=True)
    site = add_site(data["test"]).site.to_numpy() if spec["mode"] == "p7b" else None
    net = load_net(primary, device)
    extract(net, data, feat_dir, spec.get("eval", "full"), args.num_workers, device)
    m = run_metrics(load_feats(feat_dir), site_test=site)
    fb = out_root / "features_bestval" / tag
    net = load_net(best, device)
    extract(net, data, fb, spec.get("eval", "full"), args.num_workers, device)
    secondary = {"checkpoint": str(best), "epoch": ckpt_epoch(best), "metrics": run_metrics(load_feats(fb), site_test=site),
                 "features": str(fb)}
    summary = {"tag": tag, "item": spec["item"], "recipe": recipe, "lambda_adv": float(spec["lam"]), "seed": seed,
               "mode": spec["mode"], "align": spec.get("align", "dann"), "adv_shift": spec.get("adv_shift"),
               "adv_mix": spec.get("adv_mix"), "checkpoint_policy": CKPT_POLICY, "primary_checkpoint": str(primary),
               "primary_epoch": ckpt_epoch(primary), "best_checkpoint": str(best), "best_epoch": ckpt_epoch(best),
               "secondary_bestval": secondary,
               "reused_checkpoint": bool(spec.get("reuse_ckpt") and str(best).startswith(str(spec["reuse_ckpt"]))),
               "eval": spec.get("eval", "full"), "metrics": m, **spec.get("extra", {})}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("saved", run_dir / "summary.json", flush=True)


def common_args(p):
    p.add_argument("--task_id", type=int)
    p.add_argument("--list", action="store_true", help="print the job table and exit")
    p.add_argument("--batch_size", type=int, default=32, help="per stream (current setting: 32 ISIC + 32 PAD)")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p


def print_jobs(jobs):
    for i, j in enumerate(jobs):
        print(i, j["tag"], "λ={:g}".format(j["lam"]), j["mode"], j.get("align", "dann"), j["recipe"]["name"])
    print("n_jobs", len(jobs))


# ----------------------------------------------------------------------------------------------------------------------
# Bootstrap (PREREGISTER_BOOTSTRAP.json rule A, lesion-level ID groups as R3 A2)
# ----------------------------------------------------------------------------------------------------------------------
def idx_groups(g):
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


def ood_groups(name, data):
    if name in ("pad_heldout", "pad_full", "pad_adv"):
        return idx_groups(pad_patients(data[name])), "patient"
    if name == "fitzpatrick17k":
        p13 = RES / "r5" / "p13" / "fitz_clusters.csv"
        if p13.exists():
            paths = sorted(p3.FITZ_DIR.glob("*.jpg"))
            cl = pd.read_csv(p13).set_index("md5hash")["cluster_union"]
            return idx_groups([str(cl[p.stem]) for p in paths]), "P13 cluster_union"
        return [np.array([i]) for i in range(len(sorted(p3.FITZ_DIR.glob("*.jpg"))))], "image (anti-conservative)"
    meta = pd.read_csv(P7A_SETS / "{}.csv".format(name), keep_default_na=False)
    return idx_groups(meta.group.astype(str).to_numpy()), "P7a group column"


def bootstrap_cell(feat_dirs, ood_name, id_groups, o_groups, rng_seed=None, dets=None, exclude_seeds=None):
    """feat_dirs: {seed: Path}. Draws shared across seeds within the cell. Returns per-detector block with rule A.
    exclude_seeds: optional set of seeds dropped from the pooled verdict (still listed under excluded_seeds)."""
    dets = tuple(dets) if dets is not None else BOOT_DETS
    exclude_seeds = set(exclude_seeds or ())
    pre = json.loads(PREREG.read_text())
    n_boot = int(pre["n_resamples"])
    rng = np.random.default_rng(int(pre["rng_seed"]) if rng_seed is None else rng_seed)
    sc = {}
    for s, d in feat_dirs.items():
        Fs = load_feats(d, ("isic_train", "isic_test", ood_name))
        if ood_name not in Fs:
            continue
        S = Scorer(Fs["isic_train"])
        a, b = S.scores(Fs["isic_test"]), S.scores(Fs[ood_name])
        sc[s] = {k: (a[k], b[k]) for k in dets}
    if not sc:
        return None
    id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))]) for _ in range(n_boot)]
    oo_draws = [np.concatenate([o_groups[j] for j in rng.integers(0, len(o_groups), len(o_groups))]) for _ in range(n_boot)]
    seeds_all = sorted(sc)
    seeds = [s for s in seeds_all if s not in exclude_seeds]
    if not seeds:
        return {"rule_A": None, "excluded_seeds": sorted(exclude_seeds), "n_seeds": 0,
                "note": "all seeds excluded (degraded)"}
    out = {"excluded_seeds": sorted(s for s in seeds_all if s in exclude_seeds)}
    for det in dets:
        per, draws = {}, {}
        for s in seeds:
            sid, so = sc[s][det]
            draws[s] = np.array([auroc(sid[id_draws[b]], so[oo_draws[b]]) for b in range(n_boot)])
            lo, hi = float(np.quantile(draws[s], 0.025)), float(np.quantile(draws[s], 0.975))
            per[str(s)] = {"point": auroc(sid, so), "ci_lo": lo, "ci_hi": hi, "below_chance_entire_ci": hi < 0.5}
        pooled = np.mean(np.stack([draws[s] for s in seeds]), 0)
        pts = [per[str(s)]["point"] for s in seeds]
        out[det] = {"mean": float(np.mean(pts)), "sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else None,
                    "ci_lo": float(np.quantile(pooled, 0.025)), "ci_hi": float(np.quantile(pooled, 0.975)),
                    "k_below": int(sum(p["below_chance_entire_ci"] for p in per.values())), "n_seeds": len(seeds),
                    "seeds": seeds, "per_seed": per}
    m = out["mahalanobis_classcond"]
    out["rule_A"] = bool(m["ci_hi"] < 0.5 and m["k_below"] >= 3)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------------------------------------------------
T1_COLS = ("leakage", "id_bal", "id_ece", "ood_ece", "maha", "fitz", "xdom")
T1_HEAD = ("Leakage", "ID bal acc", "ID ECE", "OOD ECE (pad_heldout)", "OOD AUROC (Maha {ood})", "Fitz AUROC (Maha)",
           "X-dom bal acc (pad_heldout, 6-cls)")


def ms(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "n": len(xs)}


def pm(d, nd=3):
    if d is None:
        return "—"
    return "{:.{n}f} ± {:.{n}f}".format(d["mean"], d["sd"], n=nd) if d.get("sd") is not None else "{:.{n}f}".format(d["mean"], n=nd)


def tex_pm(d, nd=3):
    if d is None:
        return "---"
    return ("{:.{n}f}\\,$\\pm$\\,{:.{n}f}".format(d["mean"], d["sd"], n=nd) if d.get("sd") is not None
            else "{:.{n}f}".format(d["mean"], n=nd))


def metrics_for(feat_dir, run_dir=None, site_test=None, ood_primary="pad_heldout"):
    """Read summary.json metrics if present (new runs), else compute from features (reference runs)."""
    if run_dir is not None and (Path(run_dir) / "summary.json").exists():
        s = json.loads((Path(run_dir) / "summary.json").read_text())
        if "ood" in s.get("metrics", {}):
            return s["metrics"], s
    Fs = load_feats(feat_dir)
    if "isic_test" not in Fs:
        return None, None
    return run_metrics(Fs, site_test=site_test), None


def cell_table(cells, ood_primary="pad_heldout"):
    """cells: list of (label, lam, {seed: metrics}). Returns markdown lines, LaTeX lines, aggregated dict."""
    head = [h.format(ood=ood_primary) for h in T1_HEAD]
    md = ["| cell | λ | n | " + " | ".join(head) + " |", "|---|---:|---:|" + "---|" * len(head)]
    tex, agg = [], {}
    for label, lam, runs in cells:
        cols = {k: ms([table1_cols(m, ood_primary)[k] for m in runs.values()]) for k in T1_COLS} if runs else {}
        agg[(label, lam)] = cols
        n = len(runs)
        md.append("| {} | {:g} | {} | ".format(label, lam, n) + " | ".join(pm(cols.get(k)) for k in T1_COLS) + " |")
        tex.append("{} & {:g} & {} & ".format(label, lam, n) + " & ".join(tex_pm(cols.get(k)) for k in T1_COLS) + " \\\\")
    return md, tex, agg


def detector_lines(cells, boots, ood_name):
    md = ["| cell | λ | " + " | ".join(DET_LABEL[d] for d in DETS) + " | Maha k<0.5 | rule A | NV-nearest |",
          "|---|---:|" + "---|" * (len(DETS) + 3)]
    for label, lam, runs in cells:
        b = boots.get((label, lam, ood_name))
        cells_txt = []
        for d in DETS:
            pt = ms([m["ood"][ood_name][d] for m in runs.values() if ood_name in m.get("ood", {})])
            ci = "" if b is None or d not in b else " [{:.3f}, {:.3f}]".format(b[d]["ci_lo"], b[d]["ci_hi"])
            cells_txt.append(pm(pt) + ci)
        nvn = ms([m["ood"][ood_name]["nv_nearest_frac"] for m in runs.values() if ood_name in m.get("ood", {})])
        md.append("| {} | {:g} | {} | {} | {} | {} |".format(
            label, lam, " | ".join(cells_txt),
            "—" if b is None else "{}/{}".format(b["mahalanobis_classcond"]["k_below"], b["mahalanobis_classcond"]["n_seeds"]),
            "—" if b is None else ("met" if b["rule_A"] else "not met"), pm(nvn)))
    return md


def decomposition_lines(cells, ood_name="pad_heldout"):
    md = ["| cell | λ | " + " | ".join("ID {}".format(c) for c in LABELS) + " | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |",
          "|---|---:|" + "---|" * (len(LABELS) + 4)]
    for label, lam, runs in cells:
        ds = [m["ood"][ood_name]["decomposition_maha"] for m in runs.values() if ood_name in m.get("ood", {})]
        if not ds:
            continue
        idc = [pm(ms([d["id_class_auroc"].get(c) for d in ds])) for c in LABELS]
        md.append("| {} | {:g} | {} | {} | {} | {} | {} |".format(
            label, lam, " | ".join(idc), pm(ms([d["reweighted_to_ood_mix"] for d in ds])),
            pm(ms([d["identity_sum"] for d in ds])), pm(ms([d["ood_nv_vs_rest"]["nv"] for d in ds])),
            pm(ms([d["ood_nv_vs_rest"]["non_nv"] for d in ds]))))
    return md


def summary_rows(item, cells, ood_primary="pad_heldout"):
    rows = []
    for label, lam, runs in cells:
        for s, m in runs.items():
            r = {"item": item, "cell": label, "lambda": lam, "seed": s, **table1_cols(m, ood_primary),
                 "leakage_classmatched": m.get("leakage_classmatched", {}).get("bal_acc_mean"),
                 "ak_recall_pad_heldout": m["xdom_pad_heldout"]["recall_6cls"]["AK"],
                 "nv_pred_share_pad_heldout": m["xdom_pad_heldout"]["pred_share_6cls"]["NV"],
                 "id_nv_nearest": m.get("id_nv_nearest_frac")}
            if "leakage_bcn_ham" in m:
                r["leakage_bcn_ham"] = m["leakage_bcn_ham"]["bal_acc_mean"]
            for o, b in m["ood"].items():
                for d in DETS:
                    r["{}_{}".format(o, d)] = b[d]
                r["{}_nv_nearest".format(o)] = b["nv_nearest_frac"]
            rows.append(r)
    return pd.DataFrame(rows)
