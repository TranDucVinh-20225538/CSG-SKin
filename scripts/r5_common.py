"""Paper B R5 shared data, evaluation, bootstrap and report helpers (Q1, P9, P10, Q3, P11).

Evaluation follows the Table 1 / R4 definitions:
  leakage      logistic domain probe on the scoring space, ISIC test vs PAD full, 70/30, probe seeds 42/52/62,
               balanced accuracy (phase15_common.leakage_probe); also ISIC test vs pad_heldout, and the selection
               probe ISIC val vs pad_adv (training-side images only; used for P11 weight selection)
  detectors    Mahalanobis class-conditional (shared covariance + 1e-3 I), kNN k=50 (L2-normalised, mean distance),
               cosine max, MSP, Energy T=1; fit on ISIC train only; higher score = more OOD
  ECE          15 bins (phase15_common.ece_from_logits); X-dom = pad_heldout six-class bal acc (six_class_xfer)
The Mahalanobis distance is computed vectorised in float64 from ood_metrics' means/precision (same values as
ood_metrics.mahalanobis_min_squared_distances, which loops in Python).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase3_sweep as p3  # noqa: E402,F401  (installs the transformers stub)
import train_phase2_pad_holdout as p2  # noqa: E402
import train_r2_item4_lesion_split as T4  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from eval_ood_dual_branch import cosine_ood_scores, fit_class_means, fit_knn, knn_scores  # noqa: E402
from phase15_common import assert_split_hygiene, ece_from_logits, leakage_probe, six_class_xfer  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402
from src.utils import ood_metrics  # noqa: E402

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
R5 = PAPERB / "results" / "paperB" / "r5"
R7 = PAPERB / "results" / "paperB" / "r7"
R7B = PAPERB / "results" / "paperB" / "r7b"
R8 = PAPERB / "results" / "paperB" / "r8"
PREREG = PAPERB / "results" / "paperB" / "PREREGISTER_BOOTSTRAP.json"
PAD_PAT = re.compile(r"PAT_(\d+)_(\d+)_(\d+)")
P7A_SETS = R5 / "p7a" / "sets"
EXTRA_SETS = ("mednode", "sd198", "scin")
LABELS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
NV = 1
BOOT_DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")
DETS = BOOT_DETS + ("MSP", "Energy_T1")
DET_LABEL = {"mahalanobis_classcond": "Mahalanobis", "knn_k50": "kNN k=50", "cosine_max": "Cosine", "MSP": "MSP",
             "Energy_T1": "Energy"}


def git_head():
    return subprocess.run(["git", "-C", str(PAPERB), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


# ----------------------------------------------------------------------------------------------------------------------
# ISIC <-> PAD data (lesion-level split, R2 item 4)
# ----------------------------------------------------------------------------------------------------------------------


def isic_lesion_data():
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy().reset_index(drop=True)
    pad = df[df.domain == "pad_ufes"].copy().reset_index(drop=True)
    train, val, test = T4.isic_splits_by_lesion(isic)
    spec = json.loads(p3.P2_SPLIT.read_text())
    pad_adv = pad[pad.path.isin(set(spec["pad_adv_paths"]))].reset_index(drop=True)
    pad_hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    hygiene = assert_split_hygiene(train.path, val.path, test.path, pad_adv.path, pad_hold.path)
    extra = {n: pd.read_csv(P7A_SETS / "{}.csv".format(n), keep_default_na=False)
             for n in EXTRA_SETS if (P7A_SETS / "{}.csv".format(n)).exists()}
    return {"train": train, "val": val, "test": test, "pad": pad, "pad_adv": pad_adv, "pad_hold": pad_hold,
            "fitz": sorted(p3.FITZ_DIR.glob("*.jpg")), "extra": extra, "hygiene": hygiene}


def subsample_pad_adv(pad_adv, frac, seed=0):
    """Patient-stratified subsample of pad_adv that preserves class mix (approx.).

    One fixed draw per (frac, seed); default seed 0 as in decisions/paperB_r7_t2.md.
    frac=1.0 returns a copy unchanged. pad_heldout is never touched here.
    The returned frame feeds both the adversarial and the context branch (same CombinedTrainDataset rows).
    """
    frac = float(frac)
    out = pad_adv.copy().reset_index(drop=True)
    if abs(frac - 1.0) < 1e-12:
        meta = {"frac": 1.0, "seed": int(seed), "n_images": int(len(out)), "n_patients": None,
                "class_counts": {LABELS[i]: int((out.label_idx == i).sum()) for i in range(8)}}
        return out, meta
    if not (0.0 < frac < 1.0):
        raise ValueError("pad_adv_frac must be in (0, 1], got {}".format(frac))
    pids = []
    for p in out.path.astype(str):
        m = PAD_PAT.search(p)
        if not m:
            raise RuntimeError("Cannot parse patient_id from {}".format(p))
        pids.append(m.group(1))
    out = out.assign(patient_id=pids)
    # patient-level majority class for stratification
    pat = (out.groupby("patient_id")
             .agg(label_idx=("label_idx", lambda s: int(s.mode().iloc[0])), n=("label_idx", "size"))
             .reset_index())
    rng = np.random.default_rng(int(seed))
    keep = []
    for lab, g in pat.groupby("label_idx"):
        n_keep = max(1, int(round(len(g) * frac))) if len(g) else 0
        n_keep = min(n_keep, len(g))
        chosen = rng.choice(g.patient_id.to_numpy(), size=n_keep, replace=False)
        keep.extend(chosen.tolist())
    sub = out[out.patient_id.isin(set(keep))].drop(columns=["patient_id"]).reset_index(drop=True)
    meta = {"frac": frac, "seed": int(seed), "n_images": int(len(sub)), "n_patients": int(len(keep)),
            "class_counts": {LABELS[i]: int((sub.label_idx == i).sum()) for i in range(8)},
            "full_class_counts": {LABELS[i]: int((out.label_idx == i).sum()) for i in range(8)},
            "patient_ids": sorted(keep)}
    return sub, meta


def class_weights(labels, n_classes):
    """Inverse class frequency, normalised so that sum_c n_c w_c = N (mean weight 1 over training rows)."""
    cnt = np.bincount(np.asarray(labels, int), minlength=n_classes).astype(np.float64)
    w = np.where(cnt > 0, cnt.sum() / (n_classes * np.maximum(cnt, 1)), 0.0)
    return w.tolist()


class ThreeStreamISIC(Dataset):
    """recipe class_balance = sampler (P8 'three' mode): index i -> a class-balanced ISIC classifier row (drawn with
    replacement, uniform over classes, no adversary), ISIC row i at the natural mix (adversary only, domain 0) and
    PAD row i mod n_pad (adversary only, domain 1). len = n_isic, so the step count equals the two-stream design."""

    def __init__(self, isic, pad, transform, seed):
        self.ip, self.iy = isic.path.tolist(), isic.label_idx.astype(int).to_numpy()
        self.pp, self.py = pad.path.tolist(), pad.label_idx.astype(int).to_numpy()
        self.tf = transform
        w = 1.0 / np.bincount(self.iy)[self.iy]
        rng = np.random.default_rng(int(seed) + 104729)
        self.cls_rows = rng.choice(len(self.iy), len(self.iy), replace=True, p=w / w.sum())

    def __len__(self):
        return len(self.ip)

    def _load(self, p):
        return self.tf(Image.open(p).convert("RGB"))

    def __getitem__(self, i):
        c, k = int(self.cls_rows[i]), i % len(self.pp)
        return ((self._load(self.ip[c]), int(self.iy[c]), 0), (self._load(self.ip[i]), int(self.iy[i]), 0),
                (self._load(self.pp[k]), int(self.py[k]), 1))


def three_stream_collate(batch):
    """-> (x, x, y_cls, y_true, domain, adv_mask); rows = [classifier stream, ISIC adversary stream, PAD stream]."""
    rows = [b[s] for s in range(3) for b in batch]
    x = torch.stack([r[0] for r in rows])
    y = torch.tensor([r[1] for r in rows], dtype=torch.long)
    d = torch.tensor([r[2] for r in rows], dtype=torch.long)
    n = len(batch)
    y_cls = torch.full_like(y, -1)
    y_cls[:n] = y[:n]
    m = torch.ones_like(d, dtype=torch.bool)
    m[:n] = False
    return x, x, y_cls, y, d, m


def isic_train_loader(D, class_balance, batch_size, workers, seed):
    """Two-stream CombinedTrainDataset (current) or ThreeStreamISIC (class_balance = sampler)."""
    from src.datasets.skin_dataset import (CombinedTrainDataset, build_lesion_branch_transform_gray,
                                           build_train_transform_robust, csg_lite_paired_collate)

    if class_balance == "sampler":
        ds = ThreeStreamISIC(D["train"], D["pad_adv"], build_train_transform_robust(), seed)
        return p2.make_loader(ds, batch_size, workers, True, drop_last=True, collate=three_stream_collate), "three"
    ds = CombinedTrainDataset(D["train"], D["pad_adv"], transform=build_train_transform_robust(),
                              lesion_transform=build_lesion_branch_transform_gray())
    return p2.make_loader(ds, batch_size, workers, True, drop_last=True, collate=csg_lite_paired_collate), "two"


# ----------------------------------------------------------------------------------------------------------------------
# Feature collection and detectors
# ----------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def collect(forward, loader, device):
    """forward(x) -> (logits, z) with z the scoring space."""
    zs, lg, ys = [], [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        logits, z = forward(x.to(device, non_blocking=True))
        zs.append(z.float().cpu().numpy())
        lg.append(logits.float().cpu().numpy())
        ys.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
    return {"z": np.concatenate(zs), "logits": np.concatenate(lg), "labels": np.concatenate(ys)}


def auroc(sid, sood, w_ood=None):
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    w = None if w_ood is None else np.r_[np.ones(len(sid)), w_ood]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=w))


class Detectors:
    """All five detectors fit once on train (z, y)."""

    def __init__(self, ztr, ytr, n_classes):
        self.n_classes = n_classes
        self.means, self.prec = ood_metrics.compute_mahalanobis_params_from_arrays(
            ztr, np.asarray(ytr, int), num_classes=n_classes, reg_eps=1e-3)
        self.cmeans, self.present = fit_class_means(ztr, np.asarray(ytr, int), n_classes)
        self.knn = fit_knn(ztr, k=50)

    def maha_all(self, z):
        z = np.asarray(z, np.float64)
        p, m = self.prec.astype(np.float64), self.means.astype(np.float64)
        zp = z @ p
        return np.maximum(np.einsum("nd,nd->n", zp, z)[:, None] + np.einsum("cd,dk,ck->c", m, p, m)[None, :]
                          - 2.0 * zp @ m.T, 0.0)

    def scores(self, z, logits):
        lg = torch.from_numpy(np.asarray(logits)).float()
        d2 = self.maha_all(z)
        return {"mahalanobis_classcond": d2.min(1),
                "knn_k50": knn_scores(self.knn, z),
                "cosine_max": cosine_ood_scores(z, self.cmeans, self.present),
                "MSP": -torch.softmax(lg, 1).numpy().max(1).astype(np.float64),
                "Energy_T1": -torch.logsumexp(lg, 1).numpy().astype(np.float64)}, d2.argmin(1)


def ood_block(det, sc_id, y_id, pack, ood_labels=None, label_names=LABELS, majority=NV):
    """AUROCs of all detectors, per-ID-class Mahalanobis AUROC, majority-nearest fraction, and (if OOD labels map to
    ID classes) per-OOD-class AUROC plus the decomposition check (OOD reweighted to the ID class mix)."""
    sc_o, nearest = det.scores(pack["z"], pack["logits"])
    out = {d: auroc(sc_id[d], sc_o[d]) for d in DETS}
    out["n_ood"] = int(len(pack["z"]))
    out["majority_nearest_frac"] = float(np.mean(nearest == majority))
    out["nearest_class_frac"] = {label_names[c]: float(np.mean(nearest == c)) for c in range(det.n_classes)}
    m_id, m_o = sc_id["mahalanobis_classcond"], sc_o["mahalanobis_classcond"]
    out["per_id_class_maha"] = {label_names[c]: (auroc(m_id[y_id == c], m_o) if np.any(y_id == c) else None)
                                for c in range(det.n_classes)}
    if ood_labels is not None:
        ol = np.asarray(ood_labels, int)
        mapped = ol >= 0
        pc = {label_names[c]: {"n": int((ol == c).sum()), "auroc": auroc(m_id, m_o[ol == c]) if np.any(ol == c) else None}
              for c in range(det.n_classes)}
        id_mix = np.bincount(y_id, minlength=det.n_classes) / len(y_id)
        cls = [c for c in range(det.n_classes) if (ol == c).sum() >= 20]
        dec = None
        if len(cls) >= 2:
            sub = np.isin(ol, cls)
            p_id = id_mix[cls] / id_mix[cls].sum()
            p_o = np.array([(ol == c).sum() for c in cls], float)
            p_o /= p_o.sum()
            w = np.zeros(len(ol))
            for c, a, b in zip(cls, p_id, p_o):
                w[ol == c] = a / b
            dec = {"classes": [label_names[c] for c in cls], "mapped_all": auroc(m_id, m_o[sub]),
                   "reweighted_to_id_mix": auroc(m_id, m_o[sub], w[sub])}
        out["per_ood_class_maha"] = pc
        out["decomposition"] = dec
        out["n_mapped"] = int(mapped.sum())
    return out, sc_o


def isic_extra_labels(frame):
    return np.array([LABELS.index(m) if m in LABELS else -1 for m in frame.mapped], int)


def eval_isic(forward, D, device, out_dir, workers=4, batch=64, n_classes=8, mode="full"):
    """Collect ISIC-task sets, write features/<set>.npz, scores.npz and return the summary block.

    mode='full': Table 1 OOD detectors (default).
    mode='d1': R8 D1 blind — save features (incl. OOD packs for D2) + ID metrics only; no OOD AUROC/scores.
    """
    tf = build_val_transform_robust()
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)

    def L(ds):
        return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)

    sets = {"isic_train": SkinDataset(D["train"], tf), "isic_val": SkinDataset(D["val"], tf),
            "isic_test": SkinDataset(D["test"], tf), "pad_full": SkinDataset(D["pad"], tf),
            "pad_adv": SkinDataset(D["pad_adv"], tf), "pad_heldout": SkinDataset(D["pad_hold"], tf),
            "fitzpatrick17k": FolderJpegDataset(D["fitz"], tf)}
    for n, fr in D["extra"].items():
        sets[n] = FolderJpegDataset(fr.path.tolist(), tf)
    P = {}
    for name, ds in sets.items():
        P[name] = collect(forward, L(ds), device)
        np.savez_compressed(out_dir / "features" / "{}.npz".format(name), **P[name])
        print("features", name, P[name]["z"].shape, flush=True)
    tr, te = P["isic_train"], P["isic_test"]
    y_id = te["labels"].astype(int)
    pred = te["logits"].argmax(1)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    id_ece, id_conf = ece_from_logits(te["logits"], y_id)
    h = P["pad_heldout"]
    hold_ece, hold_conf = ece_from_logits(h["logits"], h["labels"])
    va = P["isic_val"]
    summ = {"val_balanced_acc": float(balanced_accuracy_score(va["labels"].astype(int), va["logits"].argmax(1))),
            "id_acc": float(accuracy_score(y_id, pred)), "id_balanced_acc": float(balanced_accuracy_score(y_id, pred)),
            "id_ece": id_ece, "id_mean_confidence": id_conf, "ood_ece_pad_heldout": hold_ece,
            "ood_mean_confidence_pad_heldout": hold_conf,
            "xfer_pad_heldout_6class": six_class_xfer(h["logits"], h["labels"]),
            "id_class_mix": {LABELS[c]: int((y_id == c).sum()) for c in range(n_classes)}, "ood": {},
            "eval_mode": mode}
    if mode == "d1":
        # Minimal leakage stub for adversary_gate; V1 comes from r5_p9v2.heldout_probe afterwards.
        summ["leakage"] = {"test_vs_pad_full": {"bal_acc_mean": None, "note": "d1_blind_skipped"},
                           "primary": "d1_blind: use heldout_probe.json V1", "z_dim": int(te["z"].shape[1])}
        print("eval_mode=d1: skipped OOD detectors; features saved for D2", flush=True)
        return summ

    det = Detectors(tr["z"], tr["labels"], n_classes)
    sc_id, _ = det.scores(te["z"], te["logits"])

    def probe(a, b):
        return leakage_probe(np.concatenate([a["z"], b["z"]]), np.r_[np.zeros(len(a["z"])), np.ones(len(b["z"]))])

    leak = {"test_vs_pad_full": probe(te, P["pad_full"]), "test_vs_pad_heldout": probe(te, h),
            "val_vs_pad_adv_selection": probe(P["isic_val"], P["pad_adv"]),
            "primary": "test_vs_pad_full (Table 1 protocol)", "z_dim": int(te["z"].shape[1])}
    if te["z"].shape[1] > 16:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=16, random_state=0).fit(tr["z"])
        leak["test_vs_pad_full_pca16"] = probe({"z": pca.transform(te["z"])}, {"z": pca.transform(P["pad_full"]["z"])})
        leak["pca16_note"] = "probe on the 16 leading ISIC-train principal components (dimension-matched to z_lesion_norm)"
    summ["leakage"] = leak
    scores = {"id__{}".format(k): v for k, v in sc_id.items()}
    for name in ("pad_heldout", "pad_full", "fitzpatrick17k") + tuple(D["extra"]):
        if name in ("pad_heldout", "pad_full"):
            ol = P[name]["labels"].astype(int)
        elif name in D["extra"]:
            ol = isic_extra_labels(D["extra"][name])
        else:
            ol = None
        blk, sc_o = ood_block(det, sc_id, y_id, P[name], ol)
        summ["ood"][name] = blk
        scores.update({"{}__{}".format(name, k): v for k, v in sc_o.items()})
        print(name, "maha {:.3f} knn {:.3f} cos {:.3f}".format(blk["mahalanobis_classcond"], blk["knn_k50"],
                                                              blk["cosine_max"]), flush=True)
    np.savez_compressed(out_dir / "scores.npz", **scores)
    return summ


# ----------------------------------------------------------------------------------------------------------------------
# Bootstrap (PREREGISTER_BOOTSTRAP.json) and rule A
# ----------------------------------------------------------------------------------------------------------------------


def idx_groups(g):
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


def _fast_auroc(sid, sood):
    s = np.r_[sid, sood]
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = len(sood), len(sid)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def bootstrap_cell(per_seed, id_groups, ood_groups, rng, n_boot):
    """per_seed: {seed: (sid, sood)}. Draws shared across seeds; ID and OOD resampled independently by group.
    Returns per-seed point/CI, pooled mean-over-seeds CI, k seeds whose CI is entirely below 0.5, and which
    PREREGISTER rule (A/B/C/above) the pooled CI meets."""
    gi = [id_groups[j] for j in range(len(id_groups))]
    go = [ood_groups[j] for j in range(len(ood_groups))]
    di = [np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))]) for _ in range(n_boot)]
    do = [np.concatenate([go[j] for j in rng.integers(0, len(go), len(go))]) for _ in range(n_boot)]
    draws, per = {}, {}
    for s, (sid, so) in per_seed.items():
        draws[s] = np.array([_fast_auroc(sid[di[b]], so[do[b]]) for b in range(n_boot)])
        lo, hi = np.quantile(draws[s], [0.025, 0.975])
        per[str(s)] = {"point": auroc(sid, so), "ci_lo": float(lo), "ci_hi": float(hi),
                       "ci_entirely_below_0.5": bool(hi < 0.5)}
    pooled = np.mean(np.stack(list(draws.values())), 0)
    lo, hi = np.quantile(pooled, [0.025, 0.975])
    k = int(sum(p["ci_entirely_below_0.5"] for p in per.values()))
    rule = ("A" if k >= 3 else "B") if hi < 0.5 else ("C" if lo < 0.5 else "above")
    return {"mean": float(np.mean([p["point"] for p in per.values()])),
            "sd": float(np.std([p["point"] for p in per.values()], ddof=1)) if len(per) > 1 else 0.0,
            "pooled_ci": [float(lo), float(hi)], "k_below": k, "n_seeds": len(per), "rule": rule, "per_seed": per}


def prereg():
    p = json.loads(PREREG.read_text())
    return int(p["n_resamples"]), int(p["rng_seed"])


def isic_test_groups():
    _rec, (_tr, _va, test) = T4.split_record()
    g, _ = T4.lesion_groups(test)
    return idx_groups(g)


def pad_heldout_groups():
    import r2_common as C

    return idx_groups(C.pad_heldout_meta().patient_id.astype(str).to_numpy())


# ----------------------------------------------------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------------------------------------------------


def ms(xs, nd=3):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return "—"
    if a.size == 1:
        return "{:.{}f}".format(a[0], nd)
    return "{:.{}f} ± {:.{}f}".format(a.mean(), nd, a.std(ddof=1), nd)


def tex_ms(xs, nd=3):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return "--"
    if a.size == 1:
        return "{:.{}f}".format(a[0], nd)
    return "{:.{}f}\\,$\\pm$\\,{:.{}f}".format(a.mean(), nd, a.std(ddof=1), nd)


def table1_values(s):
    """Table 1 columns from one ISIC-task summary.json (R4 order)."""
    return {"leakage": s["leakage"]["test_vs_pad_full"]["bal_acc_mean"], "id_bal_acc": s["id_balanced_acc"],
            "id_ece": s["id_ece"], "ood_ece": s["ood_ece_pad_heldout"],
            "ood_auroc": s["ood"]["pad_heldout"]["mahalanobis_classcond"],
            "fitz_auroc": s["ood"]["fitzpatrick17k"]["mahalanobis_classcond"],
            "xdom_bal_acc": s["xfer_pad_heldout_6class"]["balanced_accuracy"]}


ITEM4_RUNS = PAPERB / "results" / "paperB" / "r2" / "item4" / "runs"


def item4_reference():
    """Published lesion-level dual-encoder EfficientNet-B3 GRL runs (R2 item 4, Table 1 / R4): one row per run with the
    columns stored at training time (leakage = z_lesion probe ISIC test vs PAD full; Mahalanobis unrestricted)."""
    rows = []
    for f in sorted(ITEM4_RUNS.glob("runB_orth1_ladv*_s*/summary.json")):
        s = json.loads(f.read_text())
        o = s["ood"]
        rows.append({"tag": s["run_name"], "lambda_adv": float(s["lambda_adv"]), "seed": int(s["seed"]),
                     "leakage": s["leakage"]["z_lesion"]["bal_acc_mean"], "id_bal_acc": s["id_balanced_acc"],
                     "id_ece": s["id_ece"],
                     "ood_auroc": o["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"],
                     "fitz_auroc": o["fitzpatrick17k"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"],
                     "knn_pad_heldout": o["pad_heldout"]["z_lesion"]["knn_k50"],
                     "run_dir": str(f.parent)})
    return pd.DataFrame(rows)


TABLE1_COLS = ("leakage", "id_bal_acc", "id_ece", "ood_ece", "ood_auroc", "fitz_auroc", "xdom_bal_acc")
TABLE1_HEAD = ("Leakage", "ID bal acc", "ID ECE", "OOD ECE (pad_heldout)", "OOD AUROC (Maha pad_heldout)",
               "Fitz AUROC (Maha)", "X-dom bal acc")


def isic_bootstrap(groups_by_run, ood_sets=("pad_heldout", "fitzpatrick17k"), dets=BOOT_DETS):
    """groups_by_run: {cell_key: {seed: run_dir}} -> {cell_key: {ood: {det: bootstrap_cell}}} for dets.
    ID by lesion group, pad_heldout by patient, Fitzpatrick17k and extra sets by image (MED-NODE/SD-198) or case
    (SCIN, group column)."""
    n_boot, seed = prereg()
    gid = isic_test_groups()
    gpad = pad_heldout_groups()
    out = {}
    for key, runs in groups_by_run.items():
        rng = np.random.default_rng(seed)
        S = {s: dict(np.load(Path(d) / "scores.npz")) for s, d in runs.items() if (Path(d) / "scores.npz").exists()}
        if not S:
            continue
        blk = {}
        for ood in ood_sets:
            if not all("{}__mahalanobis_classcond".format(ood) in v for v in S.values()):
                continue
            n_o = len(next(iter(S.values()))["{}__mahalanobis_classcond".format(ood)])
            if ood == "pad_heldout":
                go = gpad
            elif ood == "fitzpatrick17k":
                p13 = R5 / "p13" / "fitz_clusters.csv"
                if p13.exists():
                    paths = sorted(p3.FITZ_DIR.glob("*.jpg"))
                    cl = pd.read_csv(p13).set_index("md5hash")["cluster_union"]
                    go = idx_groups([str(cl[p.stem]) for p in paths])
                else:
                    go = [np.array([i]) for i in range(n_o)]
            elif ood in EXTRA_SETS:
                go = idx_groups(pd.read_csv(P7A_SETS / "{}.csv".format(ood), keep_default_na=False).group.astype(str))
            else:
                go = [np.array([i]) for i in range(n_o)]
            blk[ood] = {d: bootstrap_cell({s: (v["id__" + d], v["{}__{}".format(ood, d)]) for s, v in S.items()},
                                          gid, go, rng, n_boot) for d in dets}
        out[key] = blk
    return out
