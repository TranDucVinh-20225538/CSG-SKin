"""Reviewer R2 shared helpers. Scoring definitions match the paper (Phase 3 / Phase 13)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
R2 = PAPERB / "results" / "paperB" / "r2"
FEAT13 = PAPERB / "results" / "paperB" / "phase13" / "features"
ISIC_SPLIT_CSV = R2 / "isic_image_split_with_lesion.csv"
PAD_HOLD_CSV = R2 / "pad_heldout_meta.csv"
P2_SPLIT = PAPERB / "results" / "paperB" / "phase2_pad_holdout" / "pad_patient_split_paths.json"
ISIC_META = Path("/data2/hpcshared/Vinh/CSG-Skin/data/ISIC_2019_Training_Metadata.csv")
LABELS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
NV = 1
PAD_CLASSES = np.array([0, 1, 2, 3, 4, 7], dtype=int)
MAHA_EPS = 1e-3
KNN_K = 50
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PAT = re.compile(r"PAT_(\d+)_(\d+)_(\d+)")


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def git_head():
    return subprocess.run(
        ["git", "-C", str(PAPERB), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def runs_with_features():
    out = []
    for d in sorted(FEAT13.glob("runB_orth1_ladv*_s*")):
        m = re.fullmatch(r"runB_orth1_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        if not all((d / "{}.npz".format(s)).exists() for s in ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k")):
            continue
        out.append((float(m.group(1).replace("p", ".")), int(m.group(2)), d.name))
    return sorted(out)


def load13(tag, split):
    return dict(np.load(FEAT13 / tag / "{}.npz".format(split)))


def auroc(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def mean_sd(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def fmt(m, s=None, nd=3):
    if m is None or not np.isfinite(m):
        return "—"
    if s is None:
        return "{:.{}f}".format(m, nd)
    return "{:.{}f} ± {:.{}f}".format(m, nd, s, nd)


class Maha:
    """Class-conditional means, shared pooled within-class covariance + eps*I (paper standard)."""

    def __init__(self, ztr, ytr, n_classes=8, eps=MAHA_EPS):
        z = np.asarray(ztr, np.float64)
        y = np.asarray(ytr, int)
        self.means = np.stack([z[y == c].mean(0) for c in range(n_classes)])
        cen = z - self.means[y]
        cov = cen.T @ cen / max(len(z) - n_classes, 1) + eps * np.eye(z.shape[1])
        self.prec = np.linalg.inv(cov)

    def dists(self, z):
        z = np.asarray(z, np.float64)
        zp = z @ self.prec
        zpz = np.einsum("nd,nd->n", zp, z)
        mp = self.means @ self.prec
        mpm = np.einsum("cd,cd->c", mp, self.means)
        return np.maximum(zpz[:, None] + mpm[None, :] - 2.0 * zp @ self.means.T, 0.0)

    def score(self, z):
        return self.dists(z).min(1)


class KNN:
    """k-NN on L2-normalised features, score = mean distance to k neighbours (paper standard)."""

    def __init__(self, ztr, k=KNN_K):
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=-1)
        self.nn.fit(self._n(ztr))

    @staticmethod
    def _n(z):
        z = np.asarray(z, np.float32)
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    def score(self, z):
        out = []
        zn = self._n(z)
        for i in range(0, len(zn), 2048):
            d, _ = self.nn.kneighbors(zn[i : i + 2048])
            out.append(d.mean(1))
        return np.concatenate(out)


def isic_split_table():
    return pd.read_csv(ISIC_SPLIT_CSV)


def isic_test_groups():
    """Return (lesion-disjoint mask over isic_test rows, group ids). Null lesion_id → own group."""
    s = isic_split_table()
    te = s[s.split == "test"].sort_values("row").reset_index(drop=True)
    ref = set(s[s.split != "test"].lesion_id.dropna())
    disjoint = ~te.lesion_id.isin(ref).to_numpy()
    groups = te.lesion_id.fillna(te.image).to_numpy()
    return disjoint, groups, te


def pad_heldout_meta():
    if PAD_HOLD_CSV.exists():
        return pd.read_csv(PAD_HOLD_CSV, dtype={"patient_id": str, "lesion_id": str})
    import sys

    sys.path.insert(0, str(PAPERB / "scripts"))
    import train_phase2_pad_holdout as p2

    spec = json.loads(P2_SPLIT.read_text())
    df = p2.load_master()
    pad = df[df.domain == "pad_ufes"].copy()
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    hold["patient_id"] = [PAT.search(p).group(1) for p in hold.path]
    hold["lesion_id"] = [PAT.search(p).group(2) for p in hold.path]
    R2.mkdir(parents=True, exist_ok=True)
    hold[["path", "label", "label_idx", "patient_id", "lesion_id"]].to_csv(PAD_HOLD_CSV, index=False)
    return hold


def cluster_bootstrap_auroc(sid, gid, sood, good, n_boot=2000, seed=0):
    """Resample ID groups and OOD groups independently; group = lesion (ID) / patient or image (OOD)."""
    rng = np.random.default_rng(seed)

    def idx_by_group(g):
        _, inv = np.unique(g, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        return np.split(order, bounds)

    gi, go = idx_by_group(gid), idx_by_group(good)
    vals = []
    for _ in range(n_boot):
        a = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
        b = np.concatenate([go[j] for j in rng.integers(0, len(go), len(go))])
        vals.append(auroc(sid[a], sood[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
