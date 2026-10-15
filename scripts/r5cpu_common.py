"""Shared helpers for the Paper B R5/R6 CPU items (P1–P6, P12, P13, Q4, Q5, R6 R3–R5).

Cached features of the 35 lesion-level runs: results/paperB/r2/item4/features/<tag>/<split>.npz with keys
backbone_raw_lesion, z_lesion, z_lesion_norm, z_context, logits, labels. Detector definitions follow
bootstrap_ood_ci / phase15_common.detector_block (fit on ISIC train only); bootstrap follows PREREGISTER_BOOTSTRAP.json
with the ID side resampled by lesion group (R3 A2 convention), pad_heldout by patient, Fitzpatrick by image
(or by a P13 near-duplicate cluster column).
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
import train_r2_item4_lesion_split as T  # noqa: E402  (imports train_phase3_sweep first: transformers stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import bootstrap_ood_ci as B  # noqa: E402
import r2_common as C  # noqa: E402
from phase15_common import PAD_CLASSES  # noqa: E402

PAPERB = C.PAPERB
RES = PAPERB / "results" / "paperB"
R5 = RES / "r5"
R6 = RES / "r6"
FEAT_L = C.R2 / "item4" / "features"
FEAT_I = RES / "phase13" / "features"
LAMS = C.LAMS
LABELS = C.LABELS
NV, AK = 1, 3
BOOT_DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")
DETS = BOOT_DETS + ("MSP", "Energy_T1")
DET_LABEL = {"mahalanobis_classcond": "Mahalanobis", "knn_k50": "kNN k=50", "cosine_max": "Cosine", "MSP": "MSP",
             "Energy_T1": "Energy"}
PRIMARY_REP = "z_lesion_norm"
P13_CLUSTERS = R5 / "p13" / "fitz_clusters.csv"
SPLITS = ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k")


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(PAPERB), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_tag(lam, seed):
    return "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), seed)


def runs(feat_root=FEAT_L, need=SPLITS, pattern=r"runB_orth1_ladv([0-9p]+)_s(\d+)"):
    """[(lambda, seed, tag)] for every run directory holding all `need` splits."""
    out = []
    for d in sorted(Path(feat_root).glob("*")):
        m = re.fullmatch(pattern, d.name)
        if m and all((d / "{}.npz".format(s)).exists() for s in need):
            out.append((float(m.group(1).replace("p", ".")), int(m.group(2)), d.name))
    return sorted(out)


def load(tag, split, feat_root=FEAT_L):
    return dict(np.load(Path(feat_root) / tag / "{}.npz".format(split)))


@functools.lru_cache(maxsize=1)
def lesion_frames():
    """(train, val, test) frames of the lesion-level ISIC split (metadata only)."""
    _rec, frames = T.split_record()
    return frames


def idx_groups(g):
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


@functools.lru_cache(maxsize=1)
def _id_test_groups():
    _tr, _va, test = lesion_frames()
    g, _null = T.lesion_groups(test)
    return np.asarray(g).astype(str)


def id_test_groups():
    return _id_test_groups().copy()


@functools.lru_cache(maxsize=1)
def _pad_patients():
    hold = C.pad_heldout_meta()
    assert len(hold) == 716
    return hold.patient_id.astype(str).to_numpy()


def pad_heldout_patients():
    return _pad_patients().copy()


def fitz_stems():
    return [p.stem for p in sorted(T.p3.FITZ_DIR.glob("*.jpg"))]


def fitz_groups(kind="image"):
    stems = fitz_stems()
    if kind == "image":
        return np.asarray(stems)
    cl = pd.read_csv(P13_CLUSTERS, dtype=str).set_index("md5hash")
    return cl.loc[stems, kind].astype(str).to_numpy()


def ood_groups(o, kind="image"):
    if o == "pad_heldout":
        return pad_heldout_patients()
    if o == "fitzpatrick17k":
        return fitz_groups(kind)
    raise KeyError(o)


def _fast_auroc(sid, sood):
    r = rankdata(np.concatenate([sid, sood]))
    n0, n1 = len(sid), len(sood)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def auroc(sid, sood, w_id=None, w_ood=None):
    """AUROC with OOD as positive class and larger score = more OOD. Optional per-sample weights."""
    sid, sood = np.asarray(sid, np.float64), np.asarray(sood, np.float64)
    if len(sid) == 0 or len(sood) == 0:
        return float("nan")
    if w_id is None and w_ood is None:
        return _fast_auroc(sid, sood)
    w = np.r_[np.ones(len(sid)) if w_id is None else w_id, np.ones(len(sood)) if w_ood is None else w_ood]
    y = np.r_[np.zeros(len(sid)), np.ones(len(sood))]
    return float(roc_auc_score(y, np.r_[sid, sood], sample_weight=w))


def softmax(logits):
    x = np.asarray(logits, np.float64)
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def detector_scores(tr, te, oods, rep=PRIMARY_REP):
    """Five detectors fit on ISIC train. Returns ({det: (s_id, {ood: s_ood})}, info)."""
    ztr, ytr, zte = tr[rep], tr["labels"].astype(int), te[rep]
    out = {}
    for det in BOOT_DETS:
        so = {}
        for o, f in oods.items():
            if det == "knn_k50":
                sid, so[o] = B.knn50_scores(ztr, zte, f[rep])
            else:
                sid, so[o] = B.DET_FNS[det](ztr, ytr, zte, f[rep])
        out[det] = (sid, so)
    lse = lambda L: np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)  # noqa: E731
    out["MSP"] = (-softmax(te["logits"]).max(1), {o: -softmax(f["logits"]).max(1) for o, f in oods.items()})
    out["Energy_T1"] = (-lse(np.asarray(te["logits"], np.float64)),
                        {o: -lse(np.asarray(f["logits"], np.float64)) for o, f in oods.items()})
    return out, {"fit": "ISIC train only", "rep": rep}


def ece15(logits, labels, restrict=None):
    """15-bin equal-width ECE (phase15_common.ece_from_logits); `restrict` = class columns for a k-way softmax."""
    L, y = np.asarray(logits, np.float64), np.asarray(labels).astype(int)
    if restrict is not None:
        cols = np.asarray(restrict, int)
        keep = np.isin(y, cols)
        P = softmax(L[keep][:, cols])
        pred, y = cols[P.argmax(1)], y[keep]
    else:
        P = softmax(L)
        pred = P.argmax(1)
    conf, correct = P.max(1), (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def prereg():
    return json.loads(B.PREREG.read_text())


def draws(groups, n, rng):
    """n bootstrap index draws resampling whole groups with replacement."""
    gs = idx_groups(groups)
    if all(len(g) == 1 for g in gs):
        flat = np.concatenate(gs)
        return [flat[rng.integers(0, len(flat), len(flat))] for _ in range(n)]
    return [np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))]) for _ in range(n)]


def pooled_bootstrap(per_seed, id_draws, ood_draws):
    """per_seed {seed: (s_id, s_ood)}; shared draws across seeds within a draw (PREREGISTER protocol).
    Returns mean/sd over seeds, percentile 95% CI of the seed-mean, per-seed CIs, k seeds with CI entirely < 0.5,
    rule_A (pooled CI < 0.5 and k >= 3)."""
    seeds = sorted(per_seed)
    pts = {s: auroc(*per_seed[s]) for s in seeds}
    mat = np.full((len(id_draws), len(seeds)), np.nan)
    for b, (ii, oo) in enumerate(zip(id_draws, ood_draws)):
        for j, s in enumerate(seeds):
            sid, so = per_seed[s]
            mat[b, j] = _fast_auroc(sid[ii], so[oo])
    pooled = np.nanmean(mat, 1)
    lo, hi = np.nanpercentile(pooled, [2.5, 97.5])
    per = {}
    for j, s in enumerate(seeds):
        a, c = np.nanpercentile(mat[:, j], [2.5, 97.5])
        per[int(s)] = {"point": pts[s], "lo": float(a), "hi": float(c), "ci_below_0.5": bool(c < 0.5)}
    k = int(sum(v["ci_below_0.5"] for v in per.values()))
    v = np.array([pts[s] for s in seeds])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(seeds),
            "pooled_ci_lo": float(lo), "pooled_ci_hi": float(hi), "k_seeds_ci_below_0.5": k,
            "rule_A": bool(hi < 0.5 and k >= 3), "per_seed": per}


def cluster_ci(sid, gid, sood, good, n=None, seed=None):
    """Percentile 95% CI of one AUROC, ID and OOD sides resampled by their groups."""
    pre = prereg()
    n = int(pre["n_resamples"]) if n is None else n
    rng = np.random.default_rng(int(pre["rng_seed"]) if seed is None else seed)
    di, do = draws(gid, n, rng), draws(good, n, rng)
    sid, sood = np.asarray(sid, np.float64), np.asarray(sood, np.float64)
    v = np.array([_fast_auroc(sid[a], sood[b]) for a, b in zip(di, do)])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def mean_sd(xs):
    return C.mean_sd(xs)


def fmt(m, s=None, nd=3):
    return C.fmt(m, s, nd)


def tex_pm(m, s=None, nd=3):
    if m is None or not np.isfinite(m):
        return "--"
    if s is None or not np.isfinite(s):
        return "{:.{}f}".format(m, nd)
    return "{:.{}f}\\,$\\pm$\\,{:.{}f}".format(m, nd, s, nd)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2, allow_nan=True))


def by_lambda(rows, key):
    """{lambda: (mean, sd, n)} of rows[key]."""
    df = pd.DataFrame(rows)
    return {float(l): mean_sd(d[key].tolist()) for l, d in df.groupby("lambda")}
