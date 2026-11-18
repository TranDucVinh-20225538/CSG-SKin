#!/usr/bin/env python3
"""Paper B R14 pair3 shared helpers — ISIC 2019 → MILK10k dermoscopic.

Decisions: decisions/paperB_r14_pair3.md
Does not write into r12_reweight/ or r13_pair2/.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import r6_common as R  # noqa: F401 — scripts path inserted by caller

ROOT = R.PAPERB
DATA = ROOT / "data" / "r14_pair3"
PROC = DATA / "processed"
RAW = DATA / "raw"
DERM_DIR = DATA / "dermoscopic"
OUT = ROOT / "results" / "paperB" / "r14_pair3"
CKPT = ROOT / "checkpoints" / "r14_pair3"
CACHE_API = OUT / "cache" / "isic_api"
MANIFEST = OUT / "DATA_MANIFEST.json"
STATUS = OUT / "STATUS.md"

LABELS = list(R.LABELS)  # MEL NV BCC AK BKL DF VASC SCC
NCLS = len(LABELS)
SEEDS = (42, 52, 62, 72, 82)
PRIORS = ("prior100", "prior50", "prior0")
PRIOR_SHIFT = {"prior100": 1.0, "prior50": 0.5, "prior0": 0.0}
INVARIANCE_V1_MAX = 0.60
ERM_ID_BAL = 0.558  # same ISIC split
OOD_LABELLED_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")
OOD_REPORT_ONLY = ("milk_heldout", "pad_ufes20", "scin")
OOD_ALL = ("milk_heldout", "pad_ufes20", "fitzpatrick17k", "mednode", "sd198", "scin")

# training_gt one-hot → Paper B label (drop INF/BEN_OTH/MAL_OTH)
GT_MAP = {
    "BCC": "BCC",
    "NV": "NV",
    "MEL": "MEL",
    "BKL": "BKL",
    "DF": "DF",
    "VASC": "VASC",
    "SCCKA": "SCC",
    "AKIEC": "AK",
}
GT_DROP = ("INF", "BEN_OTH", "MAL_OTH")

ISIC_GT = Path("/data2/hpcshared/Vinh/CSG-Skin/data/ISIC_2019_Training_Groundtruth.csv")
ISIC_META = Path("/data2/hpcshared/Vinh/CSG-Skin/data/ISIC_2019_Training_Metadata.csv")
MILK_ZIP_SRC = ROOT / "data" / "r13_pair2" / "milk10k" / "raw" / "milk10k.zip"
MILK_CSV_SRC = ROOT / "data" / "r13_pair2" / "milk10k" / "raw" / "milk10k.csv"

S3_PREFIX = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456"
API_LESION = "https://api.isic-archive.com/api/v2/lesions/{}/"
USER_AGENT = "python-requests"
API_RATE = 2.0  # req/s


def git_head():
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def lam_tag(lam):
    return "{:g}".format(float(lam)).replace(".", "p")


def tag_for(lam, prior, seed):
    if float(lam) == 0.0:
        return "r14_ladv0_prior100_s{}".format(int(seed))
    return "r14_ladv2_{}_s{}".format(prior, int(seed))


def jobs_main():
    """20 training jobs."""
    out = []
    for seed in SEEDS:
        out.append({"lam": 0.0, "prior": "prior100", "seed": seed, "tag": tag_for(0.0, "prior100", seed)})
    for prior in PRIORS:
        for seed in SEEDS:
            out.append({"lam": 2.0, "prior": prior, "seed": seed, "tag": tag_for(2.0, prior, seed)})
    return out


def jobs_stage1():
    """Early-abort stage: λ0 s42, λ2 prior100 s42/s52; optional prior0 s42."""
    return [
        {"lam": 0.0, "prior": "prior100", "seed": 42, "tag": tag_for(0.0, "prior100", 42)},
        {"lam": 2.0, "prior": "prior100", "seed": 42, "tag": tag_for(2.0, "prior100", 42)},
        {"lam": 2.0, "prior": "prior100", "seed": 52, "tag": tag_for(2.0, "prior100", 52)},
        {"lam": 2.0, "prior": "prior0", "seed": 42, "tag": tag_for(2.0, "prior0", 42)},
    ]


def class_mix(labels):
    y = np.asarray(labels, int)
    return np.bincount(y, minlength=NCLS).astype(float) / max(len(y), 1)


def dose_mix(shift, src_mix, tgt_mix):
    m = float(shift) * np.asarray(src_mix, float) + (1.0 - float(shift)) * np.asarray(tgt_mix, float)
    return m / m.sum()


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def load_frame(name: str) -> pd.DataFrame:
    return pd.read_csv(PROC / "{}.csv".format(name))


def load_data():
    """ISIC frames from main recipe + milk_adv/heldout; pad_* aliases for stream builders."""
    isic = R.load_data()
    milk_adv = load_frame("milk_adv")
    milk_heldout = load_frame("milk_heldout")
    milk_full = pd.concat([milk_adv, milk_heldout], ignore_index=True)
    return {
        "train": isic["train"], "val": isic["val"], "test": isic["test"],
        "pad_full": isic["pad_full"], "pad_adv": milk_adv, "pad_heldout": milk_heldout,
        "milk_adv": milk_adv, "milk_heldout": milk_heldout, "milk_full": milk_full,
        # keep original PAD for scoring as unseen
        "pad_ufes20": isic["pad_full"],
    }


def compute_priors(data=None):
    data = data or load_data()
    isic = class_mix(data["train"].label_idx)
    milk = class_mix(data["milk_adv"].label_idx)
    return {
        "prior100": np.asarray(isic, float),
        "prior50": dose_mix(0.5, isic, milk),
        "prior0": np.asarray(milk, float),
        "pi_source": np.asarray(isic, float),
        "pi_milk_adv": np.asarray(milk, float),
        "labels": LABELS,
        "note": "pi_milk from milk_adv labels at run start only; never in task loss",
    }


def write_status(text: str):
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(text if text.endswith("\n") else text + "\n")


def isic_image_set():
    """Set of ISIC 2019 training image stems (no .jpg)."""
    gt = pd.read_csv(ISIC_GT)
    col = "image" if "image" in gt.columns else gt.columns[0]
    return set(gt[col].astype(str).str.replace(r"\.jpg$", "", regex=True))


def isic_lesion_map():
    meta = pd.read_csv(ISIC_META, usecols=["image", "lesion_id"])
    meta["image"] = meta["image"].astype(str).str.replace(r"\.jpg$", "", regex=True)
    return dict(zip(meta.image, meta.lesion_id.astype(str)))
