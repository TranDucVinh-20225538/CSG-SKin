#!/usr/bin/env python3
"""Paper B R12 shared helpers: priors, weights, job tables, paths.

Decisions: decisions/paperB_r12_reweight.md
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

import r6_common as R  # noqa: F401 — caller inserts scripts on path first

OUT = R.RES / "r12_reweight"
CKPT = R.CKPT / "r12_reweight"
SEEDS = R.SEEDS
LAMBDAS_A = (0.5, 2.0)
PRIORS = ("prior100", "prior50", "prior0")
PRIOR_SHIFT = {"prior100": 1.0, "prior50": 0.5, "prior0": 0.0}
OOD_PRIMARY_UNSEEN = ("fitzpatrick17k", "mednode", "sd198")
OOD_ALL = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
ID_CLASSES = list(R.LABELS)  # fixed 8-class list for stratified macro


def git_head():
    return subprocess.run(
        ["git", "-C", str(R.PAPERB), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def compute_priors(data=None):
    data = data or R.load_data()
    isic = R.class_mix(data["train"].label_idx)
    pad = R.class_mix(data["pad_adv"].label_idx)
    return {
        "prior100": np.asarray(isic, float),
        "prior50": R.dose_mix(0.5, isic, pad),
        "prior0": np.asarray(pad, float),
        "pi_source": np.asarray(isic, float),
        "pi_pad_adv": np.asarray(pad, float),
    }


def class_weights(pi_req, pi_src, eps=1e-12):
    """w(c)=π_req/π_src; 0 if π_req=0. Invalid if π_req>0 and π_src≈0.
    Mean over source stream is Σ π_src·w = Σ π_req = 1 when support is OK."""
    pi_req = np.asarray(pi_req, float)
    pi_src = np.asarray(pi_src, float)
    invalid = []
    w = np.zeros_like(pi_req)
    for c in range(len(pi_req)):
        if pi_req[c] <= 0:
            w[c] = 0.0
        elif pi_src[c] <= eps:
            invalid.append(int(c))
        else:
            w[c] = pi_req[c] / pi_src[c]
    # normalize mean over source stream to 1 (should already be ~1)
    mean_w = float((pi_src * w).sum())
    if mean_w > eps and not invalid:
        w = w / mean_w
        mean_w = 1.0
    return w, invalid, mean_w


def tag_a(lam, prior, seed):
    return "A_ladv{}_{}_s{}".format(R.lam_tag(lam), prior, int(seed))


def tag_a0(seed):
    return "A_ladv0_baseline_s{}".format(int(seed))


def tag_b(lam, seed, init):
    return "B_ladv{}_init{}_s{}".format(R.lam_tag(lam), init, int(seed))


def jobs_block_a():
    """35 training jobs: 5 λ=0 baselines + 30 λ×prior×seed."""
    out = []
    for seed in SEEDS:
        out.append({"block": "A", "lam": 0.0, "prior": "prior100", "seed": seed,
                    "tag": tag_a0(seed), "role": "baseline", "reference_of": None})
    for lam in LAMBDAS_A:
        for prior in PRIORS:
            for seed in SEEDS:
                out.append({"block": "A", "lam": lam, "prior": prior, "seed": seed,
                            "tag": tag_a(lam, prior, seed), "role": "train", "reference_of": None})
    return out


def reference_rows_a():
    """λ=0 prior50/prior0 reference the same baseline checkpoint (not new trains)."""
    rows = []
    for prior in ("prior50", "prior0"):
        for seed in SEEDS:
            rows.append({"block": "A", "lam": 0.0, "prior": prior, "seed": seed,
                         "tag": tag_a(0.0, prior, seed), "role": "reference",
                         "reference_of": tag_a0(seed)})
    return rows


def jobs_block_b():
    out = []
    for lam in (0.0, 2.0):
        for seed in SEEDS:
            out.append({"block": "B", "lam": lam, "prior": "prior100", "seed": seed,
                        "tag": tag_b(lam, seed, "erm"), "role": "train", "init": "erm"})
    return out


def write_manifest(path=None):
    path = Path(path or (OUT / "MANIFEST.json"))
    priors = compute_priors()
    weights = {}
    for name in PRIORS:
        w, inv, mean_w = class_weights(priors[name], priors["pi_source"])
        weights[name] = {"w": w.tolist(), "invalid_classes": inv, "mean_stream_w": mean_w,
                         "pi_requested": priors[name].tolist()}
    doc = {
        "design": "decisions/paperB_r12_reweight.md",
        "commit": git_head(),
        "recipe": dict(R.CURRENT),
        "ckpt_policy": R.CKPT_POLICY,
        "seeds": list(SEEDS),
        "block_a_jobs": jobs_block_a(),
        "block_a_reference_rows": reference_rows_a(),
        "block_b_jobs": jobs_block_b(),
        "n_train_a": len(jobs_block_a()),
        "n_train_b": len(jobs_block_b()),
        "priors": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in priors.items()},
        "weights": weights,
        "endpoints": {
            "primary_detector": "mahalanobis_classcond",
            "primary_unseen": list(OOD_PRIMARY_UNSEEN),
            "aligned_domain": "pad_heldout",
            "fpr95tpr": "mean(score_ood <= tau); tau = 95th percentile of ID scores; higher score = more OOD",
            "stratified": "AUROC(ID class c vs same OOD panel); macro over fixed 8 labels",
            "bootstrap": str(R.PREREG),
        },
        "disclosure": "π_pad_adv from pad_adv labels at start; never in task loss; not label-free adaptation",
        "notes": [
            "Block A uses fixed image schedules (no adv_mix resampling); only source-adv loss weights change.",
            "align=iwdan_oracle + iw_cls=False implements DANN+reweight; not R2 IWDAN.",
            "Historical R1 resampling is cross-protocol reference.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc
