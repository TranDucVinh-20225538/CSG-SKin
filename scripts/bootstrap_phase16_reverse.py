#!/usr/bin/env python3
"""Bootstrap CI for Phase 16 B1-rev HAM→BCN Mahalanobis @ λ=1 (reviewer item 1, §4.8)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = object
    _tf.AutoTokenizer = object
    _cfg = types.ModuleType("transformers.configuration_utils")

    class _PC:
        def __eq__(self, other):
            return False

    _cfg.PretrainedConfig = _PC
    _tf.configuration_utils = _cfg
    sys.modules["transformers"] = _tf
    sys.modules["transformers.configuration_utils"] = _cfg

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
for p in (REPO, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_phase2_pad_holdout as p2
from phase16_b1_rev_coarse import load_or_create_splits
from phase16_b0 import build_eval_tf
from src.datasets.skin_dataset import SkinDataset
from src.utils import ood_metrics
from train_phase15_1_single_dann import SingleDannLightning, collect_all

OUT = ROOT / "results/paperB/reviewer_r1"
PREREG = json.loads((ROOT / "results/paperB/PREREGISTER_BOOTSTRAP.json").read_text())
SPLIT_PATH = ROOT / "results/paperB/phase16/b1_rev/b1_rev_split_seed42.json"
B1_REV = ROOT / "results/paperB/phase16/b1_rev"
LAM = 1.0
SEEDS = (42, 52, 62)
N_BOOT = int(PREREG["n_resamples"])
RNG_SEED = int(PREREG["rng_seed"])


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def auroc_from_scores(sid, sood):
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    s = np.concatenate([sid, sood])
    return float(roc_auc_score(y, s))


def maha_scores(ztr, ytr, zid, zood):
    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, 8, 1e-3)
    sid = ood_metrics.mahalanobis_min_squared_distances(zid, means, prec)
    sood = ood_metrics.mahalanobis_min_squared_distances(zood, means, prec)
    return sid.astype(np.float64), sood.astype(np.float64)


def percentile_ci(samples, alpha=0.05):
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi, float(np.mean(samples))


def load_run(seed: int, device):
    run_name = f"ham_bcn_ladv{lam_tag(LAM)}_s{seed}"
    summary_path = B1_REV / run_name / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    ckpt = summary["best_checkpoint"]
    lit = SingleDannLightning.load_from_checkpoint(ckpt, strict=False)
    net = lit.net.to(device).eval()
    _df, splits, _meta, _hygiene = load_or_create_splits()
    eval_tf = build_eval_tf(448)

    def loader(frame):
        return p2.make_loader(SkinDataset(frame, transform=eval_tf), 64, 4, False)

    train_pack = collect_all(net, loader(splits["ham_train"]), device)
    id_pack = collect_all(net, loader(splits["ham_test"]), device)
    hold_pack = collect_all(net, loader(splits["bcn_heldout"]), device)
    sid, sood = maha_scores(
        train_pack["z"], train_pack["labels"], id_pack["z"], hold_pack["z"]
    )
    point = auroc_from_scores(sid, sood)
    return sid, sood, point, run_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(RNG_SEED)
    seed_scores = {}
    n_id = None
    n_ood = None
    for seed in SEEDS:
        sid, sood, point, run_name = load_run(seed, device)
        seed_scores[seed] = (sid, sood, point, run_name)
        n_id = len(sid)
        n_ood = len(sood)
        print(f"loaded {run_name} point={point:.4f} n_id={n_id} n_ood={n_ood}", flush=True)

    id_idx_mat = rng.integers(0, n_id, size=(N_BOOT, n_id))
    ood_idx_mat = rng.integers(0, n_ood, size=(N_BOOT, n_ood))
    per_seed_draws = {str(s): np.empty(N_BOOT, dtype=np.float64) for s in SEEDS}
    per_seed_boot = {}
    k_above = 0
    for seed in SEEDS:
        sid, sood, point, _ = seed_scores[seed]
        for b in range(N_BOOT):
            per_seed_draws[str(seed)][b] = auroc_from_scores(
                sid[id_idx_mat[b]], sood[ood_idx_mat[b]]
            )
        lo = float(np.quantile(per_seed_draws[str(seed)], 0.025))
        hi = float(np.quantile(per_seed_draws[str(seed)], 0.975))
        above = lo > 0.5
        k_above += int(above)
        per_seed_boot[str(seed)] = {
            "point": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "above_chance_entire_ci": above,
        }

    mean_draws = np.mean(np.stack([per_seed_draws[str(s)] for s in SEEDS]), axis=0)
    lo, hi, mu = percentile_ci(mean_draws)
    payload = {
        "protocol": PREREG["secondary_scope"],
        "lambda_adv": LAM,
        "seeds": list(SEEDS),
        "n_resamples": N_BOOT,
        "bootstrap": "image both sides; shared indices across seeds per draw",
        "mahalanobis_fit": "HAM train only (fixed)",
        "point_mean_over_seeds": float(np.mean([per_seed_boot[str(s)]["point"] for s in SEEDS])),
        "mean_across_seeds_bootstrap_ci": {"lo": lo, "hi": hi, "mean": mu},
        "n_seeds_above_chance_ci": k_above,
        "n_seeds": len(SEEDS),
        "per_seed": per_seed_boot,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "bootstrap_phase16_ham_bcn_ladv1.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out_path)
    print(
        f"HAM→BCN λ=1 Mahalanobis: mean={payload['point_mean_over_seeds']:.4f} "
        f"CI=[{lo:.4f},{hi:.4f}] k_above_0.5={k_above}/{len(SEEDS)}"
    )


if __name__ == "__main__":
    main()
