#!/usr/bin/env python3
"""Stratified bootstrap CIs for OOD AUROC (reviewer item 1). See PREREGISTER_BOOTSTRAP.json."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
PREREG = ROOT / "results/paperB/PREREGISTER_BOOTSTRAP.json"
FEAT = ROOT / "results/paperB/phase13/features"
OUT = ROOT / "results/paperB/reviewer_r1"
REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REPO))

import train_phase2_pad_holdout as p2
import train_phase3_sweep as p3
from eval_ood_dual_branch import cosine_ood_scores, fit_class_means, fit_knn, knn_scores
from src.utils import ood_metrics

PAT = re.compile(r"PAT_(\d+)_(\d+)_(\d+)")


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


def cosine_scores(ztr, ytr, zid, zood):
    means, present = fit_class_means(ztr, ytr, 8)
    sid = cosine_ood_scores(zid, means, present)
    sood = cosine_ood_scores(zood, means, present)
    return sid.astype(np.float64), sood.astype(np.float64)


def knn50_scores(ztr, zid, zood):
    knn = fit_knn(ztr, k=50)
    sid = knn_scores(knn, zid)
    sood = knn_scores(knn, zood)
    return sid.astype(np.float64), sood.astype(np.float64)


DET_FNS = {
    "mahalanobis_classcond": maha_scores,
    "cosine_max": cosine_scores,
}


def load_npz(run_dir: Path, split: str):
    return dict(np.load(run_dir / f"{split}.npz"))


def pad_patient_groups(hold_frame) -> list[np.ndarray]:
    by_p: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(hold_frame["path"]):
        m = PAT.search(str(p))
        if not m:
            raise RuntimeError(f"bad path {p}")
        by_p[m.group(1)].append(i)
    return [np.asarray(v, dtype=np.int64) for v in by_p.values()]


def percentile_ci(samples, alpha=0.05):
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi, float(np.mean(samples))


def bootstrap_patient_image(
    sid_full,
    sood_full,
    patient_groups,
    n_id,
    rng,
    n_boot,
):
    """Returns array shape (n_boot,) of AUROC."""
    n_pat = len(patient_groups)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pid_idx = rng.integers(0, n_pat, size=n_pat)
        ood_idx = np.concatenate([patient_groups[i] for i in pid_idx])
        id_idx = rng.integers(0, n_id, size=n_id)
        out[b] = auroc_from_scores(sid_full[id_idx], sood_full[ood_idx])
    return out


def bootstrap_image_both(sid_full, sood_full, n_id, n_ood, rng, n_boot):
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        id_idx = rng.integers(0, n_id, size=n_id)
        ood_idx = rng.integers(0, n_ood, size=n_ood)
        out[b] = auroc_from_scores(sid_full[id_idx], sood_full[ood_idx])
    return out


def seeds_for_lambda(lam: float):
    return (42, 52, 62, 72, 82) if lam in (0.0, 2.0, 8.0) else (42, 52, 62)


def run_derm_sweep(prereg):
    spec = json.loads((ROOT / "results/paperB/phase2_pad_holdout/pad_patient_split_paths.json").read_text())
    df = p2.load_master()
    pad = df[df.domain == "pad_ufes"]
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    if len(hold) != 716:
        raise RuntimeError(f"expected 716 heldout, got {len(hold)}")
    patient_groups = pad_patient_groups(hold)
    if len(patient_groups) != 412:
        raise RuntimeError(f"expected 412 patients, got {len(patient_groups)}")

    n_boot = int(prereg["n_resamples"])
    rng = np.random.default_rng(int(prereg["rng_seed"]))
    results = {"pad_heldout": {}, "fitzpatrick17k": {}}

    for lam in prereg["lambda_grid"]:
        lam = float(lam)
        seeds = seeds_for_lambda(lam)
        seed_scores = {det: {} for det in prereg["detectors"]}
        for seed in seeds:
            tag = f"runB_orth1_ladv{lam_tag(lam)}_s{seed}"
            run_dir = FEAT / tag
            if not run_dir.exists():
                raise FileNotFoundError(run_dir)
            tr = load_npz(run_dir, "isic_train")
            te = load_npz(run_dir, "isic_test")
            hold_z = load_npz(run_dir, "pad_heldout")
            fitz = load_npz(run_dir, "fitzpatrick17k")
            ztr, ytr = tr["z_lesion_norm"], tr["labels"]
            zid, zood = te["z_lesion_norm"], hold_z["z_lesion_norm"]
            zfitz = fitz["z_lesion_norm"]
            n_id = len(zid)
            for det in prereg["detectors"]:
                if det == "knn_k50":
                    sid, sood = knn50_scores(ztr, zid, zood)
                    sid_f, sood_f = knn50_scores(ztr, zid, zfitz)
                else:
                    fn = DET_FNS[det]
                    sid, sood = fn(ztr, ytr, zid, zood)
                    sid_f, sood_f = fn(ztr, ytr, zid, zfitz)
                seed_scores[det][seed] = {
                    "pad": (sid, sood),
                    "fitz": (sid_f, sood_f),
                    "n_id": n_id,
                }

        for ood_name, key in (("pad_heldout", "pad"), ("fitzpatrick17k", "fitz")):
            results[ood_name].setdefault(str(lam), {})
            n_id = seed_scores[prereg["detectors"][0]][seeds[0]]["n_id"]
            n_ood = len(seed_scores[prereg["detectors"][0]][seeds[0]][key][1])
            n_pat = len(patient_groups)
            id_idx_mat = rng.integers(0, n_id, size=(n_boot, n_id))
            if ood_name == "pad_heldout":
                pat_draws = rng.integers(0, n_pat, size=(n_boot, n_pat))
                ood_idx_list = [np.concatenate([patient_groups[i] for i in row]) for row in pat_draws]
            else:
                ood_idx_mat = rng.integers(0, n_ood, size=(n_boot, n_ood))
                ood_idx_list = list(ood_idx_mat)
            for det in prereg["detectors"]:
                per_seed_draws = {str(s): np.empty(n_boot, dtype=np.float64) for s in seeds}
                for seed in seeds:
                    sid, sood = seed_scores[det][seed][key]
                    for b in range(n_boot):
                        per_seed_draws[str(seed)][b] = auroc_from_scores(
                            sid[id_idx_mat[b]], sood[ood_idx_list[b]]
                        )
                mean_draws = np.mean(np.stack([per_seed_draws[str(s)] for s in seeds]), axis=0)
                lo, hi, mu = percentile_ci(mean_draws)
                per_seed_boot = {}
                k_below = 0
                for seed in seeds:
                    draws = per_seed_draws[str(seed)]
                    sid, sood = seed_scores[det][seed][key]
                    ci_lo = float(np.quantile(draws, 0.025))
                    ci_hi = float(np.quantile(draws, 0.975))
                    below = ci_hi < 0.5
                    k_below += int(below)
                    per_seed_boot[str(seed)] = {
                        "point": auroc_from_scores(sid, sood),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "below_chance_entire_ci": below,
                    }
                results[ood_name][str(lam)][det] = {
                    "point_mean_over_seeds": float(np.mean([per_seed_boot[str(s)]["point"] for s in seeds])),
                    "mean_across_seeds_bootstrap_ci": {"lo": lo, "hi": hi, "mean": mu},
                    "n_seeds_below_chance_ci": int(k_below),
                    "n_seeds": len(seeds),
                    "per_seed": per_seed_boot,
                }
                print(f"  lam={lam} {ood_name} {det} done", flush=True)
    return results


def decide(results, prereg):
    lam = str(float(prereg["primary_lambda_adv"]))
    det = "mahalanobis_classcond"
    block = results["pad_heldout"][lam][det]
    ci = block["mean_across_seeds_bootstrap_ci"]
    k = block["n_seeds_below_chance_ci"]
    pooled_below = ci["hi"] < 0.5
    if pooled_below and k >= 3:
        rule = "A"
    elif pooled_below and k < 3:
        rule = "B"
    elif ci["lo"] < 0.5 < ci["hi"]:
        rule = "C"
    else:
        rule = "B" if pooled_below else "none_above"
    return {
        "rule": rule,
        "lambda": lam,
        "detector": det,
        "pooled_ci": ci,
        "k_of_5_seeds_entire_ci_below_0.5": k,
    }


def write_report(results, decision, prereg):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"preregister": prereg, "results": results, "decision": decision}
    (OUT / "bootstrap_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lam2 = str(float(prereg["primary_lambda_adv"]))
    maha = results["pad_heldout"][lam2]["mahalanobis_classcond"]
    ci = maha["mean_across_seeds_bootstrap_ci"]
    lines = [
        "# Bootstrap AUROC report (reviewer item 1)\n",
        f"\nPreregistration: `PREREGISTER_BOOTSTRAP.json` ({prereg['n_resamples']} resamples, {prereg['ci_method']} 95% CI).\n",
        f"\n## Primary decision (@ λ={lam2}, pad_heldout, Mahalanobis)\n",
        f"\n- Mean AUROC across seeds (point): **{maha['point_mean_over_seeds']:.4f}**\n",
        f"- Bootstrap CI on **mean across seeds**: **[{ci['lo']:.4f}, {ci['hi']:.4f}]**\n",
        f"- Seeds with entire per-seed 95% CI below 0.5: **{maha['n_seeds_below_chance_ci']} / {maha['n_seeds']}**\n",
        f"- **Decision rule {decision['rule']}** applies (see preregistration).\n",
        "\n## Manuscript action\n",
    ]
    rules = {r["id"]: r["manuscript"] for r in prereg["decision_rules"]}
    lines.append(f"\n{rules.get(decision['rule'], 'See pooled CI and per-seed counts.')}\n")
    lines.append("\n## λ grid (pad_heldout, Mahalanobis mean CI)\n\n| λ | mean AUROC | 95% CI | k below 0.5 |\n|---:|---:|---|---:|\n")
    for lam in prereg["lambda_grid"]:
        row = results["pad_heldout"][str(float(lam))]["mahalanobis_classcond"]
        c = row["mean_across_seeds_bootstrap_ci"]
        lines.append(
            f"| {lam} | {row['point_mean_over_seeds']:.3f} | [{c['lo']:.3f}, {c['hi']:.3f}] | {row['n_seeds_below_chance_ci']}/{row['n_seeds']} |\n"
        )
    lines.append(
        "\nFull detector grid (kNN, cosine) and Fitzpatrick17k (image bootstrap) are in `bootstrap_results.json`.\n"
    )
    (OUT / "BOOTSTRAP_REPORT.md").write_text("".join(lines))


def main():
    prereg = json.loads(PREREG.read_text())
    print("bootstrap: loading features and running", prereg["n_resamples"], "resamples...", flush=True)
    results = run_derm_sweep(prereg)
    decision = decide(results, prereg)
    write_report(results, decision, prereg)
    print("decision", decision)
    print("wrote", OUT / "BOOTSTRAP_REPORT.md")


if __name__ == "__main__":
    main()
