#!/usr/bin/env python3
"""R10b B4: SCIN Mahalanobis on split_stability cells (splits 7/123/2027, λ ∈ {0,2}).

Uses last.ckpt features under runs/*/features (alias z→z_lesion_norm). ID groups = lesion
groups of each split's test fold (StratifiedGroupKFold random_state=split). OOD unit = same
as SCIN elsewhere (P7a `group` / case).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402

OUT = R.RES / "r10b"
ROOT = R.RES / "r6" / "split_stability"
SPLITS = (7, 123, 2027)
LAMS = (0.0, 2.0)
SEEDS = (42, 52, 62)  # split_stability used 3 seeds
DET = "mahalanobis_classcond"


def alias_z(pack):
    if "z_lesion_norm" not in pack and "z" in pack:
        pack = dict(pack)
        pack["z_lesion_norm"] = pack["z"]
        pack.setdefault("z_lesion", pack["z"])
    return pack


def load_feats(feat_dir, names):
    out = {}
    for n in names:
        p = feat_dir / f"{n}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        out[n] = alias_z(dict(np.load(p)))
    return out


def id_groups_for_split(split_seed):
    """Rebuild lesion groups for the test fold of isic_split_seed."""
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy().reset_index(drop=True)
    groups, _ = T.lesion_groups(isic)
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(split_seed))
    _tv_i, te_i = next(sgk.split(np.zeros(len(isic)), isic["label_idx"], groups))
    test = isic.iloc[te_i].reset_index(drop=True)
    g_te, _ = T.lesion_groups(test)
    return R.idx_groups(g_te), len(test)


def bootstrap_scin(feat_dirs, id_groups, o_groups):
    """Same Rule A machinery as R.bootstrap_cell but with pre-aliased load."""
    pre = json.loads(R.PREREG.read_text())
    n_boot = int(pre["n_resamples"])
    rng = np.random.default_rng(int(pre["rng_seed"]))
    sc = {}
    for s, d in feat_dirs.items():
        Fs = load_feats(d, ("isic_train", "isic_test", "scin"))
        S = R.Scorer(Fs["isic_train"])
        a, b = S.scores(Fs["isic_test"]), S.scores(Fs["scin"])
        # length check vs id groups
        if sum(len(g) for g in id_groups) != len(a[DET]):
            raise ValueError(
                f"ID group size mismatch seed={s}: groups={sum(len(g) for g in id_groups)} "
                f"scores={len(a[DET])}"
            )
        sc[s] = {DET: (a[DET], b[DET])}
    id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))])
                for _ in range(n_boot)]
    oo_draws = [np.concatenate([o_groups[j] for j in rng.integers(0, len(o_groups), len(o_groups))])
                for _ in range(n_boot)]
    seeds = sorted(sc)
    out = {"excluded_seeds": []}
    per, draws = {}, {}
    for s in seeds:
        sid, so = sc[s][DET]
        draws[s] = np.array([R.auroc(sid[id_draws[b]], so[oo_draws[b]]) for b in range(n_boot)])
        lo, hi = float(np.quantile(draws[s], 0.025)), float(np.quantile(draws[s], 0.975))
        per[str(s)] = {"point": float(R.auroc(sid, so)), "ci_lo": lo, "ci_hi": hi,
                       "below_chance_entire_ci": hi < 0.5}
    pooled = np.mean(np.stack([draws[s] for s in seeds]), 0)
    pts = [per[str(s)]["point"] for s in seeds]
    block = {"mean": float(np.mean(pts)), "sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0,
             "ci_lo": float(np.quantile(pooled, 0.025)), "ci_hi": float(np.quantile(pooled, 0.975)),
             "k_below": int(sum(p["below_chance_entire_ci"] for p in per.values())),
             "n_seeds": len(seeds), "seeds": seeds, "per_seed": per}
    out[DET] = block
    # Rule A with 3 seeds: need ≥3 seed CIs below 0.5 — with n=3 that means all 3
    out["rule_A"] = bool(block["ci_hi"] < 0.5 and block["k_below"] >= 3)
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = R.load_data()
    og, unit = R.ood_groups("scin", data)
    print("SCIN unit", unit, flush=True)
    rows = []
    latex = []
    for split in SPLITS:
        idg, n_test = id_groups_for_split(split)
        print(f"split={split} n_test={n_test} n_id_groups={len(idg)}", flush=True)
        for lam in LAMS:
            feat_dirs = {}
            for s in SEEDS:
                tag = f"ss_s{split}_ladv{R.lam_tag(lam)}_s{s}"
                feat = ROOT / "runs" / tag / "features"
                if not (feat / "scin.npz").exists():
                    print("MISSING", tag, flush=True)
                    continue
                feat_dirs[s] = feat
            if len(feat_dirs) < 3:
                print(f"incomplete cell split={split} λ={lam}", flush=True)
                continue
            b = bootstrap_scin(feat_dirs, idg, og)
            x = b[DET]
            row = {
                "split": split, "lambda": lam, "mean": x["mean"], "sd": x["sd"],
                "ci": [x["ci_lo"], x["ci_hi"]], "k": f"{x['k_below']}/{x['n_seeds']}",
                "rule_A": b["rule_A"], "unit_ood": unit, "per_seed": x["per_seed"],
            }
            row["latex"] = (
                f"{split} & {lam:g} & SCIN & ${x['mean']:.3f}\\pm{x['sd']:.3f}$ & "
                f"$[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]$ & {x['k_below']}/{x['n_seeds']} & "
                f"{'met' if b['rule_A'] else 'not met'} \\\\"
            )
            rows.append(row)
            latex.append(row["latex"])
            print(f"  λ={lam:g}: {x['mean']:.3f}±{x['sd']:.3f} [{x['ci_lo']:.3f},{x['ci_hi']:.3f}] "
                  f"k={x['k_below']}/{x['n_seeds']} RuleA={b['rule_A']}", flush=True)

    payload = {"commit": R.C.git_head(), "rows": rows, "latex": latex, "unit_ood": unit,
               "cpu_hours": (time.time() - t0) / 3600.0, "note": "last.ckpt features; 3 seeds"}
    (OUT / "B4_splits_scin.json").write_text(json.dumps(payload, indent=1) + "\n")
    md = ["# R10b B4 — SCIN on tab:splits", "", f"Commit `{payload['commit']}`. Unit: `{unit}`.",
          "", "| split | λ | mean±sd | pooled CI | k | Rule A |",
          "|---:|---:|---:|---|---:|---|"]
    for r in rows:
        md.append(
            f"| {r['split']} | {r['lambda']:g} | {r['mean']:.3f}±{r['sd']:.3f} | "
            f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['k']} | "
            f"{'met' if r['rule_A'] else 'not met'} |"
        )
    md += ["", "LaTeX:", "```"] + latex + ["```", ""]
    (OUT / "B4_splits_scin.md").write_text("\n".join(md))
    print("wrote B4 cpu_h", round(payload["cpu_hours"], 3), flush=True)


if __name__ == "__main__":
    main()
