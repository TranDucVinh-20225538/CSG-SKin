#!/usr/bin/env python3
"""Paper B R7 C1: FPR@95TPR (Mahalanobis) for Table 1/2 last.ckpt cells (decisions/paperB_r7_c1.md).

Threshold: 95th percentile of ID-test Mahalanobis scores (higher = more OOD). FPR@95TPR =
fraction of OOD with score >= that threshold. Also: share of pad_heldout OOD accepted at the
95% ID-coverage threshold (score <= τ_ID where τ = 95th pctile of ID scores → accept as ID).
Bootstrap units as in the paper. CPU; uses results/paperB/r5/t1last/features_last.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r6_common as R6  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7 / "c1"
FEAT = R.R5 / "t1last" / "features_last"
SUM = R.R5 / "t1last" / "summary.csv"
REP = "z_lesion_norm"
OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")


def load_pack(tag, name):
    p = FEAT / tag / "{}.npz".format(name)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def maha_scores(tr, pack):
    det = R.Detectors(tr[REP], tr["labels"].astype(int), 8)
    sc, _ = det.scores(pack[REP], pack["logits"])
    return np.asarray(sc["mahalanobis_classcond"], float)


def fpr95(sid, sood):
    """FPR at 95% TPR: threshold = 5th pctile of OOD? No — standard: TPR on OOD at threshold from ID.
    Higher score = more OOD. TPR@95 means 95% of OOD above τ; τ = 5th pctile of OOD scores;
    FPR = fraction of ID above τ. Equivalent common form used in Paper B S-flip:
    τ = 95th pctile of ID; FPR = fraction OOD with score > τ (accepted as OOD / rejected as ID).
    We report the S-flip / detector convention: τ = quantile(sid, 0.95); fpr = mean(sood >= τ).
    """
    tau = float(np.quantile(sid, 0.95))
    return float(np.mean(sood >= tau)), tau


def accept_rate(sid, sood):
    """Share of OOD accepted as ID at 95% ID-coverage (score <= τ)."""
    tau = float(np.quantile(sid, 0.95))
    return float(np.mean(sood <= tau)), tau


def table1_runs():
    """last.ckpt Table 1 runs from t1last summary."""
    rows = []
    with open(SUM, newline="") as f:
        for r in csv.DictReader(f):
            if r["checkpoint"] != "last":
                continue
            rows.append(r)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    id_groups = R.isic_test_groups()
    pad_groups = R.pad_heldout_groups()
    n_boot, seed = R.prereg()
    rng = np.random.default_rng(seed)

    # OOD groups
    ood_group_fn = {
        "pad_heldout": lambda: pad_groups,
        "fitzpatrick17k": lambda: None,  # image-level → singleton groups filled later
    }

    per = defaultdict(dict)  # (lam, ood) -> {seed: (sid, sood, fpr, accept)}
    rows = []
    for r in table1_runs():
        tag, lam, s = r["run"], float(r["lambda"]), int(r["seed"])
        tr = load_pack(tag, "isic_train")
        te = load_pack(tag, "isic_test")
        if tr is None or te is None:
            print("missing features", tag)
            continue
        sid = maha_scores(tr, te)
        for ood in OODS:
            oo = load_pack(tag, ood)
            if oo is None:
                continue
            sood = maha_scores(tr, oo)
            fpr, tau = fpr95(sid, sood)
            acc, _ = accept_rate(sid, sood)
            per[(lam, ood)][s] = (sid, sood, fpr, acc, tau)
            rows.append({"run": tag, "lambda": lam, "seed": s, "ood": ood, "fpr95": fpr,
                         "ood_accept_at_95id": acc, "tau_id95": tau, "n_id": len(sid), "n_ood": len(sood)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_run.csv", index=False)

    # aggregate + bootstrap for FPR (resample ID and OOD groups, recompute FPR)
    summary, boot = [], {}
    for (lam, ood), seeds in sorted(per.items()):
        fprs = [v[2] for v in seeds.values()]
        accs = [v[3] for v in seeds.values()]
        # bootstrap FPR
        if ood == "pad_heldout":
            go = pad_groups
        else:
            # image-level: one index per image
            n_o = len(next(iter(seeds.values()))[1])
            go = [np.array([i]) for i in range(n_o)]
        gi = id_groups
        draws = []
        for _ in range(n_boot):
            di = np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))])
            do = np.concatenate([go[j] for j in rng.integers(0, len(go), len(go))])
            vals = []
            for sid, sood, *_ in seeds.values():
                tau = float(np.quantile(sid[di], 0.95))
                vals.append(float(np.mean(sood[do] >= tau)))
            draws.append(float(np.mean(vals)))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        key = "lam{:g}|{}".format(lam, ood)
        boot[key] = {"fpr95_mean": float(np.mean(fprs)), "fpr95_sd": float(np.std(fprs, ddof=1)) if len(fprs) > 1 else 0.0,
                     "pooled_ci": [float(lo), float(hi)], "n_seeds": len(fprs),
                     "ood_accept_mean": float(np.mean(accs)), "ood_accept_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}
        summary.append({"lambda": lam, "ood": ood, **boot[key]})

    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    (OUT / "BOOTSTRAP.json").write_text(json.dumps(boot, indent=2) + "\n")

    L = ["# Paper B R7 C1: FPR@95TPR (Mahalanobis, Table 1 last.ckpt)", "",
         "Design: `decisions/paperB_r7_c1.md`. Commit `{}`.".format(R.git_head()), "",
         "Score: class-conditional Mahalanobis on `z_lesion_norm`, fit on ISIC train. "
         "Higher = more OOD. τ = 95th percentile of ID-test scores. "
         "FPR@95TPR = fraction of OOD with score ≥ τ. "
         "OOD acceptance at 95% ID coverage = fraction of OOD with score ≤ τ.", "",
         "| λ | OOD set | FPR@95 mean±sd | pooled CI | OOD accept@95ID mean±sd | n |",
         "|---:|---|---|---|---|---:|"]
    for r in summary:
        L.append("| {:g} | {} | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] | {:.3f} ± {:.3f} | {} |".format(
            r["lambda"], r["ood"], r["fpr95_mean"], r["fpr95_sd"], *r["pooled_ci"],
            r["ood_accept_mean"], r["ood_accept_sd"], r["n_seeds"]))
    L += ["", "## pad_heldout acceptance at λ = 0 and 2", ""]
    for lam in (0.0, 2.0):
        b = boot.get("lam{:g}|pad_heldout".format(lam))
        if b:
            L.append("- λ = {:g}: OOD accept {:.3f} ± {:.3f}; FPR@95 {:.3f} ± {:.3f} CI [{:.3f}, {:.3f}].".format(
                lam, b["ood_accept_mean"], b["ood_accept_sd"], b["fpr95_mean"], b["fpr95_sd"], *b["pooled_ci"]))
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "rows", len(rows))


if __name__ == "__main__":
    main()
