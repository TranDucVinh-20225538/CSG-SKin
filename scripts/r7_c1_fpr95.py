#!/usr/bin/env python3
"""Paper B R7 C1: FPR@95TPR (Mahalanobis) for Table 1/2 last.ckpt cells (decisions/paperB_r7_c1.md).

Score: higher = more OOD. τ = 95th percentile of ID-test Mahalanobis scores (95% ID kept as ID).
Standard FPR@95TPR = share of OOD accepted as ID = mean(sood <= τ).
Also report ood_flagged_at_95id = mean(sood >= τ) (share of OOD scored as OOD at that τ).
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTS = (R.R7 / "c1", R.R7B / "c1")
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


def rates_at_id95(sid, sood):
    """τ = 95th pctile of ID (95% ID kept). fpr95tpr = OOD accepted; flagged = OOD rejected."""
    tau = float(np.quantile(sid, 0.95))
    fpr95tpr = float(np.mean(sood <= tau))
    flagged = float(np.mean(sood >= tau))
    return fpr95tpr, flagged, tau


def table1_runs():
    """last.ckpt Table 1 runs from t1last summary."""
    rows = []
    with open(SUM, newline="") as f:
        for r in csv.DictReader(f):
            if r["checkpoint"] != "last":
                continue
            rows.append(r)
    return rows


def write_outputs(out: Path, rows, summary, boot):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_run.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    (out / "BOOTSTRAP.json").write_text(json.dumps(boot, indent=2) + "\n")
    design = "`decisions/paperB_r7b_c1.md`" if "r7b" in str(out) else "`decisions/paperB_r7_c1.md` (R7b rename)"
    L = ["# Paper B R7b C1: FPR@95TPR (Mahalanobis, Table 1 last.ckpt)", "",
         "Design: {}. Commit `{}`.".format(design, R.git_head()), "",
         "Score: class-conditional Mahalanobis on `z_lesion_norm`, fit on ISIC train. "
         "Higher = more OOD. τ = 95th percentile of ID-test scores (95% of ID kept as ID). "
         "**FPR@95TPR** = share of OOD accepted as ID = mean(score ≤ τ). "
         "**ood_flagged_at_95id** = share of OOD with score ≥ τ (not FPR@95TPR).", "",
         "| λ | OOD set | FPR@95TPR mean±sd | FPR@95TPR pooled CI | ood_flagged_at_95id mean±sd | n |",
         "|---:|---|---|---|---|---:|"]
    for r in summary:
        L.append("| {:g} | {} | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] | {:.3f} ± {:.3f} | {} |".format(
            r["lambda"], r["ood"], r["fpr95tpr_mean"], r["fpr95tpr_sd"], *r["fpr95tpr_pooled_ci"],
            r["ood_flagged_at_95id_mean"], r["ood_flagged_at_95id_sd"], r["n_seeds"]))
    L += ["", "## pad_heldout at λ = 0 and 2", ""]
    for lam in (0.0, 2.0):
        b = boot.get("lam{:g}|pad_heldout".format(lam))
        if b:
            L.append(
                "- λ = {:g}: FPR@95TPR {:.3f} ± {:.3f} CI [{:.3f}, {:.3f}]; "
                "ood_flagged_at_95id {:.3f} ± {:.3f}.".format(
                    lam, b["fpr95tpr_mean"], b["fpr95tpr_sd"], *b["fpr95tpr_pooled_ci"],
                    b["ood_flagged_at_95id_mean"], b["ood_flagged_at_95id_sd"]))
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md")


def main():
    id_groups = R.isic_test_groups()
    pad_groups = R.pad_heldout_groups()
    n_boot, seed = R.prereg()
    rng = np.random.default_rng(seed)

    per = defaultdict(dict)  # (lam, ood) -> {seed: (sid, sood, fpr95tpr, flagged, tau)}
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
            fpr, flagged, tau = rates_at_id95(sid, sood)
            per[(lam, ood)][s] = (sid, sood, fpr, flagged, tau)
            rows.append({"run": tag, "lambda": lam, "seed": s, "ood": ood,
                         "fpr95tpr": fpr, "ood_flagged_at_95id": flagged,
                         "tau_id95": tau, "n_id": len(sid), "n_ood": len(sood)})

    summary, boot = [], {}
    for (lam, ood), seeds in sorted(per.items()):
        fprs = [v[2] for v in seeds.values()]
        flags = [v[3] for v in seeds.values()]
        if ood == "pad_heldout":
            go = pad_groups
        else:
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
                vals.append(float(np.mean(sood[do] <= tau)))
            draws.append(float(np.mean(vals)))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        key = "lam{:g}|{}".format(lam, ood)
        boot[key] = {
            "fpr95tpr_mean": float(np.mean(fprs)),
            "fpr95tpr_sd": float(np.std(fprs, ddof=1)) if len(fprs) > 1 else 0.0,
            "fpr95tpr_pooled_ci": [float(lo), float(hi)],
            "ood_flagged_at_95id_mean": float(np.mean(flags)),
            "ood_flagged_at_95id_sd": float(np.std(flags, ddof=1)) if len(flags) > 1 else 0.0,
            "n_seeds": len(fprs),
        }
        summary.append({"lambda": lam, "ood": ood, **boot[key]})

    for out in OUTS:
        write_outputs(out, rows, summary, boot)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
