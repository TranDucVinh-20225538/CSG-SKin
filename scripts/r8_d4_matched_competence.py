"""R8 / D4: paired lambda = 2 minus lambda = 0 change in ID balanced accuracy, next to the paired change in
Mahalanobis AUROC, for both dual-encoder backbones (last checkpoint, matched seeds 42-82).

Same seed bootstrap as the paired ID change in the paper (r5/t1last/UNSEEN_ECE.json: 2,000 draws,
seed 20261005, resampling seeds). CPU only; reads existing summaries.
Writes results/paperB/r8/d4/D4.json and REPORT.md.
"""
import csv
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paperB"
OUT = RES / "r8" / "d4"
SEEDS = [42, 52, 62, 72, 82]
DRAWS, RNG = 2000, 20261005


def paired(per_seed):
    d = np.array([per_seed[s] for s in SEEDS])
    rng = np.random.default_rng(RNG)
    boot = d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(d.mean()), "sd": float(statistics.stdev(d)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": len(d), "per_seed": {str(s): float(per_seed[s]) for s in SEEDS}}


def effb3():
    v = {}
    with open(RES / "r5/t1last/summary.csv") as f:
        for r in csv.DictReader(f):
            if r["checkpoint"] == "last" and r["lambda"] in ("0.0", "2.0"):
                v[(float(r["lambda"]), int(r["seed"]))] = r
    cols = {"id_bal_acc": "id_bal_acc", "pad_heldout": "pad_heldout_mahalanobis_classcond",
            "fitzpatrick17k": "fitzpatrick17k_mahalanobis_classcond"}
    return {k: paired({s: float(v[(2.0, s)][c]) - float(v[(0.0, s)][c]) for s in SEEDS}) for k, c in cols.items()}


def dinov2():
    v = {}
    with open(RES / "r7/t1/summary.csv") as f:
        for r in csv.DictReader(f):
            if r["arm"].startswith("dual"):
                v[(float(r["lambda_adv"]), int(r["seed"]))] = r
    cols = {"id_bal_acc": "id_bal_acc", "pad_heldout": "ood_auroc", "fitzpatrick17k": "fitz_auroc",
            "mednode": "maha_mednode", "sd198": "maha_sd198"}
    return {k: paired({s: float(v[(2.0, s)][c]) - float(v[(0.0, s)][c]) for s in SEEDS}) for k, c in cols.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"design": "paired lambda=2 minus lambda=0 per seed, last checkpoint; seed bootstrap {} draws, rng {}"
                     .format(DRAWS, RNG), "seeds": SEEDS, "effb3_dual": effb3(), "dinov2_dual": dinov2()}
    ref = json.loads((RES / "r5/t1last/UNSEEN_ECE.json").read_text())["paired_id_bal_l2_minus_l0"]
    res["check_effb3_id_vs_paper"] = {"paper_ci": [ref["ci_lo"], ref["ci_hi"]],
                                      "here_ci": [res["effb3_dual"]["id_bal_acc"]["ci_lo"],
                                                  res["effb3_dual"]["id_bal_acc"]["ci_hi"]]}
    (OUT / "D4.json").write_text(json.dumps(res, indent=1))

    lines = ["# R8 / D4: matched competence (paired lambda = 2 minus lambda = 0)", "",
             res["design"] + ". Mahalanobis AUROC.", "",
             "| backbone | quantity | mean Δ | 95% CI | per seed (42, 52, 62, 72, 82) |", "|---|---|---:|---|---|"]
    for bb in ("effb3_dual", "dinov2_dual"):
        for k, x in res[bb].items():
            lines.append("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {} |".format(
                bb, k, x["mean"], x["ci_lo"], x["ci_hi"], ", ".join("{:+.3f}".format(x["per_seed"][str(s)]) for s in SEEDS)))
    c = res["check_effb3_id_vs_paper"]
    lines += ["", "EffB3 ID CI here [{:+.4f}, {:+.4f}] vs paper source [{:+.4f}, {:+.4f}].".format(
        *c["here_ci"], *c["paper_ci"])]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
