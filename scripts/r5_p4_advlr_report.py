#!/usr/bin/env python3
"""Paper B R5 P4 (GPU part, report; CPU): Table 1 columns and the five detectors for the lesion-level λ = 2 runs with
adversary LR ×10 / ×100 (results/paperB/r5/p4/advlr), next to the ×30 runs of the same seeds (R2 item 4).
Writes results/paperB/r5/p4/advlr/{per_run/*.json, summary.csv, REPORT.md}."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
from phase15_common import six_class_xfer  # noqa: E402

ADV = R.R5 / "p4" / "advlr"
SEEDS = (42, 52, 62)
MULTS = (30, 10, 100)
OODS = ("pad_heldout", "fitzpatrick17k")


def where(mult, seed):
    if mult == 30:
        tag = R.run_tag(2.0, seed)
        return tag, R.FEAT_L, R.C.R2 / "item4" / "runs" / tag
    tag = "runB_orth1_ladv2_s{}_advlr{}".format(seed, mult)
    return tag, ADV / "features", ADV / "runs" / tag


def main():
    (ADV / "per_run").mkdir(parents=True, exist_ok=True)
    pre = R.prereg()
    rows, scores = [], {}
    for mult in MULTS:
        for s in SEEDS:
            tag, feat, run = where(mult, s)
            if not (feat / tag / "fitzpatrick17k.npz").exists():
                print("missing", tag)
                continue
            summ = json.loads((run / "summary.json").read_text())
            tr, te = R.load(tag, "isic_train", feat), R.load(tag, "isic_test", feat)
            oo = {o: R.load(tag, o, feat) for o in OODS}
            sc, _m = R.detector_scores(tr, te, oo)
            scores[(mult, s)] = sc
            ho = oo["pad_heldout"]
            rec = {"mult": mult, "seed": s, "run": tag, "leakage": summ["leakage"]["z_lesion"]["bal_acc_mean"],
                   "id_bal_acc": summ["id_balanced_acc"], "id_ece": R.ece15(te["logits"], te["labels"]),
                   "ood_ece": R.ece15(ho["logits"], ho["labels"]),
                   "xdom_bal_acc": six_class_xfer(ho["logits"], ho["labels"])["balanced_accuracy"]}
            for det, (sid, so) in sc.items():
                for o in OODS:
                    rec["{}|{}".format(det, o)] = R.auroc(sid, so[o])
            R.write_json(ADV / "per_run" / "{}.json".format(tag), rec)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(ADV / "summary.csv", index=False)
    rng = np.random.default_rng(int(pre["rng_seed"]))
    n = int(pre["n_resamples"])
    boot = {}
    for o in OODS:
        idd, odd = R.draws(R.id_test_groups(), n, rng), R.draws(R.ood_groups(o), n, rng)
        for mult in MULTS:
            per = {s: (scores[(mult, s)]["mahalanobis_classcond"][0], scores[(mult, s)]["mahalanobis_classcond"][1][o])
                   for s in SEEDS if (mult, s) in scores}
            if len(per) >= 2:
                boot[(mult, o)] = R.pooled_bootstrap(per, idd, odd)
    cols = ["leakage", "id_bal_acc", "id_ece", "ood_ece", "mahalanobis_classcond|pad_heldout", "mahalanobis_classcond|fitzpatrick17k", "xdom_bal_acc"]
    L = ["# Paper B R5 P4 (GPU part) — adversary LR ×10 / ×100 at λ = 2, lesion-level", "",
         "Design: `decisions/paperB_r5_p4.md`. Commit `{}`. Seeds 42/52/62. ×30 = the existing Table 1 runs of the same seeds. "
         "Rule A here needs 3/3 seeds (n = 3), stated as k/3.".format(R.git_head()), "",
         "| adv LR | n | Leakage | ID bal acc | ID ECE | OOD ECE | OOD AUROC (Maha pad) | Fitz AUROC | X-dom bal acc |", "|---|---:|" + "---|" * 7]
    tex = []
    for mult in MULTS:
        d = df[df["mult"] == mult]
        if not len(d):
            continue
        L.append("| ×{} | {} | {} |".format(mult, len(d), " | ".join(R.fmt(d[c].mean(), d[c].std(ddof=1)) for c in cols)))
        tex.append("$\\times${} & {} & {} \\\\".format(mult, len(d), " & ".join(R.tex_pm(d[c].mean(), d[c].std(ddof=1)) for c in cols)))
    L += ["", "| adv LR | OOD | " + " | ".join(R.DET_LABEL[d] for d in R.DETS) + " | Maha pooled CI | k/3 CI < 0.5 |", "|---|---|" + "---|" * (len(R.DETS) + 2)]
    for mult in MULTS:
        d = df[df["mult"] == mult]
        for o in OODS:
            if not len(d):
                continue
            b = boot.get((mult, o))
            L.append("| ×{} | {} | {} | {} | {} |".format(mult, o, " | ".join(R.fmt(d["{}|{}".format(x, o)].mean(), d["{}|{}".format(x, o)].std(ddof=1)) for x in R.DETS),
                                                   "[{:.3f}, {:.3f}]".format(b["pooled_ci_lo"], b["pooled_ci_hi"]) if b else "—",
                                                   "{}/{}".format(b["k_seeds_ci_below_0.5"], b["n_seeds"]) if b else "—"))
    L += ["", "## Plain reading", ""]
    for mult in (10, 100):
        d = df[df["mult"] == mult]
        if len(d):
            L.append("Adversary LR ×{}: leakage {:.3f}, Mahalanobis on pad_heldout {:.3f} (×30: leakage {:.3f}, AUROC {:.3f}; same seeds).".format(
                mult, d.leakage.mean(), d["mahalanobis_classcond|pad_heldout"].mean(), df[df["mult"] == 30].leakage.mean(),
                df[df["mult"] == 30]["mahalanobis_classcond|pad_heldout"].mean()))
    L += ["", "## LaTeX rows (adv LR & n & Table 1 columns)", "", "```latex"] + tex + ["```", ""]
    (ADV / "REPORT.md").write_text("\n".join(L) + "\n")
    R.write_json(ADV / "bootstrap.json", {"{}|{}".format(*k): v for k, v in boot.items()})
    print("wrote", ADV / "REPORT.md")


if __name__ == "__main__":
    main()
