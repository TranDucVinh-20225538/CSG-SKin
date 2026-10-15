#!/usr/bin/env python3
"""Paper B R5 P2: §4.4 (confidence and calibration) on the 35 lesion-level runs (decisions/paperB_r5_p2.md).

Per run, from cached logits: mean MSP on ISIC test vs pad_heldout (and Fitzpatrick17k), the gap, ID ECE (8-class),
OOD ECE on pad_heldout (8-class = Table 1 definition; 6-class restricted as secondary), per-class recall on
pad_heldout under the 6-class restriction (AK recall is the §4.4 number), predicted-NV share on pad_heldout.
Cross-check against results/paperB/r3/c/per_run (analyze_phase13, same features): max |difference| reported.
Usage: r5_p2_confidence.py [--report_only]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
from phase15_common import six_class_xfer  # noqa: E402

OUT = R.R5 / "p2"
R3C = R.RES / "r3" / "c" / "per_run"


def analyse_run(lam, seed, tag):
    te, ho, fz = (R.load(tag, s) for s in ("isic_test", "pad_heldout", "fitzpatrick17k"))
    msp = {k: R.softmax(v["logits"]).max(1) for k, v in (("isic_test", te), ("pad_heldout", ho), ("fitzpatrick17k", fz))}
    x6 = six_class_xfer(ho["logits"], ho["labels"])
    rec = {"lambda": lam, "seed": seed, "run": tag,
           "msp_mean": {k: float(v.mean()) for k, v in msp.items()},
           "msp_median": {k: float(np.median(v)) for k, v in msp.items()},
           "gap_pad_minus_id": float(msp["pad_heldout"].mean() - msp["isic_test"].mean()),
           "gap_fitz_minus_id": float(msp["fitzpatrick17k"].mean() - msp["isic_test"].mean()),
           "frac_pad_msp_above_id_median": float((msp["pad_heldout"] > np.median(msp["isic_test"])).mean()),
           "id_ece": R.ece15(te["logits"], te["labels"]),
           "ood_ece_pad_heldout_8cls": R.ece15(ho["logits"], ho["labels"]),
           "ood_ece_pad_heldout_6cls": R.ece15(ho["logits"], ho["labels"], restrict=list(R.PAD_CLASSES)),
           "xfer6": x6, "ak_recall_6cls": x6["per_class_recall"]["AK"],
           "ak_recall_8cls": float((ho["logits"][ho["labels"] == R.AK].argmax(1) == R.AK).mean()),
           "pred_nv_share_pad_8cls": float((ho["logits"].argmax(1) == R.NV).mean()),
           "id_bal_acc": float(pd.Series(te["logits"].argmax(1) == te["labels"]).groupby(te["labels"]).mean().mean())}
    old = R3C / "{}.json".format(tag)
    if old.exists():
        c = json.loads(old.read_text())["confidence"]
        rec["check_vs_r3c"] = {
            "msp_isic_test": rec["msp_mean"]["isic_test"] - c["isic_test"]["msp_mean"],
            "msp_pad_heldout": rec["msp_mean"]["pad_heldout"] - c["pad_heldout"]["msp_mean"],
            "ece_isic_test": rec["id_ece"] - c["isic_test"]["ece"],
            "ece_pad_heldout": rec["ood_ece_pad_heldout_8cls"] - c["pad_heldout"]["ece"]}
    return rec


COLS = (("msp_id", "Mean MSP, ISIC test"), ("msp_pad", "Mean MSP, pad_heldout"), ("gap_pad_minus_id", "Gap (pad − ID)"),
        ("msp_fitz", "Mean MSP, Fitzpatrick"), ("id_ece", "ID ECE"), ("ood_ece_pad_heldout_8cls", "OOD ECE (8-cls)"),
        ("ood_ece_pad_heldout_6cls", "OOD ECE (6-cls)"), ("ak_recall_6cls", "AK recall (6-cls)"),
        ("pred_nv_share_pad_8cls", "Predicted NV share (pad)"))


def report():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "per_run").glob("*.json"))]
    flat = []
    for r in rows:
        f = {"lambda": r["lambda"], "seed": r["seed"], "msp_id": r["msp_mean"]["isic_test"], "msp_pad": r["msp_mean"]["pad_heldout"],
             "msp_fitz": r["msp_mean"]["fitzpatrick17k"]}
        for k in ("gap_pad_minus_id", "gap_fitz_minus_id", "id_ece", "ood_ece_pad_heldout_8cls", "ood_ece_pad_heldout_6cls",
                  "ak_recall_6cls", "ak_recall_8cls", "pred_nv_share_pad_8cls", "frac_pad_msp_above_id_median"):
            f[k] = r[k]
        for c, v in r["xfer6"]["per_class_recall"].items():
            f["recall6_" + c] = v
        f["xdom_bal_acc6"] = r["xfer6"]["balanced_accuracy"]
        for k, v in r.get("check_vs_r3c", {}).items():
            f["check_" + k] = v
        flat.append(f)
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "summary.csv", index=False)
    lams = sorted(df["lambda"].unique())
    chk = [c for c in df.columns if c.startswith("check_")]
    L = ["# Paper B R5 P2 — §4.4 confidence and calibration, lesion-level runs", "",
         "Design: `decisions/paperB_r5_p2.md`. Commit `{}`. {} runs. MSP = max softmax over 8 classes. ECE 15 bins. "
         "AK recall: pad_heldout AK images (n = 202), 6-class restricted argmax (phase15_common.six_class_xfer). "
         "Mean ± s.d. over seeds.".format(R.git_head(), len(rows)), "",
         "| λ | n | " + " | ".join(t for _, t in COLS) + " |", "|---:|---:|" + "---|" * len(COLS)]
    tex = []
    for lam in lams:
        d = df[df["lambda"] == lam]
        cells = [R.fmt(d[k].mean(), d[k].std(ddof=1)) for k, _ in COLS]
        L.append("| {:g} | {} | {} |".format(lam, len(d), " | ".join(cells)))
        tex.append("{:g} & {} & {} \\\\".format(lam, len(d), " & ".join(R.tex_pm(d[k].mean(), d[k].std(ddof=1)) for k, _ in COLS)))
    L += ["", "Per-class recall on pad_heldout, 6-class restricted:", "",
          "| λ | " + " | ".join(c for c in ("MEL", "NV", "BCC", "AK", "BKL", "SCC")) + " | X-dom bal acc |",
          "|---:|" + "---|" * 7]
    for lam in lams:
        d = df[df["lambda"] == lam]
        L.append("| {:g} | {} | {} |".format(lam, " | ".join(R.fmt(d["recall6_" + c].mean(), d["recall6_" + c].std(ddof=1))
                                                       for c in ("MEL", "NV", "BCC", "AK", "BKL", "SCC")),
                                            R.fmt(d.xdom_bal_acc6.mean(), d.xdom_bal_acc6.std(ddof=1))))
    if chk:
        L += ["", "Cross-check vs `r3/c/per_run` (same features): max |difference| " +
              ", ".join("{} {:.1e}".format(c[6:], df[c].abs().max()) for c in chk) + "."]
    n_pos = {lam: int((df[df["lambda"] == lam].gap_pad_minus_id > 0).sum()) for lam in lams}
    d0, d2 = df[df["lambda"] == 0.0], df[df["lambda"] == 2.0]
    L += ["", "## Plain reading", "",
          "Mean confidence on pad_heldout minus mean confidence on ISIC test is {} at λ = 0 and {} at λ = 2; seeds with "
          "pad_heldout more confident than ISIC test, per λ: {}. OOD ECE (8-class) is {} at λ = 0 and {} at λ = 2 while ID ECE "
          "is {} and {}. AK recall on pad_heldout (6-class) is {} at λ = 0 and {} at λ = 2.".format(
              R.fmt(d0.gap_pad_minus_id.mean(), d0.gap_pad_minus_id.std(ddof=1)), R.fmt(d2.gap_pad_minus_id.mean(), d2.gap_pad_minus_id.std(ddof=1)),
              ", ".join("{:g}: {}/{}".format(l, n_pos[l], int((df["lambda"] == l).sum())) for l in lams),
              R.fmt(d0.ood_ece_pad_heldout_8cls.mean(), d0.ood_ece_pad_heldout_8cls.std(ddof=1)),
              R.fmt(d2.ood_ece_pad_heldout_8cls.mean(), d2.ood_ece_pad_heldout_8cls.std(ddof=1)),
              R.fmt(d0.id_ece.mean(), d0.id_ece.std(ddof=1)), R.fmt(d2.id_ece.mean(), d2.id_ece.std(ddof=1)),
              R.fmt(d0.ak_recall_6cls.mean(), d0.ak_recall_6cls.std(ddof=1)), R.fmt(d2.ak_recall_6cls.mean(), d2.ak_recall_6cls.std(ddof=1))),
          "", "## LaTeX rows (λ & n & " + " & ".join(t for _, t in COLS) + ")", "", "```latex"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    (OUT / "per_run").mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        rr = R.runs()
        assert len(rr) == 35, len(rr)
        for lam, seed, tag in rr:
            R.write_json(OUT / "per_run" / "{}.json".format(tag), analyse_run(lam, seed, tag))
    report()


if __name__ == "__main__":
    main()
