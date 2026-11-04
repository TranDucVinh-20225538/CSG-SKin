#!/usr/bin/env python3
"""R10 G3: n50 λ=0.5 table rows + class-aware (a)/(b)/(c). Writes r10/G3_n50_l05.* and appends to prior REPORT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_class_aware_heldout_probes as CA  # noqa: E402

LAM = 0.5
ARM = "iwdan_n50"
OUT = R.RES / "r10"
PRIOR = R.RES / "r6" / "r2_prior" / "cur"


def tag(seed):
    return "r2p_cur_{}_ladv{}_s{}".format(ARM, R.lam_tag(LAM), seed)


def msd(xs):
    a = np.asarray(xs, float)
    n = len(a)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if n > 1 else 0.0, "n": n,
            "vals": [float(x) for x in a]}


def fmt(st):
    return "{:.3f} ± {:.3f}".format(st["mean"], st["sd"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = PRIOR / "class_aware"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_root = PRIOR / "features"
    rows = []
    for seed in R.SEEDS:
        t = tag(seed)
        sp = PRIOR / "runs" / t / "summary.json"
        cfgp = PRIOR / "runs" / t / "config.json"
        if not sp.exists():
            raise SystemExit("missing summary {}".format(t))
        js = json.loads(sp.read_text())
        cfg = json.loads(cfgp.read_text())
        m = js["metrics"]
        o = m["ood"]
        cache = cache_dir / "{}.json".format(t)
        if cache.exists():
            ca = json.loads(cache.read_text())
        else:
            if not (feat_root / t / "isic_test.npz").exists():
                raise SystemExit("missing features {}".format(t))
            print("compute class-aware", t, flush=True)
            ca = CA.analyse_run(feat_root, t)
            cache.write_text(json.dumps(ca, indent=2) + "\n")
        rows.append({
            "seed": seed, "tag": t,
            "id": float(m["id"]["bal_acc"]),
            "tv": float(cfg["prior_tv_error"]),
            "leak": float(m["leakage"]["bal_acc_mean"]),
            "pad": float(o["pad_heldout"]["mahalanobis_classcond"]),
            "fitz": float(o["fitzpatrick17k"]["mahalanobis_classcond"]),
            "med": float(o["mednode"]["mahalanobis_classcond"]),
            "sd": float(o["sd198"]["mahalanobis_classcond"]),
            "scin": float(o["scin"]["mahalanobis_classcond"]),
            "a": float(ca["unweighted_a"]["bal_acc_mean"]),
            "b": float(ca["class_balanced_b"]["bal_acc_mean"]),
            "c": float(ca["within_class_c"]["bal_acc_mean"]),
        })
    agg = {k: msd([r[k] for r in rows]) for k in
           ["id", "tv", "leak", "pad", "fitz", "med", "sd", "scin", "a", "b", "c"]}
    payload = {"lambda": LAM, "arm": ARM, "n": len(rows), "per_seed": rows, "agg": agg,
               "commit": R.C.git_head()}
    (OUT / "G3_n50_l05.json").write_text(json.dumps(payload, indent=1) + "\n")

    lines = [
        "# R10 G3 — IWDAN n50 λ=0.5",
        "",
        "Same 50-image draws as λ=2. Commit `{}`.".format(payload["commit"]),
        "",
        "| seed | ID | TV | Leak | pad | Fitz | MED | SD-198 | SCIN | (a) | (b) | (c) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {seed} | {id:.3f} | {tv:.3f} | {leak:.3f} | {pad:.3f} | {fitz:.3f} | {med:.3f} | "
            "{sd:.3f} | {scin:.3f} | {a:.3f} | {b:.3f} | {c:.3f} |".format(**r))
    lines += [
        "",
        "**Mean ± sd (n={})**: ID {}; TV {}; Leak {}; pad {}; Fitz {}; MED {}; SD {}; SCIN {}; "
        "(a) {}; (b) {}; (c) {}.".format(
            len(rows), fmt(agg["id"]), fmt(agg["tv"]), fmt(agg["leak"]), fmt(agg["pad"]),
            fmt(agg["fitz"]), fmt(agg["med"]), fmt(agg["sd"]), fmt(agg["scin"]),
            fmt(agg["a"]), fmt(agg["b"]), fmt(agg["c"])),
        "",
        "LaTeX (tab:prior style):",
        "```",
        "n50 & 0.5 & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & "
        "${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & "
        "${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ & ${:.3f}\\pm{:.3f}$ \\\\".format(
            agg["id"]["mean"], agg["id"]["sd"], agg["tv"]["mean"], agg["tv"]["sd"],
            agg["leak"]["mean"], agg["leak"]["sd"], agg["pad"]["mean"], agg["pad"]["sd"],
            agg["fitz"]["mean"], agg["fitz"]["sd"], agg["med"]["mean"], agg["med"]["sd"],
            agg["sd"]["mean"], agg["sd"]["sd"],
            agg["a"]["mean"], agg["a"]["sd"], agg["b"]["mean"], agg["b"]["sd"],
            agg["c"]["mean"], agg["c"]["sd"]),
        "```",
        "",
    ]
    (OUT / "G3_n50_l05.md").write_text("\n".join(lines))
    # append pointer into prior REPORT
    pref = PRIOR / "REPORT.md"
    if pref.exists():
        text = pref.read_text()
        marker = "## R10 G3: n50 λ=0.5"
        section = "\n\n{}\n\nSee `results/paperB/r10/G3_n50_l05.md` (commit {}). Pad Maha {}; TV {}; (c) {}.\n".format(
            marker, payload["commit"], fmt(agg["pad"]), fmt(agg["tv"]), fmt(agg["c"]))
        if marker in text:
            text = text.split(marker)[0].rstrip() + section
        else:
            text = text.rstrip() + section
        pref.write_text(text)
    print("wrote", OUT / "G3_n50_l05.md", flush=True)


if __name__ == "__main__":
    main()
