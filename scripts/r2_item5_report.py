#!/usr/bin/env python3
"""R2 Item 5 gate evaluation and report. Rules: results/paperB/r2/PRECOMMIT_CAMELYON2.json.

--stage coarse: apply gates to the coarse scan; on failure write the negative REPORT.md and stop,
                otherwise submit the dense array (and the final report job) via sbatch.
--stage final : evaluate gates and the pre-committed prediction on the dense set; write REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C

OUT = C.R2 / "item5"
RUNS = OUT / "runs"
PRE = json.loads((C.R2 / "PRECOMMIT_CAMELYON2.json").read_text())
COARSE = PRE["coarse_scan"]["lambdas"]
SEEDS = PRE["dense_rule"]["seeds"]
ADV_CE, LEAK_DROP, ID_MIN, HEAD_LO, HEAD_HI = 0.60, 0.10, 0.85, 0.45, 0.55
DETS = (("mahalanobis_classcond_sharedcov", "Maha"), ("cosine_max", "Cosine"), ("knn_k50", "kNN"),
        ("MSP", "MSP"), ("Energy_T1", "Energy"))


def tag(lam, seed):
    return "camelyon2_ladv{}_s{}".format(C.lam_tag(lam), seed)


def load(lam, seed):
    p = RUNS / tag(lam, seed) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def cols(s):
    o = s["ood"]["b_hold"]
    r = {"leakage": s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"], "id_acc": s["id_test"]["acc"],
         "id_bal": s["id_test"]["balanced_acc"], "id_ece": s["id_test"]["ece"], "ood_ece": o["accuracy"]["ece"],
         "c_maha": s["ood"]["c"]["detectors"]["mahalanobis_classcond_sharedcov"],
         "adv_min_ce": s["adversary"]["min_window_adv_ce"]}
    for k, lab in DETS:
        r[lab] = o["detectors"][k]
    return r


def sig3(x):
    return float("{:.3g}".format(x))


def gate_table(rows):
    L = ["| Gate | Criterion | Value | Pass |", "|---|---|---|---|"]
    for g in rows:
        L.append("| {} | {} | {} | {} |".format(*g))
    return L


def adversary_log_lines(lams_seeds):
    L = ["## Per-epoch adversary logs\n", "| λ | seed | epoch | adv CE | adv acc | ID select acc | GRL coef |",
         "|---:|---:|---:|---|---|---|---|"]
    for lam, seed in lams_seeds:
        s = load(lam, seed)
        if s is None:
            continue
        for h in s["history"]:
            L.append("| {:g} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                lam, seed, h["epoch"], h["adv_ce"], h["adv_acc"], h["id_select_acc"], h["coef_end"]))
    L.append("")
    L.append("First and minimum 200-step windows (adversary CE, accuracy, mean reversed-gradient norm at the features):\n")
    L.append("| λ | seed | first window CE | first window acc | first window ‖g‖ | min window CE | last window CE | last window ‖g‖ |")
    L.append("|---:|---:|---|---|---|---|---|---|")
    for lam, seed in lams_seeds:
        s = load(lam, seed)
        if s is None:
            continue
        a = s["adversary"]
        f, l = a["first_window"], a["last_window"]
        L.append("| {:g} | {} | {:.4f} | {:.4f} | {:.3g} | {:.4f} | {:.4f} | {:.3g} |".format(
            lam, seed, f["adv_ce"], f["adv_acc"], f["grl_grad_norm"], a["min_window_adv_ce"], l["adv_ce"], l["grl_grad_norm"]))
    L.append("")
    return L


def table1(lams, seeds):
    keys = ["leakage", "id_bal", "id_ece", "ood_ece"] + [lab for _, lab in DETS] + ["c_maha"]
    heads = ["Leakage (site probe bal acc)", "ID bal acc", "ID ECE", "OOD ECE (B_heldout)"] + \
            ["{} B_heldout".format(lab) for _, lab in DETS] + ["Hospital C Maha"]
    L = ["| λ | n | " + " | ".join(heads) + " |", "|---:|---:|" + "---|" * len(heads)]
    for lam in lams:
        rs = [cols(s) for s in (load(lam, sd) for sd in seeds) if s is not None]
        if not rs:
            continue
        L.append("| {:g} | {} | ".format(lam, len(rs)) + " | ".join(
            C.fmt(*C.mean_sd([r[k] for r in rs])[:2]) for k in keys) + " |")
    return L


def write(lines):
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


def header(verdict):
    return ["# R2 Item 5 — Camelyon17 restricted to two hospitals\n",
            "Commit: `{}`. Pre-commit: `results/paperB/r2/PRECOMMIT_CAMELYON2.json`.\n".format(C.git_head()),
            "**Verdict:** {}\n".format(verdict),
            "Hospitals: A = 3 (labelled ID), B = 0 (B_adv slides 0,2,3,4,6,7,9; B_heldout slides 1,5,8), C = 2 (never adversarially seen). "
            "Same-modality contrast: the pre-committed prediction is collapse yes, inversion no. This tests whether the collapse "
            "generalises outside dermatology; it does not test the inversion.\n"]


def coarse():
    runs = {lam: load(lam, 42) for lam in COARSE}
    missing = [lam for lam, s in runs.items() if s is None]
    if missing:
        raise SystemExit("coarse runs missing: {}".format(missing))
    c = {lam: cols(s) for lam, s in runs.items()}
    base = c[0]
    head_ok = not (HEAD_LO <= base["Maha"] <= HEAD_HI)
    learned = {lam: c[lam]["adv_min_ce"] < ADV_CE for lam in COARSE if lam > 0}
    moved = {lam: c[lam]["leakage"] <= base["leakage"] - LEAK_DROP for lam in COARSE if lam > 0}
    star = next((lam for lam in COARSE if lam > 0 and learned[lam] and moved[lam]), None)
    id_ok0 = base["id_acc"] > ID_MIN
    id_ok_star = star is not None and c[star]["id_acc"] > ID_MIN
    gates = [
        ("Adversary learned", "some λ>0 with min window CE < 0.60",
         "; ".join("λ={:g}: {:.3f}".format(l, c[l]["adv_min_ce"]) for l in learned), "yes" if any(learned.values()) else "no"),
        ("Leakage moved", "site probe ≤ λ=0 − 0.10 at a λ where the adversary learned (λ=0: {:.3f})".format(base["leakage"]),
         "; ".join("λ={:g}: {:.3f}".format(l, c[l]["leakage"]) for l in moved), "yes" if star is not None else "no"),
        ("ID competence", "A test acc > 0.85 at λ=0 and λ*",
         "λ=0: {:.3f}; λ*: {}".format(base["id_acc"], "{:.3f}".format(c[star]["id_acc"]) if star is not None else "—"),
         "yes" if id_ok0 and id_ok_star else "no"),
        ("Detector headroom", "λ=0 Maha on B_heldout outside [0.45, 0.55]", "{:.3f}".format(base["Maha"]), "yes" if head_ok else "no"),
    ]
    dec = {"lambda_star": star, "gates": gates, "coarse": {str(k): v for k, v in c.items()}}
    failed = [g[0] for g in gates if g[3] == "no"]
    if failed:
        dec["stop"] = True
        (OUT / "coarse_decision.json").write_text(json.dumps(dec, indent=2) + "\n")
        verdict = "Gate failed ({}) -> negative reported; stop. No third design, no fourth dataset.".format(", ".join(failed))
        L = header(verdict) + ["## Gate-by-gate (coarse scan, seed 42)\n"] + gate_table(gates) + [""]
        L += ["## Coarse scan, Table 1 columns (seed 42)\n"] + table1(COARSE, [42]) + [""]
        L += adversary_log_lines([(l, 42) for l in COARSE])
        write(L)
        return
    i = COARSE.index(star)
    dense = [0.0, sig3(star / math.sqrt(3)), float(star)] + ([float(COARSE[i + 1])] if i + 1 < len(COARSE) else [])
    dec.update({"stop": False, "dense": dense})
    (OUT / "coarse_decision.json").write_text(json.dumps(dec, indent=2) + "\n")
    todo = [(l, s) for l in dense for s in SEEDS if load(l, s) is None]
    lams = " ".join("{:g}".format(l) for l, _ in todo)
    seeds = " ".join(str(s) for _, s in todo)
    jid = subprocess.run(["sbatch", "--parsable", "--array=0-{}%4".format(len(todo) - 1),
                          "--export=ALL,LAMS={},SEEDS={}".format(lams.replace(" ", ":"), seeds.replace(" ", ":")),
                          str(C.PAPERB / "slurm" / "r2_item5.sbatch")], capture_output=True, text=True, check=True).stdout.strip()
    rep = subprocess.run(["sbatch", "--parsable", "--dependency=afterany:{}".format(jid), "-J", "paperB-r2-i5f",
                          "--cpus-per-task=4", "--mem=16G", "--time=01:00:00", str(C.PAPERB / "slurm" / "r2_cpu.sbatch"),
                          "scripts/r2_item5_report.py", "--stage", "final"], capture_output=True, text=True, check=True).stdout.strip()
    print("lambda*", star, "dense", dense, "submitted", jid, "final report", rep, flush=True)


def final():
    dec = json.loads((OUT / "coarse_decision.json").read_text())
    dense, star = dec["dense"], dec["lambda_star"]
    m = {}
    for lam in dense:
        rs = [cols(s) for s in (load(lam, sd) for sd in SEEDS) if s is not None]
        m[lam] = {k: C.mean_sd([r[k] for r in rs]) for k in rs[0]}
    base = m[0.0]
    head_ok = not (HEAD_LO <= base["Maha"][0] <= HEAD_HI)
    id_ok = all(m[l]["id_acc"][0] > ID_MIN for l in dense)
    adv_ok = m[star]["adv_min_ce"][0] < ADV_CE
    leak_ok = m[star]["leakage"][0] <= base["leakage"][0] - LEAK_DROP
    gates = [
        ("Adversary learned", "mean min-window CE at λ* < 0.60", "{:.3f}".format(m[star]["adv_min_ce"][0]), "yes" if adv_ok else "no"),
        ("Leakage moved", "mean site probe at λ* ≤ λ=0 − 0.10", "{:.3f} vs {:.3f}".format(m[star]["leakage"][0], base["leakage"][0]),
         "yes" if leak_ok else "no"),
        ("ID competence", "mean A test acc > 0.85 at every dense λ",
         "; ".join("λ={:g}: {:.3f}".format(l, m[l]["id_acc"][0]) for l in dense), "yes" if id_ok else "no"),
        ("Detector headroom", "mean λ=0 Maha on B_heldout outside [0.45, 0.55]", "{:.3f}".format(base["Maha"][0]), "yes" if head_ok else "no"),
    ]
    failed = [g[0] for g in gates if g[3] == "no"]
    a0 = base["Maha"][0] - 0.5
    collapse = abs(m[star]["Maha"][0] - 0.5) <= 0.5 * abs(a0)
    inversion = any(np.sign(m[l]["Maha"][0] - 0.5) == -np.sign(a0) and abs(m[l]["Maha"][0] - 0.5) >= 0.05 for l in dense if l > 0)
    if failed:
        verdict = "Gate failed on the dense set ({}) -> negative reported; stop.".format(", ".join(failed))
    else:
        verdict = "Collapse {}, inversion {} -> prediction (collapse yes, inversion no) {}.".format(
            "yes" if collapse else "no", "yes" if inversion else "no",
            "confirmed" if (collapse and not inversion) else "not confirmed")
    L = header(verdict)
    L += ["λ* = {:g}; dense set {}; seeds {}.\n".format(star, dense, SEEDS)]
    L += ["## Gate-by-gate\n"] + gate_table(gates) + [""]
    L += ["Coarse-scan gates (seed 42), from `coarse_decision.json`:\n"] + gate_table(dec["gates"]) + [""]
    L += ["## Prediction\n", "| Quantity | Value |", "|---|---|",
          "| Maha B_heldout, λ=0 | {} |".format(C.fmt(*base["Maha"][:2])),
          "| Maha B_heldout, λ* | {} |".format(C.fmt(*m[star]["Maha"][:2])),
          "| Collapse (|AUROC−0.5| at λ* ≤ half of λ=0) | {} |".format("yes" if collapse else "no"),
          "| Inversion (opposite side of 0.5 by ≥ 0.05 at any dense λ>0) | {} |".format("yes" if inversion else "no"), ""]
    L += ["## Table 1 columns, dense set (mean ± s.d. over seeds)\n"] + table1(dense, SEEDS) + [""]
    L += ["## Coarse scan (seed 42)\n"] + table1(COARSE, [42]) + [""]
    L += adversary_log_lines([(l, s) for l in dense for s in SEEDS] + [(l, 42) for l in COARSE if l not in dense])
    L += ["## Caveats\n", "- A's ID test is patch-level (WILDS id_val convention) and shares slides with A train.",
          "- B_heldout has 3 slides; detector AUROCs are over patches, not slides."]
    write(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("coarse", "final"), required=True)
    a = ap.parse_args()
    coarse() if a.stage == "coarse" else final()
