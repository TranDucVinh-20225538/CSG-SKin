#!/usr/bin/env python3
"""R2 Item 5 gate evaluation and report. Rules: results/paperB/r2/PRECOMMIT_CAMELYON2.json (incl. amendments 1-3).

--stage coarse --assign primary : headroom failure -> submit the fallback coarse scan (A=3,B=2,C=1), no negative.
                                  other gate failure -> negative REPORT.md, stop. All pass -> submit dense set.
--stage coarse --assign fallback: any gate failure -> negative REPORT.md, stop. All pass -> submit dense set.
--stage final  --assign X       : gates and pre-committed prediction on the dense set; REPORT.md.
Every report carries the λ-placement diagnostic verdict (amendment 3).
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
DIAG = OUT / "diag"
PRE = json.loads((C.R2 / "PRECOMMIT_CAMELYON2.json").read_text())
COARSE = PRE["coarse_scan"]["lambdas"]
SEEDS = PRE["dense_rule"]["seeds"]
ADV_CE, LEAK_DROP, ID_MIN, HEAD_LO, HEAD_HI = 0.60, 0.10, 0.85, 0.45, 0.55
STUCK_CE = 0.65
DIAG_LAMS, DIAG_STEPS = (10, 100), 3000
HOSP = {"primary": (3, 0, 2), "fallback": (3, 2, 1)}
PREFIX = {"primary": "camelyon2", "fallback": "camelyon2fb"}
DETS = (("mahalanobis_classcond_sharedcov", "Maha"), ("cosine_max", "Cosine"), ("knn_k50", "kNN"),
        ("MSP", "MSP"), ("Energy_T1", "Energy"))


def tag(lam, seed, assign):
    return "{}_ladv{}_s{}".format(PREFIX[assign], C.lam_tag(lam), seed)


def load(lam, seed, assign):
    p = RUNS / tag(lam, seed, assign) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def cols(s):
    o = s["ood"]["b_hold"]
    r = {"leakage": s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"],
         "leakage_clean": s["leakage_site_probe_aclean_vs_bhold"]["balanced_acc_mean"],
         "id_acc": s["id_test"]["acc"], "id_bal": s["id_test"]["balanced_acc"], "id_ece": s["id_test"]["ece"],
         "id_acc_clean": s["id_test_slide_disjoint"]["acc"], "id_bal_clean": s["id_test_slide_disjoint"]["balanced_acc"],
         "id_ece_clean": s["id_test_slide_disjoint"]["ece"], "ood_ece": o["accuracy"]["ece"],
         "c_maha": s["ood"]["c"]["detectors"]["mahalanobis_classcond_sharedcov"],
         "c_maha_clean": s["ood"]["c"]["detectors_slide_disjoint_id"]["mahalanobis_classcond_sharedcov"],
         "adv_min_ce": s["adversary"]["min_window_adv_ce"]}
    for k, lab in DETS:
        r[lab] = o["detectors"][k]
        r[lab + "_clean"] = o["detectors_slide_disjoint_id"][k]
    return r


def sig3(x):
    return float("{:.3g}".format(x))


def gate_table(rows):
    return ["| Gate | Criterion | Value | Pass |", "|---|---|---|---|"] + ["| {} | {} | {} | {} |".format(*g) for g in rows]


def diag_windows(lam, placement):
    t = "camelyon2_ladv{}_s42{}_diag{}".format(C.lam_tag(lam), "_placeloss" if placement == "loss" else "", DIAG_STEPS)
    p = DIAG / t / "adversary_windows.json"
    return json.loads(p.read_text()) if p.exists() else None


def placement_section():
    L = ["## λ-placement hypothesis (Phase 12.1b / iWildCam)\n",
         "Phase 12.1b and iWildCam (`train_phase15b_iwildcam.py`, `loss = loss_cls + self.lambda_adv * loss_adv`) multiply λ into the "
         "domain head's own loss, with the GRL at coefficient α and the head at lr ×30. Diagnostic: primary assignment, seed 42, "
         "3000 steps, identical schedule/batches/optimiser; 'loss' = 12.1b form, 'grl' = λ on the reversed gradient only. "
         "Both codebases use AdamW, which is close to invariant to a constant rescaling of the loss.\n",
         "| λ | placement | windows | first-window CE | min-window CE | max-window CE | first-window acc | last-window acc | first ‖g‖ | last ‖g‖ |",
         "|---:|---|---:|---|---|---|---|---|---|---|"]
    res = {}
    for lam in DIAG_LAMS:
        for pl in ("loss", "grl"):
            w = diag_windows(lam, pl)
            res[(lam, pl)] = w
            if not w:
                L.append("| {} | {} | — | missing | | | | | | |".format(lam, pl))
                continue
            ces = [x["adv_ce"] for x in w]
            L.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.3f} | {:.3g} | {:.3g} |".format(
                lam, pl, len(w), ces[0], min(ces), max(ces), w[0]["adv_acc"], w[-1]["adv_acc"], w[0]["grl_grad_norm"], w[-1]["grl_grad_norm"]))
    L.append("")
    if any(v is None for v in res.values()):
        verdict = "INCOMPLETE — diagnostic runs missing; hypothesis not evaluated."
    else:
        loss_stuck = {l: min(x["adv_ce"] for x in res[(l, "loss")]) >= STUCK_CE for l in DIAG_LAMS}
        loss_learn = {l: min(x["adv_ce"] for x in res[(l, "loss")]) < ADV_CE for l in DIAG_LAMS}
        grl_learn = {l: min(x["adv_ce"] for x in res[(l, "grl")]) < ADV_CE for l in DIAG_LAMS}
        grl_stuck = {l: min(x["adv_ce"] for x in res[(l, "grl")]) >= STUCK_CE for l in DIAG_LAMS}
        if all(loss_stuck.values()) and all(grl_learn.values()):
            verdict = ("CONFIRMED. With λ in the head's loss the adversary sits at chance from the first window; with λ on the reversed "
                       "gradient only it learns. The Camelyon17 (Phase 12.1b) and iWildCam failures are implementation failures, not "
                       "dataset properties. The two limitations that attribute them to the datasets must be rewritten. Phase 12 is not rerun.")
        elif any(loss_learn.values()):
            verdict = ("NOT CONFIRMED. The 12.1b placement still trains the adversary on this setup (λ = {}), so placement alone does not "
                       "reproduce the failure. The Camelyon17 and iWildCam limitations stay as written.".format(
                           ", ".join(str(l) for l, v in loss_learn.items() if v)))
        elif all(loss_stuck.values()) and all(grl_stuck.values()):
            verdict = ("NOT CONFIRMED. Both placements stay at chance, so the failure is independent of where λ sits. "
                       "The Camelyon17 and iWildCam limitations stay as written.")
        else:
            bad = ["λ={} {}".format(l, pl) for l in DIAG_LAMS for pl, ok in
                   (("loss stuck", loss_stuck[l]), ("grl learned", grl_learn[l])) if not ok]
            verdict = "INCONCLUSIVE. Cells disagreeing with the confirmation rule: {}.".format("; ".join(bad))
    L.append("**λ-placement verdict:** {}\n".format(verdict))
    return L, verdict


def adversary_log_lines(pairs, assign):
    L = ["## Per-epoch adversary logs\n", "| λ | seed | epoch | adv CE | adv acc | ID select acc | GRL coef |",
         "|---:|---:|---:|---|---|---|---|"]
    for lam, seed in pairs:
        s = load(lam, seed, assign)
        for h in (s or {}).get("history", []):
            L.append("| {:g} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                lam, seed, h["epoch"], h["adv_ce"], h["adv_acc"], h["id_select_acc"], h["coef_end"]))
    L += ["", "200-step windows (full series in `runs/<tag>/adversary_windows.json`):\n",
          "| λ | seed | first CE | first acc | first ‖g‖ | min CE | last CE | last ‖g‖ |", "|---:|---:|---|---|---|---|---|---|"]
    for lam, seed in pairs:
        s = load(lam, seed, assign)
        if s is None:
            continue
        a = s["adversary"]
        f, l = a["first_window"], a["last_window"]
        L.append("| {:g} | {} | {:.4f} | {:.4f} | {:.3g} | {:.4f} | {:.4f} | {:.3g} |".format(
            lam, seed, f["adv_ce"], f["adv_acc"], f["grl_grad_norm"], a["min_window_adv_ce"], l["adv_ce"], l["grl_grad_norm"]))
    L.append("")
    return L


def table1(lams, seeds, assign, clean=False):
    sfx = "_clean" if clean else ""
    keys = ["leakage" + sfx, "id_acc" + sfx, "id_bal" + sfx, "id_ece" + sfx, "ood_ece"] + [lab + sfx for _, lab in DETS] + ["c_maha" + sfx]
    heads = ["Leakage (site probe bal acc)", "ID acc", "ID bal acc", "ID ECE", "OOD ECE (B_heldout)"] + \
            ["{} B_heldout".format(lab) for _, lab in DETS] + ["Hospital C Maha"]
    L = ["| λ | n | " + " | ".join(heads) + " |", "|---:|---:|" + "---|" * len(heads)]
    for lam in lams:
        rs = [cols(s) for s in (load(lam, sd, assign) for sd in seeds) if s is not None]
        if rs:
            L.append("| {:g} | {} | ".format(lam, len(rs)) + " | ".join(
                C.fmt(*C.mean_sd([r[k] for r in rs])[:2]) for k in keys) + " |")
    return L


def tables(lams, seeds, assign, title):
    return (["## {} — ID reference = patch-level A test (primary)\n".format(title)] + table1(lams, seeds, assign) + [""] +
            ["## {} — ID reference = slide-disjoint A slides 30, 32\n".format(title),
             "OOD ECE is not an ID-side number and is repeated unchanged. The slide-disjoint set is 11.3% tumour; compare balanced accuracy.\n"] +
            table1(lams, seeds, assign, clean=True) + [""])


def header(verdict, assign, placement_verdict):
    a, b, c = HOSP[assign]
    return ["# R2 Item 5 — Camelyon17 restricted to two hospitals\n",
            "Commit: `{}`. Pre-commit: `results/paperB/r2/PRECOMMIT_CAMELYON2.json` (with amendments 1–3).\n".format(C.git_head()),
            "**Verdict:** {}\n".format(verdict),
            "**λ-placement hypothesis:** {}\n".format(placement_verdict),
            "Assignment: {} (A = {}, B = {}, C = {}). Same-modality contrast: the pre-committed prediction is collapse yes, inversion no. "
            "This tests whether the collapse generalises outside dermatology; it does not test the inversion.\n".format(assign, a, b, c)]


def write(lines):
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]), flush=True)


def submit(args_list):
    return subprocess.run(["sbatch", "--parsable"] + args_list, capture_output=True, text=True, check=True).stdout.strip()


def submit_runs(pairs, assign):
    lams = ":".join("{:g}".format(l) for l, _ in pairs)
    seeds = ":".join(str(s) for _, s in pairs)
    return submit(["--array=0-{}%4".format(len(pairs) - 1), "--export=ALL,LAMS={},SEEDS={},ASSIGN={}".format(lams, seeds, assign),
                   str(C.PAPERB / "slurm" / "r2_item5.sbatch")])


def submit_report(after, stage, assign, name):
    return submit(["--dependency=afterany:{}".format(after), "-J", name, "--cpus-per-task=4", "--mem=16G", "--time=01:00:00",
                   str(C.PAPERB / "slurm" / "r2_cpu.sbatch"), "scripts/r2_item5_report.py", "--stage", stage, "--assign", assign])


def coarse(assign):
    runs = {lam: load(lam, 42, assign) for lam in COARSE}
    missing = [lam for lam, s in runs.items() if s is None]
    if missing:
        raise SystemExit("coarse runs missing: {}".format(missing))
    c = {lam: cols(s) for lam, s in runs.items()}
    base = c[0]
    head_ok = not (HEAD_LO <= base["Maha"] <= HEAD_HI)
    learned = {lam: c[lam]["adv_min_ce"] < ADV_CE for lam in COARSE if lam > 0}
    moved = {lam: c[lam]["leakage"] <= base["leakage"] - LEAK_DROP for lam in COARSE if lam > 0}
    star = next((lam for lam in COARSE if lam > 0 and learned[lam] and moved[lam]), None)
    id_ok = base["id_acc"] > ID_MIN and star is not None and c[star]["id_acc"] > ID_MIN
    gates = [
        ("Adversary learned", "some λ>0 with min window CE < 0.60",
         "; ".join("λ={:g}: {:.3f}".format(l, c[l]["adv_min_ce"]) for l in learned), "yes" if any(learned.values()) else "no"),
        ("Leakage moved", "site probe ≤ λ=0 − 0.10 where the adversary learned (λ=0: {:.3f})".format(base["leakage"]),
         "; ".join("λ={:g}: {:.3f}".format(l, c[l]["leakage"]) for l in moved), "yes" if star is not None else "no"),
        ("ID competence", "A test acc > 0.85 at λ=0 and λ*",
         "λ=0: {:.3f}; λ*: {}".format(base["id_acc"], "{:.3f}".format(c[star]["id_acc"]) if star is not None else "—"),
         "yes" if id_ok else "no"),
        ("Detector headroom", "λ=0 Maha on B_heldout outside [0.45, 0.55]", "{:.3f}".format(base["Maha"]), "yes" if head_ok else "no"),
    ]
    failed = [g[0] for g in gates if g[3] == "no"]
    dec = {"assign": assign, "lambda_star": star, "gates": gates, "coarse": {str(k): v for k, v in c.items()}}
    pl, pv = placement_section()
    coarse_block = (["## Gate-by-gate (coarse scan, seed 42, assignment {})\n".format(assign)] + gate_table(gates) + [""] +
                    tables(COARSE, [42], assign, "Coarse scan (seed 42)") + adversary_log_lines([(l, 42) for l in COARSE], assign))

    if assign == "primary" and not head_ok:
        pairs = [(l, 42) for l in COARSE]
        jid = submit_runs(pairs, "fallback")
        rep = submit_report(jid, "coarse", "fallback", "paperB-r2-i5c-fb")
        dec.update({"stop": False, "switch_to_fallback": True, "fallback_coarse_job": jid, "fallback_report_job": rep})
        (OUT / "coarse_decision_primary.json").write_text(json.dumps(dec, indent=2) + "\n")
        write(header("Detector-headroom gate failed under the primary assignment (λ=0 Maha {:.3f}) -> not a negative; "
                     "switched to the fallback A=3, B=2, C=1 and reran the coarse scan once (jobs {}, {}).".format(base["Maha"], jid, rep),
                     assign, pv) + coarse_block + pl)
        return
    if failed:
        dec["stop"] = True
        (OUT / "coarse_decision_{}.json".format(assign)).write_text(json.dumps(dec, indent=2) + "\n")
        prior = []
        if assign == "fallback":
            p = OUT / "coarse_decision_primary.json"
            if p.exists():
                prior = ["## Primary assignment (A=3, B=0, C=2): headroom gate failed, switched to fallback\n"] + \
                        gate_table(json.loads(p.read_text())["gates"]) + [""] + \
                        tables(COARSE, [42], "primary", "Primary coarse scan (seed 42)")
        write(header("Gate failed ({}) under the {} assignment -> negative reported; stop. No third design, no fourth dataset.".format(
            ", ".join(failed), assign), assign, pv) + coarse_block + prior + pl)
        return
    i = COARSE.index(star)
    dense = [0.0, sig3(star / math.sqrt(3)), float(star)] + ([float(COARSE[i + 1])] if i + 1 < len(COARSE) else [])
    todo = [(l, s) for l in dense for s in SEEDS if load(l, s, assign) is None]
    jid = submit_runs(todo, assign)
    rep = submit_report(jid, "final", assign, "paperB-r2-i5f")
    dec.update({"stop": False, "dense": dense, "dense_job": jid, "final_report_job": rep})
    (OUT / "coarse_decision_{}.json".format(assign)).write_text(json.dumps(dec, indent=2) + "\n")
    print("lambda*", star, "dense", dense, "submitted", jid, rep, flush=True)


def final(assign):
    dec = json.loads((OUT / "coarse_decision_{}.json".format(assign)).read_text())
    dense, star = dec["dense"], dec["lambda_star"]
    m = {}
    for lam in dense:
        rs = [cols(s) for s in (load(lam, sd, assign) for sd in SEEDS) if s is not None]
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
            "yes" if collapse else "no", "yes" if inversion else "no", "confirmed" if (collapse and not inversion) else "not confirmed")
    pl, pv = placement_section()
    L = header(verdict, assign, pv) + ["λ* = {:g}; dense set {}; seeds {}.\n".format(star, dense, SEEDS)]
    L += ["## Gate-by-gate (dense set)\n"] + gate_table(gates) + [""]
    L += ["Coarse-scan gates (seed 42):\n"] + gate_table(dec["gates"]) + [""]
    L += ["## Prediction\n", "| Quantity | Value |", "|---|---|",
          "| Maha B_heldout, λ=0 | {} |".format(C.fmt(*base["Maha"][:2])),
          "| Maha B_heldout, λ* | {} |".format(C.fmt(*m[star]["Maha"][:2])),
          "| Collapse (|AUROC−0.5| at λ* ≤ half of λ=0) | {} |".format("yes" if collapse else "no"),
          "| Inversion (opposite side of 0.5 by ≥ 0.05 at any dense λ>0) | {} |".format("yes" if inversion else "no"), ""]
    L += tables(dense, SEEDS, assign, "Dense set (mean ± s.d. over seeds)")
    L += tables(COARSE, [42], assign, "Coarse scan (seed 42)")
    L += adversary_log_lines([(l, s) for l in dense for s in SEEDS] + [(l, 42) for l in COARSE if l not in dense], assign)
    L += pl
    L += ["## Caveats\n", "- The primary A ID test is patch-level and shares slides with A train; the slide-disjoint tables use withheld slides 30 and 32.",
          "- B_heldout has 3 slides; detector AUROCs are over patches, not slides."]
    write(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("coarse", "final"), required=True)
    ap.add_argument("--assign", choices=("primary", "fallback"), default="primary")
    a = ap.parse_args()
    coarse(a.assign) if a.stage == "coarse" else final(a.assign)
