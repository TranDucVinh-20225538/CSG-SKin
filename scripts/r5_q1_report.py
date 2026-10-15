#!/usr/bin/env python3
"""Paper B R5 Q1 report: headroom gate, λ sweep, conditional dose test (decisions/paperB_r5_q1.md).

Reads results/paperB/r5/q1/{gate,sweep,dose}/*/summary.json and scores.npz (CPU only, no model or image access).
Writes results/paperB/r5/q1/{summary.csv, GATE.json, INVERSION.json, BOOTSTRAP.json, REPORT.md}.

Headroom gate (λ = 0, seeds 42, 52, 62, Shenzhen Mahalanobis AUROC, PREREGISTER_BOOTSTRAP draws, ID resampled by NIH
patient, Shenzhen by image): "clearly above chance" = mean ≥ 0.60 AND pooled-over-seeds 95 % CI lower bound ≥ 0.55
AND every seed's 95 % CI lower bound > 0.50.
Inversion (decides the conditional dose test): rule A on Shenzhen Mahalanobis at λ = 2 (pooled CI entirely below 0.5
and ≥ 3 of 5 per-seed CIs entirely below 0.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import r5_q1_cxr_data as Q  # noqa: E402

Q1 = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5/q1")
PREREG = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/PREREGISTER_BOOTSTRAP.json")
BOOT_DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")
GATE_SEEDS = (42, 52, 62)
GATE_MEAN, GATE_POOLED_LO, GATE_SEED_LO = 0.60, 0.55, 0.50
REPLACEMENT_OOD = [
    ("Shenzhen TB subset only (336)", "DST-Skin/data/staged/p2a_ood_256/shenzhen.tar, *_1.png",
     "same file; already scored as a descriptive subset; label-shifted (disease) not domain-only"),
    ("Kermany held-out patients (1,758 images / 933 patients)", "kerm_heldout index",
     "the adversary's own domain at new patients (pad_heldout analogue); already scored"),
    ("Kermany OCT (retina, cross-modality far-OOD)", "DST-Skin/data/staged/kermany_256.tar",
     "different modality; easy far-OOD, not a same-modality test"),
    ("Brain MRI (Cheng)", "DST-Skin/data/staged/brain_cheng_256.tar", "different modality; far-OOD"),
    ("PneumoniaMNIST / OCTMNIST", "MedMNIST derivatives of Kermany CXR / OCT", "duplicates of Kermany at 28-224 px"),
]
NOT_ON_DISK = "CheXpert, PadChest, MIMIC-CXR, Montgomery: not on disk (would need a download and a data-use agreement)."


def ms(xs, nd=3):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if a.size == 0:
        return "—"
    return "{:.{}f}".format(a[0], nd) if a.size == 1 else "{:.{}f} ± {:.{}f}".format(a.mean(), nd, a.std(ddof=1), nd)


def tex(xs, nd=3):
    return ms(xs, nd).replace(" ± ", "\\,$\\pm$\\,").replace("—", "--")


def load_runs(recipe_suffix):
    rows = []
    for st in ("gate", "sweep", "dose"):
        for f in sorted((Q1 / st).glob("*/summary.json")):
            s = json.loads(f.read_text())
            tag = s["tag"]
            if recipe_suffix and not tag.endswith(recipe_suffix):
                continue
            if not recipe_suffix and ("_rec" in tag):
                continue
            s["_dir"] = str(f.parent)
            rows.append(s)
    return rows


def flat(s):
    o = s["ood"]
    return {"tag": s["tag"], "stage": s["stage"], "seed": s["seed"], "lambda_adv": s["lambda_adv"],
            "adv_mix_shift": s.get("adv_mix_shift"), "leakage": s["leakage"]["test_vs_kerm_full"]["bal_acc_mean"],
            "leakage_kerm_heldout": s["leakage"]["test_vs_kerm_heldout"]["bal_acc_mean"],
            "id_bal_acc": s["id"]["balanced_acc"], "id_auroc": s["id"]["auroc_pneumonia"], "id_ece": s["id"]["ece"],
            "shz_maha": o["shenzhen"]["mahalanobis_classcond"], "shz_knn": o["shenzhen"]["knn_k50"],
            "shz_cos": o["shenzhen"]["cosine_max"], "shz_msp": o["shenzhen"]["MSP"],
            "shz_tb_maha": o["shenzhen"]["subset_tb"]["mahalanobis_classcond"],
            "shz_normal_maha": o["shenzhen"]["subset_normal"]["mahalanobis_classcond"],
            "shz_majority_nearest": o["shenzhen"]["majority_nearest_frac"],
            "kh_maha": o["kerm_heldout"]["mahalanobis_classcond"], "kh_knn": o["kerm_heldout"]["knn_k50"],
            "kh_reweighted": (o["kerm_heldout"].get("decomposition") or {}).get("reweighted_to_id_mix"),
            "kh_normal_maha": o["kerm_heldout"]["per_ood_class_maha"]["normal"]["auroc"],
            "kh_pneu_maha": o["kerm_heldout"]["per_ood_class_maha"]["pneumonia"]["auroc"],
            "xdom_bal_acc": s["xdom_kerm_heldout"]["balanced_acc"], "ood_ece": s["xdom_kerm_heldout"]["ece"],
            "gate_status": s["gate"]["status"], "gpu_hours": s.get("gpu_hours")}


def groups(ids):
    _, inv = np.unique(np.asarray(ids).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


def fast_auroc(sid, sood):
    r = pd.Series(np.r_[sid, sood]).rank().to_numpy()
    n0, n1 = len(sid), len(sood)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def boot(per_seed, gi, go, rng, n_boot):
    di = [np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))]) for _ in range(n_boot)]
    do = [np.concatenate([go[j] for j in rng.integers(0, len(go), len(go))]) for _ in range(n_boot)]
    per, draws = {}, {}
    for s, (a, b) in per_seed.items():
        d = np.array([fast_auroc(a[di[k]], b[do[k]]) for k in range(n_boot)])
        draws[s] = d
        lo, hi = np.quantile(d, [0.025, 0.975])
        per[str(s)] = {"point": fast_auroc(a, b), "ci_lo": float(lo), "ci_hi": float(hi), "below": bool(hi < 0.5)}
    pooled = np.mean(np.stack(list(draws.values())), 0)
    lo, hi = np.quantile(pooled, [0.025, 0.975])
    k = int(sum(p["below"] for p in per.values()))
    rule = ("A" if k >= 3 else "B") if hi < 0.5 else ("C" if lo < 0.5 else "above")
    return {"mean": float(np.mean([p["point"] for p in per.values()])), "pooled_ci": [float(lo), float(hi)],
            "k_below": k, "n_seeds": len(per), "rule": rule, "per_seed": per}


def cell_key(s):
    sh = s.get("adv_mix_shift")
    return ("dose" if sh is not None else "main", float(s["lambda_adv"]), None if sh is None else float(sh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="current")
    a = ap.parse_args()
    suffix = "" if a.recipe == "current" else ("_recp8" if a.recipe == "p8" else "_rec" + Path(a.recipe).stem)
    runs = load_runs(suffix)
    out = Q1 if not suffix else Q1 / ("recipe" + suffix)
    out.mkdir(parents=True, exist_ok=True)
    if not runs:
        print("no Q1 runs yet")
        return
    df = pd.DataFrame([flat(s) for s in runs]).sort_values(["stage", "adv_mix_shift", "lambda_adv", "seed"],
                                                           na_position="first")
    df.to_csv(out / "summary.csv", index=False)

    p = json.loads(PREREG.read_text())
    n_boot, rseed = int(p["n_resamples"]), int(p["rng_seed"])
    S = Q.load_index()
    g_id = groups(S["nih_test"].patient)
    g_shz = [np.array([i]) for i in range(len(S["shenzhen"]))]
    g_kh = groups(S["kerm_heldout"].patient)
    cells = defaultdict(dict)
    for s in runs:
        cells[cell_key(s)][int(s["seed"])] = s["_dir"]
    B = {}
    for key, seeds in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        sc = {sd: dict(np.load(Path(d) / "scores.npz")) for sd, d in seeds.items()}
        rng = np.random.default_rng(rseed)
        blk = {}
        for ood, go in (("shenzhen", g_shz), ("kerm_heldout", g_kh)):
            blk[ood] = {d: boot({sd: (v["id__" + d], v["{}__{}".format(ood, d)]) for sd, v in sc.items()},
                                g_id, go, rng, n_boot) for d in BOOT_DETS}
        B["{}|lam{}|shift{}".format(*key)] = blk

    gate_runs = {sd: d for sd, d in cells.get(("main", 0.0, None), {}).items() if sd in GATE_SEEDS}
    gate = {"criterion": {"mean_ge": GATE_MEAN, "pooled_ci_lo_ge": GATE_POOLED_LO, "every_seed_ci_lo_gt": GATE_SEED_LO},
            "seeds": sorted(gate_runs)}
    if len(gate_runs) == len(GATE_SEEDS):
        sc = {sd: dict(np.load(Path(d) / "scores.npz")) for sd, d in gate_runs.items()}
        gb = boot({sd: (v["id__mahalanobis_classcond"], v["shenzhen__mahalanobis_classcond"]) for sd, v in sc.items()},
                  g_id, g_shz, np.random.default_rng(rseed), n_boot)
        met = (gb["mean"] >= GATE_MEAN and gb["pooled_ci"][0] >= GATE_POOLED_LO
               and all(v["ci_lo"] > GATE_SEED_LO for v in gb["per_seed"].values()))
        gate.update({"bootstrap": gb, "met": bool(met)})
    else:
        gate.update({"met": None, "note": "gate seeds incomplete"})
    (out / "GATE.json").write_text(json.dumps(gate, indent=2) + "\n")

    l2 = B.get("main|lam2.0|shiftNone", {}).get("shenzhen", {}).get("mahalanobis_classcond")
    inv = {"criterion": "rule A on Shenzhen Mahalanobis at λ=2 (5 seeds)",
           "n_seeds": None if l2 is None else l2["n_seeds"], "rule": None if l2 is None else l2["rule"],
           "inverted": None if l2 is None or l2["n_seeds"] < 5 else bool(l2["rule"] == "A"),
           "bootstrap": l2}
    (out / "INVERSION.json").write_text(json.dumps(inv, indent=2) + "\n")
    (out / "BOOTSTRAP.json").write_text(json.dumps(B, indent=2) + "\n")

    L = ["# Paper B R5 Q1: chest X-ray second task (NIH -> Kermany, OOD Shenzhen)", "",
         "Recipe: `{}`. Runs: {}. Bootstrap: {} draws, seed {}, ID by NIH patient, Shenzhen by image, kerm_heldout by "
         "patient. Mean ± SD over seeds.".format(a.recipe, len(runs), n_boot, rseed), ""]
    mix = Q.OUT / "CLASS_MIX.json"
    if mix.exists():
        m = json.loads(mix.read_text())
        L += ["## Class mix per domain", "", "| set | n | patients | normal | pneumonia | pneumonia frac |",
              "|---|---:|---:|---:|---:|---:|"]
        for n, v in m.items():
            if n == "shenzhen":
                L.append("| shenzhen (OOD) | {} | {} | {} normal | {} TB | — |".format(v["n"], v["n_patients"], v["normal"],
                                                                                     v["tb"]))
            else:
                L.append("| {} | {} | {} | {} | {} | {:.3f} |".format(n, v["n"], v["n_patients"], v["normal"],
                                                                       v["pneumonia"], v["pneumonia_frac"]))
        L.append("")
    L += ["## Headroom gate (λ = 0, seeds 42, 52, 62, Shenzhen Mahalanobis)", ""]
    if gate.get("bootstrap"):
        gb = gate["bootstrap"]
        L.append("Mean {:.3f}; pooled 95 % CI [{:.3f}, {:.3f}]; per-seed CI lower bounds {}. Criterion: mean ≥ {:.2f}, "
                 "pooled lower bound ≥ {:.2f}, every seed lower bound > {:.2f}. Gate met: **{}**.".format(
                     gb["mean"], *gb["pooled_ci"], ", ".join("{:.3f}".format(v["ci_lo"]) for v in gb["per_seed"].values()),
                     GATE_MEAN, GATE_POOLED_LO, GATE_SEED_LO, gate["met"]))
        if not gate["met"]:
            L += ["", "The sweep stage does not launch (decisions file stop rule). Replacement OOD candidates on disk:", ""]
            L += ["- {} — `{}` — {}".format(*r) for r in REPLACEMENT_OOD]
            L += ["- " + NOT_ON_DISK]
    else:
        L.append("Gate seeds incomplete ({} of 3).".format(len(gate_runs)))
    L += ["", "## λ sweep (paired mode)", "",
          "| λ | n | leakage | ID bal acc | ID AUROC | ID ECE | Shenzhen Maha | kNN | Cos | MSP | Shz-TB Maha | "
          "Shz-normal Maha | kerm_heldout Maha | reweighted to ID mix | X-dom bal acc | majority-nearest (Shz) |",
          "|---:|---:|" + "---:|" * 14]
    main_df = df[df.adv_mix_shift.isna()]
    tex_rows = []
    for lam, g in main_df.groupby("lambda_adv"):
        L.append("| {:g} | {} | {} |".format(lam, len(g), " | ".join(ms(g[c]) for c in (
            "leakage", "id_bal_acc", "id_auroc", "id_ece", "shz_maha", "shz_knn", "shz_cos", "shz_msp", "shz_tb_maha",
            "shz_normal_maha", "kh_maha", "kh_reweighted", "xdom_bal_acc", "shz_majority_nearest"))))
        tex_rows.append("CXR $\\lambda={:g}$ & {} \\\\".format(lam, " & ".join(tex(g[c]) for c in (
            "leakage", "id_bal_acc", "id_ece", "ood_ece", "shz_maha", "kh_maha", "xdom_bal_acc"))))
    L += ["", "Bootstrap (Mahalanobis):", "", "| cell | OOD | mean | pooled 95 % CI | seeds CI < 0.5 | rule |",
          "|---|---|---:|---|---:|---|"]
    for k, blk in B.items():
        for ood, v in blk.items():
            b = v["mahalanobis_classcond"]
            L.append("| {} | {} | {:.3f} | [{:.3f}, {:.3f}] | {}/{} | {} |".format(k, ood, b["mean"], *b["pooled_ci"],
                                                                             b["k_below"], b["n_seeds"], b["rule"]))
    L += ["", "LaTeX rows (leakage, ID bal acc, ID ECE, OOD ECE kerm_heldout, Shenzhen Maha, kerm_heldout Maha, "
          "X-dom bal acc):", "", "```latex"] + tex_rows + ["```", ""]
    dose_df = df[df.adv_mix_shift.notna()]
    if len(dose_df):
        L += ["## Dose test (λ = 2, adversary-stream class-mix shift kept)", "",
              "| shift kept | n | leakage | ID bal acc | Shenzhen Maha | kerm_heldout Maha | reweighted |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for sh, g in dose_df.groupby("adv_mix_shift"):
            L.append("| {:g} | {} | {} |".format(sh, len(g), " | ".join(ms(g[c]) for c in (
                "leakage", "id_bal_acc", "shz_maha", "kh_maha", "kh_reweighted"))))
        L.append("")
    L += ["## Results in plain words", ""]
    if gate.get("bootstrap"):
        L.append("Headroom. At λ = 0 the NIH-trained model separates Shenzhen from NIH test with Mahalanobis AUROC "
                 "{:.3f} (pooled 95 % CI {:.3f}-{:.3f}); the pre-set criterion for 'clearly above chance' is {}.".format(
                     gate["bootstrap"]["mean"], *gate["bootstrap"]["pooled_ci"],
                     "met" if gate["met"] else "not met"))
        L.append("")
    for lam, g in main_df.groupby("lambda_adv"):
        b = B.get("main|lam{}|shiftNone".format(float(lam)), {}).get("shenzhen", {}).get("mahalanobis_classcond")
        L.append("λ = {:g} ({} seeds). Leakage {}; ID balanced accuracy {}; Shenzhen Mahalanobis AUROC {}{}; "
                 "kerm_heldout {} (reweighted to the NIH class mix: {}).".format(
                     lam, len(g), ms(g.leakage), ms(g.id_bal_acc), ms(g.shz_maha),
                     "" if b is None else " (pooled CI {:.3f}-{:.3f}, {}/{} seeds below 0.5, rule {})".format(
                         *b["pooled_ci"], b["k_below"], b["n_seeds"], b["rule"]),
                     ms(g.kh_maha), ms(g.kh_reweighted)))
        L.append("")
    if inv["inverted"] is not None:
        L.append("Prediction check. §4.8 predicts collapse toward 0.5 without inversion on same-modality Shenzhen; "
                 "the label-shift account predicts inversion. At λ = 2 the Shenzhen Mahalanobis cell meets rule {} "
                 "(inverted = {}).".format(inv["rule"], inv["inverted"]))
        L.append("")
    for sh, g in dose_df.groupby("adv_mix_shift"):
        L.append("Dose, shift kept = {:g} ({} seeds): Shenzhen Mahalanobis {}, kerm_heldout {}, leakage {}.".format(
            sh, len(g), ms(g.shz_maha), ms(g.kh_maha), ms(g.leakage)))
        L.append("")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
