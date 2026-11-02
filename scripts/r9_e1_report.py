#!/usr/bin/env python3
"""Paper B R9 E1 report: six-class vs R1 shift 100% / shift 0% (decisions/paperB_r9_e1.md).

Case from λ=2 reading rules first; λ=0.5 reported beside. pad_heldout reported, not counted.
Paired seed diffs + seed-bootstrap 95% CI (2000 draws, rng 20261005).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_report as RR  # noqa: E402

OUT = R.RES / "r9" / "e1"
R1 = R.RES / "r6" / "r1" / "cur"
LABELLED = ("fitzpatrick17k", "mednode", "sd198")
ALL_OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
DRAWS, RNG = 2000, 20261005
DET = "mahalanobis_classcond"
# Paper alignment: R1 shift 0% λ=2 seed 52 is degraded (ID bal < 0.25) but older summary.json
# omitted the flag; force it here so means / paired CIs / Rule A exclude it (as R1).
FORCE_DEGRADED = (
    # (arm_key, lambda, seed) — arm_key in {"six", "s100", "s0"}
    ("s0", 2.0, 52),
)


def load_cell(root, fmt, lam):
    """{seed: metrics} from summary.json (+ features for boot).

    If summary.json is missing but features exist (known R1 shift0 λ=0.5 s82 hole),
    recompute metrics from features so the reference cell still has 5 seeds.
    """
    ms, feats = {}, {}
    for s in R.SEEDS:
        tag = fmt.format(lam=R.lam_tag(lam), s=s)
        sp = root / "runs" / tag / "summary.json"
        fp = root / "features" / tag
        if not (fp / "isic_test.npz").exists():
            continue
        run = root / "runs" / tag if sp.exists() else None
        m, _ = R.metrics_for(fp, run)
        if m is None or "ood" not in m:
            if (fp / "isic_train.npz").exists():
                m = R.run_metrics(R.load_feats(fp))
            else:
                continue
        if m is None or "ood" not in m:
            continue
        ms[s] = m
        feats[s] = fp
    return ms, feats


def seed_means(ms, key_fn, skip_degraded=True):
    vals = []
    for s, m in sorted(ms.items()):
        if skip_degraded and m.get("degraded"):
            continue
        vals.append(float(key_fn(m)))
    if not vals:
        return None
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a)), "per_seed": vals}


def paired_delta(a_ms, b_ms, key_fn, skip_degraded=True):
    """mean(a−b) over matched seeds; seed-bootstrap CI."""
    seeds = sorted(set(a_ms) & set(b_ms))
    used, d = [], []
    for s in seeds:
        if skip_degraded and (a_ms[s].get("degraded") or b_ms[s].get("degraded")):
            continue
        used.append(s)
        d.append(float(key_fn(a_ms[s])) - float(key_fn(b_ms[s])))
    if not d:
        return None
    d = np.asarray(d, float)
    rng = np.random.default_rng(RNG)
    boot = d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(d)),
            "per_seed": {str(s): float(x) for s, x in zip(used, d)}}


def recovery(six, sh100, sh0):
    """(six − 100) / (0 − 100) from seed means."""
    if six is None or sh100 is None or sh0 is None:
        return None
    den = sh0["mean"] - sh100["mean"]
    if abs(den) < 1e-12:
        return None
    return float((six["mean"] - sh100["mean"]) / den)


def rule_a_met(boots, lab, lam, ood):
    b = boots.get((lab, lam, ood))
    if not b or DET not in b or b.get("rule_A") is None:
        return None
    return bool(b["rule_A"])


def decide_case(recov, rule_a):
    """λ=2 reading rules on labelled unseen sets."""
    sets = list(LABELLED)
    ra = {o: rule_a.get(o) for o in sets}
    rv = {o: recov.get(o) for o in sets}
    if any(v is None for v in rv.values()) or any(v is None for v in ra.values()):
        return "reported as found; it settles neither reading", {
            "reason": "missing recovery or Rule A on a labelled set", "recovery": rv, "rule_A": ra}

    met = [o for o in sets if ra[o]]
    # Support not the driver
    if len(met) >= 2 and all(rv[o] < 0.25 for o in met):
        return "Support not the driver", {"met_sets": met, "recovery": rv, "rule_A": ra}
    # Support explains the effect
    if sum(1 for o in sets if rv[o] >= 0.5) >= 2 and len(met) == 0:
        return "Support explains the effect", {"recovery": rv, "rule_A": ra}
    # Support explains part
    if sum(1 for o in sets if rv[o] >= 0.25) >= 2 and len(met) >= 1:
        return "Support explains part", {"met_sets": met, "recovery": rv, "rule_A": ra}
    return "reported as found; it settles neither reading", {"recovery": rv, "rule_A": ra}


def pm(x):
    if x is None:
        return "—"
    return "{:.3f} ± {:.3f}".format(x["mean"], x["sd"])


def ci(d):
    if d is None:
        return "—"
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(d["mean"], d["ci_lo"], d["ci_hi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_boot", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = R.load_data()
    idg = R.idx_groups(R.T.lesion_groups(data["test"])[0])

    cells_src = []
    store = {}  # (arm, lam) -> ms
    for lam in (2.0, 0.5):
        six_ms, six_f = load_cell(OUT, "e1_6cls_ladv{lam}_s{s}", lam)
        s100_ms, s100_f = load_cell(R1, "r1_cur_shift100_ladv{lam}_s{s}", lam)
        s0_ms, s0_f = load_cell(R1, "r1_cur_shift0_ladv{lam}_s{s}", lam)
        store[("six", lam)] = six_ms
        store[("s100", lam)] = s100_ms
        store[("s0", lam)] = s0_ms
        cells_src.append(("six-class", lam, {s: (six_f[s], OUT / "runs" / "e1_6cls_ladv{}_s{}".format(R.lam_tag(lam), s))
                                            for s in six_f}))
        cells_src.append(("shift 100%", lam, {s: (s100_f[s], R1 / "runs" / "r1_cur_shift100_ladv{}_s{}".format(R.lam_tag(lam), s))
                                              for s in s100_f}))
        cells_src.append(("shift 0%", lam, {s: (s0_f[s], R1 / "runs" / "r1_cur_shift0_ladv{}_s{}".format(R.lam_tag(lam), s))
                                            for s in s0_f}))

    for arm, lam, seed in FORCE_DEGRADED:
        ms = store.get((arm, lam))
        if ms is not None and seed in ms:
            ms[seed] = dict(ms[seed], degraded=True)

    missing = [k for k, v in store.items() if len(v) < 5]
    if missing:
        raise SystemExit("incomplete cells: {}".format(missing))

    arm_lab = {"six": "six-class", "s100": "shift 100%", "s0": "shift 0%"}
    excl = {}
    for (arm, lam), ms in store.items():
        lab = arm_lab[arm]
        deg = {s for s, m in ms.items() if m.get("degraded")}
        if deg:
            excl[(lab, lam)] = deg

    boots = {} if a.no_boot else RR.boots_for(cells_src, list(ALL_OODS), data, idg, exclude_by_cell=excl)

    # Build tables + case at λ=2
    rows = []
    recov2, ra2 = {}, {}
    for lam in (2.0, 0.5):
        six, s100, s0 = store[("six", lam)], store[("s100", lam)], store[("s0", lam)]
        for ood in ALL_OODS:
            key = lambda m, o=ood: m["ood"][o][DET]
            m6 = seed_means(six, key)
            m100 = seed_means(s100, key)
            m0 = seed_means(s0, key)
            d100 = paired_delta(six, s100, key)
            d0 = paired_delta(six, s0, key)
            rec = recovery(m6, m100, m0)
            ra = rule_a_met(boots, "six-class", lam, ood)
            rows.append({
                "lambda": lam, "set": ood, "six": m6, "shift100": m100, "shift0": m0,
                "delta_vs_100": d100, "delta_vs_0": d0, "recovery": rec, "rule_A_six": ra,
            })
            if lam == 2.0 and ood in LABELLED:
                recov2[ood] = rec
                ra2[ood] = ra

    case, case_detail = decide_case(recov2, ra2)

    # Leakage / ID / NV-nearest extras for λ=2
    extras = {}
    for lam in (2.0, 0.5):
        six = store[("six", lam)]
        extras[str(lam)] = {
            "id_bal": seed_means(six, lambda m: m["id"]["bal_acc"]),
            "leakage_v1": seed_means(six, lambda m: m["leakage"]["bal_acc_mean"]),
            "nv_nearest_pad": seed_means(six, lambda m: m["ood"]["pad_heldout"]["nv_nearest_frac"]),
            "degraded_seeds": sorted(s for s, m in six.items() if m.get("degraded")),
        }

    # π_6cls check from one config
    cfg = json.loads((OUT / "runs" / "e1_6cls_ladv2_s42" / "config.json").read_text())
    pi_info = {
        "pi_6cls": cfg.get("extra", {}).get("pi_6cls") or cfg.get("adv_mix"),
        "pi_6cls_ref": cfg.get("extra", {}).get("pi_6cls_ref"),
        "tv_to_pi_isic": cfg.get("extra", {}).get("tv_to_pi_isic"),
        "tv_to_pi_pad": cfg.get("extra", {}).get("tv_to_pi_pad"),
        "streams_adv_expected": next(
            (s.get("expected_mix") for s in cfg.get("streams", []) if s.get("name") == "isic_adv"), None),
    }

    outcome = {"case": case, "detail": case_detail, "commit": R.C.git_head(), "pi_6cls": pi_info,
               "rows": rows, "extras": extras,
               "force_degraded": [{"arm": a, "lambda": lam, "seed": s} for a, lam, s in FORCE_DEGRADED],
               "excluded_degraded": {("{}|{:g}".format(k[0], k[1])): sorted(v) for k, v in excl.items()}}
    (OUT / "OUTCOME.json").write_text(json.dumps(outcome, indent=1, default=float) + "\n")

    L = ["# Paper B R9 E1: six-class control", "",
         "Design: `decisions/paperB_r9_e1.md`. Commit `{}`.".format(R.C.git_head()), "",
         "## Reading-rule case (λ = 2 decides)", "",
         "**{}.**".format(case), "",
         "Detail: recovery (labelled) = {}; Rule A (six-class) = {}.".format(
             {k: (None if v is None else round(v, 3)) for k, v in recov2.items()},
             {k: ("met" if v else "not met") if v is not None else None for k, v in ra2.items()}),
         "",
         "Degraded excluded from means (as R1): R1 shift 0% λ=2 seed 52 "
         "(ID bal < 0.25; forced in report because the run summary omitted the flag).",
         "",
         "π_6cls (run config): {}; TV→ISIC={:.3f}, TV→PAD={:.3f} (ref 0.019 / 0.579).".format(
             [round(x, 3) for x in (pi_info["pi_6cls"] or [])],
             float(pi_info["tv_to_pi_isic"] or 0), float(pi_info["tv_to_pi_pad"] or 0)),
         "",
         "## Table (Mahalanobis mean ± sd; paired Δ six−ref with seed-bootstrap 95% CI; recovery; Rule A on six-class)",
         "",
         "| λ | set | six-class | shift 100% | shift 0% | Δ vs 100% | Δ vs 0% | recovery | Rule A |",
         "|---:|---|---:|---:|---:|---|---|---:|---|"]
    for r in rows:
        rec_s = "—" if r["recovery"] is None else "{:.3f}".format(r["recovery"])
        ra_s = "—" if r["rule_A_six"] is None else ("met" if r["rule_A_six"] else "not met")
        counted = "" if r["set"] in LABELLED else " (not counted)"
        L.append("| {:g} | {}{} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["lambda"], r["set"], counted, pm(r["six"]), pm(r["shift100"]), pm(r["shift0"]),
            ci(r["delta_vs_100"]), ci(r["delta_vs_0"]), rec_s, ra_s))
    L += ["", "## ID / leakage (six-class)", ""]
    for lam, ex in extras.items():
        L.append("- λ={}: ID bal {}, leakage V1 {}, NV-nearest pad {}, degraded seeds {}.".format(
            lam, pm(ex["id_bal"]), pm(ex["leakage_v1"]), pm(ex["nv_nearest_pad"]), ex["degraded_seeds"]))
    L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("case:", case)
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
