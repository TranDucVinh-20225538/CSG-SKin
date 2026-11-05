#!/usr/bin/env python3
"""Rebuild the Rule A verdicts of Paper B's main tables from the bootstrap result files and compare
them with paper/midl/paper_b.tex.

    python scripts/reproduce_rule_a.py            # check; exit 1 on any mismatch
    python scripts/reproduce_rule_a.py --latex    # also print the rebuilt LaTeX rows

Level (a): starts from the stored bootstrap summaries (pooled 95% CI, k = seeds whose own CI upper
bound is below 0.5). It recomputes Rule A (pooled CI upper < 0.5 and k >= 3), formats each cell as the
paper prints it, and checks mean, interval, k and verdict against the .tex. It does not redraw the
bootstrap; that needs the cached features (level (b), HPC).

Tables: tab:dial (Table 1), tab:unseen (main and A4 rows), tab:shiftfull (shift 50%), tab:splits (SCIN).
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paperB"
TEX = ROOT / "paper" / "midl" / "paper_b.tex"
LAMS = ("0", "0.25", "0.5", "1", "2", "4", "8")
SETS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
SET_TEX = {"pad_heldout": r"\texttt{pad\_heldout}", "fitzpatrick17k": "Fitzpatrick17k",
           "mednode": "MED-NODE", "sd198": "SD-198", "scin": "SCIN"}
UNSEEN_UNIT = {"mednode": "group(=image)", "sd198": "cluster_union", "scin": "case"}


def load(rel):
    return json.loads((RES / rel).read_text())


def rule_a(ci_hi, k):
    return ci_hi < 0.5 and k >= 3


def f3(x):
    return "{:.3f}".format(x)


def table_rows(tex, label):
    """Data rows (list of cell lists) of the tabular that carries \\label{label}."""
    i = tex.index("\\label{%s}" % label)
    a = tex.index("\\midrule", i)
    b = tex.index("\\bottomrule", a)
    rows = []
    for line in tex[a + len("\\midrule"):b].split("\\\\"):
        line = line.strip()
        if "&" in line and not line.startswith("\\midrule"):
            rows.append([c.strip() for c in line.replace("\\midrule", "").split("&")])
    return rows


def nums(cell):
    return re.findall(r"\d+\.\d+", cell)


def cell_sources():
    """(lam, set) -> dict(mean, ci_lo, ci_hi, k) for the Table 1 runs (last checkpoint)."""
    boot = load("r5/t1last/BOOTSTRAP.json")
    uns = load("r5/t1last/UNSEEN_ECE.json")["bootstrap"]
    out = {}
    for lam in LAMS:
        for s in SETS:
            if s in ("pad_heldout", "fitzpatrick17k"):
                m = boot["{}|last|{}".format(lam, s)]["mahalanobis_classcond"]
            else:
                m = uns["{}|{}|{}".format(s, lam, UNSEEN_UNIT[s])]
            out[lam, s] = {"mean": m["mean"], "sd": m.get("sd"), "ci_lo": m["ci_lo"], "ci_hi": m["ci_hi"],
                           "k": int(m["k_below"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    tex = TEX.read_text()
    bad, n = [], 0

    def check(where, ok, msg):
        nonlocal n
        n += 1
        if not ok:
            bad.append("{}: {}".format(where, msg))

    src = cell_sources()

    # ---- Table 1 (tab:dial): mean, k, bold = Rule A ----
    rows = {r[0]: r for r in table_rows(tex, "tab:dial")}
    for lam in LAMS:
        r = rows[lam]
        for j, s in enumerate(SETS):
            c, kcell = r[3 + 2 * j], r[4 + 2 * j]
            v = src[lam, s]
            ra = rule_a(v["ci_hi"], v["k"])
            w = "tab:dial λ={} {}".format(lam, s)
            check(w, nums(c)[0] == f3(v["mean"]), "mean {} vs source {}".format(nums(c)[0], f3(v["mean"])))
            check(w, kcell == "{}/5".format(v["k"]), "k {} vs source {}/5".format(kcell, v["k"]))
            check(w, ("\\mathbf" in c) == ra, "bold={} but Rule A={}".format("\\mathbf" in c, ra))

    # ---- Table 2 (tab:unseen): main and A4 rows ----
    a4 = {(str(int(r["lambda"])), r["set"]): r for r in load("r10b/B3_a4_ci.json")["rows"]}
    for r in table_rows(tex, "tab:unseen"):
        recipe = r[0]
        s = next((k for k, t in SET_TEX.items() if t == r[1]), None)
        if recipe not in ("main", "A4") or s is None:
            continue
        if recipe == "main":
            v0, v2 = src["0", s], src["2", s]
            m0, m2, lo, hi, k = v0["mean"], v2["mean"], v2["ci_lo"], v2["ci_hi"], v2["k"]
        else:
            b0, b2 = a4["0", s], a4["2", s]
            m0, m2, (lo, hi), k = b0["mean"], b2["mean"], b2["ci"], int(b2["k"].split("/")[0])
        w = "tab:unseen {} {}".format(recipe, s)
        got = nums(r[2]) + nums(r[3])
        check(w, got == [f3(m0), f3(m2), f3(lo), f3(hi)], "cells {} vs source {}".format(
            got, [f3(m0), f3(m2), f3(lo), f3(hi)]))
        check(w, r[4] == "{}/5".format(k), "k {} vs {}/5".format(r[4], k))
        check(w, (r[5] == "yes") == rule_a(hi, k), "verdict {} but Rule A={}".format(r[5], rule_a(hi, k)))
        if args.latex:
            print("{} & {} & ${}$ & ${}$ $[{}, {}]$ & {}/5 & {} \\\\".format(
                recipe, r[1], f3(m0), f3(m2), f3(lo), f3(hi), k, "yes" if rule_a(hi, k) else "no"))

    # ---- shift 50% rows (R10b B1) ----
    rows_sf = table_rows(tex, "tab:shiftfull")
    for b in load("r10b/B1_shift50_boots.json")["rows"]:
        lam = "0.5" if b["lambda"] == 0.5 else "2"
        k = int(b["k"].split("/")[0])
        want = [f3(b["mean"]), f3(b["sd"]), f3(b["ci"][0]), f3(b["ci"][1])]
        hit = [r for r in rows_sf if r[1].replace("$", "").replace("\\", "") == "50%" and r[2] == lam
               and SET_TEX[b["set"]] in r[0]]
        w = "tab:shiftfull 50% λ={} {}".format(lam, b["set"])
        if not hit:
            check(w, False, "row not found")
            continue
        r = hit[0]
        check(w, nums(r[3]) + nums(r[4]) == want, "cells {} vs source {}".format(nums(r[3]) + nums(r[4]), want))
        check(w, r[5] == "{}/5".format(k), "k {} vs {}/5".format(r[5], k))
        check(w, (r[6] == "met") == rule_a(b["ci"][1], k), "verdict {} but Rule A={}".format(
            r[6], rule_a(b["ci"][1], k)))

    # ---- tab:splits SCIN column (R10b B4) ----
    rows_sp = {(r[0], r[1]): r for r in table_rows(tex, "tab:splits")}
    for b in load("r10b/B4_splits_scin.json")["rows"]:
        lam = "0" if b["lambda"] == 0 else "2"
        r = rows_sp.get((str(b["split"]), lam))
        w = "tab:splits split {} λ={} SCIN".format(b["split"], lam)
        want = [f3(b["mean"]), f3(b["ci"][0]), f3(b["ci"][1])]
        check(w, r is not None and nums(r[-1]) == want, "cells {} vs source {}".format(
            nums(r[-1]) if r else None, want))
        k = int(b["k"].split("/")[0])
        check(w, rule_a(b["ci"][1], k) == bool(b["rule_A"]), "stored Rule A disagrees with recomputed")

    print("checked {} cells; {} mismatch(es)".format(n, len(bad)))
    for m in bad:
        print("  " + m)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
