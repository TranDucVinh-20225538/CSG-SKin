#!/usr/bin/env python3
"""Appendix B LaTeX: full R1 Maha dose table + regression (a) VIFs.

Writes results/paperB/r6/r1/cur/APPENDIX_B_LATEX.md (+ .tex snippet).
Primary rule A = excl. degraded (same as dose tables in REPORT.md).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r6" / "r1" / "cur"
SETS = [
    ("pad_heldout", "pad_heldout"),
    ("Fitzpatrick17k", "fitzpatrick17k"),
    ("MED-NODE", "mednode"),
    ("SD-198", "sd198"),
    ("SCIN", "scin"),
]
# λ ∈ {0, 0.5, 2}; λ=0 only at shift 100% baseline in R1
ROW_RE = re.compile(
    r"^\|\s*([0-9.]+)\s*\|\s*(shift [^|]+)\|\s*"
    r"([0-9.]+) ± ([0-9.]+) \[([0-9.]+), ([0-9.]+)\],\s*"
    r"(\d+)/(\d+) seed CIs below 0\.5,\s*rule A (met|not met)"
)
BASE_RE = re.compile(
    r"λ = 0 baseline \(shift 100%\): Mahalanobis pad_heldout "
    r"([0-9.]+) ± ([0-9.]+) \[([0-9.]+), ([0-9.]+)\],\s*"
    r"(\d+)/(\d+) seed CIs below 0\.5,\s*rule A (met|not met)"
)


def parse_dose(report: str):
    """Return list of dicts from Dose — <set> tables (excl. degraded column)."""
    rows = []
    # per-set detector tables also have λ=0; prefer dose + baseline line + detector λ=0
    for title, key in SETS:
        m = re.search(
            r"### Dose — {}\n\n\|[^\n]+\n\|[^\n]+\n((?:\|[^\n]+\n)+)".format(re.escape(title)),
            report,
        )
        if not m:
            continue
        for line in m.group(1).strip().splitlines():
            mm = ROW_RE.match(line.strip())
            if not mm:
                continue
            lam, shift, mean, sd, lo, hi, k, n, rule = mm.groups()
            rows.append({
                "set": key, "set_tex": title.replace("_", r"\_"),
                "lambda": float(lam), "shift": shift.strip(),
                "mean": float(mean), "sd": float(sd),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "k": int(k), "n": int(n),
                "rule_a": rule == "met",
            })
    # λ=0 baselines from detector sections (shift 100%)
    for title, key in SETS:
        # Detectors on <key>
        pat = re.compile(
            r"### Detectors on {}\b.*?^\| λ=0 baseline \(shift 100%\) \| 0 \| "
            r"([0-9.]+) ± ([0-9.]+) \[([0-9.]+), ([0-9.]+)\] \|.*?\| "
            r"(\d+)/(\d+) \| (met|not met) \|".format(re.escape(key)),
            re.M | re.S,
        )
        mm = pat.search(report)
        if not mm:
            continue
        mean, sd, lo, hi, k, n, rule = mm.groups()
        rows.append({
            "set": key, "set_tex": title.replace("_", r"\_"),
            "lambda": 0.0, "shift": "shift 100%",
            "mean": float(mean), "sd": float(sd),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "k": int(k), "n": int(n),
            "rule_a": rule == "met",
        })
    return rows


def latex_maha(rows):
    lines = [
        "% Appendix B — R1 Mahalanobis (excl. degraded; primary rule A)",
        "% columns: set & shift & $\\lambda$ & mean $\\pm$ sd & 95\\% CI & $k/n$ (CI$<0.5$) & rule A \\\\",
    ]
    order = ["pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin"]
    shift_ord = {"shift 100%": 0, "shift 50%": 1, "shift 0%": 2}
    rows = sorted(rows, key=lambda r: (order.index(r["set"]), r["lambda"], shift_ord.get(r["shift"], 9)))
    for r in rows:
        if r["lambda"] not in (0.0, 0.5, 2.0):
            continue
        if r["lambda"] == 0.0 and r["shift"] != "shift 100%":
            continue
        rule = r"met" if r["rule_a"] else r"not met"
        sh = r["shift"].replace("%", r"\%")
        lines.append(
            "{set} & {sh} & {lam:g} & {mean:.3f}\\,$\\pm$\\,{sd:.3f} & "
            "[{lo:.3f}, {hi:.3f}] & {k}/{n} & {rule} \\\\".format(
                set=r["set"].replace("_", r"\_"), sh=sh, lam=r["lambda"],
                mean=r["mean"], sd=r["sd"], lo=r["ci_lo"], hi=r["ci_hi"],
                k=r["k"], n=r["n"], rule=rule,
            )
        )
    return lines


def latex_reg_a(reg_path: Path):
    reg = json.loads(reg_path.read_text())
    lines = [
        "% Appendix B — regression (a): Maha ~ leakage + shift_level + recipe",
        "% set & n & shift coef & 95\\% CI & excl.\\ 0 & VIF$_{\\mathrm{shift|leak}}$ & "
        "VIF$_{\\mathrm{shift|recipe}}$ & identifiable \\\\",
    ]
    for ood in ("fitzpatrick17k", "mednode", "sd198", "pad_heldout", "scin"):
        if ood not in reg:
            continue
        r = reg[ood]
        c = r["coefs"]["shift_level"]
        note = r" (not counted)" if ood == "pad_heldout" else ""
        lines.append(
            "{ood}{note} & {n} & {coef:.3f} & [{lo:.3f}, {hi:.3f}] & {ex} & "
            "{v1:.2f} & {v2:.2f} & {ident} \\\\".format(
                ood=ood.replace("_", r"\_"), note=note, n=r["n"],
                coef=c["coef"], lo=c["ci_lo"], hi=c["ci_hi"],
                ex="yes" if c["excludes_0"] else "no",
                v1=r["vif_shift_vs_leakage"], v2=r["vif_shift_vs_recipe"],
                ident="yes" if r["identifiable"] else "no",
            )
        )
    return lines


def main():
    report = (OUT / "REPORT.md").read_text()
    rows = parse_dose(report)
    maha = latex_maha(rows)
    reg_p = OUT / "leakage_matched" / "regression.json"
    if not reg_p.exists():
        raise SystemExit("missing " + str(reg_p))
    reg = latex_reg_a(reg_p)
    body = [
        "# Appendix B — LaTeX rows (R1)",
        "",
        "Source: `r6/r1/cur/REPORT.md` dose + detector λ=0 baselines; "
        "`leakage_matched/regression.json`. Commit `{}`.".format(R.C.git_head()),
        "",
        "## Full Mahalanobis table (shift × λ ∈ {0, 0.5, 2} × sets)",
        "",
        "```latex",
        *maha,
        "```",
        "",
        "## Regression (a) with VIFs",
        "",
        "```latex",
        *reg,
        "```",
        "",
    ]
    out = OUT / "APPENDIX_B_LATEX.md"
    out.write_text("\n".join(body) + "\n")
    (OUT / "appendix_b_maha.tex").write_text("\n".join(maha) + "\n")
    (OUT / "appendix_b_reg_a.tex").write_text("\n".join(reg) + "\n")
    print("wrote", out, "n_maha_rows", len(maha) - 2, flush=True)


if __name__ == "__main__":
    main()
