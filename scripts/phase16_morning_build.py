#!/usr/bin/env python3
"""One-shot Phase 16B acceptance bundle (partial OK)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

P16 = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16")
SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")


def run_py(name: str) -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": "/data2/hpcshared/Vinh/CSG-Skin:" + str(SCRIPTS)},
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def count_rev_coarse():
    root = P16 / "b1_rev"
    need = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8]

    def tag(l):
        return "{:g}".format(l).replace(".", "p")

    ok = sum(1 for l in need if (root / f"ham_bcn_ladv{tag(l)}_s42" / "summary.json").exists())
    return ok, len(need)


def main():
    lines = [
        "# Phase 16B — morning acceptance bundle",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Checklist",
        "",
    ]

    b0 = P16 / "b0" / "summary.json"
    lines.append(f"- [x] B0 probe + label-only fix: `{b0}`" if b0.exists() else "- [ ] B0 missing")

    dense_agg = P16 / "b1" / "dense3seed_aggregate.json"
    if dense_agg.exists():
        d = json.loads(dense_agg.read_text())
        lines.append(f"- [x] B1 dense 3-seed (BCN→HAM): `{dense_agg}`")
        for row in d.get("rows", []):
            lines.append(
                f"  - λ={row['lambda']:g} leakage {row['leakage_mean']:.3f}±{row['leakage_std']:.3f}, "
                f"kNN {row['knn50_mean']:.3f}±{row['knn50_std']:.3f}"
            )
    else:
        lines.append("- [ ] B1 dense aggregate missing — run phase16_b1_dense_aggregate.py")

    csv_coarse = P16 / "b1" / "coarse_scan_seed42_maha_columns.csv"
    lines.append(f"- [x] B1 coarse 8λ CSV: `{csv_coarse}`" if csv_coarse.exists() else "- [ ] B1 coarse CSV")

    gate = P16 / "b1_rev" / "ham_bcn_ladv0_s42" / "summary.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        maha = g["ood_detectors_bcn_heldout"]["mahalanobis_classcond"]
        passed = g["rev_maha_gate"]["pass_continue_coarse"]
        lines.append(f"- [x] B1-rev λ=0 gate: Maha **{maha:.3f}**, pass={passed}")
    else:
        lines.append("- [ ] B1-rev λ=0 gate missing")

    ok_rev, n_rev = count_rev_coarse()
    lines.append(f"- [{'x' if ok_rev == n_rev else ' '}] B1-rev coarse 8λ: **{ok_rev}/{n_rev}** summaries")

    rc, out = run_py("phase16_b1_rev_aggregate.py")
    lines.append("")
    lines.append("## B1-rev aggregate script")
    lines.append("```")
    lines.append(out.strip() or f"exit {rc}")
    lines.append("```")

    rev_md = P16 / "b1_rev" / "B1_REV_AGGREGATE.md"
    if rev_md.exists():
        lines.append("")
        lines.append(rev_md.read_text())

    lines.append("")
    lines.append("## Narrative hooks (locked)")
    lines.append("")
    lines.append("1. **BCN→HAM:** site probe ~0.96; adv drops leakage ~0.17 (plateau ~0.80); kNN collapses with small multi-seed σ; Maha below chance → not inversion axis.")
    lines.append("2. **HAM→BCN:** λ=0 Maha ~0.85 → headroom; test whether adv inverts (coarse scan + dense).")
    lines.append("3. Label-only cohort reference **~0.668** (not 0.5).")

    out = P16 / "MORNING_ACCEPTANCE.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)
    return 0 if ok_rev == n_rev and dense_agg.exists() else 2


if __name__ == "__main__":
    sys.exit(main())
