#!/usr/bin/env python3
"""Consolidate results/paperB/r9/REPORT.md from E1/E2 reports + GPU hours."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

ROOT = R.RES / "r9"


def gpu_hours():
    stamp = ROOT / "GPU_HOURS.json"
    if stamp.exists():
        return json.loads(stamp.read_text())
    return {"note": "fill from sacct after jobs finish"}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    parts = ["# Paper B R9 consolidated report", "",
             "Generated at commit `{}`.".format(R.C.git_head()), ""]
    e1 = ROOT / "e1" / "REPORT.md"
    e2 = ROOT / "e2" / "REPORT.md"
    parts.append("| item | report |")
    parts.append("|---|---|")
    parts.append("| e1 | {} |".format("yes" if e1.exists() else "pending"))
    parts.append("| e2 | {} |".format("yes" if e2.exists() else "pending"))
    parts.append("")
    parts.append("GPU hours: {}.".format(gpu_hours()))
    parts.append("")
    if e1.exists():
        parts += ["---", "", e1.read_text()]
    if e2.exists():
        parts += ["---", "", e2.read_text()]
    (ROOT / "REPORT.md").write_text("\n".join(parts))
    print("wrote", ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
