#!/usr/bin/env python3
"""Build results/paperB/r8/REPORT.md from d1/d2/d4."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

OUT = R.R8


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(R.PAPERB))


def main():
    if (OUT / "d2" / "runs").exists() and list((OUT / "d2" / "runs").glob("*/summary.json")):
        run([sys.executable, "scripts/r8_d2_report.py"])
    head = R.git_head()
    status = {k: (OUT / k / "REPORT.md").exists() for k in ("d1", "d2", "d4")}
    L = ["# Paper B R8 consolidated report", "",
         "Generated at commit `{}`.".format(head), "",
         "| item | report |", "|---|---|"]
    for k, ok in status.items():
        L.append("| {} | {} |".format(k, "yes" if ok else "pending"))
    gpu = 0.0
    for p in OUT.glob("**/summary.json"):
        try:
            gpu += float(json.loads(p.read_text()).get("gpu_hours") or 0)
        except Exception:
            pass
    L += ["", "GPU hours (sum under r8): {:.2f}.".format(gpu), ""]
    for item in ("d1", "d2", "d4"):
        rep = OUT / item / "REPORT.md"
        if rep.exists():
            L += ["", "---", "", "## {}".format(item.upper()), "", *rep.read_text().splitlines()[:60]]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
