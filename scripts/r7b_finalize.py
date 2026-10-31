#!/usr/bin/env python3
"""Build results/paperB/r7b/REPORT.md from c1 / c3b / t1c item reports."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

OUT = R.R7B


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(R.PAPERB))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "t1c" / "SELECTION.json").exists():
        run([sys.executable, "scripts/r7b_t1c_select.py"])
    if (OUT / "t1c" / "runs").exists() or (OUT / "t1c" / "SELECTION.json").exists():
        run([sys.executable, "scripts/r7b_t1c_report.py"])

    status = {
        "c1": (OUT / "c1" / "REPORT.md").exists(),
        "c3b": (OUT / "c3b" / "REPORT.md").exists(),
        "t1c": (OUT / "t1c" / "REPORT.md").exists(),
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    head = R.git_head()
    L = ["# Paper B R7b consolidated report", "",
         "Generated at commit `{}`.".format(head), "",
         "## Item status", "",
         "| item | report |",
         "|---|---|"]
    for item in ("c1", "c3b", "t1c"):
        L.append("| {} | {} |".format(item, "yes" if status[item] else "pending"))
    gpu = 0.0
    for p in OUT.glob("**/summary.json"):
        try:
            gpu += float(json.loads(p.read_text()).get("gpu_hours") or 0)
        except Exception:
            pass
    L += ["", "GPU hours (sum of available run summaries under r7b): {:.2f}.".format(gpu), ""]
    for item in ("c1", "c3b", "t1c"):
        rep = OUT / item / "REPORT.md"
        if rep.exists():
            L += ["", "---", "", "## {}".format(item.upper()), "", *rep.read_text().splitlines()[:80]]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
