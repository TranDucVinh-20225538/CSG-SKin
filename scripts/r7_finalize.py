#!/usr/bin/env python3
"""After T1/T1b/T2 runs finish: build per-item reports + consolidated results/paperB/r7/REPORT.md."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

ROOT = R.PAPERB
OUT = R.R7


def n_summaries(glob_pat):
    return len(list(OUT.glob(glob_pat)))


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    status = {
        "t1_dual": n_summaries("t1/runs/*/summary.json"),
        "t1_erm": n_summaries("t1/erm/*/summary.json"),
        "t1b": n_summaries("t1b/runs/*/summary.json"),
        "t2": n_summaries("t2/runs/*/summary.json"),
        "c1": (OUT / "c1" / "REPORT.md").exists(),
        "c2": (OUT / "c2" / "REPORT.md").exists(),
        "c3": (OUT / "c3" / "REPORT.md").exists(),
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    print("status", status, flush=True)

    if status["t1_dual"] >= 10 and status["t1_erm"] >= 3:
        run([sys.executable, "scripts/r7_t1_report.py"])
        # class-aware probes on T1 dual λ=0/2 if script accepts paths
        try:
            run([sys.executable, "scripts/r6_class_aware_heldout_probes.py",
                 "--out", str(OUT / "t1" / "probes")])
        except Exception as e:
            print("probes skipped:", e, flush=True)
    if status["t1b"] >= 3:
        run([sys.executable, "scripts/r7_t1b_report.py"])
    if status["t2"] >= 9:
        run([sys.executable, "scripts/r7_t2_report.py"])

    # consolidated
    head = R.git_head()
    L = ["# Paper B R7 consolidated report", "",
         "Generated at commit `{}`.".format(head), "",
         "## Item hashes and status", "",
         "| item | summaries | report |",
         "|---|---:|---|"]
    for item, need in (("t1", 13), ("t1b", 3), ("t2", 9)):
        n = status.get(item, status.get("t1_dual", 0) if item == "t1" else 0)
        if item == "t1":
            n = status["t1_dual"] + status["t1_erm"]
        rep = OUT / item / "REPORT.md"
        L.append("| {} | {}/{} | {} |".format(item, n, need, "yes" if rep.exists() else "pending"))
    for c in ("c1", "c2", "c3"):
        L.append("| {} | — | {} |".format(c, "yes" if status[c] else "no"))

    gpu = 0.0
    for p in OUT.glob("**/summary.json"):
        try:
            gpu += float(json.loads(p.read_text()).get("gpu_hours") or 0)
        except Exception:
            pass
    L += ["", "GPU hours (sum of available run summaries): {:.2f}.".format(gpu), ""]

    for item in ("c1", "c2", "c3", "t1", "t1b", "t2"):
        rep = OUT / item / "REPORT.md"
        if rep.exists():
            L += ["", "---", "", "## {}".format(item.upper()), ""]
            # first ~40 lines
            lines = rep.read_text().splitlines()
            L.extend(lines[:60])
            if len(lines) > 60:
                L.append("… (see `results/paperB/r7/{}/REPORT.md`)".format(item))
            case = OUT / item / "CASE.json"
            outcome = OUT / item / "OUTCOME.json"
            if case.exists():
                L += ["", "CASE.json: `{}`".format(case.read_text().strip())]
            if outcome.exists():
                L += ["", "OUTCOME.json: `{}`".format(outcome.read_text().strip())]

    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
