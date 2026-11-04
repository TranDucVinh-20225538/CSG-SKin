#!/usr/bin/env python3
"""R10b B2: regenerate R2 SCIN report with bootstrap (wrapper around r6_r2_scin_report)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r10b"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    before = (R.RES / "r6" / "r2" / "cur" / "REPORT_SCIN.md").read_text() if (
        R.RES / "r6" / "r2" / "cur" / "REPORT_SCIN.md").exists() else ""
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "r6_r2_scin_report.py")]
    print("running", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    after = (R.RES / "r6" / "r2" / "cur" / "REPORT_SCIN.md").read_text()
    scin_json = R.RES / "r6" / "r2" / "cur" / "scin_report.json"
    payload = {
        "commit": R.C.git_head(),
        "cpu_hours": (time.time() - t0) / 3600.0,
        "report_path": str(R.RES / "r6" / "r2" / "cur" / "REPORT_SCIN.md"),
        "scin_report_json": str(scin_json) if scin_json.exists() else None,
        "before_bytes": len(before), "after_bytes": len(after),
    }
    (OUT / "B2_scin_boots.json").write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "B2_scin_boots.md").write_text(
        "# R10b B2 — R2 SCIN with bootstrap\n\n"
        f"Commit `{payload['commit']}`. Regenerated `r6/r2/cur/REPORT_SCIN.md` "
        f"(with PREREGISTER boots). CPU hours {payload['cpu_hours']:.3f}.\n"
    )
    print("wrote B2 cpu_h", round(payload["cpu_hours"], 3), flush=True)


if __name__ == "__main__":
    main()
