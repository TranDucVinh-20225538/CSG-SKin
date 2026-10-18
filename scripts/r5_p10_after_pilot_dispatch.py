#!/usr/bin/env python3
"""After Camelyon adversary ladder pilot: queue rule (decisions/paperB_r5_p10_ladder_pilot.md).

If pilot/SELECTED.json exists → submit full Camelyon 10 runs (5 seeds × natural/induced at the
selected level) on both GPUs, then release held R2 after that array finishes (afterok).
If pilot failed (no SELECTED.json) → release held R2 immediately.

Width (Camelyon arms at the selected level) outranks remaining R2 robustness when the pilot
passes. Agents must not invert this order.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
SEL = ROOT / "results" / "paperB" / "r5" / "p10" / "pilot" / "SELECTED.json"
ARMS_SBATCH = ROOT / "slurm" / "r5_p10_shift_arms_pilot.sbatch"
RELEASE_SBATCH = ROOT / "slurm" / "r6_release_r2_held.sbatch"
LOG = ROOT / "results" / "paperB" / "r5" / "p10" / "pilot" / "AFTER_PILOT_DISPATCH.json"
TZ = timezone(timedelta(hours=7))


def sbatch(*extra: str) -> str:
    cmd = ["sbatch", *extra]
    out = subprocess.check_output(cmd, text=True).strip()
    print(out, flush=True)
    return out.split()[-1]


def main() -> int:
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (+07)")
    rec = {"when": now, "selected_exists": SEL.exists()}
    if SEL.exists():
        sel = json.loads(SEL.read_text())
        rung = sel["rung"]
        if rung not in ("P_L1", "P_L2", "P_L3", "P_L4", "P_L5"):
            raise SystemExit("SELECTED.json rung not a pilot level: {}".format(rung))
        rec["selected"] = sel
        rec["action"] = "camelyon_10_then_release_r2"
        print("=== pilot PASSED level", rung, "→ Camelyon 10 before R2 ===", flush=True)
        arms_jid = sbatch(
            "--export=ALL,SELECTED_RUNG={}".format(rung),
            str(ARMS_SBATCH),
        )
        rel_jid = sbatch("--dependency=afterok:{}".format(arms_jid), str(RELEASE_SBATCH))
        rec["camelyon_array_job"] = arms_jid
        rec["r2_release_job"] = rel_jid
    else:
        rec["action"] = "release_r2_now"
        print("=== pilot FAILED (no SELECTED.json) → release R2 as scheduled ===", flush=True)
        rel_jid = sbatch(str(RELEASE_SBATCH))
        rec["r2_release_job"] = rel_jid
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote", LOG, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
