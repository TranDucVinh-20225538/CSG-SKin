#!/usr/bin/env python3
"""R3 C: Phase 13 per-run analysis (scripts/analyze_phase13.py::analyze_run, unchanged) on the lesion-level
R2 Item 4 features. Writes results/paperB/r3/c/per_run/<tag>.json; skips runs already done."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import analyze_phase13 as A  # noqa: E402

A.FEAT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r2/item4/features")
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r3/c/per_run")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for d in sorted(A.FEAT.glob("runB_orth1_ladv*_s*")):
        m = re.fullmatch(r"runB_orth1_ladv([0-9p]+)_s(\d+)", d.name)
        dest = OUT / "{}.json".format(d.name)
        if not m or dest.exists() or not A.has_features(d.name):
            continue
        rec = A.analyze_run(d.name, float(m.group(1).replace("p", ".")), int(m.group(2)))
        dest.write_text(json.dumps(rec, indent=1, default=float) + "\n")
        print("wrote", dest.name, flush=True)


if __name__ == "__main__":
    main()
