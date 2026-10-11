#!/usr/bin/env python3
"""R4: Phase 13 per-run analysis (analyze_phase13.analyze_run, unchanged) for the image-level runs added after
the original sweep (λ ∈ {0.25, 1}, seeds 72/82). Writes phase13/per_run/<tag>.json; skips runs already done."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import analyze_phase13 as A  # noqa: E402

OUT = A.PAPERB / "results" / "paperB" / "phase13" / "per_run"
EXTRA = [(0.25, 72), (0.25, 82), (1.0, 72), (1.0, 82)]


def main():
    for lam, seed in EXTRA:
        tag = "runB_orth1_ladv{}_s{}".format("0p25" if lam == 0.25 else "1", seed)
        dest = OUT / "{}.json".format(tag)
        if dest.exists() or not A.has_features(tag):
            print("skip", tag, "exists" if dest.exists() else "no features", flush=True)
            continue
        dest.write_text(json.dumps(A.analyze_run(tag, lam, seed), indent=2) + "\n")
        print("wrote", dest.name, flush=True)


if __name__ == "__main__":
    main()
