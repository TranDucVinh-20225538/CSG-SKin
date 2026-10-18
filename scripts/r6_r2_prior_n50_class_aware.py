#!/usr/bin/env python3
"""CPU: class-balanced (b) + within-class (c) held-out probes for R2-prior n50 (all 5 seeds).

Writes caches under r2_prior/cur/class_aware/ and a focused table
r2_prior/cur/n50_class_aware_probes.md. Use --force to recompute even if cache exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_class_aware_heldout_probes as CA  # noqa: E402
import r6_r2_prior_report as PR  # noqa: E402

ARM = "iwdan_n50"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if cache exists")
    args = ap.parse_args()
    feat_root = PR.root() / "features"
    cache_dir = PR.root() / "class_aware"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in R.SEEDS:
        t = PR.tag(ARM, seed)
        cache = cache_dir / "{}.json".format(t)
        if cache.exists() and not args.force:
            rec = json.loads(cache.read_text())
            print("cache", t, flush=True)
        else:
            if not (feat_root / t / "isic_test.npz").exists() or not (feat_root / t / "pad_heldout.npz").exists():
                print("MISSING features", t, flush=True)
                return 2
            print("compute", t, flush=True)
            rec = CA.analyse_run(feat_root, t)
            cache.write_text(json.dumps(rec, indent=2) + "\n")
        rows.append({
            "seed": seed,
            "tag": t,
            "a": rec["unweighted_a"]["bal_acc_mean"],
            "b": rec["class_balanced_b"]["bal_acc_mean"],
            "c": rec["within_class_c"]["bal_acc_mean"],
            "n_eligible": rec["within_class_c"].get("n_eligible_classes"),
            "eligible": rec["within_class_c"].get("eligible_classes"),
        })
    lines = [
        "# R2-prior n50 — class-aware held-out probes (b)/(c)",
        "",
        "Design: `decisions/paperB_r6_class_aware_heldout.md`. "
        "(a) unweighted; (b) class-balanced; (c) within-class (≥20/side).",
        "",
        "| seed | (a) unweighted | (b) class-balanced | (c) within-class | n_eligible |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            r["seed"], r["a"], r["b"], r["c"], r["n_eligible"]))
    lines.append("")
    out = PR.root() / "n50_class_aware_probes.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
