#!/usr/bin/env python3
"""Paper B R11 X3: point the frozen-ERM feature cache at X2 M2 packs (no re-extract).

Decisions: decisions/paperB_r11_x3.md — cache once, no augmentation.
Source: results/paperB/r5/p8/erm_ref/features/single_dann_ladv0_s{seed}/ (val transform only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r11" / "x3" / "cache"
ERM = R.P8_DIR / "erm_ref" / "features"
NEED = ("isic_train", "isic_val", "isic_test", "pad_adv", "pad_heldout", "pad_full",
        "fitzpatrick17k", "mednode", "sd198", "scin")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in R.SEEDS:
        src = ERM / "single_dann_ladv0_s{}".format(seed)
        dst = OUT / "s{}".format(seed)
        dst.mkdir(parents=True, exist_ok=True)
        missing = []
        for n in NEED:
            sp, dp = src / "{}.npz".format(n), dst / "{}.npz".format(n)
            if not sp.exists():
                missing.append(n)
                continue
            if dp.exists() or dp.is_symlink():
                dp.unlink()
            dp.symlink_to(sp.resolve())
        rows.append({"seed": seed, "src": str(src), "dst": str(dst), "missing": missing})
        print("cache", seed, "missing", missing or "none", flush=True)
    meta = {
        "design": "decisions/paperB_r11_x3.md",
        "note": "Frozen ERM EffB3 last.ckpt features; val transform only — no augmentation. "
                "Symlinks to r5/p8/erm_ref/features (filled by R5 P8 + R11 X2 M2 forwards).",
        "in_dim": 1536,
        "seeds": list(R.SEEDS),
        "sets": list(NEED),
        "rows": rows,
        "commit": __import__("subprocess").run(
            ["git", "-C", str(R.PAPERB), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
    }
    (OUT / "CACHE.json").write_text(json.dumps(meta, indent=2) + "\n")
    (OUT / "CACHE.md").write_text(
        "# X3 frozen-ERM feature cache\n\n"
        "No augmentation (val transform only). Symlinks → `r5/p8/erm_ref/features/`.\n"
        "Seeds {}; sets {}.\n".format(list(R.SEEDS), list(NEED))
    )
    print("wrote", OUT / "CACHE.json", flush=True)


if __name__ == "__main__":
    main()
