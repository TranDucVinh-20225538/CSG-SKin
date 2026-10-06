#!/usr/bin/env python3
"""Phase 1.6b: identify and export the 3 Maha-boundary images (1 ID, 2 OOD). CPU."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import eval_ood_dual_branch as p1

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_6")
SCORES = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase1_5/scores_z_context_s42.npz")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bdir = OUT / "boundary_images"
    bdir.mkdir(parents=True, exist_ok=True)
    splits = p1.build_split_frames(p1.METADATA)
    id_df = splits["isic_test"].reset_index(drop=True)
    ood_df = splits["pad"].reset_index(drop=True)
    d = np.load(SCORES)
    id_s, ood_s = d["id"], d["ood"]
    assert len(id_s) == len(id_df) and len(ood_s) == len(ood_df)

    n_disc = int(np.sum(id_s[:, None] > ood_s[None, :]))
    n_pairs = int(len(id_s) * len(ood_s))
    auroc_exact = 1.0 - n_disc / n_pairs

    id_idx = int(np.argmax(id_s))
    ood_order = np.argsort(ood_s)
    ood_idxs = [int(ood_order[0]), int(ood_order[1])]

    recs = []
    # ID outlier
    row = id_df.iloc[id_idx]
    recs.append(
        {
            "role": "id_outlier",
            "split": "isic_test",
            "index": id_idx,
            "maha_score": float(id_s[id_idx]),
            "path": str(row.path),
            "label": str(row.label),
            "domain": str(row.domain),
        }
    )
    for k, oi in enumerate(ood_idxs, start=1):
        row = ood_df.iloc[oi]
        recs.append(
            {
                "role": "ood_low_{}".format(k),
                "split": "pad",
                "index": oi,
                "maha_score": float(ood_s[oi]),
                "path": str(row.path),
                "label": str(row.label),
                "domain": str(row.domain),
            }
        )

    thumbs = []
    for rec in recs:
        src = Path(rec["path"])
        dest = bdir / "{}_{}_{}{}".format(rec["role"], rec["label"], src.stem, src.suffix)
        shutil.copy2(src, dest)
        # also a small jpeg for inspection
        im = Image.open(src).convert("RGB")
        rec["orig_size"] = list(im.size)
        rec["copied_to"] = str(dest)
        small = im.copy()
        small.thumbnail((448, 448))
        jpg = dest.with_suffix(".preview.jpg")
        small.save(jpg, quality=90)
        rec["preview"] = str(jpg)
        thumbs.append(str(jpg))

    out = {
        "checkpoint_seed": 42,
        "n_id": int(len(id_s)),
        "n_ood": int(len(ood_s)),
        "n_pairs": n_pairs,
        "n_discordant_pairs": n_disc,
        "auroc_exact": auroc_exact,
        "reporting": "AUROC > 0.9999 ({} discordant pairs / {:.1f}M)".format(n_disc, n_pairs / 1e6),
        "id_p99": float(np.percentile(id_s, 99)),
        "ood_p01": float(np.percentile(ood_s, 1)),
        "boundary": recs,
        "note": "Do not print 1.0000 ± 0.0000. A bare 1.0000 reads as leakage; the pair count is the diligence.",
    }
    (OUT / "boundary_images.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ["n_discordant_pairs", "n_pairs", "auroc_exact", "reporting"]}, indent=2))
    for r in recs:
        print(r["role"], r["label"], r["maha_score"], r["path"])


if __name__ == "__main__":
    main()
