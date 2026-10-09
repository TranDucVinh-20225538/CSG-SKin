#!/usr/bin/env python3
"""R2 Item 4 sweep: full Table-1 columns on the lesion-level ISIC split, every λ.

λ ∈ {0, 2}: Item 4 runs (seeds 42, 52, 62). λ ∈ {0.25, 0.5, 1, 4, 8}: sweep runs (seeds 42, 43, 44).
Cross-domain accuracy = 6-class balanced accuracy on pad_heldout (argmax restricted to PAD classes), as in the paper.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402
from r2_item4_report import COLS, LES_FEAT, LES_RUNS, row  # noqa: E402

OUT = C.R2 / "item4"
SEEDS = {0.0: (42, 52, 62), 2.0: (42, 52, 62), 0.25: (42, 43, 44), 0.5: (42, 43, 44), 1.0: (42, 43, 44),
         4.0: (42, 43, 44), 8.0: (42, 43, 44)}
COLS_X = tuple(COLS) + (("xdom", "Cross-domain bal acc (pad_heldout, 6-cls)"),)


def xdom(tag):
    z = np.load(LES_FEAT / tag / "pad_heldout.npz")
    lg, y = z["logits"], z["labels"].astype(int)
    pred = C.PAD_CLASSES[lg[:, C.PAD_CLASSES].argmax(1)]
    return float(balanced_accuracy_score(y, pred))


def main():
    tab, missing = {}, []
    for lam in sorted(SEEDS):
        rs = []
        for s in SEEDS[lam]:
            tag = "runB_orth1_ladv{}_s{}".format(C.lam_tag(lam), s)
            if not (LES_RUNS / tag / "summary.json").exists() or not (LES_FEAT / tag / "pad_heldout.npz").exists():
                missing.append(tag)
                continue
            r = row(LES_RUNS, LES_FEAT, tag)
            r["xdom"] = xdom(tag)
            r["seed"] = s
            rs.append(r)
        tab[lam] = rs
    head = C.git_head()
    L = ["# R2 Item 4 sweep — lesion-level ISIC split, all λ\n",
         "Commit: `{}`. Same code and recipe as Item 4 (`scripts/train_r2_item4_lesion_split.py`); only λ and seed vary.\n".format(head),
         "Seeds: λ ∈ {0, 2} use 42, 52, 62 (Item 4 runs); λ ∈ {0.25, 0.5, 1, 4, 8} use 42, 43, 44 (sweep). Mean ± s.d. over seeds. "
         "Detectors on z_lesion^norm, fit on the run's ISIC train. Cross-domain: 6-class balanced accuracy on pad_heldout.\n"]
    if missing:
        L.append("**Missing runs:** {}\n".format(", ".join(missing)))
    L += ["## Table 1 columns, lesion-level split\n",
          "| λ | n | seeds | " + " | ".join(lab for _, lab in COLS_X) + " |",
          "|---:|---:|---|" + "---|" * len(COLS_X)]
    for lam, rs in tab.items():
        L.append("| {:g} | {} | {} | ".format(lam, len(rs), ",".join(str(r["seed"]) for r in rs)) + " | ".join(
            C.fmt(*C.mean_sd([r[k] for r in rs])[:2]) for k, _ in COLS_X) + " |")
    L += ["", "## Per seed\n", "| λ | seed | " + " | ".join(lab for _, lab in COLS_X) + " |", "|---:|---:|" + "---|" * len(COLS_X)]
    for lam, rs in tab.items():
        for r in rs:
            L.append("| {:g} | {} | ".format(lam, r["seed"]) + " | ".join("{:.3f}".format(r[k]) for k, _ in COLS_X) + " |")
    L += ["", "## Caveats\n",
          "- Two seed sets: λ ∈ {0, 2} (42, 52, 62) and the other λ (42, 43, 44). Only seed 42 is shared across all λ.",
          "- Lesion-level split: StratifiedGroupKFold by `lesion_id`, null `lesion_id` = own group; zero lesion overlap between train, val and test (`split_record.json`)."]
    (OUT / "SWEEP_REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "sweep_results.json").write_text(json.dumps({"commit": head, "missing": missing,
                                                        "table": {str(k): v for k, v in tab.items()}}, indent=1) + "\n")
    print("wrote", OUT / "SWEEP_REPORT.md", "missing", missing)


if __name__ == "__main__":
    main()
