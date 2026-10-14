#!/usr/bin/env python3
"""Paper B R5 P7(a): features of the unseen clinical sets for one lesion-level run (TASK_ID indexes the 35 runs).

Same path as R2 item 4 `extract_features`: fin.load_csg(best_checkpoint), build_val_transform_robust,
x13.collect_depths, REQUIRED keys. Writes <set>.npz next to the run's existing feature files.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase3_sweep as p3  # noqa: E402,F401  (installs the transformers stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import r2_common as C  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import build_val_transform_robust  # noqa: E402

ITEM4 = C.R2 / "item4"
SETS = C.PAPERB / "results" / "paperB" / "r5" / "p7a" / "sets"
NAMES = ("mednode", "sd198", "scin")
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def run_tags():
    tags = []
    for lam in LAMS:
        ds = sorted((ITEM4 / "features").glob("runB_orth1_ladv{}_s*".format(C.lam_tag(lam))))
        tags += [d.name for d in ds]
    assert len(tags) == 35, len(tags)
    return tags


def main():
    tag = run_tags()[int(sys.argv[1])]
    out = ITEM4 / "features" / tag
    todo = [n for n in NAMES if not (out / "{}.npz".format(n)).exists()]
    if not todo:
        print("skip_done", tag)
        return
    summary = json.loads((ITEM4 / "runs" / tag / "summary.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = fin.load_csg(Path(summary["best_checkpoint"]), device)
    tf = build_val_transform_robust()
    for name in todo:
        paths = pd.read_csv(SETS / "{}.csv".format(name)).path.tolist()
        loader = DataLoader(FolderJpegDataset(paths, tf), batch_size=64, shuffle=False, num_workers=8)
        with torch.no_grad():
            pack = x13.collect_depths(net, loader, device)
        assert len(pack["labels"]) == len(paths)
        np.savez_compressed(out / "{}.npz".format(name), **{k: pack[k] for k in x13.REQUIRED})
        print("features", tag, name, pack["z_lesion_norm"].shape, flush=True)


if __name__ == "__main__":
    main()
