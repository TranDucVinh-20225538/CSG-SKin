#!/usr/bin/env python3
"""Paper B R5 P3/P12/P13/Q5 stage 0 (CPU, metadata only): write the image lists that the frozen-encoder extractor
reads, in exactly the row order of the lesion-level feature files (results/paperB/r2/item4/features/<tag>/<split>.npz).

Outputs results/paperB/r5/frozen/frames/<split>.csv with columns path, label_idx, group.
Splits: isic_train, isic_test (lesion-level split, train_r2_item4_lesion_split.split_record), pad_full, pad_heldout,
fitzpatrick17k (sorted glob, = FolderJpegDataset order) and the P7(a) unseen clinical sets when their CSV exists.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "frozen" / "frames"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, _val, test = R.lesion_frames()
    spec = json.loads(R.T.p3.P2_SPLIT.read_text())
    df = R.T.p2.load_master()
    pad = df[df.domain == "pad_ufes"].copy().reset_index(drop=True)
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    pid = lambda f: f.path.str.extract(r"PAT_(\d+)_")[0]  # noqa: E731
    frames = {
        "isic_train": train.assign(group=R.T.lesion_groups(train)[0]),
        "isic_test": test.assign(group=R.T.lesion_groups(test)[0]),
        "pad_full": pad.assign(group=pid(pad)),
        "pad_heldout": hold.assign(group=pid(hold)),
    }
    fz = sorted(R.T.p3.FITZ_DIR.glob("*.jpg"))
    frames["fitzpatrick17k"] = pd.DataFrame({"path": [str(p) for p in fz], "label_idx": -1, "group": [p.stem for p in fz]})
    for name in ("mednode", "sd198", "scin"):
        p = R.R5 / "p7a" / "sets" / "{}.csv".format(name)
        if p.exists():
            s = pd.read_csv(p, keep_default_na=False)
            frames[name] = pd.DataFrame({"path": s.path, "label_idx": -1, "group": s.group.astype(str)})
    for name, f in frames.items():
        f[["path", "label_idx", "group"]].to_csv(OUT / "{}.csv".format(name), index=False)
        print(name, len(f), flush=True)
    assert len(frames["isic_test"]) == 5067 and len(frames["pad_heldout"]) == 716 and len(frames["fitzpatrick17k"]) == 3887


if __name__ == "__main__":
    main()
