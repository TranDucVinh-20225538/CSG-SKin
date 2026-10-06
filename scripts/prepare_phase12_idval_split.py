#!/usr/bin/env python3
"""Amendment A: split Camelyon17 id_val into id_val_select and id_val_score.

Stratified by hospital and label. Seed recorded. Does not write into the WILDS tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from wilds import get_dataset

DATA_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/camelyon17")
SPLIT_SEED = 42


def release_blob():
    p = DATA_ROOT / "camelyon17_v1.0" / "RELEASE_v1.0.txt"
    raw = p.read_bytes()
    return {
        "path": str(p.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": raw.decode("utf-8"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(dataset="camelyon17", root_dir=str(DATA_ROOT), download=False, split_scheme="official")
    sid = ds.split_dict["id_val"]
    id_val_idx = np.where(ds.split_array == sid)[0]
    hosp = ds.metadata_array[id_val_idx, 0].numpy().astype(int)
    y = ds.y_array[id_val_idx].numpy().astype(int)
    strata = hosp * 10 + y
    select_rel, score_rel = train_test_split(
        np.arange(len(id_val_idx)),
        test_size=0.5,
        random_state=args.seed,
        stratify=strata,
    )
    select_idx = id_val_idx[np.sort(select_rel)]
    score_idx = id_val_idx[np.sort(score_rel)]
    assert set(select_idx).isdisjoint(set(score_idx))
    assert set(select_idx).union(set(score_idx)) == set(id_val_idx)

    def counts(idx):
        h = ds.metadata_array[idx, 0].numpy().astype(int)
        yy = ds.y_array[idx].numpy().astype(int)
        out = {"n": int(len(idx)), "by_hospital": {}, "by_label": {}, "by_hospital_label": {}}
        for hv in sorted(set(h.tolist())):
            out["by_hospital"][str(hv)] = int((h == hv).sum())
        for lab in sorted(set(yy.tolist())):
            out["by_label"][str(lab)] = int((yy == lab).sum())
        for hv in sorted(set(h.tolist())):
            for lab in (0, 1):
                out["by_hospital_label"]["h{}_y{}".format(hv, lab)] = int(((h == hv) & (yy == lab)).sum())
        return out

    rec = {
        "dataset": "camelyon17",
        "split_scheme": "official",
        "parent_split": "id_val",
        "n_id_val": int(len(id_val_idx)),
        "fraction_score": 0.5,
        "seed": int(args.seed),
        "stratify": "hospital x label (6 strata)",
        "data_root": str(DATA_ROOT.resolve()),
        "read_only_data_root": True,
        "release": release_blob(),
        "id_val_select": {
            "role": "checkpoint selection only — never used as ID reference for OOD detectors",
            **counts(select_idx),
        },
        "id_val_score": {
            "role": "ID accuracy + ID reference distribution for all OOD detectors",
            **counts(score_idx),
        },
        "disjoint": True,
        "note": "Do not construct an id_test. These are halves of official id_val only.",
    }
    print(json.dumps({k: rec[k] for k in ("n_id_val", "id_val_select", "id_val_score", "seed")}, indent=2))
    if args.dry_run:
        return
    np.savez_compressed(
        OUT / "id_val_split.npz",
        select_idx=select_idx.astype(np.int64),
        score_idx=score_idx.astype(np.int64),
        id_val_idx=id_val_idx.astype(np.int64),
        seed=np.array(args.seed),
    )
    (OUT / "id_val_split.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote", OUT / "id_val_split.json")


if __name__ == "__main__":
    main()
