#!/usr/bin/env python3
"""Write the fixed PAD patient-level 70/30 split used by Phase 2 (and later Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import train_phase2_pad_holdout as p2

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase2_pad_holdout")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths_file = OUT / "pad_patient_split_paths.json"
    meta_file = OUT / "pad_patient_split.json"
    df = p2.load_master()
    pad = df[df.domain == "pad_ufes"].copy()
    if paths_file.exists():
        spec = json.loads(paths_file.read_text())
        print("existing split n_adv={} n_heldout={}".format(spec.get("n_adv"), spec.get("n_heldout")))
        return
    adv, hold = p2.pad_patient_split(pad, seed=42, heldout_frac=0.3)
    spec = {
        "heldout_frac": 0.3,
        "split_seed": 42,
        "n_pad": int(len(pad)),
        "n_adv": int(len(adv)),
        "n_heldout": int(len(hold)),
        "n_adv_patients": int(adv.patient_id.nunique()),
        "n_heldout_patients": int(hold.patient_id.nunique()),
        "n_adv_lesions": int(adv.groupby(["patient_id", "lesion_id"]).ngroups),
        "n_heldout_lesions": int(hold.groupby(["patient_id", "lesion_id"]).ngroups),
        "patient_overlap": sorted(set(adv.patient_id) & set(hold.patient_id)),
        "lesion_pair_overlap": sorted(set(zip(adv.patient_id, adv.lesion_id)) & set(zip(hold.patient_id, hold.lesion_id))),
        "class_counts_adv": adv.label.value_counts().to_dict(),
        "class_counts_heldout": hold.label.value_counts().to_dict(),
        "pad_adv_paths": adv.path.tolist(),
        "pad_heldout_paths": hold.path.tolist(),
    }
    if spec["patient_overlap"] or spec["lesion_pair_overlap"]:
        raise RuntimeError("PAD split leaked patients or lesions")
    meta = {k: v for k, v in spec.items() if k not in ("pad_adv_paths", "pad_heldout_paths")}
    meta_file.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    paths_file.write_text(json.dumps(spec, default=str) + "\n")
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
