#!/usr/bin/env python3
"""Paper B R5 Q1 chest X-ray data: index, patient-level splits, class mix, and node-local staging.

Sources (all already resized to 256×256 grey PNG by DST-Skin r3 item 8; nothing is re-encoded here):
  NIH ChestX-ray14   DST-Skin/data/staged/nih_256/images_0{01..12}.tar, members <Image Index>.png
                     metadata DST-Skin/data/raw/nih_cxr14/Data_Entry_2017_v2020.csv
  Kermany pediatric  DST-Skin/data/staged/p2a_ood_256/kermany_ped.tar, members kped_%05d.png
                     (Kermany et al. 2018 CXR v2, 5,856 images; mapping to original file names in
                     DST-Skin/results/r3/8/p2a/ood.csv.gz, column member)
  Shenzhen           DST-Skin/data/staged/p2a_ood_256/shenzhen.tar, members CHNCXR_####_{0,1}.png (1 = TB)

Label definitions (class 0 = normal, 1 = pneumonia):
  NIH normal     Finding Labels == "No Finding"
  NIH pneumonia  "Pneumonia" is one of the '|'-separated Finding Labels (other findings may co-occur)
  NIH excluded   every other image (a finding but not Pneumonia)
  Kermany        folder NORMAL -> 0, PNEUMONIA (bacteria or virus) -> 1
  Shenzhen       not a task label; TB flag kept for the descriptive normal / TB subsets

Splits:
  NIH      patient-level (Patient ID). StratifiedGroupKFold(5, shuffle, seed 42) on the image label: fold 0 = test;
           on the rest, StratifiedGroupKFold(5, shuffle, seed 42) fold 0 = val; the remainder = train
           (the R2 item-4 lesion-split procedure with patient as the group).
  Kermany  patient-level, our own split (the official train/test split shares 170 patient IDs). Patient ID parsed
           from the original file name: pneumonia 'person<N>_{bacteria,virus}_<k>[_<j>].jpeg' -> person<N>;
           normal '[NORMAL2-]IM-<N>-<k>[-<j>].jpeg' -> [NORMAL2-]IM-<N>. StratifiedGroupKFold(10, seed 42) on
           the folder label: folds 0-2 = kerm_heldout (≈30 %, the pad_heldout analogue), folds 3-9 = kerm_adv
           (the adversary's target-domain stream). Normal and pneumonia IDs live in disjoint name spaces, so a
           child with images in both folders cannot be linked (stated limitation).
  Shenzhen OOD only, all 662 images (one image per patient).

CLI:
  --index [--dry_run]   build results/paperB/r5/q1/index/*.csv + CLASS_MIX.json + SPLIT_AUDIT.json (metadata only)
  --stage_to DIR        extract the needed PNGs into DIR/{nih,p2a} (node-local; run at job start, never on login)
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DST = Path("/data2/cmdir/home/toandq/DST-Skin")
NIH_META = DST / "data" / "raw" / "nih_cxr14" / "Data_Entry_2017_v2020.csv"
NIH_TARS = sorted((DST / "data" / "staged" / "nih_256").glob("images_*.tar"))
P2A_DIR = DST / "data" / "staged" / "p2a_ood_256"
P2A_MAP = DST / "results" / "r3" / "8" / "p2a" / "ood.csv.gz"
OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5/q1/index")
CLASSES = ("normal", "pneumonia")
SETS = ("nih_train", "nih_val", "nih_test", "kerm_adv", "kerm_heldout", "kerm_full", "shenzhen")

RX_PNEU = re.compile(r"^(person\d+)_(bacteria|virus)_\d+(?:_\d+)?\.jpe?g$", re.I)
RX_NORM = re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)?\.jpe?g$", re.I)


def nih_frame():
    d = pd.read_csv(NIH_META, usecols=["Image Index", "Finding Labels", "Patient ID", "Patient Age", "View Position"])
    fl = d["Finding Labels"].astype(str)
    normal = fl.eq("No Finding")
    pneu = fl.str.split("|").map(lambda s: "Pneumonia" in s)
    keep = normal | pneu
    f = pd.DataFrame({"member": d.loc[keep, "Image Index"].to_numpy(), "patient": d.loc[keep, "Patient ID"].astype(str).to_numpy(),
                      "label_idx": pneu[keep].astype(int).to_numpy(), "age": d.loc[keep, "Patient Age"].to_numpy(),
                      "view": d.loc[keep, "View Position"].to_numpy(),
                      "pneumonia_only": fl[keep].eq("Pneumonia").astype(int).to_numpy()})
    f["source"] = "nih"
    f["path"] = "nih/" + f.member
    excl = {"n_images_all": int(len(d)), "n_excluded_other_findings": int((~keep).sum()),
            "n_patients_all": int(d["Patient ID"].nunique())}
    return f.reset_index(drop=True), excl


def kermany_frame():
    o = pd.read_csv(P2A_MAP)
    k = o[o.source == "kermany_ped"].copy()
    name = k.member.map(lambda m: Path(m).name)
    folder = k.member.map(lambda m: Path(m).parent.name.upper())
    if not set(folder) <= {"NORMAL", "PNEUMONIA"}:
        raise SystemExit("unexpected Kermany folders {}".format(sorted(set(folder))))
    pid, bad = [], []
    for n, fo in zip(name, folder):
        m = (RX_PNEU if fo == "PNEUMONIA" else RX_NORM).match(n)
        if m is None:
            bad.append(n)
            pid.append(None)
        else:
            pid.append(("P:" if fo == "PNEUMONIA" else "N:") + m.group(1))
    if bad:
        raise SystemExit("{} Kermany names did not parse, e.g. {}".format(len(bad), bad[:5]))
    f = pd.DataFrame({"member": (k.key + ".png").to_numpy(), "orig": k.member.to_numpy(), "patient": pid,
                      "label_idx": (folder == "PNEUMONIA").astype(int).to_numpy(),
                      "subtype": [("virus" if "virus" in n.lower() else "bacteria") if fo == "PNEUMONIA" else "normal"
                                  for n, fo in zip(name, folder)],
                      "official_split": k.member.map(lambda m: Path(m).parts[1]).to_numpy()})
    f["source"] = "kermany"
    f["path"] = "p2a/" + f.member
    return f.reset_index(drop=True)


def shenzhen_frame():
    o = pd.read_csv(P2A_MAP)
    s = o[o.source == "shenzhen"].copy()
    f = pd.DataFrame({"member": (s.key + ".png").to_numpy(), "patient": s.key.to_numpy(),
                      "tb": s.shenzhen_tb.astype(int).to_numpy(), "label_idx": -1})
    f["source"] = "shenzhen"
    f["path"] = "p2a/" + f.member
    return f.reset_index(drop=True)


def _sgkf_fold(frame, n_splits, folds, seed=42):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.zeros(len(frame), bool)
    for k, (_tr, te) in enumerate(sg.split(np.zeros(len(frame)), frame.label_idx, frame.patient)):
        if k in folds:
            idx[te] = True
    return idx


def build_splits():
    nih, excl = nih_frame()
    te = _sgkf_fold(nih, 5, {0})
    rest = nih[~te].reset_index(drop=True)
    va = _sgkf_fold(rest, 5, {0})
    S = {"nih_train": rest[~va].reset_index(drop=True), "nih_val": rest[va].reset_index(drop=True),
         "nih_test": nih[te].reset_index(drop=True)}
    ker = kermany_frame()
    ho = _sgkf_fold(ker, 10, {0, 1, 2})
    S["kerm_heldout"] = ker[ho].reset_index(drop=True)
    S["kerm_adv"] = ker[~ho].reset_index(drop=True)
    S["kerm_full"] = ker
    S["shenzhen"] = shenzhen_frame()
    return S, excl


def class_mix(S, excl):
    mix = {}
    for n, f in S.items():
        if n == "shenzhen":
            mix[n] = {"n": int(len(f)), "n_patients": int(f.patient.nunique()), "normal": int((f.tb == 0).sum()),
                      "tb": int((f.tb == 1).sum()), "task_labels": "none (OOD)"}
            continue
        c = np.bincount(f.label_idx, minlength=2)
        mix[n] = {"n": int(len(f)), "n_patients": int(f.patient.nunique()), "normal": int(c[0]),
                  "pneumonia": int(c[1]), "pneumonia_frac": float(c[1] / max(len(f), 1))}
        if "view" in f:
            mix[n]["view_counts"] = {str(k): int(v) for k, v in f.view.value_counts().items()}
            mix[n]["pneumonia_only_label"] = int(f.pneumonia_only.sum())
            mix[n]["age_lt18"] = {"normal": int(((f.age < 18) & (f.label_idx == 0)).sum()),
                                  "pneumonia": int(((f.age < 18) & (f.label_idx == 1)).sum())}
        if "subtype" in f:
            mix[n]["subtype_counts"] = {str(k): int(v) for k, v in f.subtype.value_counts().items()}
    pat = {n: set(f.patient) for n, f in S.items()}
    audit = {"nih_exclusions": excl,
             "patient_overlap": {"nih_train_test": len(pat["nih_train"] & pat["nih_test"]),
                                 "nih_train_val": len(pat["nih_train"] & pat["nih_val"]),
                                 "nih_val_test": len(pat["nih_val"] & pat["nih_test"]),
                                 "kerm_adv_heldout": len(pat["kerm_adv"] & pat["kerm_heldout"])},
             "nih_patients_with_both_classes": int(
                 pd.concat([S["nih_train"], S["nih_val"], S["nih_test"]]).groupby("patient").label_idx.nunique().gt(1).sum()),
             "kermany_official_split_shared_ids": int(len(
                 set(S["kerm_full"].patient[S["kerm_full"].official_split == "train"])
                 & set(S["kerm_full"].patient[S["kerm_full"].official_split == "test"]))),
             "kermany_patient_link_limitation": "normal (IM-*) and pneumonia (person*) IDs are disjoint name spaces"}
    assert all(v == 0 for v in audit["patient_overlap"].values()), audit["patient_overlap"]
    return mix, audit


def write_index(dry_run=False):
    S, excl = build_splits()
    mix, audit = class_mix(S, excl)
    print(json.dumps({"class_mix": mix, "audit": audit}, indent=2))
    if dry_run:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for n, f in S.items():
        f.to_csv(OUT / "{}.csv".format(n), index=False)
    (OUT / "CLASS_MIX.json").write_text(json.dumps(mix, indent=2) + "\n")
    (OUT / "SPLIT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")


def load_index():
    if not (OUT / "nih_train.csv").exists():
        raise SystemExit("run scripts/r5_q1_cxr_data.py --index first (slurm/r5_q1_index.sbatch)")
    return {n: pd.read_csv(OUT / "{}.csv".format(n), dtype={"patient": str}) for n in SETS}


def stage_to(dst):
    """Extract only the indexed PNGs to dst/{nih,p2a}. Idempotent (existing files are kept)."""
    dst = Path(dst)
    S = load_index()
    need_nih = set(pd.concat([S["nih_train"], S["nih_val"], S["nih_test"]]).member)
    (dst / "nih").mkdir(parents=True, exist_ok=True)
    (dst / "p2a").mkdir(parents=True, exist_ok=True)
    n = 0
    for t in NIH_TARS:
        with tarfile.open(t) as tf:
            for m in tf:
                if m.isfile() and m.name in need_nih and not (dst / "nih" / m.name).exists():
                    tf.extract(m, dst / "nih")
                    n += 1
    for t in ("kermany_ped.tar", "shenzhen.tar"):
        with tarfile.open(P2A_DIR / t) as tf:
            tf.extractall(dst / "p2a")
    have = sum(1 for _ in (dst / "nih").glob("*.png"))
    if have != len(need_nih):
        raise SystemExit("staged {} of {} NIH images".format(have, len(need_nih)))
    print("staged", dst, "nih", have, "(new {})".format(n), "p2a", sum(1 for _ in (dst / "p2a").glob("*.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stage_to", default="")
    a = ap.parse_args()
    if a.index:
        write_index(a.dry_run)
    if a.stage_to:
        stage_to(a.stage_to)


if __name__ == "__main__":
    main()
