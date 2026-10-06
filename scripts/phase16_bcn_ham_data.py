"""BCN vs HAM ISIC site cohort for Phase 16B. Lesion-level splits only."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO = Path("/data2/hpcshared/Vinh/CSG-Skin")
ISIC_META = REPO / "data" / "ISIC_2019_Training_Metadata.csv"
MASTER = REPO / "data" / "master_metadata_lesion_only_soft.csv"


def remap_path(p: str) -> str:
    s = str(p)
    if s.startswith("/mnt/data2/Vinh/"):
        return "/data2/hpcshared/Vinh/" + s[len("/mnt/data2/Vinh/") :]
    return s


def site_from_lesion(lid) -> str:
    if pd.isna(lid):
        return "empty"
    s = str(lid)
    if s.startswith("BCN_"):
        return "BCN"
    if s.startswith("HAM_"):
        return "HAM"
    if s.startswith("MSK"):
        return "MSK4"
    return "other"


def load_bcn_ham_frame() -> pd.DataFrame:
    from src.datasets.constants import LABEL_TO_INDEX

    master = pd.read_csv(MASTER)
    master = master[master["domain"] == "isic"].copy()
    master["path"] = master["path"].map(remap_path)
    master = master[master["path"].map(os.path.isfile)].reset_index(drop=True)
    master["image"] = master["path"].map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    meta = pd.read_csv(ISIC_META)[["image", "lesion_id"]]
    df = master.merge(meta, on="image", how="left")
    df["site"] = df["lesion_id"].map(site_from_lesion)
    df = df[df["site"].isin(["BCN", "HAM"])].copy()
    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    df["site_id"] = df["site"].map({"BCN": 0, "HAM": 1}).astype(int)
    return df.reset_index(drop=True)


def lesion_split(df: pd.DataFrame, seed: int, train_frac=0.70, val_frac=0.15):
    """Split by lesion_id; images from one lesion never cross splits."""
    lesions = df[["lesion_id", "site_id"]].drop_duplicates("lesion_id")
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_idx, rest_idx = next(gss1.split(lesions, groups=lesions["lesion_id"]))
    rest = lesions.iloc[rest_idx]
    val_rel = val_frac / (1.0 - train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=1.0 - val_rel, random_state=seed + 1)
    val_idx, probe_idx = next(gss2.split(rest, groups=rest["lesion_id"]))
    train_l = set(lesions.iloc[tr_idx]["lesion_id"])
    val_l = set(rest.iloc[val_idx]["lesion_id"])
    probe_l = set(rest.iloc[probe_idx]["lesion_id"])
    assert not (train_l & val_l or train_l & probe_l or val_l & probe_l)
    train = df[df["lesion_id"].isin(train_l)].reset_index(drop=True)
    val = df[df["lesion_id"].isin(val_l)].reset_index(drop=True)
    probe = df[df["lesion_id"].isin(probe_l)].reset_index(drop=True)
    return train, val, probe


def majority_site_per_class(frame: pd.DataFrame) -> pd.Series:
    """Per class index, site_id (0=BCN, 1=HAM) with more training images."""
    ct = pd.crosstab(frame["label_idx"], frame["site_id"])
    return ct.idxmax(axis=1)


def label_only_majority_reference(frame: pd.DataFrame, eval_frame: pd.DataFrame | None = None):
    """Predict site from class majority (no image). Cohort rule for manuscript ~0.668."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    rule = majority_site_per_class(frame)
    ev = eval_frame if eval_frame is not None else frame
    pred = ev["label_idx"].map(rule)
    y = ev["site_id"].astype(int)
    bcn_rec = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    ham_rec = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    return {
        "protocol": "each class → site with more images in reference frame; predict site from label only",
        "reference_frame": "full_bcn_ham_cohort" if eval_frame is None else "eval_frame",
        "plain_acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "bcn_recall": bcn_rec,
        "ham_recall": ham_rec,
        "majority_plain_acc": float(max((y == 0).mean(), (y == 1).mean())),
        "per_class_majority_site": {int(k): int(v) for k, v in rule.items()},
    }


def prepare_b1_splits(df: pd.DataFrame, seed: int, ham_heldout_frac: float = 0.30):
    """BCN: lesion train/val/test. HAM: lesion adv vs heldout (heldout never in train)."""
    bcn = df[df["site"] == "BCN"].copy()
    ham = df[df["site"] == "HAM"].copy()
    bcn_les = bcn[["lesion_id"]].drop_duplicates()
    g1 = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed)
    tr_i, rest_i = next(g1.split(bcn_les, groups=bcn_les["lesion_id"]))
    rest = bcn_les.iloc[rest_i]
    g2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    va_i, te_i = next(g2.split(rest, groups=rest["lesion_id"]))
    bcn_train_l = set(bcn_les.iloc[tr_i]["lesion_id"])
    bcn_val_l = set(rest.iloc[va_i]["lesion_id"])
    bcn_test_l = set(rest.iloc[te_i]["lesion_id"])

    ham_les = ham[["lesion_id"]].drop_duplicates()
    g3 = GroupShuffleSplit(n_splits=1, train_size=1.0 - ham_heldout_frac, random_state=seed + 2)
    adv_i, hold_i = next(g3.split(ham_les, groups=ham_les["lesion_id"]))
    ham_adv_l = set(ham_les.iloc[adv_i]["lesion_id"])
    ham_hold_l = set(ham_les.iloc[hold_i]["lesion_id"])

    splits = {
        "bcn_train": bcn[bcn["lesion_id"].isin(bcn_train_l)].reset_index(drop=True),
        "bcn_val": bcn[bcn["lesion_id"].isin(bcn_val_l)].reset_index(drop=True),
        "bcn_test": bcn[bcn["lesion_id"].isin(bcn_test_l)].reset_index(drop=True),
        "ham_adv": ham[ham["lesion_id"].isin(ham_adv_l)].reset_index(drop=True),
        "ham_heldout": ham[ham["lesion_id"].isin(ham_hold_l)].reset_index(drop=True),
    }
    meta = {
        "seed": seed,
        "ham_heldout_frac": ham_heldout_frac,
        "n_bcn_train": int(len(splits["bcn_train"])),
        "n_bcn_val": int(len(splits["bcn_val"])),
        "n_bcn_test": int(len(splits["bcn_test"])),
        "n_ham_adv": int(len(splits["ham_adv"])),
        "n_ham_heldout": int(len(splits["ham_heldout"])),
        "lesion_disjoint": True,
    }
    return splits, meta


def prepare_b1_rev_splits(df: pd.DataFrame, seed: int, bcn_heldout_frac: float = 0.30):
    """Mirror of prepare_b1_splits: ID=HAM (train/val/test), OOD=BCN heldout, BCN adv in training."""
    ham = df[df["site"] == "HAM"].copy()
    bcn = df[df["site"] == "BCN"].copy()
    ham_les = ham[["lesion_id"]].drop_duplicates()
    g1 = GroupShuffleSplit(n_splits=1, train_size=0.80, random_state=seed)
    tr_i, rest_i = next(g1.split(ham_les, groups=ham_les["lesion_id"]))
    rest = ham_les.iloc[rest_i]
    g2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    va_i, te_i = next(g2.split(rest, groups=rest["lesion_id"]))
    ham_train_l = set(ham_les.iloc[tr_i]["lesion_id"])
    ham_val_l = set(rest.iloc[va_i]["lesion_id"])
    ham_test_l = set(rest.iloc[te_i]["lesion_id"])

    bcn_les = bcn[["lesion_id"]].drop_duplicates()
    g3 = GroupShuffleSplit(n_splits=1, train_size=1.0 - bcn_heldout_frac, random_state=seed + 2)
    adv_i, hold_i = next(g3.split(bcn_les, groups=bcn_les["lesion_id"]))
    bcn_adv_l = set(bcn_les.iloc[adv_i]["lesion_id"])
    bcn_hold_l = set(bcn_les.iloc[hold_i]["lesion_id"])

    splits = {
        "ham_train": ham[ham["lesion_id"].isin(ham_train_l)].reset_index(drop=True),
        "ham_val": ham[ham["lesion_id"].isin(ham_val_l)].reset_index(drop=True),
        "ham_test": ham[ham["lesion_id"].isin(ham_test_l)].reset_index(drop=True),
        "bcn_adv": bcn[bcn["lesion_id"].isin(bcn_adv_l)].reset_index(drop=True),
        "bcn_heldout": bcn[bcn["lesion_id"].isin(bcn_hold_l)].reset_index(drop=True),
    }
    meta = {
        "seed": seed,
        "bcn_heldout_frac": bcn_heldout_frac,
        "n_ham_train": int(len(splits["ham_train"])),
        "n_ham_val": int(len(splits["ham_val"])),
        "n_ham_test": int(len(splits["ham_test"])),
        "n_bcn_adv": int(len(splits["bcn_adv"])),
        "n_bcn_heldout": int(len(splits["bcn_heldout"])),
        "lesion_disjoint": True,
        "orientation": "ID=HAM OOD=BCN_heldout",
    }
    return splits, meta
