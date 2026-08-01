# src/datasets/splits.py
"""Train/val/test splits from master_metadata.csv (ISIC + PAD-UFES)."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets.constants import LABEL_TO_INDEX
from src.datasets.skin_dataset import SkinDataset


def load_filtered_master(metadata_csv):
    df = pd.read_csv(metadata_csv)
    required = {"path", "label", "domain"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required metadata columns: {}".format(sorted(missing)))

    df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering to expected lesion labels.")

    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
    df = df[df["path"].map(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with existing image paths.")
    return df


def build_csg_train_dataframe(datamodule):
    """ISIC train (domain 0) + all PAD (label_idx=-1 for lesion CE, domain 1)."""
    df = load_filtered_master(datamodule.metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    pad_df = df[df["domain"] == "pad_ufes"].copy()
    if isic_df.empty or pad_df.empty:
        raise ValueError("CSG training expects both ISIC and pad_ufes rows in metadata.")

    isic_train_val, _isic_test = train_test_split(
        isic_df,
        test_size=datamodule.split_config.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=datamodule.split_config.random_state,
    )
    isic_train, _isic_val = train_test_split(
        isic_train_val,
        test_size=datamodule.split_config.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=datamodule.split_config.random_state,
    )

    isic_train = isic_train.copy()
    isic_train["domain_idx"] = 0

    pad_df = pad_df.copy()
    pad_df["label_idx"] = -1
    pad_df["domain_idx"] = 1

    return pd.concat([isic_train, pad_df], ignore_index=True)


def build_id_ood_test_dataloaders(datamodule):
    """
    Same splits as SkinDataModule.setup:
    - ID test = held-out ISIC test
    - OOD test = PAD-UFES only
    """
    df = load_filtered_master(datamodule.metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    pad_df = df[df["domain"] == "pad_ufes"].copy()

    isic_train_val, isic_test = train_test_split(
        isic_df,
        test_size=datamodule.split_config.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=datamodule.split_config.random_state,
    )
    _a, _b = train_test_split(
        isic_train_val,
        test_size=datamodule.split_config.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=datamodule.split_config.random_state,
    )

    id_test_ds = SkinDataset(isic_test, transform=datamodule.eval_transform)
    ood_test_ds = SkinDataset(pad_df, transform=datamodule.eval_transform)

    common = dict(
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=datamodule.num_workers,
        pin_memory=True,
        persistent_workers=datamodule.num_workers > 0,
    )
    return DataLoader(id_test_ds, **common), DataLoader(ood_test_ds, **common)
