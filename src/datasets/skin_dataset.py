# src/datasets/skin_dataset.py
# Master CSV dataset + Lightning DataModule (ISIC train/val, ISIC+PAD test).
# CSG-lite (stage 1): optional paired ISIC + PAD train stream (domain 0 / 1), length=max with repeat.

import os
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.datasets.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INDEX_TO_LABEL,
    LABELS,
    LABEL_TO_INDEX,
)
from src.utils.paths import DATA_ROOT as _DATA_ROOT

# Domain ids for CSG-lite (match splits.build_csg_train_dataframe / CSGTrainDataset).
DOMAIN_ISIC = 0
DOMAIN_PAD_UFES = 1


def _find_isic_training_root(isic_df):
    """Resolve a directory under ISIC_*Training* that contains training JPEGs."""
    if isic_df.empty:
        return None
    p = Path(str(isic_df.iloc[0]["path"])).resolve()
    cur = p if p.is_dir() else p.parent
    for _ in range(12):
        if cur is None or cur == cur.parent:
            break
        if "Training" in cur.name and "ISIC" in cur.name:
            return cur
        cur = cur.parent
    return p.parent


def _build_sanity_isic_eval_items(isic_df, n=5):
    """
    First n JPEG paths under the ISIC training folder (sorted), with labels from metadata
    (basename lookup). Rows stay image–label aligned; no separate label shuffle.
    """
    root = _find_isic_training_root(isic_df)
    if root is None or not root.is_dir():
        return []
    lookup = {}
    for _, row in isic_df.iterrows():
        lookup[Path(str(row["path"])).name] = (int(row["label_idx"]), str(row["label"]))
    paths = sorted(root.rglob("*.jpg"), key=lambda q: q.as_posix())[:n]
    out = []
    for p in paths:
        key = p.name
        if key not in lookup:
            continue
        li, lab = lookup[key]
        out.append((str(p.resolve()), li, lab))
        if len(out) >= n:
            break
    if len(out) < n:
        # Fallback: first n metadata rows whose path contains Training_Input (same labels)
        sub = isic_df[isic_df["path"].astype(str).str.contains("ISIC_2019_Training_Input")].sort_values(
            "path"
        )
        out = []
        for _, row in sub.head(n).iterrows():
            out.append((str(Path(str(row["path"])).resolve()), int(row["label_idx"]), str(row["label"])))
    return out


def _verify_split_rows_paired(train_df, val_df, parent_df):
    """train_test_split keeps each row intact; labels are not shuffled independently of paths."""
    if len(train_df) + len(val_df) != len(parent_df):
        raise RuntimeError("Split size mismatch: train + val != parent.")
    train_ix = set(train_df.index.tolist())
    val_ix = set(val_df.index.tolist())
    if train_ix & val_ix:
        raise RuntimeError("Data leakage: train and val share row indices.")
    if train_ix | val_ix != set(parent_df.index.tolist()):
        raise RuntimeError("Split does not partition parent index set.")
    # Spot-check: label_idx still matches label string per row
    for name, part in [("train", train_df), ("val", val_df)]:
        if part.empty:
            continue
        exp = part["label"].map(LABEL_TO_INDEX)
        if not (part["label_idx"].values == exp.values).all():
            raise ValueError("label / label_idx mismatch after split in {}.".format(name))


def build_train_transform_robust():
    """
    Match Ban_sao_datn/scripts/train_resnet50_robust.py (strong aug + RandomErasing).
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )


def build_val_transform_robust():
    """Resize + CenterCrop 224 — same as robust ResNet50 val pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )


def build_lesion_branch_transform_gray():
    """
    Lesion branch transform for color invariance:
    grayscale -> tensor -> ImageNet normalization.
    Keep output 3 channels for EfficientNet input compatibility.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
    )


def audit_label_consistency_and_print_head(datamodule, n_rows=5):
    """
    Verify string `label` <-> label_idx matches LABEL_TO_INDEX; print first rows of train split.
    Call after datamodule.setup().
    """
    if datamodule.train_dataset is None:
        datamodule.setup()

    df = datamodule.train_dataset.data
    bad_mask = ~df["label"].isin(LABEL_TO_INDEX.keys())
    if bad_mask.any():
        raise ValueError(
            "Train split has labels not in LABEL_TO_INDEX: {}".format(
                df.loc[bad_mask, "label"].unique().tolist()
            )
        )

    expected = df["label"].map(LABEL_TO_INDEX)
    if not (df["label_idx"].values == expected.values).all():
        mism = df[df["label_idx"].values != expected.values]
        raise ValueError(
            "label_idx does not match LABEL_TO_INDEX for label column:\n{}".format(mism.head(10))
        )

    print("\n=== Label mapping (index -> name) ===")
    for i in range(len(LABELS)):
        print("  {} -> {} (verify metadata uses this exact spelling)".format(i, INDEX_TO_LABEL[i]))
    print("\n=== Train split: first {} rows (path, label, label_idx) ===".format(n_rows))
    print(df[["path", "label", "label_idx"]].head(n_rows).to_string())
    print("\n=== Train label counts (by name) ===")
    print(df["label"].value_counts().sort_index())
    print("\n=== Train label_idx counts (should align 0..7) ===")
    vc = df["label_idx"].value_counts().sort_index()
    print(vc)
    missing_classes = set(range(len(LABELS))) - set(vc.index.astype(int).tolist())
    if missing_classes:
        print(
            "\n[WARN] Classes with no training samples (indices): {}".format(sorted(missing_classes))
        )


class SkinDataset(Dataset):
    """
    Rows from master_metadata.csv (columns: path, label, label_idx, ...).
    Same pattern as Ban_sao_datn ISICDataset: load by path, apply transform, return tensor label.
    """

    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["path"])
        if not os.path.isfile(img_path):
            raise FileNotFoundError("Image not found: {}".format(img_path))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(row["label_idx"]), dtype=torch.long)
        return image, label


class CombinedTrainDataset(Dataset):
    """
    One training step = one ISIC sample + one PAD-UFES sample (paired indices).
    Shorter domain repeats so every index has both domains (same idea as max(len_isic, len_pad) cycling).

    __getitem__ returns:
        ((image_isic_ctx, image_isic_lesion, label_isic, domain_isic),
         (image_pad_ctx, image_pad_lesion, label_pad, domain_pad))
    with label_isic in 0..num_classes-1 and domain_pad == DOMAIN_PAD_UFES (scalar long tensor).

    DataLoader(default_collate) stacks this to nested batches; use csg_lite_paired_collate to merge
    into (images_ctx, images_lesion, y_cls, y_supcon, domain) for CSGLiteLightning.
    - y_cls: lesion CE targets (PAD rows set to -1)
    - y_supcon: true labels for both ISIC and PAD (for cross-domain supervised contrastive loss)
    """

    def __init__(self, isic_dataframe, pad_dataframe, transform=None, lesion_transform=None):
        self.isic_df = isic_dataframe.reset_index(drop=True).copy()
        self.pad_df = pad_dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.lesion_transform = lesion_transform
        self.n_isic = len(self.isic_df)
        self.n_pad = len(self.pad_df)
        if self.n_isic == 0:
            raise ValueError("CombinedTrainDataset: empty ISIC split.")
        if self.n_pad == 0:
            raise ValueError("CombinedTrainDataset: empty PAD-UFES set (need all pad_ufes rows in metadata).")

    def __len__(self):
        # Repeat the shorter list so each step always has one ISIC + one PAD (see modulo in __getitem__).
        return max(self.n_isic, self.n_pad)

    @property
    def data(self):
        """ISIC rows only — for audit_label_consistency_and_print_head (same as SkinDataset.data)."""
        return self.isic_df

    def __getitem__(self, idx):
        isic_row = self.isic_df.iloc[idx % self.n_isic]
        pad_row = self.pad_df.iloc[idx % self.n_pad]

        def _load_item(row, domain_idx):
            img_path = str(row["path"])
            if not os.path.isfile(img_path):
                raise FileNotFoundError("Image not found: {}".format(img_path))
            pil = Image.open(img_path).convert("RGB")
            image_ctx = self.transform(pil) if self.transform else pil
            image_lesion = self.lesion_transform(pil) if self.lesion_transform else image_ctx
            y = torch.tensor(int(row["label_idx"]), dtype=torch.long)
            d = torch.tensor(int(domain_idx), dtype=torch.long)
            return image_ctx, image_lesion, y, d

        pair_isic = _load_item(isic_row, DOMAIN_ISIC)
        pair_pad = _load_item(pad_row, DOMAIN_PAD_UFES)
        return pair_isic, pair_pad


def csg_lite_paired_collate(batch):
    """
    Collate list of ((x_i_ctx, x_i_les, y_i, d_i), (x_p_ctx, x_p_les, y_p, d_p)) into:
      images_ctx, images_lesion, y_cls, y_supcon, domain
    """
    x_isic_ctx = torch.stack([b[0][0] for b in batch], dim=0)
    x_isic_les = torch.stack([b[0][1] for b in batch], dim=0)
    y_isic = torch.stack([b[0][2] for b in batch], dim=0)
    d_isic = torch.stack([b[0][3] for b in batch], dim=0)
    x_pad_ctx = torch.stack([b[1][0] for b in batch], dim=0)
    x_pad_les = torch.stack([b[1][1] for b in batch], dim=0)
    y_pad = torch.stack([b[1][2] for b in batch], dim=0)
    d_pad = torch.stack([b[1][3] for b in batch], dim=0)

    images_ctx = torch.cat([x_isic_ctx, x_pad_ctx], dim=0)
    images_lesion = torch.cat([x_isic_les, x_pad_les], dim=0)
    y_cls_pad = torch.full((y_isic.shape[0],), -1, dtype=torch.long)
    y_cls = torch.cat([y_isic, y_cls_pad], dim=0)
    y_supcon = torch.cat([y_isic, y_pad], dim=0)
    domain = torch.cat([d_isic, d_pad], dim=0)
    return images_ctx, images_lesion, y_cls, y_supcon, domain


class SplitConfig:
    """Same random splits across scripts when instantiated with defaults."""

    def __init__(self, val_fraction=0.2, isic_test_fraction=0.2, random_state=42):
        self.val_fraction = val_fraction
        self.isic_test_fraction = isic_test_fraction
        self.random_state = random_state


class SkinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_csv=None,
        batch_size=32,
        num_workers=8,
        split_config=None,
        use_robust_transforms=True,
        csg_lite_train=False,
    ):
        super().__init__()
        if metadata_csv is None:
            metadata_csv = str(_DATA_ROOT / "master_metadata.csv")
        self.metadata_csv = Path(metadata_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_config = split_config or SplitConfig()
        self.use_robust_transforms = use_robust_transforms
        self.csg_lite_train = csg_lite_train

        self.train_dataset = None
        self.val_dataset = None
        self.ood_test_dataset = None
        self.sanity_isic_eval_items = []
        self.pad_df_all = None

        if use_robust_transforms:
            self.train_transform = build_train_transform_robust()
            self.eval_transform = build_val_transform_robust()
            self.lesion_branch_transform = build_lesion_branch_transform_gray()
        else:
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            self.eval_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            self.lesion_branch_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    def setup(self, stage=None):
        df = pd.read_csv(self.metadata_csv)
        required = {"path", "label", "domain"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError("Missing columns: {}".format(sorted(missing)))

        df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
        if df.empty:
            raise ValueError("No valid rows after label filter.")

        df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)
        df["label_name"] = df["label"]
        df = df[df["path"].map(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
        if df.empty:
            raise ValueError("No existing image paths in metadata.")

        print("\n=== LABEL_TO_INDEX (string label -> class index) ===")
        print(dict(LABEL_TO_INDEX))
        print("\n=== After mapping: first 10 rows (label_name vs label_idx) ===")
        print(df[["label_name", "label_idx"]].head(10).to_string())

        isic_df = df[df["domain"] == "isic"].copy()
        pad_df = df[df["domain"] == "pad_ufes"].copy()
        if isic_df.empty:
            raise ValueError("No ISIC rows in metadata.")

        # Stratify by label_idx so val mirrors train class proportions (no separate balancing step).
        # Rows are split atomically; image path and label stay on the same row (no label shuffle).
        isic_train_val, isic_test = train_test_split(
            isic_df,
            test_size=self.split_config.isic_test_fraction,
            stratify=isic_df["label_idx"],
            random_state=self.split_config.random_state,
        )
        isic_train, isic_val = train_test_split(
            isic_train_val,
            test_size=self.split_config.val_fraction,
            stratify=isic_train_val["label_idx"],
            random_state=self.split_config.random_state,
        )
        _verify_split_rows_paired(isic_train, isic_val, isic_train_val)

        train_vc = isic_train["label_idx"].value_counts().sort_index()
        val_vc = isic_val["label_idx"].value_counts().sort_index()
        train_frac = train_vc / train_vc.sum()
        val_frac = val_vc / val_vc.sum()
        print("\n=== Split check: train vs val class counts (stratify=label_idx; proportions should match) ===")
        print("Train counts:\n{}".format(train_vc))
        print("Val counts:\n{}".format(val_vc))
        print("Train fraction per class:\n{}".format(train_frac.round(4)))
        print("Val fraction per class:\n{}".format(val_frac.round(4)))

        self.sanity_isic_eval_items = _build_sanity_isic_eval_items(isic_df, n=5)

        # All PAD-UFES rows (auxiliary domain for CSG-lite paired training).
        self.pad_df_all = pad_df.reset_index(drop=True).copy()

        ood_test = pd.concat([isic_test, pad_df], ignore_index=True)
        if self.csg_lite_train:
            self.train_dataset = CombinedTrainDataset(
                isic_train,
                self.pad_df_all,
                transform=self.train_transform,
                lesion_transform=self.lesion_branch_transform,
            )
            print(
                "\n=== CSG-lite paired train: ISIC train n={} | PAD-UFES (all) n={} | steps/epoch={} (max×repeat) ===".format(
                    len(isic_train),
                    len(self.pad_df_all),
                    len(self.train_dataset),
                )
            )
        else:
            self.train_dataset = SkinDataset(isic_train, transform=self.train_transform)
        self.val_dataset = SkinDataset(isic_val, transform=self.eval_transform)
        self.ood_test_dataset = SkinDataset(ood_test, transform=self.eval_transform)

    def _make_loader(self, dataset, shuffle, drop_last=False, collate_fn=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first.")
        collate = csg_lite_paired_collate if self.csg_lite_train else None
        return self._make_loader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Call setup() first.")
        return self._make_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        if self.ood_test_dataset is None:
            raise RuntimeError("Call setup() first.")
        return self._make_loader(self.ood_test_dataset, shuffle=False, drop_last=False)
