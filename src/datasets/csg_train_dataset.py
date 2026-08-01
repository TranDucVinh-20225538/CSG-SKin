# src/datasets/csg_train_dataset.py
"""Training CSV that mixes ISIC (lesion labels) + PAD (ignored lesion CE, domain=1)."""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CSGTrainDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = str(row["path"])
        if not os.path.isfile(path):
            raise FileNotFoundError("Image not found: {}".format(path))
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        y = int(row["label_idx"])
        domain = int(row["domain_idx"])
        return image, torch.tensor(y, dtype=torch.long), torch.tensor(domain, dtype=torch.long)
