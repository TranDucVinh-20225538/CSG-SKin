#!/usr/bin/env python3
"""R2 Item 1: frozen ImageNet ResNet-50 features on the Phase 13 split frames.

Outputs results/paperB/r2/features/imagenet_resnet50/<split>.npz with keys z, labels.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = object
    _tf.AutoTokenizer = object
    _cfg = types.ModuleType("transformers.configuration_utils")

    class _PC:
        def __eq__(self, other):
            return False

    _cfg.PretrainedConfig = _PC
    _tf.configuration_utils = _cfg
    sys.modules["transformers"] = _tf
    sys.modules["transformers.configuration_utils"] = _cfg

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C
import train_phase2_pad_holdout as p2
import train_phase3_sweep as p3
from eval_fitz_domain_axis import FolderJpegDataset
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust

OUT = C.R2 / "features" / "imagenet_resnet50"
CACHE16 = C.PAPERB / "results" / "paperB" / "phase1_6" / "features" / "imagenet_resnet50_raw.npz"


@torch.no_grad()
def run(net, loader, device):
    zs, ys = [], []
    for images, labels in loader:
        zs.append(net(images.to(device, non_blocking=True)).float().cpu().numpy())
        ys.append(np.asarray(labels))
    return np.concatenate(zs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()

    spec = json.loads(p3.P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy()
    train, _val, test = p2.isic_splits(isic)
    hold = pad[pad.path.isin(set(spec["pad_heldout_paths"]))].reset_index(drop=True)
    tf = build_val_transform_robust()

    def ld(ds):
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=device.type == "cuda")

    frames = {
        "isic_train": ld(SkinDataset(train, transform=tf)),
        "isic_test": ld(SkinDataset(test, transform=tf)),
        "pad_heldout": ld(SkinDataset(hold, transform=tf)),
        "fitzpatrick17k": ld(FolderJpegDataset(sorted(p3.FITZ_DIR.glob("*.jpg")), tf)),
    }
    for name, loader in frames.items():
        z, y = run(net, loader, device)
        np.savez_compressed(OUT / "{}.npz".format(name), z=z, labels=y)
        print("saved", name, z.shape, flush=True)

    cache = np.load(CACHE16)
    for name, key in (("isic_train", "train"), ("isic_test", "id")):
        z = np.load(OUT / "{}.npz".format(name))["z"]
        diff = float(np.abs(z - cache[key]).max())
        print("check vs phase1_6 cache", name, "max abs diff", diff, flush=True)


if __name__ == "__main__":
    main()
