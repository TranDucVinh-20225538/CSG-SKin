#!/usr/bin/env python3
"""Paper B R11 X2 GPU forwards: M2 missing OOD features + M3 ImageNet EffB3.

M2: fill mednode/sd198/scin under results/paperB/r5/p8/erm_ref/features/<tag>/
M3: write results/paperB/r11/x2/m3_imagenet_effb3/<split>.npz
     with z_lesion_norm = penultimate (1536-d); logits zeros (no skin head).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402
from train_phase15_1_single_dann import SingleDannLightning, collect_all  # noqa: E402

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
ERM_FEAT = R.P8_DIR / "erm_ref" / "features"
ERM_CK = R.CKPT / "r5_p8" / "erm_ref"
M3_OUT = R.RES / "r11" / "x2" / "m3_imagenet_effb3"
OOD_MISS = ("mednode", "sd198", "scin")
M3_SPLITS = ("isic_train", "isic_test", "isic_val", "pad_heldout", "fitzpatrick17k",
             "mednode", "sd198", "scin")
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def erm_tag(seed: int) -> str:
    return "single_dann_ladv0_s{}".format(seed)


def m2_datasets(tf):
    data = R.load_data()
    out = {}
    for n in OOD_MISS:
        csv = R.P7A_SETS / "{}.csv".format(n)
        out[n] = R.FolderJpegDataset(pd.read_csv(csv, keep_default_na=False).path.tolist(), tf)
    return out


def run_m2(seeds, num_workers, device, skip_done=True):
    tf = build_val_transform_robust()
    sets = m2_datasets(tf)
    for seed in seeds:
        tag = erm_tag(seed)
        ck = ERM_CK / tag / "last.ckpt"
        if not ck.exists():
            raise FileNotFoundError(ck)
        feat_dir = ERM_FEAT / tag
        feat_dir.mkdir(parents=True, exist_ok=True)
        need = [n for n in OOD_MISS if not (feat_dir / "{}.npz".format(n)).exists()]
        if skip_done and not need:
            print("skip_m2", tag, flush=True)
            continue
        print("m2_load", tag, ck, "need", need, flush=True)
        net = SingleDannLightning.load_from_checkpoint(
            str(ck), strict=False, map_location=device).net.to(device).eval()
        for name in need:
            out = feat_dir / "{}.npz".format(name)
            pack = collect_all(net, DataLoader(sets[name], batch_size=64, shuffle=False,
                                               num_workers=num_workers), device)
            z = pack["z"]
            np.savez_compressed(
                out,
                z_lesion=z,
                z_lesion_norm=z,
                z_context=np.zeros_like(z),
                logits=pack["logits"],
                labels=pack["labels"],
            )
            print("m2_wrote", out, pack["labels"].shape, flush=True)


def m3_datasets(tf):
    data = R.load_data()
    out = {
        "isic_train": SkinDataset(data["train"], transform=tf),
        "isic_val": SkinDataset(data["val"], transform=tf),
        "isic_test": SkinDataset(data["test"], transform=tf),
        "pad_heldout": SkinDataset(data["pad_heldout"], transform=tf),
        "fitzpatrick17k": R.FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf),
    }
    for n in OOD_MISS:
        csv = R.P7A_SETS / "{}.csv".format(n)
        out[n] = R.FolderJpegDataset(pd.read_csv(csv, keep_default_na=False).path.tolist(), tf)
    return out


@torch.no_grad()
def run_m3(num_workers, device, skip_done=True):
    M3_OUT.mkdir(parents=True, exist_ok=True)
    need = [n for n in M3_SPLITS if not (M3_OUT / "{}.npz".format(n)).exists()]
    if skip_done and not need:
        print("skip_m3", flush=True)
        return
    net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    d = net.classifier[1].in_features
    net.classifier = torch.nn.Identity()
    net = net.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*IMNET),
    ])
    sets = m3_datasets(tf)
    print("m3_dim", d, "need", need, flush=True)
    for name in need:
        ds = sets[name]
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
        zs, ys = [], []
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            zs.append(net(x).float().cpu().numpy())
            ys.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
        z = np.concatenate(zs).astype(np.float32)
        y = np.concatenate(ys)
        # Scorer layout; no skin classifier → logits zeros (MSP/Energy N/A for M3).
        lg = np.zeros((len(z), R.NCLS), dtype=np.float32)
        out = M3_OUT / "{}.npz".format(name)
        np.savez_compressed(
            out,
            z_lesion=z,
            z_lesion_norm=z,
            z_context=np.zeros_like(z),
            logits=lg,
            labels=y,
        )
        print("m3_wrote", out, z.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("m2", "m3", "all"), default="all")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(R.SEEDS))
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_skip_done", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    skip = not a.no_skip_done
    if a.which in ("m2", "all"):
        run_m2(a.seeds, a.num_workers, device, skip_done=skip)
    if a.which in ("m3", "all"):
        run_m3(a.num_workers, device, skip_done=skip)
    print("DONE_X2_FORWARDS", flush=True)


if __name__ == "__main__":
    main()
