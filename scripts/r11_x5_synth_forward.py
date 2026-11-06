#!/usr/bin/env python3
"""Paper B R11 X5: smartphone-transform ISIC features through frozen ERM.

Decisions: decisions/paperB_r11_x5.md
  blur σ=1.5, colour cast +10% R / −10% B, JPEG quality 40; then val-normalize; EffB3 ERM last.ckpt.
Writes results/paperB/r11/x5/synth/s{seed}/{isic_train,isic_val,isic_test}.npz
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
from train_phase15_1_single_dann import SingleDannLightning, collect_all  # noqa: E402

OUT = R.RES / "r11" / "x5" / "synth"
ERM_CK = R.CKPT / "r5_p8" / "erm_ref"
SEEDS = (42, 52, 62)  # three seeds per X5 lock
SPLITS = ("train", "val", "test")
NAMES = {"train": "isic_train", "val": "isic_val", "test": "isic_test"}
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
BLUR_SIGMA = 1.5
JPEG_Q = 40


def git_head():
    return subprocess.run(
        ["git", "-C", str(R.PAPERB), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"


def smartphone(img: Image.Image) -> Image.Image:
    """Fixed transform: Gaussian blur σ=1.5, +10% R / −10% B, JPEG q=40."""
    img = img.convert("RGB")
    # PIL GaussianBlur radius ≈ σ
    img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))
    arr = np.asarray(img).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 1.10, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * 0.90, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_Q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class SmartphoneIsic(Dataset):
    def __init__(self, frame, tf):
        self.paths = frame.path.tolist()
        self.y = frame.label_idx.to_numpy().astype(int)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = smartphone(Image.open(self.paths[i]))
        return self.tf(img), int(self.y[i])


def erm_tag(seed):
    return "single_dann_ladv0_s{}".format(seed)


def run_seed(seed, num_workers, skip_done=True):
    out = OUT / "s{}".format(seed)
    out.mkdir(parents=True, exist_ok=True)
    need = [NAMES[s] for s in SPLITS if not (out / "{}.npz".format(NAMES[s])).exists()]
    if skip_done and not need:
        print("skip_synth", seed, flush=True)
        return
    ck = ERM_CK / erm_tag(seed) / "last.ckpt"
    if not ck.exists():
        raise FileNotFoundError(ck)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SingleDannLightning.load_from_checkpoint(
        str(ck), strict=False, map_location=device).net.to(device).eval()
    # Match ERM val preprocess after the smartphone corruption.
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*IMNET),
    ])
    data = R.load_data()
    print("synth_seed", seed, "device", device, "need", need, flush=True)
    for split in SPLITS:
        name = NAMES[split]
        dest = out / "{}.npz".format(name)
        if skip_done and dest.exists():
            continue
        ds = SmartphoneIsic(data[split], tf)
        pack = collect_all(net, DataLoader(ds, batch_size=64, shuffle=False,
                                           num_workers=num_workers), device)
        z = pack["z"]
        np.savez_compressed(
            dest,
            z_lesion=z, z_lesion_norm=z, z_context=np.zeros_like(z),
            logits=pack["logits"], labels=pack["labels"],
        )
        print("wrote", dest, z.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int, default=None, help="0..len(SEEDS)-1")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_skip_done", action="store_true")
    a = ap.parse_args()
    seeds = list(SEEDS) if a.seeds is None else a.seeds
    if a.task_id is not None:
        seeds = [SEEDS[a.task_id]]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "design": "decisions/paperB_r11_x5.md",
        "blur_sigma": BLUR_SIGMA, "jpeg_quality": JPEG_Q,
        "colour_cast": {"R": 1.10, "B": 0.90},
        "seeds": seeds, "commit": git_head(),
    }
    (OUT / "SYNTH.json").write_text(json.dumps(meta, indent=2) + "\n")
    for s in seeds:
        run_seed(s, a.num_workers, skip_done=not a.no_skip_done)
    print("DONE_X5_SYNTH", flush=True)


if __name__ == "__main__":
    main()
