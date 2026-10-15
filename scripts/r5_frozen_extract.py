#!/usr/bin/env python3
"""Paper B R5 P3/P12/Q5 stage 1 (GPU-light; runs on CPU too, slower): frozen-encoder features on the frame lists.

Does not import the CSG code (no transformers stub), so MONET can use the real transformers package.
Usage:
  r5_frozen_extract.py --encoder dinov2_b14 [--frames_dir DIR] [--splits isic_train isic_test ...] [--out_root DIR]
Encoders:
  imagenet_r50  torchvision ResNet-50 IMAGENET1K_V1, fc removed (2048-d) — same as R2 item 1 (local cache)
  dinov2_b14    DINOv2 ViT-B/14, CLS token after final norm (768-d); patch-mean stored as z_patchmean (local hub cache)
  dinov2_l14    DINOv2 ViT-L/14, same (1024-d) (local hub cache)
  monet         MONET (suinleelab/monet, CLIP ViT-L/14), get_image_features (768-d). Downloads ~1.7 GB from Hugging Face
                on first use (ungated; no licence file on the model card — see decisions/paperB_r5_p12.md)
  panderm_l     PanDerm ViT-L/16. Weights are NOT on the cluster (Google Drive, CC BY-NC-ND 4.0) and need the PanDerm
                repository's model code; set PANDERM_REPO and PANDERM_CKPT. Raises if absent.
Output: <out_root>/<encoder>/<split>.npz with z (float32), labels, paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
FRAMES = ROOT / "results" / "paperB" / "r5" / "frozen" / "frames"
OUT = ROOT / "results" / "paperB" / "r5" / "frozen" / "features"
HUB_DINO = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
DEFAULT_SPLITS = ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k", "pad_full")


class Frame(Dataset):
    def __init__(self, df, tf):
        self.p, self.y, self.tf = list(df.path), df.label_idx.astype(int).to_numpy(), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


def build(encoder, device):
    bic = transforms.InterpolationMode.BICUBIC
    if encoder == "imagenet_r50":
        from torchvision.models import ResNet50_Weights, resnet50
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(*IMNET)])
        return net.to(device).eval(), tf, lambda n, x: {"z": n(x)}
    if encoder in ("dinov2_b14", "dinov2_l14"):
        name = {"dinov2_b14": "dinov2_vitb14", "dinov2_l14": "dinov2_vitl14"}[encoder]
        net = torch.hub.load(str(HUB_DINO), name, source="local", pretrained=True)
        tf = transforms.Compose([transforms.Resize(256, interpolation=bic), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(*IMNET)])

        def fwd(n, x):
            f = n.forward_features(x)
            return {"z": f["x_norm_clstoken"], "z_patchmean": f["x_norm_patchtokens"].mean(1)}
        return net.to(device).eval(), tf, fwd
    if encoder == "monet":
        from transformers import CLIPModel
        net = CLIPModel.from_pretrained("suinleelab/monet")
        tf = transforms.Compose([transforms.Resize(224, interpolation=bic), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(*CLIP)])
        return net.to(device).eval(), tf, lambda n, x: {"z": n.get_image_features(pixel_values=x)}
    if encoder == "panderm_l":
        repo, ckpt = os.environ.get("PANDERM_REPO"), os.environ.get("PANDERM_CKPT")
        if not (repo and ckpt and Path(ckpt).exists()):
            raise SystemExit("PanDerm weights not available: download panderm_ll_data6_checkpoint-499.pth (Google Drive, "
                             "CC BY-NC-ND 4.0) and clone github.com/SiyuanYan1/PanDerm; set PANDERM_REPO, PANDERM_CKPT.")
        sys.path.insert(0, repo)
        raise SystemExit("PanDerm loader not wired: confirm the model constructor in the PanDerm repo "
                         "(linear-probe entry point) before extraction; see decisions/paperB_r5_p12.md.")
    raise SystemExit("unknown encoder " + encoder)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--frames_dir", type=Path, default=FRAMES)
    ap.add_argument("--out_root", type=Path, default=OUT)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, tf, fwd = build(a.encoder, device)
    out = a.out_root / a.encoder
    out.mkdir(parents=True, exist_ok=True)
    splits = a.splits or [s for s in DEFAULT_SPLITS if (a.frames_dir / "{}.csv".format(s)).exists()] + \
        [p.stem for p in sorted(a.frames_dir.glob("*.csv")) if p.stem not in DEFAULT_SPLITS]
    for s in splits:
        dest = out / "{}.npz".format(s)
        if dest.exists():
            print("skip", dest, flush=True)
            continue
        df = pd.read_csv(a.frames_dir / "{}.csv".format(s), keep_default_na=False)
        dl = DataLoader(Frame(df, tf), batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                        pin_memory=device.type == "cuda")
        acc = {}
        for x, _y in dl:
            for k, v in fwd(net, x.to(device, non_blocking=True)).items():
                acc.setdefault(k, []).append(v.float().cpu().numpy())
        pack = {k: np.concatenate(v).astype(np.float32) for k, v in acc.items()}
        np.savez_compressed(dest, labels=df.label_idx.to_numpy().astype(np.int64), paths=df.path.to_numpy().astype(str),
                            groups=df.group.astype(str).to_numpy(), **pack)
        print("saved", a.encoder, s, pack["z"].shape, flush=True)


if __name__ == "__main__":
    main()
