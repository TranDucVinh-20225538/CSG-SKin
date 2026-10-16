#!/usr/bin/env python3
"""Download NCT-CRC-HE for P10 semantic OOD and write a fixed eval-path manifest.

Decisions: decisions/paperB_r5_p10.md (CC-BY-4.0, Zenodo 10.5281/zenodo.1214456).
Prefers NCT-CRC-HE-100K; also fetches CRC-VAL-HE-7K as a smaller backup. Writes
results/paperB/r5/p10/nct_crc_he_eval_paths.txt (up to N stratified paths).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = ROOT / "results" / "paperB" / "r5" / "p10"
DATA = ROOT / "data" / "nct_crc_he"
ZENODO = {
    "NCT-CRC-HE-100K.zip": "https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content",
    "CRC-VAL-HE-7K.zip": "https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content",
}


def download(name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print("exists", dest, dest.stat().st_size, flush=True)
        return dest
    url = ZENODO[name]
    tmp = dest.with_suffix(dest.suffix + ".part")
    print("download", name, "from", url, flush=True)
    urlretrieve(url, tmp)
    tmp.rename(dest)
    print("saved", dest, dest.stat().st_size, flush=True)
    return dest


def extract(zip_path: Path) -> Path:
    dest = DATA / zip_path.stem
    if dest.exists() and any(dest.rglob("*.tif")) or any(dest.rglob("*.png")) or any(dest.rglob("*.jpg")):
        print("extracted", dest, flush=True)
        return dest
    print("extract", zip_path, "->", dest, flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def collect_images(root: Path):
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def write_manifest(images, n_eval: int, seed: int = 20261016):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    imgs = list(images)
    rng.shuffle(imgs)
    # keep class folders if present: stratified by parent name
    by = {}
    for p in imgs:
        by.setdefault(p.parent.name, []).append(p)
    picked = []
    if len(by) > 1:
        per = max(1, n_eval // len(by))
        for cls, ps in sorted(by.items()):
            picked.extend(ps[:per])
        rng.shuffle(picked)
        picked = picked[:n_eval]
    else:
        picked = imgs[:n_eval]
    man = OUT / "nct_crc_he_eval_paths.txt"
    man.write_text("\n".join(str(p.resolve()) for p in picked) + "\n")
    meta = {
        "licence": "CC-BY-4.0",
        "zenodo": "10.5281/zenodo.1214456",
        "n_eval": len(picked),
        "seed": seed,
        "manifest": str(man),
        "source_n_images": len(images),
        "classes": {k: len(v) for k, v in by.items()},
    }
    (OUT / "nct_crc_he_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", man, "n=", len(picked), flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--skip_100k", action="store_true", help="only CRC-VAL-HE-7K (faster)")
    a = ap.parse_args()
    zips = ["CRC-VAL-HE-7K.zip"] if a.skip_100k else ["CRC-VAL-HE-7K.zip", "NCT-CRC-HE-100K.zip"]
    images = []
    for name in zips:
        try:
            z = download(name)
            root = extract(z)
            images = collect_images(root)
            print(name, "n_images", len(images), flush=True)
            if name.startswith("NCT-CRC-HE-100K") and images:
                break
        except Exception as e:
            print("WARN", name, e, flush=True)
    if not images:
        raise SystemExit("no NCT/CRC images available")
    write_manifest(images, a.n_eval)
    print("done", flush=True)


if __name__ == "__main__":
    main()
