#!/usr/bin/env python3
"""R14 §2.4: pHash near-dup of MILK dermoscopic vs ISIC 2019 and OOD panels.

Drop MILK images matching ISIC 2019 (Hamming ≤ 6). List (do not drop) OOD matches.
Appends to OVERLAP.md; updates processed milk_adv/heldout if drops occur.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r14_pair3_common as C  # noqa: E402
import r6_common as R  # noqa: E402

PHASH_HAMMING = 6
HASH_SIZE = 8  # 64-bit


def phash(path, hash_size=HASH_SIZE):
    import imagehash
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"), hash_size=hash_size)


def build_index(paths, label, cache_path: Path | None = None):
    if cache_path and cache_path.exists():
        print("load hash cache", cache_path, flush=True)
        rows = json.loads(cache_path.read_text())
        path_hash = {r["path"]: r["hash"] for r in rows}
        by_hash = defaultdict(list)
        for p, h in path_hash.items():
            by_hash[h].append(p)
        return path_hash, by_hash
    path_hash = {}
    by_hash = defaultdict(list)
    n_fail = 0
    for i, p in enumerate(paths):
        try:
            h = str(phash(p))
            path_hash[p] = h
            by_hash[h].append(p)
        except Exception:
            n_fail += 1
        if (i + 1) % 1000 == 0:
            print(label, "hashed", i + 1, "/", len(paths), "fail", n_fail, flush=True)
    print(label, "done", len(path_hash), "fail", n_fail, flush=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            [{"path": p, "hash": h} for p, h in path_hash.items()]) + "\n")
    return path_hash, by_hash


def hamming(h1, h2):
    import imagehash
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def match_against(query_hash: dict, gallery_by_hash: dict, gallery_path_hash: dict, max_d=PHASH_HAMMING):
    """Brute Hamming on 64-bit ints via numpy uint64 (OK for ~25k×5k)."""
    q_paths = list(query_hash.keys())
    g_paths = list(gallery_path_hash.keys())
    if not q_paths or not g_paths:
        return []
    q = np.array([int(query_hash[p], 16) for p in q_paths], dtype=np.uint64)
    g = np.array([int(gallery_path_hash[p], 16) for p in g_paths], dtype=np.uint64)
    def popcount_u64(x):
        m1 = np.uint64(0x5555555555555555)
        m2 = np.uint64(0x3333333333333333)
        m4 = np.uint64(0x0f0f0f0f0f0f0f0f)
        h01 = np.uint64(0x0101010101010101)
        x = x - ((x >> np.uint64(1)) & m1)
        x = (x & m2) + ((x >> np.uint64(2)) & m2)
        x = (x + (x >> np.uint64(4))) & m4
        return ((x * h01) >> np.uint64(56)).astype(np.int16)

    hits = []
    chunk = 512
    for i0 in range(0, len(q), chunk):
        qi = q[i0:i0 + chunk]
        xor = np.bitwise_xor(qi[:, None], g[None, :])
        dist = popcount_u64(xor)
        ys, xs = np.where(dist <= max_d)
        for yi, xi in zip(ys.tolist(), xs.tolist()):
            hits.append({
                "query": q_paths[i0 + yi],
                "gallery": g_paths[xi],
                "hamming": int(dist[yi, xi]),
            })
    return hits


def panel_paths():
    panels = {}
    # ISIC 2019 all training images
    isic_root = Path("/data2/hpcshared/Vinh/CSG-Skin/data/lesion_only_images/ISIC_2019_Training_Input")
    panels["isic2019"] = sorted(str(p) for p in isic_root.glob("*.jpg"))
    # PAD
    master = R.p2.load_master()
    pad = master[master.domain == "pad_ufes"]
    panels["pad_ufes20"] = pad.path.tolist()
    # Fitz
    for cand in (R.p3.FITZ_DIR, Path("/data2/hpcshared/Vinh/CSG-Skin/data/fitzpatrick17k")):
        if cand.exists():
            panels["fitzpatrick17k"] = [str(p) for p in sorted(cand.glob("*.jpg"))]
            break
    p7a = R.P7A_SETS
    for n in ("mednode", "sd198"):
        csv = p7a / "{}.csv".format(n)
        if csv.exists():
            panels[n] = pd.read_csv(csv, keep_default_na=False).path.tolist()
    return panels


def main():
    cache_dir = C.OUT / "cache" / "phash"
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(C.PROC / "milk_dermoscopic_clean.csv")
    milk_paths = clean["path"].tolist()
    milk_hash, _ = build_index(milk_paths, "milk", cache_dir / "milk.json")

    panels = panel_paths()
    all_hits = {}
    drop_isic = set()
    for name, paths in panels.items():
        print("panel", name, len(paths), flush=True)
        g_hash, g_by = build_index(paths, name, cache_dir / "{}.json".format(name))
        hits = match_against(milk_hash, g_by, g_hash, PHASH_HAMMING)
        all_hits[name] = hits
        print("hits", name, len(hits), flush=True)
        if name == "isic2019":
            for h in hits:
                drop_isic.add(h["query"])

    # apply drops
    before = len(clean)
    clean2 = clean[~clean["path"].isin(drop_isic)].reset_index(drop=True)
    print("drop ISIC near-dups", before - len(clean2), "remain", len(clean2), flush=True)

    # re-split? Work order: drop MILK images that match ISIC; then splits already done.
    # Re-filter existing adv/heldout rather than redraw.
    for split in ("milk_adv", "milk_heldout"):
        df = pd.read_csv(C.PROC / "{}.csv".format(split))
        n0 = len(df)
        df2 = df[~df["path"].isin(drop_isic)].reset_index(drop=True)
        df2.to_csv(C.PROC / "{}.csv".format(split), index=False)
        print(split, n0, "->", len(df2), flush=True)

    clean2.to_csv(C.PROC / "milk_dermoscopic_clean.csv", index=False)

    # update overlap_lesions.csv with phash rows (append section)
    phash_rows = []
    for name, hits in all_hits.items():
        for h in hits:
            phash_rows.append({
                "kind": "phash",
                "panel": name,
                "milk_path": h["query"],
                "match_path": h["gallery"],
                "hamming": h["hamming"],
                "dropped": name == "isic2019",
            })
    phash_df = pd.DataFrame(phash_rows)
    phash_path = C.OUT / "overlap_phash.csv"
    phash_df.to_csv(phash_path, index=False)

    lines = [
        "# OVERLAP — R14 ISIC 2019 → MILK10k dermoscopic",
        "",
        "Generated: {}".format(datetime.now(timezone.utc).isoformat()),
        "Commit: `{}`".format(C.git_head()),
        "",
        "## ISIC API lesion overlap (§2.3)",
        "",
    ]
    api = json.loads((C.OUT / "OVERLAP_API.json").read_text()) if (C.OUT / "OVERLAP_API.json").exists() else {}
    lines += [
        "- Overlap lesions removed: **{}** (expected 859)".format(api.get("n_overlap_lesions")),
        "- Linked ISIC 2019 images: **{}** (expected 1,895)".format(api.get("n_overlap_isic2019_images")),
        "- Remaining mapped lesions: **{}**".format(api.get("n_remaining")),
        "- Remaining counts: `{}`".format(api.get("remaining_counts")),
        "- Diff from box check: `{}`".format(api.get("diff_from_expected")),
        "",
        "Patient overlap note: Vienna/HAM10000 patient sharing cannot be removed (no patient id); "
        "only shared lesions removed. milk_heldout not in verdict.",
        "",
        "## pHash near-duplicates (64-bit, Hamming ≤ {})".format(PHASH_HAMMING),
        "",
        "| Panel | Matches | Action |",
        "|---|---:|---|",
    ]
    for name in ("isic2019", "pad_ufes20", "fitzpatrick17k", "mednode", "sd198"):
        n = len(all_hits.get(name, []))
        act = "DROP from MILK" if name == "isic2019" else "list only"
        lines.append("| {} | {} | {} |".format(name, n, act))
    lines += [
        "",
        "MILK images dropped by pHash vs ISIC 2019: **{}**".format(len(drop_isic)),
        "",
        "Files: `overlap_lesions.csv` (API), `overlap_phash.csv` (pHash).",
        "",
    ]
    (C.OUT / "OVERLAP.md").write_text("\n".join(lines) + "\n")
    (C.OUT / "OVERLAP_PHASH.json").write_text(json.dumps({
        "hamming_max": PHASH_HAMMING,
        "n_drop_isic": len(drop_isic),
        "counts": {k: len(v) for k, v in all_hits.items()},
        "commit": C.git_head(),
    }, indent=2) + "\n")
    print("wrote OVERLAP.md", flush=True)


if __name__ == "__main__":
    main()
