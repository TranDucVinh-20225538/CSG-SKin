#!/usr/bin/env python3
"""R14 §2: download/hardlink MILK10k, dermoscopic extract, labels, ISIC API overlap, split, PRIORS.

Writes only under data/r14_pair3/ and results/paperB/r14_pair3/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r14_pair3_common as C  # noqa: E402
import r6_common as R  # noqa: E402

SEED = 42
EXPECTED_OVERLAP_LESIONS = 859
EXPECTED_OVERLAP_ISIC_IMGS = 1895
EXPECTED_REMAINING = {
    "MEL": 230, "NV": 347, "BCC": 2429, "AK": 293, "BKL": 442, "DF": 33, "VASC": 31, "SCC": 473,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw():
    C.RAW.mkdir(parents=True, exist_ok=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    # hardlink/copy zip + csv from r13 raw if present
    for src, name in ((C.MILK_ZIP_SRC, "milk10k.zip"), (C.MILK_CSV_SRC, "milk10k.csv")):
        dst = C.RAW / name
        if dst.exists() and dst.stat().st_size > 0:
            print("have", dst, flush=True)
            continue
        if src.exists():
            print("link", src, "->", dst, flush=True)
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        else:
            url = "{}/{}".format(C.S3_PREFIX, name)
            print("download", url, flush=True)
            urllib.request.urlretrieve(url, dst)
    # training_gt from zip or S3
    gt_dst = C.RAW / "training_gt.csv"
    if not gt_dst.exists():
        zpath = C.RAW / "milk10k.zip"
        with zipfile.ZipFile(zpath) as zf:
            with zf.open("supplements/training_gt.csv") as src, gt_dst.open("wb") as out:
                shutil.copyfileobj(src, out)
        print("extracted training_gt.csv", flush=True)
    # try supplements path on S3 for sha (optional)
    return {
        "milk10k.zip": C.RAW / "milk10k.zip",
        "milk10k.csv": C.RAW / "milk10k.csv",
        "training_gt.csv": C.RAW / "training_gt.csv",
    }


def extract_dermoscopic():
    C.DERM_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.RAW / "milk10k.csv")
    want = set(meta.loc[meta["image_type"] == "dermoscopic", "isic_id"].astype(str))
    print("want dermoscopic", len(want), flush=True)
    zpath = C.RAW / "milk10k.zip"
    n_have = sum(1 for i in want if (C.DERM_DIR / "{}.jpg".format(i)).exists())
    if n_have >= len(want):
        print("dermoscopic already extracted", n_have, flush=True)
        return len(want)
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = Path(name).stem
            if stem not in want:
                continue
            dest = C.DERM_DIR / "{}.jpg".format(stem)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
    n = sum(1 for p in C.DERM_DIR.glob("*.jpg"))
    print("extracted dermoscopic", n, flush=True)
    missing = [i for i in want if not (C.DERM_DIR / "{}.jpg".format(i)).exists()]
    if missing:
        raise RuntimeError("missing {} dermoscopic images e.g. {}".format(len(missing), missing[:5]))
    return n


def write_manifest(files: dict):
    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": C.git_head(),
        "sources": {
            "s3_prefix": C.S3_PREFIX,
            "doi": "10.34970/648456",
            "note": "Do not redistribute images. CC-BY-NC 4.0.",
        },
        "files": {},
        "filter": {"image_type": "dermoscopic", "n_images_expected": 5240},
    }
    for name, path in files.items():
        doc["files"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    n_img = sum(1 for _ in C.DERM_DIR.glob("*.jpg"))
    doc["dermoscopic_extracted"] = n_img
    C.MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", C.MANIFEST, flush=True)
    return doc


def map_labels():
    meta = pd.read_csv(C.RAW / "milk10k.csv")
    derm = meta[meta["image_type"] == "dermoscopic"].copy()
    gt = pd.read_csv(C.RAW / "training_gt.csv")
    # one-hot → mapped label
    mapped = []
    drop_reason = []
    for _, row in gt.iterrows():
        lid = str(row["lesion_id"])
        active = [c for c in C.GT_MAP if int(row.get(c, 0)) == 1]
        dropped = [c for c in C.GT_DROP if int(row.get(c, 0)) == 1]
        if dropped and not active:
            mapped.append((lid, None, dropped[0]))
            drop_reason.append(dropped[0])
            continue
        if len(active) != 1:
            mapped.append((lid, None, "ambiguous:{}".format(active)))
            drop_reason.append("ambiguous")
            continue
        mapped.append((lid, C.GT_MAP[active[0]], active[0]))
    gt_map = pd.DataFrame(mapped, columns=["lesion_id", "label", "raw"])
    df = derm.merge(gt_map, on="lesion_id", how="left")
    df["path"] = df["isic_id"].map(lambda i: str(C.DERM_DIR / "{}.jpg".format(i)))
    drop_counts = Counter(drop_reason)
    kept = df[df["label"].isin(C.LABELS)].copy()
    kept["label_idx"] = kept["label"].map({lab: i for i, lab in enumerate(C.LABELS)}).astype(int)
    kept["domain"] = "milk10k_dermoscopy"
    print("mapped kept", len(kept), "drop", dict(drop_counts), flush=True)
    return kept.reset_index(drop=True), dict(drop_counts)


def fetch_lesion(lesion_id: str) -> dict:
    C.CACHE_API.mkdir(parents=True, exist_ok=True)
    cache = C.CACHE_API / "{}.json".format(lesion_id)
    if cache.exists():
        return json.loads(cache.read_text())
    url = C.API_LESION.format(lesion_id)
    req = urllib.request.Request(url, headers={"User-Agent": C.USER_AGENT})
    # rate limit outside
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    cache.write_text(json.dumps(data) + "\n")
    return data


def collect_isic_ids(payload: dict) -> list:
    """Extract isic_ids / image ids from lesion API JSON (schema-tolerant)."""
    ids = set()
    if not isinstance(payload, dict):
        return []
    # common shapes
    for key in ("images", "isic_images", "related_images"):
        arr = payload.get(key)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, str):
                    ids.add(item.replace(".jpg", ""))
                elif isinstance(item, dict):
                    for k in ("isic_id", "id", "name", "image_id"):
                        if k in item and item[k]:
                            ids.add(str(item[k]).replace(".jpg", ""))
    for k in ("isic_ids", "image_ids"):
        arr = payload.get(k)
        if isinstance(arr, list):
            for x in arr:
                ids.add(str(x).replace(".jpg", ""))
    # nested under 'lesion'
    if "lesion" in payload and isinstance(payload["lesion"], dict):
        ids |= set(collect_isic_ids(payload["lesion"]))
    return sorted(ids)


def api_overlap(kept: pd.DataFrame, all_derm: pd.DataFrame):
    """Query every dermoscopic lesion_id (5,240); remove overlaps from mapped kept."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    isic2019 = C.isic_image_set()
    lesions = sorted(all_derm["lesion_id"].astype(str).unique())
    print("API overlap query n_lesions", len(lesions), "rate", C.API_RATE, "/s", flush=True)
    overlap_rows = []
    min_interval = 1.0 / C.API_RATE
    lock = threading.Lock()
    last = [0.0]
    n_cache = [0]
    n_net = [0]
    n_done = [0]

    def rate_limited_fetch(lid: str):
        cache = C.CACHE_API / "{}.json".format(lid)
        if cache.exists():
            with lock:
                n_cache[0] += 1
            return lid, json.loads(cache.read_text())
        with lock:
            now = time.monotonic()
            wait = min_interval - (now - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            n_net[0] += 1
        payload = fetch_lesion(lid)
        return lid, payload

    # fill missing via small pool; cache hits are instant
    missing = [lid for lid in lesions if not (C.CACHE_API / "{}.json".format(lid)).exists()]
    print("api missing", len(missing), "cached", len(lesions) - len(missing), flush=True)
    if missing:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(rate_limited_fetch, lid) for lid in missing]
            for fut in as_completed(futs):
                fut.result()
                with lock:
                    n_done[0] += 1
                    if n_done[0] % 200 == 0 or n_done[0] == 1:
                        print("api fetch progress", n_done[0], "/", len(missing),
                              "cache", n_cache[0], "net", n_net[0], flush=True)

    # scan all cached payloads for overlap
    for i, lid in enumerate(lesions):
        cache = C.CACHE_API / "{}.json".format(lid)
        payload = json.loads(cache.read_text())
        n_cache[0] += 1
        ids = collect_isic_ids(payload)
        hit = [x for x in ids if x in isic2019]
        if hit:
            overlap_rows.append({
                "lesion_id": lid,
                "n_isic_ids": len(ids),
                "n_isic2019": len(hit),
                "isic2019_ids": ";".join(hit),
            })
        if (i + 1) % 1000 == 0 or i == 0:
            print("api scan", i + 1, "/", len(lesions), "overlap_so_far", len(overlap_rows), flush=True)
    # refine HAM: lesion_id starts with HAM_ or linked images historically from HAM10000
    ov = pd.DataFrame(overlap_rows)
    if len(ov):
        ov["lesion_ham"] = ov["lesion_id"].astype(str).str.startswith("HAM_")
        # also check metadata lesion prefixes of hit images
        lid_map = C.isic_lesion_map()
        def hamish(row):
            if row["lesion_ham"]:
                return True
            lids = [lid_map.get(x, "") for x in str(row["isic2019_ids"]).split(";") if x]
            return any(str(l).startswith("HAM_") for l in lids)
        ov["all_ham10000"] = ov.apply(hamish, axis=1)
    else:
        ov = pd.DataFrame(columns=["lesion_id", "n_isic_ids", "n_isic2019", "isic2019_ids",
                                   "lesion_ham", "all_ham10000"])
    C.OUT.mkdir(parents=True, exist_ok=True)
    ov_path = C.OUT / "overlap_lesions.csv"
    ov.to_csv(ov_path, index=False)
    overlap_lesions = set(ov["lesion_id"].astype(str)) if len(ov) else set()
    n_isic_imgs = int(ov["n_isic2019"].sum()) if len(ov) else 0
    clean = kept[~kept["lesion_id"].astype(str).isin(overlap_lesions)].reset_index(drop=True)
    counts = Counter(clean["label"])
    summary = {
        "n_dermoscopic_input": int(len(kept)),
        "n_overlap_lesions": int(len(overlap_lesions)),
        "n_overlap_isic2019_images": n_isic_imgs,
        "all_ham10000_flag_true": int(ov["all_ham10000"].sum()) if len(ov) and "all_ham10000" in ov else None,
        "n_remaining": int(len(clean)),
        "remaining_counts": {k: int(counts.get(k, 0)) for k in C.LABELS},
        "expected_box_check_2026_11_16": {
            "n_overlap_lesions": EXPECTED_OVERLAP_LESIONS,
            "n_overlap_isic2019_images": EXPECTED_OVERLAP_ISIC_IMGS,
            "remaining_counts": EXPECTED_REMAINING,
        },
        "diff_from_expected": {
            "n_overlap_lesions": int(len(overlap_lesions)) - EXPECTED_OVERLAP_LESIONS,
            "n_overlap_isic2019_images": n_isic_imgs - EXPECTED_OVERLAP_ISIC_IMGS,
            "remaining_counts": {k: int(counts.get(k, 0)) - EXPECTED_REMAINING[k] for k in C.LABELS},
        },
        "api_cache_hits": n_cache,
        "api_network_calls": n_net,
        "commit": C.git_head(),
    }
    (C.OUT / "OVERLAP_API.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return clean, summary, ov


def split_milk(clean: pd.DataFrame):
    C.PROC.mkdir(parents=True, exist_ok=True)
    lesions = clean[["lesion_id"]].drop_duplicates().reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    idx = np.arange(len(lesions))
    adv_i, hold_i = next(gss.split(idx, groups=lesions["lesion_id"]))
    adv_l = set(lesions.iloc[adv_i]["lesion_id"].astype(str))
    hold_l = set(lesions.iloc[hold_i]["lesion_id"].astype(str))
    milk_adv = clean[clean["lesion_id"].astype(str).isin(adv_l)].reset_index(drop=True)
    milk_heldout = clean[clean["lesion_id"].astype(str).isin(hold_l)].reset_index(drop=True)
    cols = ["path", "label", "label_idx", "domain", "lesion_id", "isic_id"]
    if "attribution" in clean.columns:
        cols.append("attribution")
    for extra in ("anatom_site_general", "sex", "age_approx"):
        if extra in clean.columns:
            cols.append(extra)
    milk_adv[cols].to_csv(C.PROC / "milk_adv.csv", index=False)
    milk_heldout[cols].to_csv(C.PROC / "milk_heldout.csv", index=False)
    print("split milk_adv", len(milk_adv), "lesions", milk_adv.lesion_id.nunique(),
          "milk_heldout", len(milk_heldout), "lesions", milk_heldout.lesion_id.nunique(), flush=True)
    return milk_adv, milk_heldout


def write_priors(milk_adv, milk_heldout):
    isic = R.load_data()
    pi_isic = C.class_mix(isic["train"].label_idx)
    pi_adv = C.class_mix(milk_adv["label_idx"])
    pi_hold = C.class_mix(milk_heldout["label_idx"])
    priors = {
        "prior100": pi_isic.tolist(),
        "prior50": C.dose_mix(0.5, pi_isic, pi_adv).tolist(),
        "prior0": pi_adv.tolist(),
        "pi_source": pi_isic.tolist(),
        "pi_milk_adv": pi_adv.tolist(),
        "pi_milk_heldout": pi_hold.tolist(),
        "labels": C.LABELS,
        "note": "pi_milk from milk_adv labels at run start only; never in task loss",
        "tv_isic_train_milk_adv": C.tv(pi_isic, pi_adv),
        "counts": {
            "isic_train": {C.LABELS[i]: int(np.bincount(isic["train"].label_idx, minlength=C.NCLS)[i])
                           for i in range(C.NCLS)},
            "milk_adv": {C.LABELS[i]: int(np.bincount(milk_adv["label_idx"], minlength=C.NCLS)[i])
                         for i in range(C.NCLS)},
            "milk_heldout": {C.LABELS[i]: int(np.bincount(milk_heldout["label_idx"], minlength=C.NCLS)[i])
                             for i in range(C.NCLS)},
        },
        "support_small": {
            "DF_milk_adv": int((milk_adv["label"] == "DF").sum()),
            "VASC_milk_adv": int((milk_adv["label"] == "VASC").sum()),
            "DF_milk_heldout": int((milk_heldout["label"] == "DF").sum()),
            "VASC_milk_heldout": int((milk_heldout["label"] == "VASC").sum()),
        },
        "site_field": None,
        "commit": C.git_head(),
    }
    # site/attribution
    if "attribution" in milk_adv.columns:
        sites = sorted(set(milk_adv["attribution"].astype(str)) | set(milk_heldout["attribution"].astype(str)))
        if len(sites) > 1:
            priors["site_field"] = "attribution"
            priors["per_site_lesion_counts"] = {
                "milk_adv": milk_adv.groupby("attribution")["lesion_id"].nunique().astype(int).to_dict(),
                "milk_heldout": milk_heldout.groupby("attribution")["lesion_id"].nunique().astype(int).to_dict(),
            }
        else:
            priors["site_field"] = "attribution_uniform"
            priors["attribution_value"] = sites[0] if sites else None
            priors["note_site"] = "milk10k.csv attribution is uniform; no per-site split recorded."
    (C.OUT / "PRIORS.json").write_text(json.dumps(priors, indent=2) + "\n")
    print("PRIORS TV", priors["tv_isic_train_milk_adv"], "support", priors["support_small"], flush=True)
    return priors


def main():
    C.write_status("# R14 STATUS\n\nphase: prepare_data\nstarted: {}\n".format(
        datetime.now(timezone.utc).isoformat()))
    files = ensure_raw()
    extract_dermoscopic()
    man = write_manifest(files)
    kept, drop_counts = map_labels()
    meta = pd.read_csv(C.RAW / "milk10k.csv")
    all_derm = meta[meta["image_type"] == "dermoscopic"].copy()
    clean, api_sum, ov = api_overlap(kept, all_derm)
    # gate: report diffs before continuing — we continue but flag
    diffs = api_sum["diff_from_expected"]
    big_diff = (abs(diffs["n_overlap_lesions"]) > 20
                or abs(diffs["n_overlap_isic2019_images"]) > 50
                or any(abs(v) > 20 for v in diffs["remaining_counts"].values()))
    if big_diff:
        print("WARNING: overlap counts differ from box check — see OVERLAP_API.json", flush=True)
    milk_adv, milk_heldout = split_milk(clean)
    priors = write_priors(milk_adv, milk_heldout)
    # persist mapped kept before overlap for audit
    kept.to_csv(C.PROC / "milk_dermoscopic_mapped_pre_overlap.csv", index=False)
    clean.to_csv(C.PROC / "milk_dermoscopic_clean.csv", index=False)
    (C.OUT / "PREPARE_SUMMARY.json").write_text(json.dumps({
        "manifest_sha_keys": list(man["files"].keys()),
        "drop_counts": drop_counts,
        "api": api_sum,
        "priors_tv": priors["tv_isic_train_milk_adv"],
        "n_milk_adv": len(milk_adv),
        "n_milk_heldout": len(milk_heldout),
        "big_diff_from_expected": big_diff,
        "commit": C.git_head(),
    }, indent=2) + "\n")
    C.write_status(
        "# R14 STATUS\n\nphase: prepare_data_done\n"
        "n_milk_adv: {}\nn_milk_heldout: {}\nTV: {}\n"
        "overlap_lesions: {}\nbig_diff: {}\ncommit: {}\n".format(
            len(milk_adv), len(milk_heldout), priors["tv_isic_train_milk_adv"],
            api_sum["n_overlap_lesions"], big_diff, C.git_head()))
    print("DONE prepare_data", flush=True)


if __name__ == "__main__":
    main()
