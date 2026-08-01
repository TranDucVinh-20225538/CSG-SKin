# src/datasets/preprocess_metadata.py
# Build master_metadata.csv: ISIC 2019 (one-hot labels) + PAD-UFES (mapped labels).
#
# ISIC 2019 challenge usually ships:
#   - ISIC_2019_Training_Metadata.csv  (demographics; often NO class columns)
#   - ISIC_2019_Training_GroundTruth.csv (image + MEL, NV, BCC, ... one-hot)
# If one-hot columns are missing from metadata, GroundTruth is required (same folder as DATA_ROOT).

import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.paths import DATA_ROOT

# ================= Paths (under DATA_ROOT) =================
ISIC_META_CSV = DATA_ROOT / "ISIC_2019_Training_Metadata.csv"
ISIC_INPUT_DIR = DATA_ROOT / "ISIC_2019_Training_Input"
PAD_META_CSV = DATA_ROOT / "pad_ufes20" / "metadata.csv"
PAD_IMAGES_DIR = DATA_ROOT / "pad_ufes20" / "images"
OUT_CSV = DATA_ROOT / "master_metadata.csv"

ISIC_ONE_HOT_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
PAD_DIAG_MAP = {
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NV",
    "ACK": "AK",
    "SEK": "BKL",
    "SCC": "SCC",
}

IMAGE_COL_CANDIDATES = ["image", "image_id", "Image"]


def _one_hot_to_label(row, cols):
    present = [c for c in cols if c in row.index]
    if not present:
        return float("nan")
    positives = []
    for c in present:
        v = row[c]
        try:
            is_pos = int(float(v)) == 1
        except (TypeError, ValueError):
            is_pos = False
        if is_pos:
            positives.append(c)
    if len(positives) == 1:
        return positives[0]
    return float("nan")


def _find_image_column(df):
    for c in IMAGE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _dataframe_has_all_one_hot(df):
    return all(c in df.columns for c in ISIC_ONE_HOT_COLS)


def _normalize_gt_dataframe(df):
    """Map legacy column names (e.g. ISIC 2018 uses AKIEC instead of AK)."""
    df = df.copy()
    if "AKIEC" in df.columns and "AK" not in df.columns:
        df = df.rename(columns={"AKIEC": "AK"})
    return df


def _iter_groundtruth_csv_paths():
    """Preferred filenames first, then any CSV whose name contains 'groundtruth' (any case)."""
    preferred = [
        DATA_ROOT / "ISIC_2019_Training_GroundTruth.csv",
        DATA_ROOT / "ISIC_2019_Training_Groundtruth.csv",
        DATA_ROOT / "ISIC2019_Training_GroundTruth.csv",
        DATA_ROOT / "ISIC2019_Training_Groundtruth.csv",
    ]
    seen = set()
    for p in preferred:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                yield p
    # Linux globs are case-sensitive; unzip often yields e.g. ...Groundtruth.csv
    for p in sorted(DATA_ROOT.glob("*.csv")):
        if "groundtruth" not in p.name.lower():
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            yield p


def _load_isic_label_source():
    """
    Return (dataframe with one-hot columns, image column name).
    Prefer metadata CSV if it already contains lesion one-hot columns; else GroundTruth CSV.
    """
    if not ISIC_META_CSV.is_file():
        raise FileNotFoundError("Missing ISIC metadata: {}".format(ISIC_META_CSV))

    meta = pd.read_csv(ISIC_META_CSV)
    meta = _normalize_gt_dataframe(meta)
    if _dataframe_has_all_one_hot(meta):
        img_col = _find_image_column(meta)
        if img_col is None:
            raise KeyError("ISIC metadata needs an image column among {}".format(IMAGE_COL_CANDIDATES))
        return meta, img_col

    for gt_path in _iter_groundtruth_csv_paths():
        try:
            if gt_path.stat().st_size == 0:
                print(
                    "  Skipping {} (0 bytes): replace with the official ISIC 2019 Task 3 training GroundTruth CSV.".format(
                        gt_path.name
                    )
                )
                continue
            gt = _normalize_gt_dataframe(pd.read_csv(gt_path))
        except EmptyDataError:
            print("  Skipping {}: empty or invalid CSV.".format(gt_path.name))
            continue
        if not _dataframe_has_all_one_hot(gt):
            continue
        img_col = _find_image_column(gt)
        if img_col is None:
            continue
        print(
            "  Using {} for lesion labels ({} has no one-hot columns).".format(
                gt_path.name, ISIC_META_CSV.name
            )
        )
        return gt, img_col

    found_gt = list(_iter_groundtruth_csv_paths())
    extra = ""
    if found_gt:
        p0 = found_gt[0]
        if p0.is_file() and p0.stat().st_size == 0:
            extra = "\n  {} exists but is empty (0 bytes). Re-download from https://challenge.isic-archive.com/data/".format(
                p0.name
            )
        else:
            try:
                g = _normalize_gt_dataframe(pd.read_csv(p0))
                extra = "\n  Found {} but columns are {} (need {}).".format(
                    p0.name, list(g.columns), ISIC_ONE_HOT_COLS
                )
            except Exception:
                extra = "\n  Found GroundTruth file(s) but could not read them as expected."

    raise FileNotFoundError(
        "ISIC lesion labels not found.\n"
        "  Your {} only has demographics (no MEL/NV/...).\n"
        "  Download the ISIC 2019 Task 3 **training** ground-truth CSV (8-class one-hot) and save as e.g.:\n"
        "    {}/ISIC_2019_Training_GroundTruth.csv\n"
        "  Official challenge data: https://challenge.isic-archive.com/data/\n"
        "  Alternatively, merge one-hot columns into the metadata CSV.\n"
        "  Expected columns: {}"
        "{}".format(ISIC_META_CSV.name, DATA_ROOT, ISIC_ONE_HOT_COLS, extra)
    )


def resolve_isic_image_path(image_id):
    """
    Map ISIC `image` id to a local jpg under ISIC_INPUT_DIR (flat or nested unzip layout).
    """
    raw = str(image_id).strip()
    if raw.lower().endswith(".jpg"):
        raw = raw[: -4]

    nested = ISIC_INPUT_DIR / "ISIC_2019_Training_Input"
    search_roots = [ISIC_INPUT_DIR, nested]
    direct_names = ["{}.jpg".format(raw), "{}_downsampled.jpg".format(raw)]

    for root in search_roots:
        if not root.is_dir():
            continue
        for name in direct_names:
            p = root / name
            if p.is_file():
                return p

    for root in search_roots:
        if not root.is_dir():
            continue
        matches = sorted(root.glob("{}*.jpg".format(raw)))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            preferred = root / "{}.jpg".format(raw)
            if preferred.is_file():
                return preferred
            return matches[0]

    return None


def load_isic():
    df, img_col = _load_isic_label_source()

    records = []
    skipped_label = 0
    skipped_path = 0
    for _, row in df.iterrows():
        label = _one_hot_to_label(row, ISIC_ONE_HOT_COLS)
        if isinstance(label, float) and pd.isna(label):
            skipped_label += 1
            continue
        image_id = str(row[img_col]).strip()
        resolved = resolve_isic_image_path(image_id)
        if resolved is None:
            skipped_path += 1
            continue
        records.append(
            {
                "path": str(resolved.resolve()),
                "label": str(label),
                "domain": "isic",
            }
        )
    if skipped_label:
        print("  (ISIC) skipped {} rows: no single positive one-hot label.".format(skipped_label))
    if skipped_path:
        print("  (ISIC) skipped {} rows: image file not found under {}.".format(skipped_path, ISIC_INPUT_DIR))

    return pd.DataFrame.from_records(records)


def load_pad_ufes():
    if not PAD_META_CSV.is_file():
        raise FileNotFoundError("Missing PAD-UFES metadata: {}".format(PAD_META_CSV))

    df = pd.read_csv(PAD_META_CSV)
    if "diagnostic" not in df.columns:
        raise KeyError("PAD-UFES metadata must contain 'diagnostic'; found: {}".format(list(df.columns)))
    if "img_id" not in df.columns:
        raise KeyError("PAD-UFES metadata must contain 'img_id'; found: {}".format(list(df.columns)))

    df = df.copy()
    df = df[df["diagnostic"].isin(PAD_DIAG_MAP.keys())]
    df["label"] = df["diagnostic"].map(PAD_DIAG_MAP)

    records = []
    for _, row in df.iterrows():
        img_id = str(row["img_id"]).strip()
        # Keep paths under project data/ (do not resolve symlink targets to another repo).
        path_str = str(PAD_IMAGES_DIR / img_id)
        records.append(
            {
                "path": path_str,
                "label": str(row["label"]),
                "domain": "pad_ufes",
            }
        )
    return pd.DataFrame.from_records(records)


def filter_existing_paths(df):
    paths = df["path"].astype(str)
    exists = paths.map(lambda p: Path(p).is_file())
    n_drop = int((~exists).sum())
    return df.loc[exists].reset_index(drop=True), n_drop


def main():
    for p, name in ((ISIC_META_CSV, "ISIC metadata"), (PAD_META_CSV, "PAD-UFES metadata")):
        if not p.is_file():
            print("ERROR: {} not found: {}".format(name, p), file=sys.stderr)
            sys.exit(1)

    print("Loading ISIC 2019 labels + paths...")
    isic_df = load_isic()
    print("  rows (before file check): {}".format(len(isic_df)))

    print("Loading PAD-UFES metadata...")
    pad_df = load_pad_ufes()
    print("  rows (before file check): {}".format(len(pad_df)))

    master = pd.concat([isic_df, pad_df], ignore_index=True)
    master, n_bad = filter_existing_paths(master)
    if n_bad:
        print("Dropped {} rows whose image file was missing on disk.".format(n_bad))

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)
    print("\nSaved {} rows to {}".format(len(master), out_path))

    print("\nLabel counts by domain (rows = domain, columns = label):")
    summary = (
        master.groupby(["domain", "label"])
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .sort_index(axis=1)
    )
    print(summary.to_string())
    print("\nRow totals per domain:")
    print(master.groupby("domain").size().to_string())


if __name__ == "__main__":
    main()
