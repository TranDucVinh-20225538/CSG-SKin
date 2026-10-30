#!/usr/bin/env python3
"""Paper B R7 C3: Fitzpatrick17k coverage (headers only) (decisions/paperB_r7_c3.md)."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = R.R7 / "c3"
CANDIDATES = [
    R.PAPERB / "results" / "paperB" / "phase11_third_domain" / "fitzpatrick17k.csv",
    R.PAPERB / "results" / "paperB" / "r5" / "frozen" / "frames" / "fitzpatrick17k.csv",
    R.PAPERB / "data" / "fitzpatrick17k.csv",
]


def find_csv():
    for p in CANDIDATES:
        if p.exists():
            return p
    # search
    hits = list((R.PAPERB / "results").rglob("fitzpatrick17k.csv"))
    return hits[0] if hits else None


def find_download_log():
    roots = [R.PAPERB / "logs", R.PAPERB / "results" / "paperB" / "phase11_third_domain",
             R.PAPERB / "data", Path.home() / "downloads"]
    pats = ("*fitz*log*", "*fitz*download*", "*fitzpatrick*fail*", "*fitz*http*")
    found = []
    for root in roots:
        if not root.exists():
            continue
        for pat in pats:
            found.extend(root.glob(pat))
    return found


def tv(p, q, keys):
    """Total variation distance between two discrete distributions over keys."""
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def mix(series):
    c = Counter(str(x) for x in series if pd.notna(x) and str(x) != "")
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}, n


def head_status(url, timeout=8):
    """Prefer HEAD; on 405/406 fall back to GET with Range: bytes=0-0 (still no image body download)."""
    import urllib.error
    import urllib.request

    def _one(method, headers=None):
        req = urllib.request.Request(url, method=method, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), "ok"

    try:
        return _one("HEAD")
    except Exception as e:
        msg = str(e)
        if "404" in msg:
            return 404, "removed_or_missing"
        if "403" in msg:
            return 403, "forbidden"
        if "timed out" in msg.lower() or "Name or service" in msg or "Network is unreachable" in msg:
            return None, "host_unreachable"
        # retry as ranged GET (many atlas hosts reject HEAD with 406)
        try:
            return _one("GET", {"Range": "bytes=0-0"})
        except urllib.error.HTTPError as he:
            if he.code == 404:
                return 404, "removed_or_missing"
            if he.code == 403:
                return 403, "forbidden"
            return he.code, "http_{}".format(he.code)
        except Exception as e2:
            msg2 = str(e2)
            if "timed out" in msg2.lower() or "Name or service" in msg2 or "unreachable" in msg2.lower():
                return None, "host_unreachable"
            return None, "error:" + msg2[:80]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = find_csv()
    if csv_path is None:
        (OUT / "REPORT.md").write_text("# C3 blocked: no fitzpatrick17k.csv found\n")
        print("no csv")
        return
    df = pd.read_csv(csv_path, low_memory=False)
    n_full = len(df)
    # kept images: those present on disk under FITZ_DIR if available
    import train_phase3_sweep as p3
    kept_paths = {p.name for p in p3.FITZ_DIR.glob("*.jpg")} if p3.FITZ_DIR.exists() else set()
    # try url / md5 / filename columns
    url_col = next((c for c in ("url",) if c in df.columns), None) or next(
        (c for c in df.columns if "url" in c.lower() and "alphanum" not in c.lower()), None)
    label_col = "label" if "label" in df.columns else next(
        (c for c in df.columns if c.lower() in ("nine_partition_label", "diagnosis", "dx")), None)
    fitz_col = "fitzpatrick_scale" if "fitzpatrick_scale" in df.columns else next(
        (c for c in df.columns if "fitzpatrick" in c.lower()), None)
    # on-disk files are {md5hash}.jpg
    if "md5hash" in df.columns and kept_paths:
        df["_fname"] = df["md5hash"].astype(str) + ".jpg"
        df["_kept"] = df["_fname"].isin(kept_paths)
    else:
        name_col = next((c for c in df.columns if c.lower() in ("filename", "file", "image")), None)
        df["_kept"] = df[name_col].astype(str).map(lambda x: Path(x).name in kept_paths) if name_col else False

    n_kept = int(df["_kept"].sum()) if hasattr(df["_kept"], "sum") else 0
    (OUT / "note.json").write_text(json.dumps({
        "csv": str(csv_path), "n_full": n_full, "n_disk_jpg": len(kept_paths), "n_kept_matched": n_kept,
        "url_col": url_col, "label_col": label_col, "fitz_col": fitz_col,
        "columns": list(df.columns)[:40]}, indent=2) + "\n")

    # atlas host from URL
    atlas = Counter()
    missing_status = Counter()
    logs = find_download_log()
    header_sample = []
    if url_col and n_kept < n_full:
        missing = df.loc[~df["_kept"]] if df["_kept"].any() else df.sample(min(200, len(df)), random_state=0)
        # sample up to 150 missing URLs for HEAD
        urls = missing[url_col].dropna().astype(str).unique().tolist()[:150]
        for u in urls:
            host = urlparse(u).netloc or "unknown"
            atlas[host] += 1
            code, reason = head_status(u)
            missing_status[reason] += 1
            header_sample.append({"url": u, "host": host, "status": code, "reason": reason})

    # class / fitz mix TV
    kept_df = df[df["_kept"]] if bool(df["_kept"].any()) else df.iloc[0:0]
    tv_rows = []
    map_path = csv_path.parent / "fitzpatrick17k_label_map.json"
    isic_map = {}
    if map_path.exists():
        isic_map = {k.lower(): v for k, v in json.loads(map_path.read_text()).get("mapping_rule", {}).items()}

    def to_isic(series):
        return series.astype(str).str.lower().map(lambda x: isic_map.get(x, "UNMAPPED"))

    if label_col and len(kept_df):
        p_full, _ = mix(df[label_col])
        p_kept, _ = mix(kept_df[label_col])
        keys = sorted(set(p_full) | set(p_kept))
        tv_rows.append({"variable": "raw_label ({})".format(label_col), "tv": tv(p_full, p_kept, keys),
                        "n_full": n_full, "n_kept": int(len(kept_df))})
        if isic_map:
            p_full, _ = mix(to_isic(df[label_col]))
            p_kept, _ = mix(to_isic(kept_df[label_col]))
            keys = sorted(set(p_full) | set(p_kept))
            tv_rows.append({"variable": "isic_mapped_label", "tv": tv(p_full, p_kept, keys),
                            "n_full": n_full, "n_kept": int(len(kept_df)),
                            "full_mix": p_full, "kept_mix": p_kept})
    if fitz_col and len(kept_df):
        p_full, _ = mix(df[fitz_col])
        p_kept, _ = mix(kept_df[fitz_col])
        keys = sorted(set(p_full) | set(p_kept))
        tv_rows.append({"variable": "fitzpatrick ({})".format(fitz_col), "tv": tv(p_full, p_kept, keys),
                        "n_full": n_full, "n_kept": int(len(kept_df)),
                        "full_mix": p_full, "kept_mix": p_kept})

    coverage = {
        "csv": str(csv_path), "n_full_csv": n_full, "n_kept": n_kept, "n_disk_jpg": len(kept_paths),
        "download_logs_found": [str(p) for p in logs],
        "missing_by_reason": dict(missing_status),
        "missing_by_host_sample": dict(atlas),
        "header_sample_n": len(header_sample),
        "tv": tv_rows,
    }
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if header_sample:
        (OUT / "header_sample.json").write_text(json.dumps(header_sample, indent=2) + "\n")

    L = ["# Paper B R7 C3: Fitzpatrick17k coverage", "",
         "Design: `decisions/paperB_r7_c3.md`. Commit `{}`.".format(R.git_head()), "",
         "CSV: `{}` ({} rows). On-disk jpgs under FITZ_DIR: {}. Matched kept rows: {}.".format(
             csv_path, n_full, len(kept_paths), n_kept), "",
         "## Missing-image reasons (HEAD sample)", "",
         "| reason | count |", "|---|---:|"]
    for k, v in sorted(missing_status.items(), key=lambda x: -x[1]):
        L.append("| {} | {} |".format(k, v))
    if not missing_status:
        L.append("| (no HEAD sample — see coverage.json; download logs: {}) | — |".format(
            len(logs)))
    L += ["", "## Class / Fitzpatrick mix: full vs kept (TV)", "",
          "| variable | n_full | n_kept | TV |", "|---|---:|---:|---:|"]
    for r in tv_rows:
        L.append("| {} | {} | {} | {:.4f} |".format(r["variable"], r["n_full"], r["n_kept"], r["tv"]))
    if not tv_rows:
        L.append("| (could not map kept rows to csv; see note.json / coverage.json) | {} | {} | — |".format(
            n_full, len(kept_paths)))
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", "full", n_full, "kept", n_kept, "disk", len(kept_paths))


if __name__ == "__main__":
    main()
