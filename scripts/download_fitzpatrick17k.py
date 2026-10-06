#!/usr/bin/env python3
"""Download Fitzpatrick17k images for evaluation only. Do not redistribute."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CSV_PATH = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase11_third_domain/fitzpatrick17k.csv")
OUT_DIR = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/data/fitzpatrick17k/images")
META_OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase11_third_domain/fitzpatrick17k_download_status.json")
UA = "CSG-Skin-paperB/phase11-eval-only (research; no redistribution)"


def fetch_one(md5, url, dest, timeout=20):
    if dest.exists() and dest.stat().st_size > 1000:
        return md5, "exists", dest.stat().st_size
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 500:
            return md5, "too_small", len(data)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return md5, "ok", len(data)
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        return md5, "fail:{}".format(type(e).__name__), 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timeout", type=int, default=20)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
    print("queued", len(rows), flush=True)
    counts = {"ok": 0, "exists": 0, "fail": 0, "too_small": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for row in rows:
            md5 = row["md5hash"]
            url = row["url"]
            dest = OUT_DIR / "{}.jpg".format(md5)
            futs.append(ex.submit(fetch_one, md5, url, dest, args.timeout))
        done = 0
        for fut in as_completed(futs):
            md5, status, nbytes = fut.result()
            key = "fail" if status.startswith("fail") else status
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 500 == 0:
                print(done, "/", len(rows), counts, flush=True)
    META_OUT.write_text(
        json.dumps(
            {
                "n_listed": len(rows),
                "counts": counts,
                "image_dir": str(OUT_DIR),
                "licence": "eval only; mixed photograph copyright; do not redistribute",
            },
            indent=2,
        )
        + "\n"
    )
    print("done", counts, "->", META_OUT)


if __name__ == "__main__":
    main()
