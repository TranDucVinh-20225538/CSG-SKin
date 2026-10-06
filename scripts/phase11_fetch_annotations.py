#!/usr/bin/env python3
"""Download Fitzpatrick17k annotation CSV only (no images) and draft an 8-class mapping."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen, Request

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase11_third_domain")
# Official annotation table from Groh et al. (CC BY-NC-SA 3.0).
URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/master/fitzpatrick17k.csv"

# Conservative string matches into ISIC-8. Unmapped rows stay label=None (still valid OOD images).
MAP = {
    "melanoma": "MEL",
    "melanoma metastasis": "MEL",
    "nevus": "NV",
    "atypical nevus": "NV",
    "dysplastic nevus": "NV",
    "melanocytic nevus": "NV",
    "basal cell carcinoma": "BCC",
    "actinic keratosis": "AK",
    "seborrheic keratosis": "BKL",
    "lichenoid keratosis": "BKL",
    "solar lentigo": "BKL",
    "dermatofibroma": "DF",
    "pyogenic granuloma": "VASC",
    "angioma": "VASC",
    "hemangioma": "VASC",
    "squamous cell carcinoma": "SCC",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "fitzpatrick17k.csv"
    if not dest.exists():
        req = Request(URL, headers={"User-Agent": "CSG-Skin-paperB/phase11"})
        print("GET", URL)
        with urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
    print("csv bytes", dest.stat().st_size)
    with dest.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    print("n_rows", len(rows), "cols", list(rows[0].keys()) if rows else None)
    labels = Counter()
    mapped = Counter()
    for row in rows:
        lab = (row.get("label") or row.get("dx") or row.get("nine_partition_label") or "").strip().lower()
        labels[lab] += 1
        hit = None
        for k, v in MAP.items():
            if k in lab:
                hit = v
                break
        mapped[hit or "UNMAPPED"] += 1
    report = {
        "source_url": URL,
        "n_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "n_mapped_to_isic8": int(sum(v for k, v in mapped.items() if k != "UNMAPPED")),
        "n_unmapped": int(mapped.get("UNMAPPED", 0)),
        "mapped_counts": dict(mapped),
        "top_raw_labels": labels.most_common(30),
        "mapping_rule": MAP,
        "note": (
            "Images are NOT downloaded. Annotation licence is CC BY-NC-SA 3.0. "
            "Underlying photos are web-scraped; do not redistribute. "
            "Phase 11 OOD detection on z_context does not require the 8-class mapping; "
            "the mapping is only for optional diagnostic transfer."
        ),
        "images_acquired": False,
    }
    (OUT / "fitzpatrick17k_label_map.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["n_rows", "n_mapped_to_isic8", "n_unmapped", "columns"]}, indent=2))


if __name__ == "__main__":
    main()
