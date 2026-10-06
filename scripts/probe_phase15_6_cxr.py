#!/usr/bin/env python3
"""15.6 — report chest X-ray availability before any compute."""

from __future__ import annotations

from pathlib import Path

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase15/PHASE15_6_AVAILABILITY.md")
CANDIDATES = {
    "CheXpert": [
        Path("/data2/cmdir/home/toandq/data/CheXpert"),
        Path("/data2/cmdir/home/toandq/data/chexpert"),
        Path("/data2/hpcshared/Vinh/data/CheXpert"),
        Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/data/CheXpert"),
    ],
    "NIH ChestX-ray14": [
        Path("/data2/cmdir/home/toandq/data/NIH"),
        Path("/data2/cmdir/home/toandq/data/chestxray14"),
        Path("/data2/hpcshared/Vinh/data/NIH"),
    ],
    "MIMIC-CXR": [
        Path("/data2/cmdir/home/toandq/data/MIMIC-CXR"),
        Path("/data2/cmdir/home/toandq/data/mimic-cxr"),
        Path("/data2/hpcshared/Vinh/data/MIMIC-CXR"),
    ],
}


def exists(p: Path):
    return p.exists()


def main():
    lines = [
        "# Phase 15.6 — chest X-ray availability (before compute)",
        "",
        "MIMIC-CXR needs PhysioNet credentialing. CheXpert needs Stanford registration.",
        "Do not start training until a readable pair exists.",
        "",
        "| Dataset | Found | Paths tried |",
        "|---|---|---|",
    ]
    any_pair = []
    for name, paths in CANDIDATES.items():
        hits = [str(p) for p in paths if exists(p)]
        found = "yes" if hits else "no"
        if hits:
            any_pair.append(name)
        lines.append("| {} | {} | {} |".format(name, found, "; ".join(hits) if hits else ", ".join(str(p) for p in paths)))
    lines.append("")
    if len(any_pair) >= 2:
        lines.append("At least two sources present ({}). 15.6 may be scheduled after 15.1–15.3.".format(", ".join(any_pair)))
    else:
        lines.append(
            "Not enough local data (found: {}). "
            "15.6 is future work until CheXpert↔NIH or MIMIC↔CheXpert is on disk and licensed.".format(
                ", ".join(any_pair) if any_pair else "none"
            )
        )
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("found", any_pair)


if __name__ == "__main__":
    main()
