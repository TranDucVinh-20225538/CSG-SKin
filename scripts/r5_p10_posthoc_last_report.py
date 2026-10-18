#!/usr/bin/env python3
"""Roll up summary.posthoc_last for pilot + shift_arms → POSTHOC_LAST.md (descriptive)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5/p10")


def rows():
    out = []
    for stage in ("pilot", "shift_arms"):
        for sp in sorted((ROOT / stage).glob("camelyon2_*/summary.json")):
            s = json.loads(sp.read_text())
            ph = s.get("posthoc_last")
            out.append({
                "stage": stage,
                "tag": sp.parent.name,
                "rung": s.get("rung"),
                "seed": s.get("seed"),
                "best_epoch": s.get("best_epoch"),
                "locked_drop": s.get("probe_drop_vs_lambda0"),
                "locked_id_bal": (s.get("id_test") or {}).get("balanced_acc"),
                "last": ph,
            })
    return out


def main() -> int:
    rs = rows()
    missing = [r for r in rs if not r["last"]]
    lines = [
        "# P10 / pilot — posthoc last-epoch site probe & ID bal",
        "",
        "Descriptive only. Locked gate still uses ID-select `best.pt`. Derm R1/R2 use `last.ckpt`.",
        "",
        "| stage | tag | best_ep | locked drop | locked ID bal | last drop | last ID bal | last probe |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rs:
        ph = r["last"] or {}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r["stage"], r["tag"], r["best_epoch"],
                "{:.4f}".format(r["locked_drop"]) if r["locked_drop"] is not None else "—",
                "{:.4f}".format(r["locked_id_bal"]) if r["locked_id_bal"] is not None else "—",
                "{:.4f}".format(ph["probe_drop_vs_lambda0"]) if ph else "**pending**",
                "{:.4f}".format(ph["id_bal"]) if ph else "**pending**",
                "{:.4f}".format(ph["site_probe"]) if ph else "**pending**",
            )
        )
    lines += [
        "",
        "Pending: {} / {} runs without `posthoc_last`.".format(len(missing), len(rs)),
        "",
        "Reading guide (not locked): if last-epoch probe drop is large while ID bal collapses, "
        "the Camelyon story is that DANN removes site only by breaking the classifier — "
        "unlike derm R1/R2 last.ckpt readouts.",
        "",
    ]
    out = ROOT / "POSTHOC_LAST.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out, "missing", len(missing), flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
