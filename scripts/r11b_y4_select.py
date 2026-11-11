#!/usr/bin/env python3
"""Y4 screen selection: smaller λ ∈ {4,8} with two-seed mean V1 ≤ 0.55.

Write SELECTION.json before any main submit. Do not submit from this script.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/y4")


def main():
    rows = []
    d = OUT / "screen" / "runs"
    for p in sorted(d.glob("*/summary.json")) if d.exists() else []:
        rows.append(json.loads(p.read_text()))
    by_lam = {}
    for r in rows:
        if abs(r.get("shift", 1) - 1.0) > 1e-9:
            continue
        by_lam.setdefault(float(r["lam"]), []).append(r)

    table = []
    selected = None
    for lam in (4.0, 8.0):
        rs = by_lam.get(lam, [])
        v1s = [float(r["heldout_leakage_V1"]) for r in rs]
        mean = sum(v1s) / len(v1s) if v1s else float("nan")
        ok = len(rs) == 2 and mean <= 0.55
        table.append({"lam": lam, "n": len(rs), "v1_mean": mean, "ok": ok,
                      "runs": [{"tag": r["tag"], "V1": r["heldout_leakage_V1"],
                                "id_bal": r["id_balanced_acc"]} for r in rs]})
        if ok and selected is None:
            selected = lam

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    verdict = ("selected λ={}".format(selected) if selected is not None
               else "neither λ meets two-seed mean V1 ≤ 0.55; stop")
    reason = ("Smaller λ with two-seed mean V1 ≤ 0.55." if selected is not None
              else "Neither λ=4 nor λ=8 has two-seed mean V1 ≤ 0.55. Stop Y4 main.")
    doc = {
        "locked_at_utc": now,
        "table": table,
        "selected_lam": selected,
        "selected": None if selected is None else {"lam": selected},
        "verdict": verdict,
        "reason": reason,
        "rule": "smaller λ with two-seed mean V1 ≤ 0.55; commit SELECTION before main",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_table.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "SELECTION.json").write_text(json.dumps({
        "locked_at_utc": now,
        "lam": selected,
        "selected": doc["selected"],
        "verdict": verdict,
        "reason": reason,
        "rule": doc["rule"],
    }, indent=1) + "\n")
    lines = ["# Y4 screen selection", "",
             "Locked at `{}` (UTC) — before any Y4 main submit.".format(now), "",
             "## Verdict", "", "**{}**".format(verdict), "", reason, "",
             "| λ | n | V1 mean | ok |", "|--:|--:|--------:|:--:|"]
    for r in table:
        lines.append("| {lam:g} | {n} | {v1:.4f} | {ok} |".format(
            lam=r["lam"], n=r["n"], v1=r["v1_mean"], ok=r["ok"]))
    (OUT / "SELECTION.md").write_text("\n".join(lines) + "\n")
    (OUT / "SCREEN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "selected_lam": selected}))


if __name__ == "__main__":
    main()
