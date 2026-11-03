#!/usr/bin/env python3
"""R10 C0: audit paper reports/stores for cells that still include degraded runs.

Writes results/paperB/r10/DEGRADED_AUDIT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

OUT = R.RES / "r10"
DEGRADED_ID_BAL = R.DEGRADED_ID_BAL
# Known degraded runs (tag substring or exact)
KNOWN = {
    "r1_cur_shift0_ladv2_s52": {"id_bal": 0.139, "cell": "R1 shift0% λ=2"},
}


def scan_summaries():
    hits = []
    roots = [
        R.RES / "r5", R.RES / "r6", R.RES / "r7", R.RES / "r7b", R.RES / "r8", R.RES / "r9",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for sp in root.rglob("summary.json"):
            if "/lightning_logs/" in str(sp):
                continue
            try:
                s = json.loads(sp.read_text())
            except Exception:
                continue
            m = s.get("metrics") or {}
            idb = None
            if isinstance(m.get("id"), dict) and "bal_acc" in m["id"]:
                idb = float(m["id"]["bal_acc"])
            elif "id_balanced_acc" in s:
                idb = float(s["id_balanced_acc"])
            elif "id_bal_acc" in s:
                try:
                    idb = float(s["id_bal_acc"])
                except Exception:
                    idb = None
            if idb is None:
                continue
            deg = m.get("degraded")
            if deg is None:
                deg = bool(idb < DEGRADED_ID_BAL)
            if not deg and idb >= DEGRADED_ID_BAL:
                continue
            if not deg and idb < DEGRADED_ID_BAL:
                deg = True
            if not deg:
                continue
            rel = str(sp.relative_to(R.RES))
            hits.append({"file": rel, "tag": s.get("tag") or sp.parent.name,
                         "seed": s.get("seed"), "id_bal": idb, "degraded_flag": m.get("degraded"),
                         "lambda": s.get("lambda_adv", s.get("lam"))})
    return hits


def class_aware_status():
    p = R.RES / "r6" / "class_aware_heldout" / "DEGRADED_EXCLUDED.json"
    cells = R.RES / "r6" / "class_aware_heldout" / "cells.json"
    rep = R.RES / "r6" / "class_aware_heldout" / "REPORT.md"
    out = {"degraded_excluded_file": p.exists(), "cells_file": cells.exists(), "report": rep.exists()}
    if p.exists():
        out["excluded"] = json.loads(p.read_text())
    if cells.exists():
        cj = json.loads(cells.read_text())
        r1 = next((c for c in cj if c.get("cell") == "r1_shift0_l2"), None)
        out["r1_shift0_l2"] = r1
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hits = scan_summaries()
    ca = class_aware_status()
    # R9 E1 already force-excludes s52 in shift0 reference
    e1 = R.RES / "r9" / "e1" / "OUTCOME.json"
    e1_note = None
    if e1.exists():
        o = json.loads(e1.read_text())
        e1_note = {"force_degraded": o.get("force_degraded"), "excluded_degraded": o.get("excluded_degraded")}

    lines = [
        "# Paper B R10 C0 — degraded-run audit",
        "",
        "Rule: ID test balanced accuracy < {:.2f}: report the run, exclude from every mean / CI / probe / regression.".format(
            DEGRADED_ID_BAL),
        "",
        "## Class-aware held-out probes",
        "",
    ]
    if ca.get("excluded"):
        ex = ca["excluded"]
        lines.append("Excluded n={}: {}".format(ex.get("n_excluded"),
            ", ".join("{} s{}".format(r["cell"], r["seed"]) for r in ex.get("rows", []))))
        r1 = ca.get("r1_shift0_l2") or {}
        if r1:
            lines.append(
                "R1 shift0% λ=2 cell means after exclusion (n={}): "
                "(a) {:.3f}, (b) {:.3f}, (c) {:.3f} "
                "(work-order target without s52: 0.630 / 0.522 / 0.555).".format(
                    r1.get("n"), r1["a"]["mean"], r1["b"]["mean"], r1["c"]["mean"]))
            lines.append("Excluded seeds in that cell: {}.".format(r1.get("excluded_seeds")))
            status = "fixed"
        else:
            status = "cells.json missing r1_shift0_l2"
        lines.append("")
        lines.append("| file | cell | run | status |")
        lines.append("|---|---|---|---|")
        lines.append("| `r6/class_aware_heldout/` | r1_shift0_l2 | r1_cur_shift0_ladv2_s52 | {} |".format(status))
    else:
        lines.append("DEGRADED_EXCLUDED.json not found — regenerate class-aware report.")
        status = "not fixed"

    lines += ["", "## R9 E1 shift references", ""]
    if e1_note:
        lines.append("OUTCOME force_degraded / excluded: `{}`.".format(json.dumps(e1_note)))
        lines.append("| file | cell | run | status |")
        lines.append("|---|---|---|---|")
        lines.append("| `r9/e1/OUTCOME.json` | shift 0% λ=2 ref | seed 52 | fixed (forced exclude) |")
    else:
        lines.append("R9 E1 OUTCOME missing.")

    lines += ["", "## All degraded summaries under results/paperB/r{{5–9}}", "",
              "| file | tag | seed | ID bal | degraded flag in summary |",
              "|---|---|---:|---:|---|"]
    for h in sorted(hits, key=lambda x: (x["file"], str(x.get("seed")))):
        lines.append("| `{}` | {} | {} | {:.3f} | {} |".format(
            h["file"], h["tag"], h.get("seed"), h["id_bal"], h.get("degraded_flag")))
    if not hits:
        lines.append("| — | — | — | — | none found |")

    lines += ["", "## Notes", "",
              "- R1 report (`r6_report.py --item r1`) already excludes degraded from primary Rule A.",
              "- R10 regenerates class-aware probes with exclusion; audit lists remaining degraded summaries on disk.",
              "- Paper number stores are rebuilt only when their source reports are regenerated (G1 follow-ups).",
              ""]
    (OUT / "DEGRADED_AUDIT.md").write_text("\n".join(lines) + "\n")
    (OUT / "DEGRADED_AUDIT.json").write_text(json.dumps({"hits": hits, "class_aware": ca, "e1": e1_note}, indent=1) + "\n")
    print("wrote", OUT / "DEGRADED_AUDIT.md", "n_hits", len(hits))


if __name__ == "__main__":
    main()
