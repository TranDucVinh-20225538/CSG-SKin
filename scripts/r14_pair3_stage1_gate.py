#!/usr/bin/env python3
"""Score stage-1 summaries → STAGE1.md and apply early-abort rules."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r14_pair3_common as C  # noqa: E402

NEEDED = [
    C.tag_for(0.0, "prior100", 42),
    C.tag_for(2.0, "prior100", 42),
    C.tag_for(2.0, "prior100", 52),
]
OPTIONAL = [C.tag_for(2.0, "prior0", 42)]


def load_summary(tag):
    p = C.OUT / "runs" / tag / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def maha(s, panel):
    try:
        return float(s["metrics"]["ood"][panel]["mahalanobis_classcond"])
    except Exception:
        return None


def main():
    rows = {}
    for tag in NEEDED + OPTIONAL:
        rows[tag] = load_summary(tag)

    missing = [t for t in NEEDED if rows[t] is None]
    lines = [
        "# STAGE1 — R14 early-abort",
        "",
        "Generated: {}".format(datetime.now(timezone.utc).isoformat()),
        "Commit: `{}`".format(C.git_head()),
        "",
    ]
    if missing:
        lines += ["Status: **WAITING** — missing summaries: {}".format(missing), ""]
        (C.OUT / "STAGE1.md").write_text("\n".join(lines) + "\n")
        C.write_status("# R14 STATUS\n\nphase: stage1_waiting\nmissing: {}\n".format(missing))
        print("WAITING", missing)
        return 2

    # table
    lines += [
        "| tag | degenerate | ID bal | V1 | Maha Fitz | Maha MED | Maha SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for tag in NEEDED + OPTIONAL:
        s = rows[tag]
        if s is None:
            lines.append("| {} | — | — | — | — | — | — |".format(tag))
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                tag,
                s.get("degenerate"),
                "{:.3f}".format(s["id_balanced_acc"]) if "id_balanced_acc" in s else "—",
                "{:.3f}".format(s["heldout_leakage_V1"]) if "heldout_leakage_V1" in s else "—",
                "{:.3f}".format(maha(s, "fitzpatrick17k")) if maha(s, "fitzpatrick17k") is not None else "—",
                "{:.3f}".format(maha(s, "mednode")) if maha(s, "mednode") is not None else "—",
                "{:.3f}".format(maha(s, "sd198")) if maha(s, "sd198") is not None else "—",
            ))

    s0 = rows[NEEDED[0]]
    s42 = rows[NEEDED[1]]
    s52 = rows[NEEDED[2]]

    decision = "CONTINUE"
    reasons = []

    # STOP-degenerate
    for tag in NEEDED:
        s = rows[tag]
        if s.get("degenerate"):
            decision = "STOP-degenerate"
            reasons.append("{} degenerate: {}".format(tag, s.get("degenerate_reason")))
        elif s.get("id_balanced_acc", 1) < 0.25:
            decision = "STOP-degenerate"
            reasons.append("{} ID bal {:.3f} < 0.25".format(tag, s["id_balanced_acc"]))

    # STOP-gap: V1 > 0.70 at λ=2 prior100 on both s42 and s52
    if decision == "CONTINUE":
        v42 = s42.get("heldout_leakage_V1")
        v52 = s52.get("heldout_leakage_V1")
        if v42 is not None and v52 is not None and v42 > 0.70 and v52 > 0.70:
            decision = "STOP-gap"
            reasons.append("V1 λ2 prior100 s42={:.3f} s52={:.3f} both > 0.70".format(v42, v52))

    # STOP-headroom: λ=0 Maha < 0.65 on all three labelled unseen
    if decision == "CONTINUE":
        vals = [maha(s0, p) for p in ("fitzpatrick17k", "mednode", "sd198")]
        if all(v is not None and v < 0.65 for v in vals):
            decision = "STOP-headroom"
            reasons.append("λ0 Maha all < 0.65: Fitz={:.3f} MED={:.3f} SD={:.3f}".format(*vals))

    lines += ["", "## Decision", "", "**{}**".format(decision), ""]
    for r in reasons:
        lines.append("- {}".format(r))
    if decision == "CONTINUE":
        lines.append("- Submit remaining 16–17 runs (array %2). Do not retune from AUROC.")
    lines.append("")

    (C.OUT / "STAGE1.md").write_text("\n".join(lines) + "\n")
    (C.OUT / "STAGE1.json").write_text(json.dumps({
        "decision": decision, "reasons": reasons,
        "tags": {t: rows[t] for t in NEEDED + OPTIONAL},
        "commit": C.git_head(),
    }, indent=1, default=float) + "\n")
    C.write_status("# R14 STATUS\n\nphase: stage1_done\ndecision: {}\nreasons: {}\n".format(
        decision, reasons))
    print(decision, reasons)
    return 0 if decision == "CONTINUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
