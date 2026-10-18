#!/usr/bin/env python3
"""Roll up the 2-run trajectory smoke → TRAJ_SMOKE.md (descriptive)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5/p10")
TAGS = (
    ("shift_arms", "camelyon2_natural_l2_ladv2_s42"),
    ("pilot", "camelyon2_P_L4_ladv10_s42"),
)


def main() -> int:
    lines = [
        "# P10 trajectory smoke (natural s42 + pilot L4 s42)",
        "",
        "Descriptive. Locked gate still uses ID-select `best.pt`. Fast probe n=2000 per epoch.",
        "Question: is there any epoch with site drop ≥ 0.10 **and** ID bal within 0.05 of λ=0?",
        "",
    ]
    any_pilot = False
    any_p10 = False
    missing = []
    for stage, tag in TAGS:
        sp = ROOT / stage / tag / "summary.json"
        if not sp.exists():
            missing.append(tag)
            continue
        s = json.loads(sp.read_text())
        et = s.get("epoch_traj") or {}
        ph = s.get("posthoc_last") or {}
        any_pilot = any_pilot or bool(et.get("any_fast_would_pass_pilot"))
        any_p10 = any_p10 or bool(et.get("any_fast_would_pass_p10_gate"))
        lines += [
            "## `{}`".format(tag),
            "",
            "- best_epoch (locked): **{}**".format(s.get("best_epoch")),
            "- locked drop / ID bal: {:.4f} / {:.4f}".format(
                float(s.get("probe_drop_vs_lambda0") or 0),
                float((s.get("id_test") or {}).get("balanced_acc") or 0)),
            "- posthoc last drop / ID bal: {} / {}".format(
                "{:.4f}".format(ph["probe_drop_vs_lambda0"]) if ph else "—",
                "{:.4f}".format(ph["id_bal"]) if ph else "—"),
            "- any epoch fast-pass P10 gate: **{}** → {}".format(
                et.get("any_fast_would_pass_p10_gate"), et.get("epochs_fast_pass_p10_gate")),
            "- any epoch fast-pass pilot: **{}** → {}".format(
                et.get("any_fast_would_pass_pilot"), et.get("epochs_fast_pass_pilot")),
            "",
            "| ep | warm | coef | id_select | probe_fast | drop_fast | id_bal_fast | pass_p10 | pass_pilot |",
            "|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for r in s.get("history") or []:
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {} | {} | {} | {} | {} |".format(
                    r["epoch"], r.get("warmup"), float(r.get("coef_end") or 0),
                    float(r.get("id_select_acc") or 0),
                    "{:.3f}".format(r["site_probe_fast"]) if "site_probe_fast" in r else "—",
                    "{:.3f}".format(r["probe_drop_vs_lambda0_fast"]) if "probe_drop_vs_lambda0_fast" in r else "—",
                    "{:.3f}".format(r["id_bal_fast"]) if "id_bal_fast" in r else "—",
                    r.get("fast_would_pass_p10_gate"), r.get("fast_would_pass_pilot"),
                )
            )
        lines.append("")
    lines += [
        "## Decision fork",
        "",
        "- If **any** epoch fast-passes pilot/P10 gate → write a **new** decisions file before changing "
        "checkpoint selection; then consider full 15-run trajectory.",
        "- If **none** → keep locked ID-select rule; do not expand to 15 runs for checkpoint-rule change.",
        "",
        "Smoke any_fast_pass_pilot={} any_fast_pass_p10_gate={} missing={}".format(
            any_pilot, any_p10, missing),
        "",
    ]
    out = ROOT / "TRAJ_SMOKE.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out, flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
