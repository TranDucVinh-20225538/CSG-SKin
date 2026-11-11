#!/usr/bin/env python3
"""Select admissible Y2 screen config (decisions/paperB_r11b.md).

Admissible at λ=2, both seeds:
  - not collapsed: acc NOT (within 0.005 of 0.5 AND loss within 0.001 of ln 2)
  - competent: ID val balanced accuracy ≥ 0.528 (validation, not test)
Among admissible: lowest two-seed mean V1; ties within 0.01 → smaller latent, then A < B < C.
No admissible → verdict "adversary collapse not resolved"; write SELECTION with selected=null; do not submit main.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB-r11b/results/paperB/r11b/y2")
LN2 = float(math.log(2.0))
COMPETENCE = 0.528


def load_summaries(phase="screen"):
    d = OUT / phase / "runs"
    rows = []
    if not d.exists():
        return rows
    for p in sorted(d.glob("*/summary.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def collapsed(acc, loss):
    """Order definition: collapsed iff acc∈[0.5±0.005] AND loss∈[ln2±0.001]."""
    return abs(float(acc) - 0.5) <= 0.005 and abs(float(loss) - LN2) <= 0.001


def np_mean(xs):
    return sum(xs) / max(len(xs), 1)


def main():
    rows = load_summaries("screen")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["cfg"], int(r["latent"]))].append(r)

    table = []
    admissible = []
    for (cfg, lat), rs in sorted(by.items(), key=lambda x: (x[0][1], "ABC".index(x[0][0]))):
        lam2 = [r for r in rs if abs(r["lam"] - 2.0) < 1e-9 and abs(r["shift"] - 1.0) < 1e-9]
        lam0 = [r for r in rows if abs(r["lam"] - 0.0) < 1e-9 and int(r["latent"]) == lat]

        seeds_ok = True
        notes = []
        for r in sorted(lam2, key=lambda x: x["seed"]):
            if collapsed(r["final_acc_adv"], r["final_loss_adv"]):
                seeds_ok = False
                notes.append("{} collapsed (acc={:.4f}, loss={:.4f})".format(
                    r["tag"], r["final_acc_adv"], r["final_loss_adv"]))
            idv = float(r["id_val_balanced_acc"])  # validation, not test
            if idv < COMPETENCE:
                seeds_ok = False
                notes.append("{} id_val={:.3f}<{:.3f}".format(r["tag"], idv, COMPETENCE))
        if len(lam2) < 2:
            seeds_ok = False
            notes.append("missing λ=2 seeds (have {})".format(len(lam2)))

        v1_mean = float(np_mean([r["heldout_leakage_V1"] for r in lam2])) if lam2 else float("nan")
        id_val_mean = float(np_mean([r["id_val_balanced_acc"] for r in lam2])) if lam2 else float("nan")
        rec = {
            "cfg": cfg, "latent": lat,
            "n_lam2": len(lam2),
            "v1_mean_lam2": v1_mean,
            "id_val_mean_lam2": id_val_mean,
            "admissible": bool(seeds_ok and len(lam2) == 2),
            "notes": notes,
            "lam2": [{"tag": r["tag"], "seed": r["seed"],
                      "acc_adv": r["final_acc_adv"], "loss_adv": r["final_loss_adv"],
                      "collapsed": collapsed(r["final_acc_adv"], r["final_loss_adv"]),
                      "V1": r["heldout_leakage_V1"],
                      "id_val": r["id_val_balanced_acc"],
                      "id_test": r["id_balanced_acc"]} for r in sorted(lam2, key=lambda x: x["seed"])],
            "lam0": [{"tag": r["tag"], "id_val": r["id_val_balanced_acc"],
                      "id_test": r["id_balanced_acc"]} for r in sorted(lam0, key=lambda x: x["seed"])],
        }
        table.append(rec)
        if rec["admissible"]:
            admissible.append(rec)

    selected = None
    tie_note = None
    if admissible:
        admissible.sort(key=lambda r: (r["v1_mean_lam2"], r["latent"], "ABC".index(r["cfg"])))
        best_v1 = admissible[0]["v1_mean_lam2"]
        tied = [r for r in admissible if abs(r["v1_mean_lam2"] - best_v1) <= 0.01]
        tied.sort(key=lambda r: (r["latent"], "ABC".index(r["cfg"])))
        selected = tied[0]
        if len(tied) > 1:
            tie_note = ("tie within 0.01 of V1={:.4f}: candidates {}; chose latent={}, cfg={} "
                        "(smaller latent, then A<B<C)").format(
                best_v1,
                [("{}{}".format(r["cfg"], r["latent"])) for r in tied],
                selected["latent"], selected["cfg"])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if selected is None:
        verdict = "adversary collapse not resolved"
        reason = (
            "No admissible configuration at λ=2 (both seeds). "
            "Admissible requires: (i) not collapsed on either seed "
            "(collapse = acc within 0.005 of 0.5 AND loss within 0.001 of ln 2); "
            "(ii) ID val balanced accuracy ≥ 0.528 on both seeds. "
            "Stop Y2 main and Y3; do not submit further jobs."
        )
    else:
        verdict = "selected cfg={} latent={}".format(selected["cfg"], selected["latent"])
        reason = (
            "Admissible at λ=2 both seeds (not collapsed; ID val ≥ 0.528). "
            "Lowest two-seed mean V1={:.4f}.".format(selected["v1_mean_lam2"])
        )
        if tie_note:
            reason = reason + " " + tie_note

    sel_doc = {
        "locked_at_utc": now,
        "selected": None if selected is None else {
            "cfg": selected["cfg"], "latent": selected["latent"],
            "v1_mean_lam2": selected["v1_mean_lam2"],
            "id_val_mean_lam2": selected["id_val_mean_lam2"],
        },
        "verdict": verdict,
        "reason": reason,
        "rules": {
            "collapse": "acc within 0.005 of 0.5 AND loss within 0.001 of ln 2, both seeds at λ=2",
            "competence": "ID val balanced accuracy ≥ 0.528 (validation, not test), both seeds",
            "select": "lowest two-seed mean V1 at λ=2; ties within 0.01 → smaller latent, then A < B < C",
            "no_admissible": "stop Y2 and Y3; do not submit main",
        },
        "admissible": [{"cfg": r["cfg"], "latent": r["latent"],
                        "v1_mean_lam2": r["v1_mean_lam2"],
                        "id_val_mean_lam2": r["id_val_mean_lam2"]} for r in admissible],
        "design": "decisions/paperB_r11b.md",
    }

    doc = {
        "locked_at_utc": now,
        "table": table,
        "admissible": sel_doc["admissible"],
        "selected": sel_doc["selected"],
        "verdict": verdict,
        "reason": reason,
        "rules": sel_doc["rules"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_table.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "SELECTION.json").write_text(json.dumps(sel_doc, indent=1) + "\n")

    lines = [
        "# Y2 screen selection",
        "",
        "Locked at `{}` (UTC) — before any Y2 main submit.".format(now),
        "",
        "## Verdict",
        "",
        "**{}**".format(verdict),
        "",
        reason,
        "",
        "## Rules (order)",
        "",
        "- Collapse (both seeds at λ=2): acc within 0.005 of 0.5 **and** loss within 0.001 of ln 2.",
        "- Competence: **ID val** balanced accuracy ≥ 0.528 (not test).",
        "- Select: lowest two-seed mean V1; ties within 0.01 → smaller latent, then A < B < C.",
        "- No admissible → stop Y2 and Y3; do not submit main.",
        "",
        "## Screen table (λ=2, shift 100 %)",
        "",
        "| cfg | latent | admissible | V1 mean | id_val mean | per-seed | notes |",
        "|-----|-------:|:----------:|--------:|------------:|----------|-------|",
    ]
    for r in table:
        per = "; ".join(
            "s{} acc={:.3f} loss={:.4f} coll={} id_val={:.3f} V1={:.3f}".format(
                x["seed"], x["acc_adv"], x["loss_adv"], x["collapsed"], x["id_val"], x["V1"])
            for x in r["lam2"])
        lines.append(
            "| {cfg} | {lat} | {adm} | {v1:.4f} | {idv:.4f} | {per} | {notes} |".format(
                cfg=r["cfg"], lat=r["latent"], adm=r["admissible"],
                v1=r["v1_mean_lam2"], idv=r["id_val_mean_lam2"],
                per=per, notes="; ".join(r["notes"]) if r["notes"] else "—"))
    lines += ["", "## Selected", "",
              "```json", json.dumps(sel_doc["selected"], indent=2), "```", ""]
    (OUT / "SELECTION.md").write_text("\n".join(lines) + "\n")
    (OUT / "SCREEN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "selected": sel_doc["selected"]}, indent=2))


if __name__ == "__main__":
    main()
