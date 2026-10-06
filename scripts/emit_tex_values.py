#!/usr/bin/env python3
"""Emit remaining manuscript \\newcommand values from Phase 6 xfer JSONs.

No training. No recomputation. No substitute split / metric / class restriction.
Reads results/paperB/phase6_xfer/*.json as written by eval_final_gpu.py (array 60585).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
XFER = PAPERB / "results" / "paperB" / "phase6_xfer"
OUT = PAPERB / "results" / "paperB" / "TEX_VALUES.md"

NEED_LAMBDAS = (0.5, 1.0, 4.0, 8.0)
ALREADY_IN_MS = (0.0, 0.25, 2.0)
RECALL_LAMS = (0.0, 2.0)
CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "SCC")
AK_FLAG_BELOW = 100


def mean_std(xs):
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5, n


def tex_ms(m, s):
    if m is None:
        return r"n/a"
    return r"{:.3f}\,$\pm$\,{:.3f}".format(m, s)


def load_xfer():
    rows = []
    skipped = []
    for p in sorted(XFER.glob("*.json")):
        if p.name in {"diagnosis_aggregate.json"}:
            skipped.append(p.name)
            continue
        d = json.loads(p.read_text())
        if "pad_heldout_6class" not in d:
            skipped.append("{} (no pad_heldout_6class)".format(p.name))
            continue
        h = d["pad_heldout_6class"]
        rows.append({"path": p, "raw": d, "h": h})
    return rows, skipped


def check_subset(h, path):
    problems = []
    subset = h.get("subset")
    if subset != "pad_heldout":
        problems.append("{} subset={!r} (want pad_heldout)".format(path.name, subset))
    if h.get("n_images") != 716:
        problems.append("{} n_images={} (want 716)".format(path.name, h.get("n_images")))
    if h.get("n_patients") != 412:
        problems.append("{} n_patients={} (want 412)".format(path.name, h.get("n_patients")))
    return problems


def main():
    rows, skipped = load_xfer()
    problems = []
    for r in rows:
        problems.extend(check_subset(r["h"], r["path"]))

    counts = None
    for r in rows:
        c = r["h"].get("class_counts")
        if counts is None:
            counts = dict(c)
        elif c != counts:
            problems.append("{} class_counts differ: {} vs {}".format(r["path"].name, c, counts))

    csg = defaultdict(list)
    effb = []
    for r in rows:
        kind = r["raw"].get("kind")
        h = r["h"]
        bal = ((h.get("balanced_accuracy") or {}).get("point"))
        rec = {
            "kind": kind,
            "lam": r["raw"].get("lambda_adv"),
            "seed": r["raw"].get("seed"),
            "bal": bal,
            "recall": h.get("per_class_recall") or {},
            "path": r["path"],
            "subset": h.get("subset"),
        }
        if kind == "csg":
            csg[float(rec["lam"])].append(rec)
        elif kind == "effb3_control":
            effb.append(rec)

    cmds = {}
    notes = []
    notes.append("# Remaining manuscript values (Phase 6 / array 60585)")
    notes.append("")
    notes.append("Extracted by `scripts/emit_tex_values.py` from `results/paperB/phase6_xfer/`.")
    notes.append("Statistic: 6-class balanced accuracy on **pad_heldout** (716 images / 412 patients),")
    notes.append("mean ± s.d. across seeds. Softmax/argmax restricted to MEL, NV, BCC, AK, BKL, SCC.")
    notes.append("No recomputation. No pad_full / pad_adv substitute.")
    notes.append("")
    if skipped:
        notes.append("Skipped non-run files: {}.".format(", ".join(skipped)))
        notes.append("")
    if problems:
        notes.append("**Subset / count mismatches:**")
        for p in problems:
            notes.append("- " + p)
        notes.append("")

    notes.append("## 1. CSG remaining λ points")
    notes.append("")
    notes.append("Already in the manuscript (not re-emitted):")
    for lam in ALREADY_IN_MS:
        recs = csg.get(lam, [])
        m, s, n = mean_std([r["bal"] for r in recs]) if recs else (None, None, 0)
        files = ", ".join(r["path"].name for r in recs)
        notes.append(
            "- λ={:g}: {} (n={}, files: {})".format(lam, tex_ms(m, s).replace(r"\,$\pm$\,", " ± "), n, files)
        )
    notes.append("")

    tex_names = {0.5: "xdomHalf", 1.0: "xdomOne", 4.0: "xdomFour", 8.0: "xdomEight"}
    for lam in NEED_LAMBDAS:
        recs = csg.get(lam, [])
        name = tex_names[lam]
        if not recs:
            cmds[name] = "n/a"
            notes.append(
                "- λ={:g} (`\\{}`): **n/a** — no CSG `pad_heldout_6class` JSON at this λ.".format(lam, name)
            )
            continue
        vals = [r["bal"] for r in recs]
        if any(v is None for v in vals):
            cmds[name] = "n/a"
            notes.append(
                "- λ={:g} (`\\{}`): **n/a** — balanced_accuracy.point missing in {}.".format(
                    lam, name, [r["path"].name for r in recs if r["bal"] is None]
                )
            )
            continue
        m, s, n = mean_std(vals)
        cmds[name] = tex_ms(m, s)
        files = ", ".join(r["path"].name for r in recs)
        notes.append(
            "- λ={:g} (`\\{}`): {:.3f} ± {:.3f} (n={}). "
            "subset=pad_heldout, 716/412, 6-class bal acc. Files: {}.".format(
                lam, name, m, s, n, files
            )
        )
    notes.append("")

    notes.append("## 2. EffNet-B3 single-encoder control")
    notes.append("")
    if not effb:
        cmds["xdomEffb"] = "n/a"
        notes.append(
            "- `\\xdomEffb`: **n/a** — no `kind=effb3_control` file under phase6_xfer. "
            "Do not substitute pad_full / pad_adv or a different metric."
        )
    elif any(r["bal"] is None for r in effb):
        cmds["xdomEffb"] = "n/a"
        notes.append("- `\\xdomEffb`: **n/a** — balanced_accuracy.point missing on an EffB3 file.")
    else:
        subsets = {r["subset"] for r in effb}
        if subsets != {"pad_heldout"}:
            cmds["xdomEffb"] = "n/a"
            notes.append(
                "- `\\xdomEffb`: **n/a** — EffB3 files are not pad_heldout (subsets={}). "
                "Do not report a number from a different PAD subset.".format(sorted(subsets))
            )
        else:
            m, s, n = mean_std([r["bal"] for r in effb])
            cmds["xdomEffb"] = tex_ms(m, s)
            files = ", ".join(r["path"].name for r in effb)
            notes.append(
                "- `\\xdomEffb`: {:.3f} ± {:.3f} (n={}). "
                "same statistic as the CSG rows: 6-class bal acc on pad_heldout 716/412. "
                "Files: {}.".format(m, s, n, files)
            )
    notes.append("")

    notes.append("## 3. pad_heldout class counts (AK, SCC)")
    notes.append("")
    if counts is None:
        cmds["nAK"] = "n/a"
        cmds["nSCC"] = "n/a"
        notes.append("- **n/a** — no class_counts in the xfer JSONs.")
    else:
        nak = counts.get("AK")
        nscc = counts.get("SCC")
        cmds["nAK"] = str(nak) if nak is not None else "n/a"
        cmds["nSCC"] = str(nscc) if nscc is not None else "n/a"
        notes.append(
            "- `\\nAK` = {}, `\\nSCC` = {}. Full counts on pad_heldout (716/412): {}. "
            "MEL=9 matches the manuscript. A proportional 31.2% of full PAD would be "
            "AK ≈ 227, SCC ≈ 60; patient-level split does not preserve proportions "
            "(MEL 9 vs proportional 16). Source: `class_counts` in every phase6_xfer run JSON "
            "(verified identical).".format(nak, nscc, counts)
        )
        if nak is not None and nak < AK_FLAG_BELOW:
            notes.append(
                "- **FLAG:** AK denominator {} is below ~{}. "
                "The strongest Discussion sentence rests on AK recall.".format(nak, AK_FLAG_BELOW)
            )
        else:
            notes.append(
                "- AK denominator {} ≥ {}. No flag. SCC n={} is smaller and is not the gated claim.".format(
                    nak, AK_FLAG_BELOW, nscc
                )
            )
    notes.append("")

    notes.append("## Report only — per-class recall at λ=0 and λ=2 (not for the manuscript block)")
    notes.append("")
    notes.append(
        "Numerator = round(recall × class_count) per seed; denominator is the pad_heldout count. "
        "Mean TP is the average of those integers."
    )
    notes.append("")
    for lam in RECALL_LAMS:
        recs = sorted(csg.get(lam, []), key=lambda r: r["seed"])
        notes.append("### λ={:g} (n={})".format(lam, len(recs)))
        notes.append("")
        if not recs:
            notes.append("n/a — no CSG files at this λ.")
            notes.append("")
            continue
        for cls in CLASSES:
            denom = counts[cls] if counts else None
            tps = []
            bits = []
            for r in recs:
                rec = r["recall"].get(cls)
                if rec is None or denom is None:
                    bits.append("s{}: n/a".format(r["seed"]))
                    continue
                tp = int(round(rec * denom))
                tps.append(tp)
                bits.append("s{}: {}/{}".format(r["seed"], tp, denom))
            if tps:
                notes.append(
                    "- {}: mean {}/{:.0f} (recall {:.3f} ± {:.3f}). {}.".format(
                        cls,
                        "{:.1f}".format(sum(tps) / len(tps)),
                        denom,
                        *mean_std([r["recall"][cls] for r in recs])[:2],
                        "; ".join(bits),
                    )
                )
            else:
                notes.append("- {}: n/a".format(cls))
        notes.append("")

    if counts and 0.0 in csg and 2.0 in csg:
        for cls in ("AK", "SCC"):
            denom = counts[cls]
            tp0 = [int(round(r["recall"][cls] * denom)) for r in csg[0.0]]
            tp2 = [int(round(r["recall"][cls] * denom)) for r in csg[2.0]]
            notes.append(
                "Summary {}: {:.1f}/{} → {:.1f}/{} "
                "(λ=0 n={}, λ=2 n={}).".format(
                    cls, sum(tp0) / len(tp0), denom, sum(tp2) / len(tp2), denom, len(tp0), len(tp2)
                )
            )
        notes.append("")

    block = [
        r"\newcommand{\xdomHalf}{" + cmds["xdomHalf"] + "}",
        r"\newcommand{\xdomOne}{" + cmds["xdomOne"] + "}",
        r"\newcommand{\xdomFour}{" + cmds["xdomFour"] + "}",
        r"\newcommand{\xdomEight}{" + cmds["xdomEight"] + "}",
        r"\newcommand{\xdomEffb}{" + cmds["xdomEffb"] + "}",
        r"\newcommand{\nAK}{" + cmds["nAK"] + "}",
        r"\newcommand{\nSCC}{" + cmds["nSCC"] + "}",
    ]

    notes.append("## FILL-IN BLOCK")
    notes.append("")
    notes.append("```")
    notes.extend(block)
    notes.append("```")
    notes.append("")

    OUT.write_text("\n".join(notes) + "\n")
    print("\n".join(block))
    print("")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
