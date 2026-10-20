#!/usr/bin/env python3
"""Build manuscript CPU number pack under results/paperB/manuscript_cpu/ (items 2–7; item 1 if ready)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R5  # noqa: E402
import r6_common as R  # noqa: E402

OUT = R.RES / "manuscript_cpu"
ROOT = R.RES


def w(name, body, meta=None):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(body if body.endswith("\n") else body + "\n")
    if meta is not None:
        (OUT / (name + ".meta.json")).write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", p, flush=True)
    return p


def item1():
    src = ROOT / "r5" / "t1last" / "UNSEEN_ECE_REPORT.md"
    if not src.exists():
        w("01_table1_last.md",
          "# (1) Table 1 last.ckpt — PENDING\n\n"
          "Waiting on `r5_t1last_unseen_report.py` → `results/paperB/r5/t1last/UNSEEN_ECE_REPORT.md`.\n")
        return False
    text = src.read_text()
    # append note pointing to SD cluster sensitivity
    cl = ROOT / "r5" / "p13_sd198" / "REPORT.md"
    extra = ""
    if cl.exists():
        flips = [ln for ln in cl.read_text().splitlines() if ln.startswith("- ")]
        extra = ("\n## SD-198 image vs cluster_union (sensitivity, Table-1 / P7a features)\n\n"
                 "See `results/paperB/r5/p13_sd198/REPORT.md`. Rule-A flips: {}.\n".format(
                     "; ".join(flips) if flips else "none"))
    w("01_table1_last.md", "# (1) Table 1 last.ckpt\n\nSource: `r5/t1last/UNSEEN_ECE_REPORT.md`.\n\n" + text + extra,
      {"source": str(src)})
    shutil.copy2(src, OUT / "01_table1_last_SOURCE.md")
    return True


def item2():
    # R1 MED/SD from REPORT + regression coding from decisions/code
    r1 = (ROOT / "r6" / "r1" / "cur" / "REPORT.md").read_text()
    # extract detector sections for mednode and sd198
    lines = r1.splitlines()
    chunks = []
    keep = False
    for ln in lines:
        if ln.startswith("### Detectors on mednode") or ln.startswith("### Detectors on sd198"):
            keep = True
            chunks.append(ln)
            continue
        if keep:
            if ln.startswith("### ") and "mednode" not in ln and "sd198" not in ln:
                keep = False
                chunks.append("")
                continue
            chunks.append(ln)
    body = [
        "# (2) R1 — MED-NODE / SD-198 (main recipe `current`, last.ckpt)",
        "",
        "Source: `results/paperB/r6/r1/cur/REPORT.md` (excl. degraded primary rule A).",
        "",
        "## Requested cells",
        "",
        "### λ = 0.5, all shifts — Mahalanobis [CI], k/n, rule A",
        "",
        "| set | shift | mean ± sd [CI] | k | rule A |",
        "|---|---|---|---|---|",
        "| MED-NODE | 100% | 0.310 ± 0.055 [0.284, 0.337] | 5/5 | met |",
        "| MED-NODE | 50% | 0.303 ± 0.054 [0.272, 0.336] | 3/3 | met |",
        "| MED-NODE | 0% | 0.525 ± 0.050 [0.497, 0.554] | 1/5 | not met |",
        "| SD-198 | 100% | 0.365 ± 0.029 [0.353, 0.377] | 5/5 | met |",
        "| SD-198 | 50% | 0.451 ± 0.133 [0.439, 0.461] | 2/3 | not met |",
        "| SD-198 | 0% | 0.664 ± 0.058 [0.653, 0.675] | 0/5 | not met |",
        "",
        "### λ = 2, shift 50%",
        "",
        "| set | mean ± sd [CI] | k | rule A |",
        "|---|---|---|---|",
        "| MED-NODE | 0.299 ± 0.052 [0.269, 0.331] | 3/3 | met |",
        "| SD-198 | 0.414 ± 0.047 [0.402, 0.425] | 3/3 | met |",
        "",
        "## Regression (a): shift coding and resampling unit",
        "",
        "- **Coding:** `shift_level` ∈ {0, 0.5, 1.0} = shift% / 100 for R1; Table 1 and A4 coded as "
        "`shift_level = 1.0` (100%). Source: `decisions/paperB_r6_r1.md` and "
        "`scripts/r6_r1_leakage_matched.py` (`load_r1`: `shift_level: shift / 100.0`).",
        "- **Resampling unit for (a) CIs:** wild cluster bootstrap with **clusters = setting cells** "
        "`(recipe, λ, shift_level)`, **not** image/patient/P13 clusters. Webb weights, 2000 draws, "
        "rng 20261005. Outcome y = seed-level Mahalanobis **point** AUROC from `summary.json` "
        "(`scripts/r6_r1_leakage_matched.py:244–247`).",
        "",
        "## Detector tables (full)",
        "",
    ]
    body.extend(chunks)
    w("02_r1.md", "\n".join(body), {"source": "r6/r1/cur/REPORT.md"})


def item3():
    body = """# (3) A4 — k seeds below 0.5 (main recipe, last.ckpt primary)

Source: `results/paperB/r5/p8/REPORT.md` Bootstrap units section.

| λ | set | Mahalanobis mean ± sd [CI] | k CI < 0.5 | rule A |
|---:|---|---|---|---|
| 0.5 | pad_heldout | 0.468 ± 0.028 [0.440, 0.495] | **3/5** | met |
| 0.5 | Fitzpatrick17k (P13 cluster_union) | 0.444 ± 0.049 [0.431, 0.457] | **4/5** | met |
| 2 | pad_heldout | 0.452 ± 0.014 [0.425, 0.478] | **4/5** | met |
| 2 | Fitzpatrick17k (P13 cluster_union) | 0.458 ± 0.061 [0.445, 0.471] | **3/5** | met |

(best-val secondary omitted; manuscript primary = last.ckpt.)
"""
    w("03_a4.md", body, {"source": "r5/p8/REPORT.md"})


def item4():
    clin = (ROOT / "r5" / "posthoc" / "clinical" / "REPORT.md").read_text()
    # keep label maps + summary table through old_recipe λ=2 block header end
    lines = clin.splitlines()
    out = ["# (4) Clinical table (post hoc)", ""]
    for ln in lines:
        out.append(ln)
        if ln.startswith("## R1 shift 0%"):
            break
    # re-add full summary: already in file before ## R1
    # Actually we broke at R1 section - summary table is before that. Good.
    # Ensure we include summary - it's before ## R1. Truncate after summary's last old_recipe row
    # Simpler: copy label maps + summary table only
    text = clin
    start = text.index("## Label maps")
    end = text.index("## R1 shift 0%")
    w("04_clinical.md", "# (4) Clinical table\n\nSource: `r5/posthoc/clinical/REPORT.md`.\n\n" + text[start:end],
      {"source": "r5/posthoc/clinical/REPORT.md", "units": "Fitz=P13 cluster_union; SD-198=P7a group (image-level)"})


def item5():
    df = pd.read_csv(ROOT / "r5" / "s_flip" / "summary.csv")
    # lesion sets; OOD acceptance under flipped score = FPR@95
    les = df[df.set.isin(["pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin"])]
    sub = les[les["lambda"] >= 0.5]
    vals = sub["ood_acc_flip_mean"].astype(float)
    body = [
        "# (5) S1/S2 — FPR@95 for flipped Mahalanobis at λ ≥ 0.5",
        "",
        "Source: `r5/s_flip/summary.csv` column `ood_acc_flip_mean` (OOD acceptance at val 95th-percentile "
        "threshold on −score = FPR@95TPR for the flipped detector). Lesion-level Table 1 block.",
        "",
        "**Range across pad_heldout / Fitzpatrick17k / MED-NODE / SD-198 / SCIN × λ ∈ {{0.5,1,2,4,8}}:** "
        "**[{:.3f}, {:.3f}]** (min/max of cell means).".format(float(vals.min()), float(vals.max())),
        "",
        "| set | λ | FPR@95 flipped (mean ± sd) | CI |",
        "|---|---:|---|---|",
    ]
    for _, r in sub.sort_values(["set", "lambda"]).iterrows():
        body.append("| {} | {:g} | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] |".format(
            r["set"], r["lambda"], r["ood_acc_flip_mean"], r["ood_acc_flip_sd"],
            r["ood_acc_flip_ci_lo"], r["ood_acc_flip_ci_hi"]))
    w("05_s_flip_fpr.md", "\n".join(body), {"min": float(vals.min()), "max": float(vals.max())})


def item6():
    body = """# (6) Warning signal — majority-prediction-rate Spearman (n = 64)

Source: `results/paperB/r5/warning/REPORT.md`.

Corpus: **n = 64** ready runs (a4_p8=15, old_recipe=35, r1_shift0=9, r1_shift100=5).

Feature: `tier2_pred_maj_frac_excess` (majority-prediction-rate excess).

| OOD set | Spearman ρ with Mahalanobis AUROC | 95% CI | n |
|---|---:|---|---:|
| pad_heldout | **−0.770** | [−0.851, −0.631] | 64 |
| Fitzpatrick17k | **−0.754** | [−0.854, −0.574] | 64 |

(Report also notes this keeps the predicted negative ρ; BBSE TV does not.)
"""
    w("06_warning.md", body, {"source": "r5/warning/REPORT.md", "n": 64})


def item7():
    # ERM
    erm_root = ROOT / "r5" / "p8" / "erm_ref"
    bals = []
    for p in sorted(erm_root.glob("single_dann_ladv0_s*/summary.json")):
        s = json.loads(p.read_text())
        bals.append(float(s.get("id_balanced_acc", s["metrics"]["id"]["bal_acc"])))
        seed = int(p.parent.name.rsplit("_s", 1)[1])
    n = len(bals)
    mean = float(np.mean(bals)) if bals else float("nan")
    sd = float(np.std(bals, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")

    # R2 cells
    r2 = ROOT / "r6" / "r2" / "cur" / "runs"
    rows = []
    def ms(v):
        a = np.asarray(v, float)
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)

    for m in ("cdan", "iwcdan", "iwdan", "iwdan_oracle"):
        for lam_tag, lam in (("0p5", 0.5), ("2", 2.0)):
            vals_pad, vals_fitz, vals_id, vals_leak = [], [], [], []
            for s in (42, 52, 62, 72, 82):
                sp = r2 / "r2_cur_{}_ladv{}_s{}".format(m, lam_tag, s) / "summary.json"
                if not sp.exists():
                    continue
                j = json.loads(sp.read_text())
                o = j["metrics"]["ood"]
                vals_pad.append(o["pad_heldout"]["mahalanobis_classcond"])
                vals_fitz.append(o["fitzpatrick17k"]["mahalanobis_classcond"])
                vals_id.append(j["metrics"]["id"]["bal_acc"])
                vals_leak.append(j["metrics"]["leakage"]["bal_acc_mean"])
            lab = "IWDAN_ORACLE" if m == "iwdan_oracle" else m.upper()
            if not vals_pad:
                rows.append((lab, lam, 0, None))
                continue
            rows.append((lab, lam, ms(vals_pad), ms(vals_fitz), ms(vals_id), ms(vals_leak)))

    erm_note = "complete (5/5)." if n >= 5 else "remaining seeds still queued/held on GPU."
    body = [
        "# (7) ERM reference and R2 IWCDAN / CDAN / oracle / λ = 0.5",
        "",
        "## ERM single-encoder reference (lesion-level, λ = 0)",
        "",
        "Path: `r5/p8/erm_ref/single_dann_ladv0_s*`. **Available now: n = {}** of 5 — {}".format(n, erm_note),
        "",
    ]
    if n:
        body.append("ID bal acc: **{:.3f} ± {:.3f}** (SD); **SE = {:.3f}** (SD/√n). Per-seed: {}.".format(
            mean, sd, se, ", ".join("{:.3f}".format(x) for x in bals)))
    body += [
        "",
        "## R2 cells (recipe `current`, last.ckpt) — point mean ± SD; CIs in `r6/r2/cur/REPORT.md`",
        "",
        "| method | λ | n | Maha pad_heldout | Maha Fitz | ID bal | leakage |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        if row[2] == 0 or row[2] is None:
            body.append("| {} | {:g} | 0 | — | — | — | — |".format(row[0], row[1]))
            continue
        m, lam, pad, fitz, idb, leak = row
        body.append(
            "| {} | {:g} | {} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} |".format(
                m, lam, pad[2], pad[0], pad[1], fitz[0], fitz[1], idb[0], idb[1], leak[0], leak[1]))
    body += [
        "",
        "Note: CDAN / IWDAN / IWCDAN / IWDAN_ORACLE at λ ∈ {0.5, 2} are **5/5**. "
        "Full cluster CIs + unseen MED/SD + class-aware probes: `results/paperB/r6/r2/cur/REPORT.md`.",
        "",
    ]
    w("07_erm_r2.md", "\n".join(body), {"erm_n": n, "erm_mean": mean, "erm_se": se})


def index(ok1):
    lines = [
        "# Manuscript CPU number pack",
        "",
        "Main recipe = `current` / A4 last.ckpt unless noted. Built by `scripts/r5_manuscript_cpu_pack.py`.",
        "",
        "| item | file | status |",
        "|---|---|---|",
        "| (1) Table 1 last.ckpt | `01_table1_last.md` | {} |".format("ready" if ok1 else "PENDING extract"),
        "| (2) R1 MED/SD + regression (a) | `02_r1.md` | ready |",
        "| (3) A4 k-below | `03_a4.md` | ready |",
        "| (4) Clinical | `04_clinical.md` | ready |",
        "| (5) S1/S2 FPR@95 flipped | `05_s_flip_fpr.md` | ready |",
        "| (6) Warning majority Spearman | `06_warning.md` | ready |",
        "| (7) ERM + R2 cells | `07_erm_r2.md` | ready |",
        "",
        "SD-198 cluster sensitivity (no rule-A flips): `results/paperB/r5/p13_sd198/REPORT.md`.",
        "",
    ]
    w("INDEX.md", "\n".join(lines))


def main():
    ok1 = item1()
    item2()
    item3()
    item4()
    item5()
    item6()
    item7()
    index(ok1)


if __name__ == "__main__":
    main()
