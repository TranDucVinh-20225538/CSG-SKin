#!/usr/bin/env python3
"""R3 C: Figures 1, 2, 3 and 5 from the lesion-level sweep, same code and spec as plot_manuscript_figures.py.

Data: R2 Item 4 summaries (results/paperB/r2/item4/runs/*/summary.json) for Fig 1a–c and Fig 3; Phase 13 per-run
analysis of the lesion-level features (results/paperB/r3/c/per_run/*.json) for Fig 1d, 2 and 5. The image-level
versions are kept as <stem>_imagelevel.{pdf,png}. Figure 6 is not touched."""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import paper_figure_style as pfs  # noqa: E402
import plot_manuscript_figures as M  # noqa: E402

RES = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB")
LES_RUNS = RES / "r2" / "item4" / "runs"
OUT = RES / "r3" / "c"
PER_RUN = OUT / "per_run"
STEMS = ("fig1_intervention", "fig2_confidence", "fig3_detectors", "fig5_leakage_no_mediation")


def summaries():
    return [json.loads(p.read_text()) for p in sorted(LES_RUNS.glob("runB_orth1_ladv*/summary.json"))]


def per_run():
    return [json.loads(p.read_text()) for p in sorted(PER_RUN.glob("*.json"))]


def ms(xs):
    return M.mean_std(xs)


def main():
    fig = Path(pfs.ROOT_FIG)
    for s in STEMS:
        for ext in ("pdf", "png"):
            src, dst = fig / "{}.{}".format(s, ext), fig / "{}_imagelevel.{}".format(s, ext)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    rows, runs = summaries(), per_run()
    by = defaultdict(list)
    for r in rows:
        by[float(r["lambda_adv"])].append(r)
    by13 = defaultdict(list)
    for r in runs:
        by13[float(r["lambda_adv"])].append(r)

    # Fig 1c reads the dual-column file; give it the lesion-level pad_heldout Mahalanobis instead.
    root = OUT / "fig_inputs"
    (root / "phase16").mkdir(parents=True, exist_ok=True)
    dual = {}
    for lam in pfs.LAMS:
        m, s = ms([r["ood"]["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"] for r in by.get(lam, [])])
        dual[M._dual_pad_key(lam)] = {"pad_heldout": {"mean": m, "std": s, "n": len(by.get(lam, []))}}
    (root / "phase16" / "table1_ood_pad_dual_column.json").write_text(json.dumps({"by_lambda": dual}, indent=1) + "\n")

    M.load_phase3_summaries = summaries
    M.load_phase13_runs = per_run
    M.PAPERB = root

    # Shared y-range for Fig 1 (a)(b)(c) that contains every plotted mean ± s.d.
    pts = []
    for lam in pfs.LAMS:
        for fn in (lambda r: r["leakage"]["z_lesion"]["bal_acc_mean"], lambda r: r["id_balanced_acc"],
                   lambda r: r["ood"]["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]):
            m, s = ms([fn(r) for r in by.get(lam, [])])
            if np.isfinite(m):
                pts += [m - s, m + s]
    lo = np.floor(min(min(pts), 0.4) / 0.05) * 0.05
    ylim = (float(lo), 1.0)

    missing = []
    need = {
        "Fig 1a-c, Fig 3 (summary.json)": by,
        "Fig 1d, Fig 2, Fig 5 (per-run analysis)": by13,
    }
    for what, d in need.items():
        for lam in pfs.LAMS:
            n = len(d.get(lam, []))
            if n < 3:
                missing.append("{}: λ = {:g} has {} seed(s)".format(what, lam, n))

    pfs.setup_matplotlib()
    M.fig1_intervention(shared_ylim=ylim)
    M.fig2_confidence()
    M.fig3_detectors()
    M.fig5_no_mediation()
    from PIL import Image
    for s in STEMS:
        Image.open(fig / "{}.png".format(s)).convert("L").save(fig / "{}_gray.png".format(s))

    seeds = {"{:g}".format(l): sorted(int(r["seed"]) for r in by.get(l, [])) for l in pfs.LAMS}
    seeds13 = {"{:g}".format(l): sorted(int(r["seed"]) for r in by13.get(l, [])) for l in pfs.LAMS}
    L = ["# R3 C — Figures 1, 2, 3, 5 from the lesion-level sweep", "",
         "Code: `scripts/r3_c_lesion_figures.py` → `scripts/plot_manuscript_figures.py` (unchanged plotting, palette "
         "#0072B2/#D55E00/#009E73/#000000/#E69F00, constrained_layout, 300 dpi, PDF + PNG, grayscale PNG). Figure 6 untouched. "
         "Image-level versions kept as `<stem>_imagelevel.{pdf,png}`.", "",
         "**Verdict:** {}".format("All seven λ plotted in every panel." if not missing else
                                  "Points plotted where they exist; no interpolation. Missing: see below."), "",
         "Fig 1 shared y-range (a)(b)(c): {:.2f}–{:.2f}, the smallest 0.05 step that contains every mean ± s.d. "
         "(lowest {:.3f}).".format(ylim[0], ylim[1], min(pts)), "",
         "| λ | seeds (summary.json: Fig 1a–c, 3) | seeds (per-run analysis: Fig 1d, 2, 5) |", "|---:|---|---|"]
    for l in pfs.LAMS:
        k = "{:g}".format(l)
        L.append("| {} | {} | {} |".format(k, ", ".join(map(str, seeds[k])), ", ".join(map(str, seeds13[k]))))
    L += ["", "## Missing", ""] + (["- " + m for m in missing] if missing else ["- None."])
    L += ["", "## Caveats", "",
          "- Fig 1c is Mahalanobis on pad_heldout from the lesion-level summaries; the image-level Fig 1c used the Phase 16 "
          "dual-column file.",
          "- λ ∈ {0, 2} use seeds 42/52/62/72/82; other λ use 42/43/44."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
