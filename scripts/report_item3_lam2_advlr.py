#!/usr/bin/env python3
"""Aggregate Item 3 λ=2 adv_lr retrain runs → markdown + JSON verdict."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P3 = ROOT / "results/paperB/phase3_sweep"
OUT = ROOT / "results/paperB/reviewer_r1"

RUNS = []
for mult in (30, 10, 100):
    for seed in (42, 52, 62):
        tag = f"runB_orth1_ladv2_s{seed}" + ("" if mult == 30 else f"_advlr{mult}")
        RUNS.append((mult, seed, tag))


def load(tag):
    p = P3 / tag / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def maha_unrestricted(d):
    return d["ood"]["pad_heldout"]["z_lesion"]["mahalanobis_classcond"]["unrestricted"]


def main():
    rows = []
    missing = []
    for mult, seed, tag in RUNS:
        d = load(tag)
        if d is None:
            missing.append(tag)
            continue
        z = d["ood"]["pad_heldout"]["z_lesion"]
        rows.append(
            {
                "adv_lr_mult": mult,
                "seed": seed,
                "run": tag,
                "reused_phase2_ckpt": d.get("reused_phase2_ckpt"),
                "id_balanced_acc": d.get("id_balanced_acc"),
                "leakage_bal": d["leakage"]["z_lesion"]["bal_acc_mean"],
                "mahalanobis": maha_unrestricted(d),
                "knn_k50": z["knn_k50"],
                "cosine_max": z["cosine_max"],
            }
        )

    def agg(mult, key):
        vals = [r[key] for r in rows if r["adv_lr_mult"] == mult]
        if not vals:
            return None
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a), "vals": vals}

    baseline = agg(30, "mahalanobis")
    x10 = agg(10, "mahalanobis")
    x100 = agg(100, "mahalanobis")

    verdict = "pending"
    lines = [
        "# Item 3 — λ=2 adversary LR sensitivity (retrained)\n",
        "\nCanonical ×30 @ λ=2 may reuse Phase-2 ckpt; ×10/×100 **must** train fresh (`reused_phase2_ckpt: false`).\n",
    ]
    if missing:
        lines.append(f"\n**Missing runs ({len(missing)}):** " + ", ".join(missing) + "\n")
        verdict = "incomplete"

    if x10 and x100 and baseline:
        sub30 = baseline["mean"] < 0.5
        sub10 = x10["mean"] < 0.5
        sub100 = x100["mean"] < 0.5
        lines.append("\n## Mahalanobis pad_heldout (3 seeds)\n\n| adv_lr × | mean ± std | per-seed |\n|---:|---:|---|\n")
        for mult in (30, 10, 100):
            a = agg(mult, "mahalanobis")
            lines.append(f"| {mult} | {a['mean']:.4f} ± {a['std']:.4f} | {a['vals']} |\n")

        if not missing:
            if sub10 and sub100:
                verdict = "robust"
                lines.append(
                    "\n**Verdict:** Sub-chance Mahalanobis **holds at ×10 and ×100** (fresh trains). "
                    "One sentence in main text; full detector grid in appendix.\n"
                )
            elif sub30 and not (sub10 and sub100):
                verdict = "boundary_x30"
                lines.append(
                    "\n**Verdict:** Sub-chance **only at ×30** (or not at ×10/×100). "
                    "Boundary condition → **main text**; claim narrows to adversarial configuration.\n"
                )
            else:
                verdict = "mixed"
                lines.append("\n**Verdict:** Mixed — report all three multipliers; do not claim robustness until inspected.\n")

    for mult in (10, 100):
        bad = [r for r in rows if r["adv_lr_mult"] == mult and r.get("reused_phase2_ckpt")]
        if bad:
            lines.append(f"\n⚠ ×{mult}: {len(bad)} run(s) still flagged `reused_phase2_ckpt` — invalidate sensitivity claim.\n")

    payload = {"rows": rows, "missing": missing, "verdict": verdict, "aggregates": {str(m): agg(m, "mahalanobis") for m in (30, 10, 100)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "item3_lam2_advlr_sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "ITEM3_LAM2_ADVLR_REPORT.md").write_text("".join(lines))
    print("wrote", OUT / "ITEM3_LAM2_ADVLR_REPORT.md", "verdict=", verdict)


if __name__ == "__main__":
    main()
