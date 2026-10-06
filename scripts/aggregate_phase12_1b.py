#!/usr/bin/env python3
"""Aggregate 12.1b: classify (a)/(b)/(c). Plot leakage + ID acc vs λ.

Matched λ=0 is required. Coarse λ=0 is not an anchor.
Leakage is 3-class balanced accuracy; floor is 1/3, not the plain-acc majority 0.437.
Divergence is a training failure, not outcome (b). Do not launch iWildCam or a dense grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/camelyon17/matched")
FIG = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/figures")
REPORT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase12/PHASE12_1B_REPORT.md")
LAMS = (0.0, 10.0, 30.0, 100.0, 300.0)
BAL_FLOOR = 1.0 / 3.0
NEAR_FLOOR = 0.50
HARD_STOP_LEAK = 0.80
ID_COLLAPSE = 0.90
WONG = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "gray": "#888888", "black": "#000000"}


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def load_run(lam, seed=42):
    p = OUT / "matched_densenet121_ladv{}_s{}".format(lam_tag(lam), seed) / "summary.json"
    if not p.exists():
        raise SystemExit(
            "missing {} — matched λ=0 is blocking for interpretation; "
            "do not use coarse λ=0 as the reference row.".format(p)
        )
    return json.loads(p.read_text())


def finite(x):
    return isinstance(x, (int, float)) and np.isfinite(x)


def is_diverged(row):
    if row.get("diverged"):
        return True, "flagged in summary"
    hist = row.get("history") or []
    if not hist:
        return False, ""
    accs, losses = [], []
    for h in hist:
        if h.get("diverged"):
            return True, "history.diverged"
        for key, bucket in (("train_loss", losses), ("id_val_select_acc", accs)):
            v = h.get(key)
            if v is None:
                continue
            if not finite(v):
                return True, "non-finite {}".format(key)
            bucket.append(float(v))
    if accs and max(accs) < 0.55:
        return True, "id_val_select never exceeded 0.55 (near chance)"
    return False, ""


def leak_of(row):
    blob = row.get("leakage_train_hospitals_3class") or {}
    return blob.get("balanced_acc")


def id_of(row):
    return (row.get("id_val_score") or {}).get("acc")


def main():
    rows = [load_run(lam) for lam in LAMS]
    by_lam = {float(r.get("lambda_adv", lam)): r for lam, r in zip(LAMS, rows)}
    if 0.0 not in by_lam:
        raise SystemExit("matched λ=0 missing; interpretation blocked")

    flags = []
    for lam, r in zip(LAMS, rows):
        d, why = is_diverged(r)
        flags.append({"lambda_adv": lam, "diverged": d, "why": why})

    leaks = [leak_of(r) for r in rows]
    ids = [id_of(r) for r in rows]
    row10 = by_lam[10.0]
    leak10 = leak_of(row10)
    div10, why10 = is_diverged(row10)
    leak0 = leak_of(by_lam[0.0])
    id0 = id_of(by_lam[0.0])
    row300 = by_lam[300.0]
    leak300 = leak_of(row300)
    div300, why300 = is_diverged(row300)

    # Downward-extension rule uses λ=10 only, and only if that run did not diverge.
    if div10 or not finite(leak10):
        range_call = "lambda10_diverged"
        next_action = (
            "λ=10 diverged ({}). Record as a training failure. Do not read ID collapse as outcome (b). "
            "Do not launch a downward grid from a failed run."
        ).format(why10 or "no finite leakage")
        next_grid = None
    elif leak10 <= NEAR_FLOOR:
        range_call = "overshot"
        next_action = (
            "Leakage at matched λ=10 is already at or near the 3-class balanced-accuracy floor "
            "({:.3f} vs floor {:.3f}). The {{10,30,100,300}} grid sits past the transition. "
            "Do not conclude from four post-cliff points. Extend downward: λ ∈ {{0.25, 0.5, 1, 2, 5}} "
            "(not launched)."
        ).format(leak10, BAL_FLOOR)
        next_grid = [0.25, 0.5, 1.0, 2.0, 5.0]
    elif leak10 > HARD_STOP_LEAK:
        range_call = "upward_was_right"
        next_action = (
            "Leakage at matched λ=10 is still high ({:.3f} > 0.8). The upward scan was right. "
            "Continue classification on this grid. Do not extend downward."
        ).format(leak10)
        next_grid = None
    else:
        range_call = "mid_range"
        next_action = (
            "Leakage at matched λ=10 is mid-range ({:.3f}). Dense-sample around λ=10. Not launched."
        ).format(leak10)
        next_grid = None

    usable = []
    for lam, r, leak, ida, fl in zip(LAMS, rows, leaks, ids, flags):
        if fl["diverged"] or not finite(leak) or not finite(ida):
            continue
        usable.append((lam, leak, ida, r))

    leak_near_floor = any(leak <= NEAR_FLOOR for _, leak, _, _ in usable)
    leak_never = bool(usable) and all(leak > HARD_STOP_LEAK for _, leak, _, _ in usable)
    id_collapsed_where_leak_drops = False
    leak_drops_id_held = False
    for lam, leak, ida, _r in usable:
        if leak <= NEAR_FLOOR:
            if ida < ID_COLLAPSE:
                id_collapsed_where_leak_drops = True
            else:
                leak_drops_id_held = True

    if not usable:
        outcome = "training_failure"
        meaning = (
            "No non-diverged matched-recipe run produced finite leakage and ID accuracy. "
            "This is a training failure, not outcome (b). Do not escalate λ. iWildCam stays gated."
        )
    elif leak_never and finite(leak300) and not div300 and leak300 > HARD_STOP_LEAK:
        outcome = "c"
        meaning = (
            "Leakage never fell below 0.8 at any non-diverged λ including 300, with a derm-matched adversary. "
            "Outcome (c): this adversarial formulation does not induce invariance on Camelyon17. "
            "Hard stop. Do not raise λ further. iWildCam stays gated."
        )
    elif leak_never:
        outcome = "c"
        meaning = (
            "Leakage stayed above 0.8 on every non-diverged λ. "
            "Outcome (c) under the hard stop. Do not escalate λ. iWildCam stays gated."
        )
    elif range_call == "overshot":
        if leak_drops_id_held:
            outcome = "a_overshot"
            meaning = (
                "Leakage is already near the balanced-accuracy floor by λ=10 while ID accuracy is retained. "
                "That is consistent with outcome (a) on a different λ scale, but the four points "
                "{{10,30,100,300}} do not locate the transition. Extend downward before claiming a cliff."
            )
        elif id_collapsed_where_leak_drops and not leak_drops_id_held:
            outcome = "b_overshot"
            meaning = (
                "Leakage is near the floor by λ=10 but ID accuracy also collapsed on the non-diverged "
                "runs that got there. Candidate outcome (b). Confirm with the downward grid; do not "
                "read diverged high-λ ID collapse as entanglement."
            )
        else:
            outcome = "deferred_overshot"
            meaning = (
                "Leakage near the floor at λ=10; ID pattern mixed or incomplete. "
                "Do not classify (a)/(b) from four post-cliff points. Extend downward."
            )
    elif leak_near_floor and id_collapsed_where_leak_drops and not leak_drops_id_held:
        outcome = "b"
        meaning = (
            "Leakage falls toward the 3-class balanced-accuracy floor (1/3) only where ID accuracy "
            "also collapses, on non-diverged runs. Outcome (b): invariance is unattainable here "
            "without destroying the task. Boundary condition — the derm claim narrows to "
            "'when invariance is achieved'. iWildCam stays gated."
        )
    elif leak_near_floor and leak_drops_id_held:
        outcome = "a"
        meaning = (
            "Leakage falls toward the 3-class balanced-accuracy floor while ID accuracy is retained. "
            "Outcome (a): the derm form is now testable on this λ scale. Dense-sample around the "
            "transition next; not launched in this round."
        )
    else:
        outcome = "c"
        meaning = (
            "Leakage moved but did not approach the balanced-accuracy floor (still > 0.8 or not near 1/3). "
            "Treat as outcome (c) under the hard stop. Do not escalate λ."
        )

    FIG.mkdir(parents=True, exist_ok=True)
    x_plot = [0.03 if lam == 0 else lam for lam in LAMS]
    y_leak = [leak if finite(leak) else np.nan for leak in leaks]
    y_id = [ida if finite(ida) else np.nan for ida in ids]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x_plot, y_leak, color=WONG["orange"], marker="o", lw=1.8, label="leakage (3-class bal acc)")
    ax.plot(x_plot, y_id, color=WONG["green"], marker="s", lw=1.8, label="ID acc (id_val_score)")
    ax.axhline(BAL_FLOOR, color=WONG["gray"], ls="--", lw=1.0, label="3-class bal-acc floor {:.3f}".format(BAL_FLOOR))
    ax.axhline(HARD_STOP_LEAK, color=WONG["orange"], ls=":", lw=0.8, label="hard-stop leak 0.8")
    ax.set_xscale("log")
    ax.set_xticks(x_plot)
    ax.set_xticklabels(["0" if lam == 0 else "{:g}".format(lam) for lam in LAMS])
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$ (derm-matched adversary; λ=0 is the matched control)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.25, 1.05)
    ax.set_title("12.1b: did invariance take?  outcome {}".format(outcome))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_12_1b_leakage_id.{}".format(ext), dpi=300)
    plt.close(fig)

    lines = []
    lines.append("# Phase 12.1b — find where leakage actually drops")
    lines.append("")
    lines.append("Audit: `PHASE12_1B_AUDIT.md`. Coarse dense grid was **not** submitted. iWildCam not started.")
    lines.append("Coarse λ=0 is **not** the reference row. Every comparison uses matched-recipe λ=0.")
    lines.append("")
    lines.append("**Outcome: ({})**".format(outcome))
    lines.append("")
    lines.append(meaning)
    lines.append("")
    lines.append("## Floors (do not mix scales)")
    lines.append("")
    lines.append(
        "Leakage is **3-class balanced accuracy**. Its chance floor is **1/3 ≈ {:.3f}**, "
        "independent of hospital imbalance. A constant predictor scores recall 1.0 on one class "
        "and 0 on the other two.".format(BAL_FLOOR)
    )
    lines.append("")
    lines.append(
        "The figure 0.437 = 132k/302k is the **plain-accuracy** majority (always predict hospital 4). "
        "It is reported in the table as `plain maj` and is **not** the invariance target. "
        "Declaring invariance against 0.437 would declare it early."
    )
    lines.append("")
    lines.append("## λ=10 range call")
    lines.append("")
    lines.append("**{}** — {}".format(range_call, next_action))
    lines.append("")
    if next_grid:
        lines.append("Proposed downward grid (not launched): `{}`".format(next_grid))
        lines.append("")
    lines.append("## Per-λ (leakage first)")
    lines.append("")
    lines.append(
        "| λ_adv | leak 3-cls bal acc | bal-acc floor | plain maj | ID acc | diverged | "
        "Maha h2 | kNN h2 | Maha h1 | kNN h1 | xfer acc h2 | xfer acc h1 |"
    )
    lines.append("|---:|---|---|---|---|---|---|---|---|---|---|---|")
    for lam, r, leak, ida, fl in zip(LAMS, rows, leaks, ids, flags):
        div_cell = "yes ({})".format(fl["why"] or "flagged") if fl["diverged"] else "no"
        if r.get("ood") is None or not finite(leak) or not finite(ida):
            lines.append(
                "| {:g} | {} | {:.3f} | — | {} | {} | — | — | — | — | — | — |".format(
                    lam,
                    "{:.3f}".format(leak) if finite(leak) else "—",
                    BAL_FLOOR,
                    "{:.3f}".format(ida) if finite(ida) else "—",
                    div_cell,
                )
            )
            continue
        d2 = r["ood"]["test"]["detectors"]
        d1 = r["ood"]["val"]["detectors"]
        leak_blob = r["leakage_train_hospitals_3class"]
        lines.append(
            "| {:g} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                lam,
                leak_blob["balanced_acc"],
                BAL_FLOOR,
                leak_blob["majority_baseline_acc"],
                r["id_val_score"]["acc"],
                div_cell,
                d2["mahalanobis_classcond_sharedcov"],
                d2["knn_k50"],
                d1["mahalanobis_classcond_sharedcov"],
                d1["knn_k50"],
                r["ood"]["test"]["accuracy"]["acc"],
                r["ood"]["val"]["accuracy"]["acc"],
            )
        )
    lines.append("")
    lines.append(
        "OOD AUROC is secondary until leakage moves. Do not call a Maha-only drop the derm phenomenon. "
        "Do not write 'AUROC fell from X to Y' unless X is matched-recipe λ=0."
    )
    lines.append("")
    lines.append("Figure: `phase12/figures/fig_12_1b_leakage_id.{png,pdf}`.")
    lines.append("")
    lines.append("**STOP.** Dense cliff grid and iWildCam remain gated until leakage moves and the range call is acted on.")
    REPORT.write_text("\n".join(lines) + "\n")
    dump = {
        "outcome": outcome,
        "balanced_acc_floor": BAL_FLOOR,
        "plain_majority_is_not_the_floor": True,
        "range_call": range_call,
        "next_action": next_action,
        "next_grid_not_launched": next_grid,
        "lambda0_leak": leak0,
        "lambda0_id": id0,
        "lambda10_leak": leak10,
        "lambda10_diverged": div10,
        "lambda300_leak": leak300,
        "lambda300_diverged": div300,
        "flags": flags,
        "leakage": list(zip(LAMS, leaks)),
        "id_acc": list(zip(LAMS, ids)),
        "meaning": meaning,
        "coarse_lambda0_used": False,
    }
    (OUT / "aggregate.json").write_text(json.dumps(dump, indent=2) + "\n")
    print("outcome", outcome)
    print("range_call", range_call)
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
