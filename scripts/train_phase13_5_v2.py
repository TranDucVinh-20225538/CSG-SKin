#!/usr/bin/env python3
"""Phase 13.5 v2 — one pre-registered revision. Do not retune.

PREREGISTER_v2.json was written before this script runs.
BN-off is primary. Heavy-tailed structure on the ID domain only.
Outputs under results/paperB/phase13_5/v2/. v1 files are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(1)

import sys

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))
import train_phase13_5_toy as v1

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase13_5/v2")
FIG = OUT / "figures"
PREREG = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase13_5/PREREGISTER_v2.json")
WONG = v1.WONG
DERM_AUROC = v1.DERM_AUROC


def sample_id_heavy(world, n, classes, beta, seed, cfg):
    """ID-only: class-dependent τ, rare classes, contamination."""
    rng = np.random.default_rng(seed * 1009 + 17)
    D = world["mu"].shape[1]
    classes = np.asarray(classes, dtype=int)
    n_rare = int(cfg.get("n_rare_classes", 2))
    n_per_rare = int(cfg.get("n_per_rare", 80))
    rare = classes[-n_rare:]
    common = classes[:-n_rare]
    n_rare_tot = n_rare * n_per_rare
    if n_rare_tot >= n:
        raise ValueError("rare mass exceeds n")
    y_parts = [np.full(n_per_rare, int(c), dtype=np.int64) for c in rare]
    rem = n - n_rare_tot
    base, extra = divmod(rem, len(common))
    for i, c in enumerate(common):
        y_parts.append(np.full(base + (1 if i < extra else 0), int(c), dtype=np.int64))
    y = np.concatenate(y_parts)
    rng.shuffle(y)

    tau_rng = np.random.default_rng(seed * 17 + 3)
    lo = float(cfg.get("tau_id_log_lo", np.log(0.3)))
    hi = float(cfg.get("tau_id_log_hi", np.log(3.0)))
    tau_k = {int(c): float(np.exp(tau_rng.uniform(lo, hi))) for c in classes}
    tau = np.array([tau_k[int(c)] for c in y], dtype=np.float64)
    eps = rng.normal(size=(n, D))
    x = world["mu"][y] + beta * world["u"][None, :] + tau[:, None] * eps

    p = float(cfg.get("contam_frac", 0.03))
    n_c = int(round(p * n))
    if n_c > 0:
        idx = rng.choice(n, n_c, replace=False)
        tau_c = float(cfg.get("contam_tau", 8.0))
        x[idx] = world["mu"][y[idx]] + beta * world["u"][None, :] + tau_c * rng.normal(size=(n_c, D))
    return x.astype(np.float32), y.astype(np.int64), {"tau_k": tau_k, "n_contaminated": n_c}


def run_cell(payload):
    cell_name, cfg, lam, seed = payload
    torch.manual_seed(seed)
    np.random.seed(seed)
    world = v1.make_world(cfg, seed)
    id_classes = list(range(cfg["K"]))
    ood_classes = list(range(cfg["K_ood"]))
    x0, y0, id_meta = sample_id_heavy(world, cfg["n0"], id_classes, 0.0, seed, cfg)
    x1, y1 = v1.sample_domain(world, cfg["n1"], ood_classes, cfg["delta"], cfg["tau1"], seed, "d1")
    x2, y2 = v1.sample_heldout_domain(world, cfg["n1"], ood_classes, cfg["delta2"], cfg["tau1"], seed)

    idx = np.arange(len(x0))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    nte = max(int(0.2 * len(idx)), 1)
    te, tr = idx[:nte], idx[nte:]
    x0tr, y0tr = x0[tr], y0[tr]
    x0te, y0te = x0[te], y0[te]

    net = v1.ToyNet(cfg["D"], cfg["m"], cfg["K"], cfg["encoder"], tuple(cfg["hidden"]), cfg["bn"], cfg["adversary"])
    opt = torch.optim.AdamW(
        [
            {"params": list(net.enc.parameters()) + list(net.bn.parameters()) + list(net.cls.parameters()), "lr": cfg["lr"]},
            {"params": list(net.adv.parameters()), "lr": cfg["lr"] * cfg["adv_lr_mult"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    x1t = torch.from_numpy(x1)
    y1t = torch.from_numpy(y1)
    x0trt = torch.from_numpy(x0tr)
    y0trt = torch.from_numpy(y0tr)
    n_steps = max(len(x0tr) // (cfg["batch"] // 2), 1)
    half = cfg["batch"] // 2
    net.train()
    for _ep in range(int(cfg["epochs"])):
        perm0 = rng.integers(0, len(x0tr), size=(n_steps, half))
        perm1 = rng.integers(0, len(x1), size=(n_steps, half))
        for s in range(n_steps):
            xb = torch.cat([x0trt[perm0[s]], x1t[perm1[s]]])
            yb = torch.cat([y0trt[perm0[s]], y1t[perm1[s]]])
            db = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
            logits, dlog, _z = net(xb, lam)
            if cfg["cls_domain0_only"]:
                loss_cls = torch.nn.functional.cross_entropy(logits[:half], yb[:half])
            else:
                loss_cls = torch.nn.functional.cross_entropy(logits, yb)
            loss = loss_cls + float(lam) * torch.nn.functional.cross_entropy(dlog, db)
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        def pack(x):
            lg, _d, z = net(torch.from_numpy(x), 0.0)
            return z.numpy(), lg.numpy()

        ztr, _ = pack(x0tr)
        zid, lid = pack(x0te)
        zood, lood = pack(x1)
        zd2, ld2 = pack(x2)
    ev = v1.evaluate(ztr, y0tr, zid, y0te, lid, zood, y1, lood, zd2, ld2, cfg["K"], ood_classes, world["u"], None)
    ev.update(
        {
            "cell": cell_name,
            "lambda_adv": float(lam),
            "seed": int(seed),
            "m": int(cfg["m"]),
            "encoder": cfg["encoder"],
            "bn": bool(cfg["bn"]),
            "tau_k": id_meta["tau_k"],
            "n_contaminated": id_meta["n_contaminated"],
        }
    )
    return ev


def mean_std(xs):
    return v1.mean_std(xs)


def fmt(m, s):
    return v1.fmt(m, s)


def validate_v2(by_lam):
    def g(lam, key):
        recs = by_lam.get(lam) or []
        if not recs:
            return float("nan")
        return mean_std([r[key] for r in recs])[0]

    a0, a025, a2 = g(0.0, "maha"), g(0.25, "maha"), g(2.0, "maha")
    C1 = bool(a0 >= 0.5)
    C2 = bool(C1 and a2 < 0.48 and a025 < a0)
    dets = [g(2.0, k) for k in ("maha", "MSP", "Energy", "cosine", "knn50")]
    C3 = bool(C1 and (np.nanmax(dets) - np.nanmin(dets) < 0.08))
    C5 = bool(C1 and g(2.0, "maha_id_reweighted_to_ood_mix") < g(2.0, "maha") - 0.03)
    C7 = bool(C1 and g(2.0, "maha_domain2") < 0.48)
    rejected = not C1
    return {
        "C1_lambda0_at_or_above_chance": {"pass": C1, "auc0": a0},
        "C2_subchance_at_lambda_ge_0.25": {"pass": C2, "auc0": a0, "auc025": a025, "auc2": a2},
        "C3_detectors_together": {"pass": C3, "detectors_at_2": dict(zip(("maha", "MSP", "Energy", "cosine", "knn50"), dets))},
        "C5_reweight_strengthens": {"pass": C5, "auc_all": g(2.0, "maha"), "auc_reweight": g(2.0, "maha_id_reweighted_to_ood_mix")},
        "C7_third_domain_inverts": {"pass": C7, "auc_domain2": g(2.0, "maha_domain2")},
        "minimum_pass": bool(C1 and C2),
        "confirmed": bool(C1 and C2 and C3 and C5 and C7),
        "rejected_as_artifact": bool(rejected),
        "n_pass": int(sum([C1, C2, C3, C5, C7])),
        "n_tested": 5,
    }


def write_report(prereg, rows, facts_primary, facts_bn):
    lines = []
    lines.append("# Phase 13.5 v2 — one pre-registered revision")
    lines.append("")
    lines.append("`PREREGISTER_v2.json` was written before any v2 fit. Not retuned. v1 files untouched.")
    lines.append("")
    lines.append("Primary cell: locked MLP, **BatchNorm OFF**, ID-only heavy tails (class-dependent τ log-uniform [0.3, 3], two rare classes n=80, 3% contamination at τ=8). BN-on is a control, not the gate.")
    lines.append("")

    def cell_rows(name):
        return [r for r in rows if r["cell"] == name]

    def table_lambda(name):
        by = {}
        for r in cell_rows(name):
            by.setdefault(r["lambda_adv"], []).append(r)
        return by

    for name, title in (
        ("heavy_id_bn_off", "Primary (BN off)"),
        ("heavy_id_bn_on", "Control (BN on) — not the pass gate"),
    ):
        lines.append("## {} vs λ".format(title))
        lines.append("")
        lines.append("| λ | Maha | MSP | Energy | cosine | kNN-50 | leak bal | var OOD/ID | Maha d2 |")
        lines.append("|---:|---|---|---|---|---|---|---|---|")
        by = table_lambda(name)
        for lam in prereg["locked_from_v1"]["lams"]:
            recs = by.get(lam, [])
            def g(k):
                return fmt(*mean_std([r[k] for r in recs]))
            lines.append(
                "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    lam, g("maha"), g("MSP"), g("Energy"), g("cosine"), g("knn50"),
                    g("leakage_bal_acc"), g("var_ood_over_id"), g("maha_domain2"),
                )
            )
        lines.append("")

    facts = facts_primary
    lines.append("## Pre-registered criteria (primary cell only)")
    lines.append("")
    if facts.get("rejected_as_artifact"):
        lines.append(
            "**REJECTED.** AUROC at λ=0 is {:.3f} < 0.5. This is the same class of artifact as v1. "
            "The revision is not interpreted. No third revision.".format(facts["C1_lambda0_at_or_above_chance"]["auc0"])
        )
    elif facts.get("confirmed"):
        lines.append("**CONFIRMED.** C1–C2 and C3, C5, C7 all pass. The tail-asymmetric ID construction reproduces the derm F-set on this grid.")
    elif facts.get("minimum_pass"):
        lines.append("**MINIMUM PASS, NOT CONFIRMED.** C1 and C2 hold. C3–C7 determine confirmation; at least one failed. The mechanism is not excluded.")
    else:
        lines.append("**FAIL.** C1 held or the start was legal, but sub-chance at λ≥0.25 did not appear. The mechanism question closes.")
    lines.append("")
    lines.append("| Criterion | Pass? | Detail |")
    lines.append("|---|---|---|")
    for key, lab in (
        ("C1_lambda0_at_or_above_chance", "C1 λ=0 ≥ 0.5"),
        ("C2_subchance_at_lambda_ge_0.25", "C2 sub-chance at λ≥0.25"),
        ("C3_detectors_together", "C3 detectors together"),
        ("C5_reweight_strengthens", "C5 reweight strengthens"),
        ("C7_third_domain_inverts", "C7 third domain inverts"),
    ):
        blob = facts[key]
        lines.append("| {} | {} | {} |".format(lab, "yes" if blob["pass"] else "no", json.dumps({k: v for k, v in blob.items() if k != "pass"})))
    lines.append("")
    if facts_bn:
        a0 = facts_bn["C1_lambda0_at_or_above_chance"]["auc0"]
        a2 = facts_bn["C2_subchance_at_lambda_ge_0.25"]["auc2"]
        lines.append("BN-on control (not gated): Maha λ=0 {:.3f}, λ=2 {:.3f}. If this starts below 0.5, BN is still an artifact on this DGP.".format(a0, a2))
        lines.append("")
    lines.append("## Mechanism statement")
    lines.append("")
    if facts.get("rejected_as_artifact"):
        lines.append(
            "The revision started below 0.5. Rejected under the pre-registration. "
            "The paper reports a robust phenomenon with five refuted explanations "
            "(image memorisation, class composition, domain specificity, dimensional collapse, latent compression) "
            "plus an explicit statement that a controlled synthetic reproduction failed under a pre-registered grid. "
            "No third revision."
        )
    elif facts.get("confirmed"):
        lines.append(
            "A locked encoder plus ID-only heavy tails produced the derm F-set from a legal λ=0 start. "
            "That is the first synthetic construction that is allowed to be compared to the derm cliff."
        )
    elif facts.get("minimum_pass"):
        lines.append(
            "Sub-chance appears from a legal start. The remaining facts did not all fire. "
            "The tail ingredient is not excluded; it is not confirmed. Not retuned."
        )
    else:
        lines.append(
            "The pre-registered heavy-tail revision did not recover the derm inversion from a legal start. "
            "Close the mechanism question. The paper reports a robust phenomenon, five refuted explanations, "
            "and a failed controlled synthetic reproduction. That is a publishable position. No third revision."
        )
    lines.append("")
    lines.append("Figures: `phase13_5/v2/figures/fig_13_5_v2_*.{png,pdf}`.")
    (OUT / "PHASE13_5_V2_REPORT.md").write_text("\n".join(lines) + "\n")
    # also place a pointer next to the preregister
    pointer = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase13_5/PHASE13_5_V2_REPORT.md")
    pointer.write_text("".join(lines) + "\n\nCanonical copy: `phase13_5/v2/PHASE13_5_V2_REPORT.md`.\n")


def write_figures(lams, rows):
    FIG.mkdir(parents=True, exist_ok=True)
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], {}).setdefault(r["lambda_adv"], []).append(r)

    def series(cell, key):
        xs, ys, es = [], [], []
        for lam in lams:
            recs = by_cell.get(cell, {}).get(lam, [])
            m, s = mean_std([r[key] for r in recs])
            xs.append(lam)
            ys.append(m)
            es.append(s)
        return xs, ys, es

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x, y, e = series("heavy_id_bn_off", "maha")
    ax.errorbar(x, y, yerr=e, color=WONG["orange"], marker="o", lw=1.8, label="v2 primary BN-off")
    if "heavy_id_bn_on" in by_cell:
        x, y, e = series("heavy_id_bn_on", "maha")
        ax.errorbar(x, y, yerr=e, color=WONG["purple"], marker="^", lw=1.2, label="v2 BN-on control")
    ax.plot(list(DERM_AUROC.keys()), list(DERM_AUROC.values()), color=WONG["blue"], marker="s", lw=1.4, ls="--", label="derm Maha")
    ax.axhline(0.5, color=WONG["gray"], ls=":", lw=0.8)
    ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
    ax.set_ylabel("Mahalanobis AUROC")
    ax.set_title("13.5 v2: pre-registered heavy-ID revision")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_13_5_v2_auroc.{}".format(ext), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if not PREREG.exists():
        raise SystemExit("PREREGISTER_v2.json missing — write it before any v2 fit")
    prereg = json.loads(PREREG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    locked = prereg["locked_from_v1"]
    base = {
        "D": locked["D"],
        "K": locked["K"],
        "K_ood": locked["K_ood"],
        "n0": locked["n0"],
        "n1": locked["n1"],
        "tau0": 1.0,
        "tau1": locked["tau1"],
        "delta": locked["delta"],
        "delta2": locked["delta2"],
        "label_scale": locked["label_scale"],
        "encoder": locked["encoder"],
        "hidden": list(locked["hidden"]),
        "m": locked["m"],
        "cls_domain0_only": locked["cls_domain0_only"],
        "adversary": locked["adversary"],
        "adv_lr_mult": locked["adv_lr_mult"],
        "epochs": locked["epochs"],
        "batch": locked["batch"],
        "lr": locked["lr"],
        "weight_decay": locked["weight_decay"],
        "n_rare_classes": 2,
        "n_per_rare": 80,
        "tau_id_log_lo": float(np.log(0.3)),
        "tau_id_log_hi": float(np.log(3.0)),
        "contam_frac": 0.03,
        "contam_tau": 8.0,
    }
    cells = {
        "heavy_id_bn_off": {**deepcopy(base), "bn": False},
        "heavy_id_bn_on": {**deepcopy(base), "bn": True},
    }
    jobs = []
    for name, cfg in cells.items():
        for lam in locked["lams"]:
            for seed in locked["seeds"]:
                jobs.append((name, cfg, float(lam), int(seed)))
    (OUT / "job_list.json").write_text(json.dumps({"n": len(jobs), "preregister": str(PREREG)}, indent=2) + "\n")
    if args.dry_run:
        print("dry_run n_jobs", len(jobs))
        return

    rows = []
    done_path = OUT / "all_rows.jsonl"
    if done_path.exists():
        done_path.unlink()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            rec["tau_k"] = {str(k): v for k, v in rec["tau_k"].items()}
            rows.append(rec)
            with done_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if i % 10 == 0 or i == len(jobs):
                print("done", i, "/", len(jobs), rec["cell"], rec["lambda_adv"], rec["seed"], "maha", rec["maha"], flush=True)

    (OUT / "all_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    def facts_for(name):
        by = {}
        for r in rows:
            if r["cell"] == name:
                by.setdefault(r["lambda_adv"], []).append(r)
        return validate_v2(by) if by else None

    facts_p = facts_for("heavy_id_bn_off")
    facts_b = facts_for("heavy_id_bn_on")
    (OUT / "facts_primary.json").write_text(json.dumps(facts_p, indent=2) + "\n")
    (OUT / "facts_bn_on.json").write_text(json.dumps(facts_b, indent=2) + "\n")
    write_figures(locked["lams"], rows)
    write_report(prereg, rows, facts_p, facts_b)
    print("primary", json.dumps({k: facts_p[k] for k in ("minimum_pass", "confirmed", "rejected_as_artifact", "n_pass")}))
    print("wrote", OUT / "PHASE13_5_V2_REPORT.md")


if __name__ == "__main__":
    main()
