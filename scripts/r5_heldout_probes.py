#!/usr/bin/env python3
"""Paper B held-out domain probes (linear / MLP / kNN) — decisions/paperB_r5_heldout_probes.md.

P6 V1 protocol on images the adversary never saw (ISIC-test ∪ pad_heldout, GroupShuffleSplit).
pad_full linear probe reported beside held-out (primary vs secondary; 94e2399 unchanged).
λ=0.25 Table-1 per-seed table + Spearman vs Mahalanobis AUROC.

Usage:
  r5_heldout_probes.py              # run all available cells + report
  r5_heldout_probes.py --report_only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402

OUT = R.R5 / "heldout_probes"
REP = "z_lesion_norm"
PSEEDS = (42, 52, 62)
BOOT_N, BOOT_SEED = 2000, 20261017
NEED = ("isic_test.npz", "pad_heldout.npz", "pad_full.npz")

# (family, feat_root, tag_glob_or_None)
FAMILIES = [
    ("table1", R.FEAT_L, None),
    ("a4_main", R.R5 / "p8" / "main" / "features", None),
    ("single_p8", R.R5 / "p8" / "erm_ref" / "features", None),
    ("single_p5", R.R5 / "p5" / "features", None),
    ("r1_cur", R.R6 / "r1" / "cur" / "features", None),
]


def _parse_meta(family: str, tag: str) -> dict:
    out = {"family": family, "tag": tag, "lambda": None, "seed": None, "shift": None}
    m = re.search(r"ladv([0-9p]+)_s(\d+)", tag)
    if m:
        out["lambda"] = float(m.group(1).replace("p", "."))
        out["seed"] = int(m.group(2))
    m = re.search(r"shift(\d+)", tag)
    if m:
        out["shift"] = int(m.group(1))
    return out


def inventory():
    rows = []
    for family, root, _ in FAMILIES:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if not all((d / n).is_file() for n in NEED):
                continue
            # single: need z_lesion_norm in npz
            try:
                z = np.load(d / "isic_test.npz")
                if REP not in z:
                    continue
            except Exception:  # noqa: BLE001
                continue
            meta = _parse_meta(family, d.name)
            meta["feat_root"] = str(root)
            rows.append(meta)
    return rows


def _bal(y, pred):
    return float(balanced_accuracy_score(y, pred))


def fit_linear(Xtr, ytr, Xte, yte, s):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
    clf.fit(Xtr, ytr)
    return _bal(yte, clf.predict(Xte))


def fit_mlp(Xtr, ytr, Xte, yte, s):
    # early_stopping uses validation_fraction of the train split (as committed)
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=s,
        ),
    )
    clf.fit(Xtr, ytr)
    return _bal(yte, clf.predict(Xte))


def fit_knn(Xtr, ytr, Xte, yte, s):
    del s  # deterministic given scaled features
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=20, weights="uniform"))
    clf.fit(Xtr, ytr)
    return _bal(yte, clf.predict(Xte))


def _summarize(vals):
    a = np.asarray(vals, float)
    return {
        "bal_acc_mean": float(a.mean()),
        "bal_acc_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "bal_acc_per_probe_seed": [float(x) for x in a],
    }


def analyse_run(feat_root: Path, tag: str) -> dict:
    te = R.load(tag, "isic_test", feat_root=feat_root)
    hold = R.load(tag, "pad_heldout", feat_root=feat_root)
    full = R.load(tag, "pad_full", feat_root=feat_root)
    gi, gp = R.id_test_groups(), R.pad_heldout_patients()
    assert len(gi) == len(te["labels"]) and len(gp) == len(hold["labels"])

    Xh = np.r_[te[REP], hold[REP]]
    dh = np.r_[np.zeros(len(te["labels"])), np.ones(len(hold["labels"]))]
    gh = np.r_[np.char.add("I", gi.astype(str)), np.char.add("P", gp.astype(str))]

    Xp = np.r_[te[REP], full[REP]]
    dp = np.r_[np.zeros(len(te["labels"])), np.ones(len(full["labels"]))]

    linear, mlp, knn, pad_full = [], [], [], []
    for s in PSEEDS:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(Xh, dh, gh))
        linear.append(fit_linear(Xh[a], dh[a], Xh[b], dh[b], s))
        mlp.append(fit_mlp(Xh[a], dh[a], Xh[b], dh[b], s))
        knn.append(fit_knn(Xh[a], dh[a], Xh[b], dh[b], s))
        atr, ate, ytr, yte = train_test_split(Xp, dp, test_size=0.3, random_state=s, stratify=dp)
        pad_full.append(fit_linear(atr, ytr, ate, yte, s))

    return {
        "rep": REP,
        "heldout_linear": _summarize(linear),
        "heldout_mlp": _summarize(mlp),
        "heldout_knn20": _summarize(knn),
        "pad_full_linear": _summarize(pad_full),
        "n": {"isic_test": int(len(te["labels"])), "pad_heldout": int(len(hold["labels"])),
              "pad_full": int(len(full["labels"]))},
    }


def maha_auroc_sets(feat_root: Path, tag: str, sets=("pad_heldout", "fitzpatrick17k", "mednode", "sd198")):
    tr = R.load(tag, "isic_train", feat_root=feat_root)
    te = R.load(tag, "isic_test", feat_root=feat_root)
    oods = {}
    for o in sets:
        p = Path(feat_root) / tag / f"{o}.npz"
        if p.is_file():
            oods[o] = dict(np.load(p))
    if not oods:
        return {}
    scores, _ = R.detector_scores(tr, te, oods, rep=REP)
    sid, so = scores["mahalanobis_classcond"]
    return {o: R.auroc(sid, so[o]) for o in so}


def seed_mean_ci(vals):
    """Bootstrap CI of the mean over training seeds (n_seeds draws with replacement)."""
    v = np.asarray(vals, float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else float("nan"), "sd": 0.0,
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(v)}
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT_N)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(v))}


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir(exist_ok=True)
    inv = inventory()
    R.write_json(OUT / "inventory.json", {"n": len(inv), "runs": inv, "commit": R.git_head(),
                                          "note": "tab:obj blocked — no cached features under phase15/objectives"})
    for meta in inv:
        key = f"{meta['family']}__{meta['tag']}"
        outp = OUT / "per_run" / f"{key}.json"
        if outp.is_file():
            continue
        print(f"[heldout_probes] {key}", flush=True)
        rec = {**meta, **analyse_run(Path(meta["feat_root"]), meta["tag"])}
        if meta["family"] == "table1" and meta["lambda"] == 0.25:
            rec["maha_auroc"] = maha_auroc_sets(Path(meta["feat_root"]), meta["tag"])
        R.write_json(outp, rec)
    report()


def _load_rows():
    rows = []
    for p in sorted((OUT / "per_run").glob("*.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def report():
    rows = _load_rows()
    if not rows:
        print("no per_run JSON yet", flush=True)
        return
    flat = []
    for r in rows:
        flat.append({
            "family": r["family"], "tag": r["tag"], "lambda": r.get("lambda"), "seed": r.get("seed"),
            "shift": r.get("shift"),
            "heldout_linear": r["heldout_linear"]["bal_acc_mean"],
            "heldout_mlp": r["heldout_mlp"]["bal_acc_mean"],
            "heldout_knn20": r["heldout_knn20"]["bal_acc_mean"],
            "pad_full_linear": r["pad_full_linear"]["bal_acc_mean"],
            **{f"maha_{k}": v for k, v in (r.get("maha_auroc") or {}).items()},
        })
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "summary.csv", index=False)

    # cell aggregates
    cells = []
    gcols = ["family", "lambda", "shift"]
    for keys, g in df.groupby(gcols, dropna=False):
        fam, lam, shift = keys
        cell = {"family": fam, "lambda": lam, "shift": shift, "n_seeds": int(len(g))}
        for col in ("heldout_linear", "heldout_mlp", "heldout_knn20", "pad_full_linear"):
            st = seed_mean_ci(g[col].tolist())
            cell[f"{col}_mean"] = st["mean"]
            cell[f"{col}_sd"] = st["sd"]
            cell[f"{col}_ci_lo"] = st["ci_lo"]
            cell[f"{col}_ci_hi"] = st["ci_hi"]
        cells.append(cell)
    cdf = pd.DataFrame(cells).sort_values(["family", "shift", "lambda"], na_position="last")
    cdf.to_csv(OUT / "cells.csv", index=False)
    R.write_json(OUT / "cells.json", cells)

    # λ=0.25 per-seed
    d025 = df[(df.family == "table1") & (df["lambda"] == 0.25)].sort_values("seed")
    spearman = {}
    if len(d025):
        # ensure maha filled
        for i, row in d025.iterrows():
            if pd.isna(row.get("maha_pad_heldout")):
                meta = next(r for r in rows if r["tag"] == row["tag"])
                if "maha_auroc" not in meta:
                    meta["maha_auroc"] = maha_auroc_sets(Path(meta["feat_root"]), meta["tag"])
                    R.write_json(OUT / "per_run" / f"{meta['family']}__{meta['tag']}.json", meta)
                for k, v in meta["maha_auroc"].items():
                    d025.at[i, f"maha_{k}"] = v
        d025.to_csv(OUT / "lambda025_per_seed.csv", index=False)
        for probe in ("heldout_linear", "heldout_mlp"):
            spearman[probe] = {}
            for ood in ("pad_heldout", "fitzpatrick17k", "mednode", "sd198"):
                col = f"maha_{ood}"
                if col not in d025.columns or d025[col].isna().all():
                    continue
                m = d025[probe].notna() & d025[col].notna()
                if m.sum() >= 3:
                    rho, p = spearmanr(d025.loc[m, probe], d025.loc[m, col])
                    spearman[probe][ood] = {"rho": float(rho), "p": float(p), "n": int(m.sum())}
        R.write_json(OUT / "lambda025_spearman.json", spearman)

    # finding: held-out vs pad_full story differs?
    findings = []
    for _, c in cdf.iterrows():
        if not np.isfinite(c["heldout_linear_mean"]) or not np.isfinite(c["pad_full_linear_mean"]):
            continue
        gap = abs(c["heldout_linear_mean"] - c["pad_full_linear_mean"])
        # different picture: one near chance (<0.6) other clearly above (≥0.75), or gap≥0.10
        ho, pf = c["heldout_linear_mean"], c["pad_full_linear_mean"]
        diverge = gap >= 0.10 or ((ho < 0.60) != (pf < 0.60) and gap >= 0.05)
        if diverge:
            findings.append({
                "family": c["family"], "lambda": c["lambda"], "shift": c["shift"],
                "heldout_linear": ho, "pad_full_linear": pf, "gap": float(gap),
            })
    R.write_json(OUT / "post_commit_findings.json", {
        "label": "finding made after decisions commit c144f7b / paperB_r5_heldout_probes.md",
        "primary_unchanged": "pad_full remains primary leakage variable (94e2399)",
        "n_divergent_cells": len(findings),
        "cells": findings,
    })

    lines = [
        "# Paper B — held-out domain probes (linear / MLP / kNN)",
        "",
        f"Design: `decisions/paperB_r5_heldout_probes.md`. Commit `{R.git_head()}`.",
        "Protocol: P6 V1 (ISIC-test ∪ pad_heldout, GroupShuffleSplit 70/30 by lesion/patient).",
        "Representation: `z_lesion_norm`. Probe seeds 42/52/62.",
        "",
        "**Primary leakage variable unchanged:** pad_full linear probe (94e2399).",
        "**Secondary:** held-out linear / MLP / kNN. If held-out ≠ pad_full story, both are reported",
        "and labelled a finding made after the decisions commit.",
        "",
        f"Inventory: {len(rows)} runs with cached features. tab:obj / P9 / P11 / R2: deferred until features exist.",
        "",
        "## Cell means (mean ± s.d. over training seeds; bootstrap CI of seed-mean, n≥2)",
        "",
        "| family | shift | λ | n | pad_full linear (primary) | heldout linear | heldout MLP | heldout kNN20 |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for _, c in cdf.iterrows():
        def cell(prefix):
            m, s, lo, hi = c[f"{prefix}_mean"], c[f"{prefix}_sd"], c[f"{prefix}_ci_lo"], c[f"{prefix}_ci_hi"]
            if not np.isfinite(m):
                return "—"
            body = f"{m:.3f} ± {s:.3f}"
            if np.isfinite(lo):
                body += f" [{lo:.3f}, {hi:.3f}]"
            return body
        sh = "" if pd.isna(c["shift"]) else int(c["shift"])
        lam = "" if pd.isna(c["lambda"]) else c["lambda"]
        lines.append(
            f"| {c['family']} | {sh} | {lam} | {int(c['n_seeds'])} | "
            f"{cell('pad_full_linear')} | {cell('heldout_linear')} | {cell('heldout_mlp')} | {cell('heldout_knn20')} |"
        )

    lines += ["", "## λ = 0.25 Table 1 — per seed (prediction: near-chance held-out ⇒ invert)", ""]
    if len(d025):
        lines += [
            "| seed | heldout linear | heldout MLP | Maha pad_heldout | Maha Fitz | Maha MED-NODE | Maha SD-198 |",
            "|---:|---|---|---|---|---|---|",
        ]
        for _, r in d025.iterrows():
            def f(k):
                v = r.get(k)
                return f"{v:.3f}" if v is not None and np.isfinite(v) else "—"
            lines.append(
                f"| {int(r['seed'])} | {f('heldout_linear')} | {f('heldout_mlp')} | "
                f"{f('maha_pad_heldout')} | {f('maha_fitzpatrick17k')} | {f('maha_mednode')} | {f('maha_sd198')} |"
            )
        lines += ["", "### Spearman (descriptive, n = 5)", ""]
        for probe, block in spearman.items():
            for ood, st in block.items():
                lines.append(f"- {probe} vs Maha {ood}: ρ = {st['rho']:.3f} (p = {st['p']:.3g}, n = {st['n']})")
    else:
        lines.append("_Not computed yet (no table1 λ=0.25 per_run files)._")

    lines += ["", "## Post-commit findings (held-out vs pad_full diverge)", ""]
    if findings:
        lines.append(f"{len(findings)} cell(s) where held-out and pad_full tell a different story "
                     "(gap ≥ 0.10, or chance vs non-chance). **pad_full stays primary.**")
        for fnd in findings[:30]:
            lines.append(
                f"- {fnd['family']} shift={fnd['shift']} λ={fnd['lambda']}: "
                f"heldout={fnd['heldout_linear']:.3f}, pad_full={fnd['pad_full_linear']:.3f}, gap={fnd['gap']:.3f}"
            )
    else:
        lines.append("No divergent cells under the gap/chance rule (or not enough runs yet).")

    lines += [
        "",
        "## Status notes",
        "",
        "- Fitzpatrick by skin type: `results/paperB/r5/p7a/fitz_skintype/REPORT.md` (not under p7a/REPORT.md).",
        "- P2 / P4 CPU: already done (`r5/p2/REPORT.md`, `r5/p4/REPORT.md`).",
        "- tab:obj: no `pad_heldout.npz` under `phase15/objectives/` — needs frozen extract before probe.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'REPORT.md'} ({len(rows)} runs)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        report()
    else:
        run_all()


if __name__ == "__main__":
    main()
