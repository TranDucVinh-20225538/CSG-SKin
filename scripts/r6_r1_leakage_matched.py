#!/usr/bin/env python3
"""Paper B R6 R1 leakage-matched test (a)/(b)/(c) — decisions/paperB_r6_r1.md @ 94e2399.

Pool: R1 three-stream (shift × λ∈{0.5,2}) + Table 1 old recipe + A4 (λ∈{0,0.5,2}).
Leakage = pad_full Table 1 probe (bal_acc_mean). Degraded R1 (ID bal < 0.25) excluded from (a)/(b).
CPU only. Outputs: results/paperB/r6/r1/cur/leakage_matched/
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r6_common as R6  # noqa: E402

OUT = R6.RES / "r6" / "r1" / "cur" / "leakage_matched"
PRIMARY = ("fitzpatrick17k", "mednode", "sd198")
ALL_SETS = PRIMARY + ("pad_heldout",)
RNG = 20261005
N_BOOT = 2000
DEGRADED = 0.25


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None or isinstance(x, str):
        return x
    return x


def leak_of(summary):
    m = summary.get("metrics", summary)
    leak = m.get("leakage") or {}
    if isinstance(leak, dict):
        # Table 1 definition: balanced accuracy of the domain probe
        for k in ("bal_acc_mean", "bal_acc", "auroc_mean"):
            if k in leak and leak[k] is not None:
                return float(leak[k])
    if isinstance(leak, (int, float)):
        return float(leak)
    return None


def maha_of(summary, ood):
    m = summary.get("metrics", summary)
    o = (m.get("ood") or {}).get(ood)
    if o is None:
        return None
    if isinstance(o, dict):
        v = o.get("mahalanobis_classcond", o.get("Mahalanobis"))
        if isinstance(v, dict):
            return float(v.get("auroc", v.get("mean")))
        if isinstance(v, (int, float)):
            return float(v)
    return None


def id_bal_of(summary):
    m = summary.get("metrics", summary)
    idb = m.get("id") or {}
    if isinstance(idb, dict) and "bal_acc" in idb:
        return float(idb["bal_acc"])
    return None


def load_r1():
    root = R6.RES / "r6" / "r1" / "cur"
    rows = []
    for d in sorted((root / "features").glob("r1_cur_shift*_ladv*_s*")):
        m = re.fullmatch(r"r1_cur_shift(\d+)_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        shift, lam, seed = int(m.group(1)), float(m.group(2).replace("p", ".")), int(m.group(3))
        if lam not in (0.5, 2.0):
            continue
        sj = root / "runs" / d.name / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        idb = id_bal_of(s)
        deg = bool(idb is not None and idb < DEGRADED)
        leak = leak_of(s)
        rec = {"recipe": "R1", "lambda": lam, "shift_pct": shift, "shift_level": shift / 100.0,
               "seed": seed, "tag": d.name, "leakage": leak, "degraded": deg, "id_bal": idb}
        for o in ALL_SETS:
            rec["maha_{}".format(o)] = maha_of(s, o)
        rows.append(rec)
    return rows


def load_a4():
    root = R6.P8_DIR / "main"
    rows = []
    for d in sorted((root / "features").glob("main_ladv*_s*")):
        m = re.fullmatch(r"main_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        lam, seed = float(m.group(1).replace("p", ".")), int(m.group(2))
        if lam not in (0.0, 0.5, 2.0):
            continue
        sj = root / "runs" / d.name / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        rec = {"recipe": "A4", "lambda": lam, "shift_pct": 100, "shift_level": 1.0,
               "seed": seed, "tag": d.name, "leakage": leak_of(s), "degraded": False,
               "id_bal": id_bal_of(s)}
        for o in ALL_SETS:
            rec["maha_{}".format(o)] = maha_of(s, o)
        rows.append(rec)
    return rows


def load_table1():
    """Old recipe item4: summaries under r2/item4/runs or features + phase13 JSON."""
    rows = []
    feat = R6.ITEM4_FEAT
    run_root = R6.RES / "r2" / "item4" / "runs"
    for d in sorted(feat.glob("runB_orth1_ladv*_s*")):
        m = re.fullmatch(r"runB_orth1_ladv([0-9p]+)_s(\d+)", d.name)
        if not m:
            continue
        lam, seed = float(m.group(1).replace("p", ".")), int(m.group(2))
        sj = run_root / d.name / "summary.json"
        s = None
        if sj.exists():
            s = json.loads(sj.read_text())
        else:
            # fall back to scoring from features (pad_full leakage + maha)
            need = ("isic_train", "isic_test", "pad_full") + ALL_SETS
            if not all((d / "{}.npz".format(n)).exists() for n in need):
                continue
            Fs = R6.load_feats(d, need)
            met = R6.run_metrics(Fs, ood_names=list(ALL_SETS) + ["pad_full"])
            s = {"metrics": met}
        leak = leak_of(s)
        # item4 leakage sometimes under different key
        if leak is None and isinstance(s.get("metrics"), dict):
            t1 = s["metrics"].get("leakage") or s.get("leakage")
            if isinstance(t1, dict):
                leak = float(t1.get("bal_acc_mean") or t1.get("bal_acc") or t1.get("auroc_mean") or "nan")
                if not np.isfinite(leak):
                    leak = None
        rec = {"recipe": "Table1", "lambda": lam, "shift_pct": 100, "shift_level": 1.0,
               "seed": seed, "tag": d.name, "leakage": leak, "degraded": False,
               "id_bal": id_bal_of(s) if s else None}
        for o in ALL_SETS:
            rec["maha_{}".format(o)] = maha_of(s, o) if s else None
            if rec["maha_{}".format(o)] is None and (d / "{}.npz".format(o)).exists():
                # score on the fly once
                pass
        rows.append(rec)
    # fill missing maha/leak from features if needed
    for rec in rows:
        if rec["leakage"] is not None and all(rec.get("maha_{}".format(o)) is not None for o in ALL_SETS):
            continue
        d = feat / rec["tag"]
        need = ("isic_train", "isic_test", "pad_full") + ALL_SETS
        if not all((d / "{}.npz".format(n)).exists() for n in need):
            continue
        Fs = R6.load_feats(d, need)
        met = R6.run_metrics(Fs, ood_names=list(ALL_SETS) + ["pad_full"])
        if rec["leakage"] is None:
            rec["leakage"] = leak_of({"metrics": met})
        for o in ALL_SETS:
            if rec.get("maha_{}".format(o)) is None:
                rec["maha_{}".format(o)] = met.get("ood", {}).get(o, {}).get("mahalanobis_classcond")
    return rows


def setting_key(r):
    return (r["recipe"], r["lambda"], r["shift_level"])


def design_matrix(df):
    """y ~ leak + shift + recipe(R1/Table1/A4). Drop one recipe (R1)."""
    recipes = sorted(df["recipe"].unique())
    # encode: intercept + leak + shift + dummies for recipes except first
    base = recipes[0]
    Xcols = ["leakage", "shift_level"] + ["recipe_{}".format(r) for r in recipes if r != base]
    X = np.column_stack([
        np.ones(len(df)),
        df["leakage"].to_numpy(float),
        df["shift_level"].to_numpy(float),
    ] + [(df["recipe"] == r).to_numpy(float) for r in recipes if r != base])
    names = ["intercept", "leakage", "shift_level"] + ["recipe_{}".format(r) for r in recipes if r != base]
    return X, names, base


def ols(X, y):
    beta, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def vif_pairwise(a, b):
    """VIF of regressing a on b (and intercept): 1/(1-R^2)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    X = np.column_stack([np.ones(len(b)), b])
    beta, _, _, _ = lstsq(X, a, rcond=None)
    pred = X @ beta
    ss_tot = np.sum((a - a.mean()) ** 2)
    ss_res = np.sum((a - pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - r2))


def webb_weights(n, rng):
    """Webb six-point weights (mean 0, var 1)."""
    # values ±√(3/2), ±1, ±√(1/2) with equal prob 1/6... common simplification: Rademacher ±1
    # Decision: Webb weights — use classic Webb: ±√1.5 with p=1/2? 
    # Davidson & MacKinnon Webb: weights in {±√(3/2), ±√(1/2), ±√(3/2)*something}
    # Standard Webb (2014): P(w=±√(3/2))=1/2? Actually: six points
    # w ∈ {±√(3/2), ±1, ±√(1/2)} each with probability 1/6? Mean of squares = (3/2+1+1/2)/3 = 1.
    vals = np.array([np.sqrt(1.5), -np.sqrt(1.5), 1.0, -1.0, np.sqrt(0.5), -np.sqrt(0.5)])
    return rng.choice(vals, size=n, replace=True)


def wild_cluster_ci(df, ycol, n_boot=N_BOOT, seed=RNG):
    """OLS + wild cluster bootstrap CI for shift_level coefficient. Clusters = setting cells."""
    d = df.dropna(subset=[ycol, "leakage"]).copy()
    d["cluster"] = d.apply(lambda r: "{}|{:g}|{}".format(r["recipe"], r["lambda"], r["shift_level"]), axis=1)
    X, names, base = design_matrix(d)
    y = d[ycol].to_numpy(float)
    beta, resid = ols(X, y)
    # map rows to clusters
    clusters = d["cluster"].to_numpy()
    uniq = sorted(set(clusters))
    rng = np.random.default_rng(seed)
    betas = np.zeros((n_boot, len(beta)))
    for b in range(n_boot):
        w_c = {c: webb_weights(1, rng)[0] for c in uniq}
        w = np.array([w_c[c] for c in clusters])
        y_star = X @ beta + w * resid
        betas[b], _ = ols(X, y_star)
    lo = np.percentile(betas, 2.5, axis=0)
    hi = np.percentile(betas, 97.5, axis=0)
    out = {}
    for i, name in enumerate(names):
        out[name] = {"coef": float(beta[i]), "ci_lo": float(lo[i]), "ci_hi": float(hi[i]),
                     "excludes_0": bool(lo[i] > 0 or hi[i] < 0)}
    # VIFs for shift
    vif_shift_leak = vif_pairwise(d["shift_level"], d["leakage"])
    # recipe as numeric codes for VIF vs shift
    recipe_code = pd.Categorical(d["recipe"]).codes.astype(float)
    vif_shift_recipe = vif_pairwise(d["shift_level"], recipe_code)
    return {
        "n": int(len(d)), "recipe_base": base, "coefs": out,
        "vif_shift_vs_leakage": vif_shift_leak,
        "vif_shift_vs_recipe": vif_shift_recipe,
        "identifiable": bool(vif_shift_leak <= 5 and vif_shift_recipe <= 5),
        "supported_direction": bool(
            out["shift_level"]["coef"] < 0 and out["shift_level"]["ci_hi"] < 0
        ),
    }


def matched_b(df, ood, n_boot=N_BOOT, seed=RNG):
    """(b) R1 shift0 λ=2 vs A4 λ=2: mean Maha difference (R1 − A4). Seed bootstrap."""
    a = df[(df.recipe == "R1") & (df.shift_level == 0.0) & (df["lambda"] == 2.0) & (~df.degraded)]
    b = df[(df.recipe == "A4") & (df["lambda"] == 2.0)]
    ya = a["maha_{}".format(ood)].dropna().to_numpy(float)
    yb = b["maha_{}".format(ood)].dropna().to_numpy(float)
    if len(ya) < 2 or len(yb) < 2:
        return None
    point = float(ya.mean() - yb.mean())
    rng = np.random.default_rng(seed)
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        diffs[i] = ya[rng.integers(0, len(ya), len(ya))].mean() - yb[rng.integers(0, len(yb), len(yb))].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # predicted: higher shift → lower AUROC ⇒ R1 shift0 should be HIGHER than A4 (shift100) ⇒ point > 0
    return {
        "n_r1": int(len(ya)), "n_a4": int(len(yb)),
        "mean_r1": float(ya.mean()), "mean_a4": float(yb.mean()),
        "diff_r1_minus_a4": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "excludes_0": bool(lo > 0 or hi < 0),
        "supported_direction": bool(point > 0 and lo > 0),
        "leakage_r1": float(a["leakage"].mean()), "leakage_a4": float(b["leakage"].mean()),
        "label": "cross-recipe",
    }


def plot_c(df, out_dir):
    shift_colors = {0.0: "#1b9e77", 0.5: "#d95f02", 1.0: "#7570b3"}
    markers = {"R1": "o", "A4": "s", "Table1": "^"}
    for ood in ALL_SETS:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for _, r in df.iterrows():
            y = r.get("maha_{}".format(ood))
            if y is None or r["leakage"] is None or not np.isfinite(y):
                continue
            c = shift_colors.get(r["shift_level"], "#999999")
            m = markers.get(r["recipe"], "o")
            edge = "red" if r["degraded"] else "none"
            ax.scatter(r["leakage"], y, c=c, marker=m, s=36, edgecolors=edge, linewidths=0.8, alpha=0.85)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("pad_full leakage (bal acc)")
        ax.set_ylabel("Mahalanobis AUROC")
        title = ood
        if ood == "pad_heldout":
            title += " (adversary target; not counted)"
        ax.set_title(title)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.2, 1.0)
        # legend proxies
        for sh, col in shift_colors.items():
            ax.scatter([], [], c=col, label="shift={:g}".format(sh))
        for rec, mk in markers.items():
            ax.scatter([], [], c="k", marker=mk, label=rec)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "plot_{}.png".format(ood), dpi=140)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_r1() + load_a4() + load_table1()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pool.csv", index=False)
    # exclude degraded from (a)/(b)
    d_ab = df[~df.degraded].copy()

    reg = {}
    for ood in ALL_SETS:
        ycol = "maha_{}".format(ood)
        reg[ood] = wild_cluster_ci(d_ab, ycol)
        print("reg", ood, reg[ood]["coefs"]["shift_level"], "VIF",
              reg[ood]["vif_shift_vs_leakage"], reg[ood]["vif_shift_vs_recipe"],
              "ident", reg[ood]["identifiable"], flush=True)

    matched = {ood: matched_b(d_ab, ood) for ood in PRIMARY}
    matched["pad_heldout"] = matched_b(d_ab, "pad_heldout")

    n_a = sum(1 for o in PRIMARY if reg[o]["identifiable"] and reg[o]["supported_direction"])
    n_a_ident = sum(1 for o in PRIMARY if reg[o]["identifiable"])
    a_supported = n_a >= 2
    a_not_ident = n_a_ident < 3 and any(not reg[o]["identifiable"] for o in PRIMARY)
    # if any primary has VIF>5, mark (a) not identifiable for that set; overall gate
    any_not_ident = any(not reg[o]["identifiable"] for o in PRIMARY)

    n_b = sum(1 for o in PRIMARY if matched[o] and matched[o]["supported_direction"])
    b_supported = n_b >= 2

    plot_c(df, OUT)

    # reading
    if any_not_ident:
        a_verdict = "not identifiable (VIF>5 on at least one primary set) — do not read as a null shift effect"
    elif a_supported:
        a_verdict = "supported (≥2/3 primary sets: shift coef negative, CI excludes 0)"
    else:
        a_verdict = "not supported"

    if b_supported:
        b_verdict = "supported (≥2/3: R1 shift0 λ2 > A4 λ2, CI excludes 0)"
    else:
        b_verdict = "not supported"

    if (not any_not_ident and not a_supported) or (any_not_ident and not b_supported):
        if any_not_ident and not b_supported:
            reading = ("(a) not identifiable and (b) null → report leakage / invariance depth as the main "
                       "variable, not adversary label shift (prediction branch).")
        elif not a_supported and not b_supported:
            reading = ("shift effect null under (a) and (b) → report leakage / invariance depth as the main "
                       "variable, not adversary label shift.")
        else:
            reading = "mixed; see per-set tables."
    elif a_supported or b_supported:
        reading = "label shift lowers Mahalanobis AUROC at matched leakage on the unseen primary sets (prediction)."
    else:
        reading = "see tables."

    payload = {
        "commit": R.git_head(),
        "decision": "decisions/paperB_r6_r1.md (94e2399 leakage-matched test)",
        "n_pool": int(len(df)),
        "n_ab": int(len(d_ab)),
        "n_degraded_excluded": int(df.degraded.sum()),
        "regression_a": reg,
        "matched_b": matched,
        "verdict_a": a_verdict,
        "verdict_b": b_verdict,
        "n_primary_a_supported": n_a,
        "n_primary_b_supported": n_b,
        "reading": reading,
        "identification_note": (
            "The shift coefficient is identified from variation within R1; A4 and Table 1 only extend "
            "the leakage range at shift = 100%."
        ),
    }
    (OUT / "regression.json").write_text(json.dumps(_jsonable(payload["regression_a"]), indent=1))
    (OUT / "matched.json").write_text(json.dumps(_jsonable(payload["matched_b"]), indent=1))
    (OUT / "results.json").write_text(json.dumps(_jsonable(payload), indent=1))

    L = [
        "# Paper B R6 R1 — leakage-matched test (label shift vs invariance depth)",
        "",
        "Design: `decisions/paperB_r6_r1.md` (committed `94e2399`). Analysis commit `{}`. "
        "Pool: R1 (shift×λ∈{{0.5,2}}) + Table 1 + A4 (λ∈{{0,0.5,2}}). Leakage = pad_full probe "
        "**bal_acc_mean**. Degraded R1 (ID bal < 0.25) excluded from (a)/(b); marked on (c) plots.".format(
            payload["commit"]),
        "",
        payload["identification_note"],
        "",
        "## (a) Regression — Mahalanobis AUROC ~ leakage + shift_level + recipe",
        "",
        "Wild cluster bootstrap (Webb weights), cluster = (recipe, λ, shift), {} draws, rng {}. "
        "VIF gate: if shift vs recipe or shift vs leakage VIF > 5 → not identifiable.".format(N_BOOT, RNG),
        "",
        "**Verdict (a):** {}".format(a_verdict),
        "",
        "| set | n | shift coef | 95% CI | excludes 0 | direction (−) | VIF shift|leak | VIF shift|recipe | identifiable |",
        "|---|---:|---:|---|:---:|:---:|---:|---:|:---:|",
    ]
    for ood in ALL_SETS:
        r = reg[ood]
        c = r["coefs"]["shift_level"]
        note = " (not counted)" if ood == "pad_heldout" else ""
        L.append(
            "| {}{} | {} | {:.3f} | [{:.3f}, {:.3f}] | {} | {} | {:.2f} | {:.2f} | {} |".format(
                ood, note, r["n"], c["coef"], c["ci_lo"], c["ci_hi"],
                "yes" if c["excludes_0"] else "no",
                "yes" if r["supported_direction"] else "no",
                r["vif_shift_vs_leakage"], r["vif_shift_vs_recipe"],
                "yes" if r["identifiable"] else "no"))
    L += [
        "",
        "## (b) Matched comparison (cross-recipe) — R1 shift 0% λ=2 vs A4 λ=2",
        "",
        "Difference = mean Maha(R1 shift0 λ2) − mean Maha(A4 λ2). Seed bootstrap {} draws. "
        "Supported if diff > 0 with CI excluding 0 on ≥ 2/3 primary sets.".format(N_BOOT),
        "",
        "**Verdict (b):** {}".format(b_verdict),
        "",
        "| set | n R1 / A4 | leakage R1 / A4 | Maha R1 / A4 | diff | 95% CI | supported |",
        "|---|---|---|---|---:|---|:---:|",
    ]
    for ood in list(PRIMARY) + ["pad_heldout"]:
        m = matched.get(ood)
        if not m:
            L.append("| {} | — | — | — | — | — | — |".format(ood))
            continue
        note = " (not counted)" if ood == "pad_heldout" else ""
        L.append(
            "| {}{} | {} / {} | {:.3f} / {:.3f} | {:.3f} / {:.3f} | {:.3f} | [{:.3f}, {:.3f}] | {} |".format(
                ood, note, m["n_r1"], m["n_a4"], m["leakage_r1"], m["leakage_a4"],
                m["mean_r1"], m["mean_a4"], m["diff_r1_minus_a4"], m["ci_lo"], m["ci_hi"],
                "yes" if m["supported_direction"] else "no"))
    L += [
        "",
        "## (c) Plots",
        "",
        "Mahalanobis AUROC vs pad_full leakage; colour = shift level; marker = recipe; "
        "red edge = degraded (excluded from a/b).",
        "",
    ]
    for ood in ALL_SETS:
        L.append("- `plot_{}.png`".format(ood))
    L += ["", "## Reading", "", reading, ""]
    (OUT / "REPORT.md").write_text("\n".join(L))

    # append short section pointer into main R1 REPORT if present
    main = R6.RES / "r6" / "r1" / "cur" / "REPORT.md"
    if main.exists():
        text = main.read_text()
        marker = "## Leakage-matched test (94e2399)"
        section = (
            "\n\n{}\n\n"
            "See `leakage_matched/REPORT.md` (commit {}). "
            "Verdict (a): {}. Verdict (b): {}. {}\n"
        ).format(marker, payload["commit"], a_verdict, b_verdict, reading)
        if marker in text:
            # replace from marker to EOF or next ## at end — append after removing old
            pre = text.split(marker)[0].rstrip()
            text = pre + section
        else:
            text = text.rstrip() + section
        main.write_text(text)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
