#!/usr/bin/env python3
"""Paper B R11 X2: score M1/M2/M3 monitors — AUROC, Rule A, FPR@95TPR, clinical MEL→NV flag.

Design: decisions/paperB_r11_x2.md. CPU after forwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r5_posthoc_scoring as PH  # noqa: E402

OUT = R.RES / "r11" / "x2"
DUAL_FEAT = R.RES / "r5" / "t1last" / "features_last"
ERM_FEAT = R.P8_DIR / "erm_ref" / "features"
M3_FEAT = OUT / "m3_imagenet_effb3"
SUM_CSV = R.RES / "r5" / "t1last" / "summary.csv"

OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198", "scin")
LABELLED = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198")  # SCIN out-of-label per X1
CLINICAL_OODS = ("fitzpatrick17k", "sd198")
DETS = R.DETS
M3_DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max")  # no skin logits
NV = 1
MEL = R.LABELS.index("MEL")
DEG = 0.25

# classifier λ → (seeds, dual tag template)
CLS = {
    0.0: ((42, 52, 62, 72, 82), "runB_orth1_ladv0_s{}"),
    0.5: ((42, 43, 44, 45, 46), "runB_orth1_ladv0p5_s{}"),
    2.0: ((42, 52, 62, 72, 82), "runB_orth1_ladv2_s{}"),
}
ERM_SEEDS = R.SEEDS  # 42/52/62/72/82


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def dual_tag(lam, seed):
    return CLS[lam][1].format(seed)


def erm_tag(seed):
    return "single_dann_ladv0_s{}".format(seed)


def id_bal_map():
    """last.ckpt id_bal_acc per (lambda, seed)."""
    out = {}
    with open(SUM_CSV) as f:
        for r in pd.read_csv(f).to_dict("records"):
            if str(r["checkpoint"]) != "last":
                continue
            out[(float(r["lambda"]), int(r["seed"]))] = float(r["id_bal_acc"])
    return out


def load_pack(feat_dir, name):
    p = Path(feat_dir) / "{}.npz".format(name)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def rates_at_id95(sid, sood):
    tau = float(np.quantile(sid, 0.95))
    return float(np.mean(sood <= tau)), float(np.mean(sood >= tau)), tau


def monitor_feat_dir(monitor, lam, seed):
    """Return feature Path for (monitor, classifier λ, seed), or None if unpaired."""
    if monitor == "M1":
        return DUAL_FEAT / dual_tag(lam, seed)
    if monitor == "M2":
        # same seed as classifier; only 42 overlaps at λ=0.5
        if seed not in ERM_SEEDS:
            return None
        return ERM_FEAT / erm_tag(seed)
    if monitor == "M3":
        return M3_FEAT
    raise ValueError(monitor)


def pool_cell(per_seed):
    """per_seed: {seed: bootstrap_one result}. Aggregate Rule A etc."""
    if not per_seed:
        return None
    seeds = sorted(per_seed)
    out = {"n_seeds": len(seeds), "seeds": seeds, "per_seed": {}}
    for det in next(iter(per_seed.values())):
        pts, draws_list, k = [], [], 0
        for s in seeds:
            d = per_seed[s][det]
            pts.append(d["point"])
            draws_list.append(d["draws"])
            if d["below_chance_entire_ci"]:
                k += 1
            out["per_seed"].setdefault(str(s), {})[det] = {
                "point": d["point"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                "below_chance_entire_ci": d["below_chance_entire_ci"],
            }
        pooled = np.mean(np.stack(draws_list), 0)
        blk = {
            "mean": float(np.mean(pts)),
            "sd": float(np.std(pts, ddof=1)) if len(pts) > 1 else None,
            "ci_lo": float(np.quantile(pooled, 0.025)),
            "ci_hi": float(np.quantile(pooled, 0.975)),
            "k_below": int(k),
            "n_seeds": len(seeds),
        }
        out[det] = blk
    m = out.get("mahalanobis_classcond")
    out["rule_A"] = bool(m and m["ci_hi"] < 0.5 and m["k_below"] >= 3)
    return out


def fpr95_cell(per_seed):
    if not per_seed:
        return None
    fprs, flags = [], []
    for s, blk in per_seed.items():
        d = blk["mahalanobis_classcond"]
        fpr, flagged, tau = rates_at_id95(d["sid"], d["sood"])
        fprs.append(fpr)
        flags.append(flagged)
    return {
        "fpr95tpr_mean": float(np.mean(fprs)),
        "fpr95tpr_sd": float(np.std(fprs, ddof=1)) if len(fprs) > 1 else None,
        "ood_flagged_at_95id_mean": float(np.mean(flags)),
        "ood_flagged_at_95id_sd": float(np.std(flags, ddof=1)) if len(flags) > 1 else None,
        "n_seeds": len(fprs),
    }


def clinical_mel_nv09(cls_logits, y_ood, mon_sid, mon_sood):
    """Share of MEL with pred=NV conf>0.9 that monitor flags at 95% ID τ."""
    y = np.asarray(y_ood, int)
    mel = y == MEL
    if not mel.any():
        return {"n_mel": 0, "n_nv09": 0, "share_flagged": None}
    lg = np.asarray(cls_logits)[mel]
    prob = np.exp(lg - lg.max(1, keepdims=True))
    prob = prob / prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    conf = prob.max(1)
    nv09 = (pred == NV) & (conf > 0.9)
    n_nv09 = int(nv09.sum())
    if n_nv09 == 0:
        return {"n_mel": int(mel.sum()), "n_nv09": 0, "share_flagged": None}
    tau = float(np.quantile(mon_sid, 0.95))
    so = np.asarray(mon_sood)[mel][nv09]
    return {
        "n_mel": int(mel.sum()),
        "n_nv09": n_nv09,
        "share_flagged": float(np.mean(so >= tau)),
        "tau": tau,
    }


def write_report(cells, clinical, id_bal):
    L = [
        "# Paper B R11 X2 — OOD monitors M1 / M2 / M3",
        "",
        "Design: `decisions/paperB_r11_x2.md`. Commit `{}`.".format(R.git_head()),
        "",
        "Monitors: **M1** dual latent; **M2** ERM EffB3 (same seed); **M3** ImageNet EffB3 frozen "
        "(MSP/Energy N/A — no skin head). Primary = Mahalanobis. Rule A = pooled CI upper < 0.5 and k≥3.",
        "Reading: avoids inversion at λ=2 if Rule A on no labelled set and AUROC within 0.03 of own λ=0 "
        "(M3: first condition only).",
        "",
        "## Cells (Mahalanobis)",
        "",
        "| monitor | λ | OOD | AUROC mean±sd | pooled CI | k | n | Rule A | FPR@95TPR |",
        "|---|---:|---|---|---|---:|---:|---|---|",
    ]
    for mon in ("M1", "M2", "M3"):
        for lam in (0.0, 0.5, 2.0):
            for ood in OODS:
                key = "{}|lam{:g}|{}".format(mon, lam, ood)
                c = cells.get(key)
                if not c:
                    L.append("| {} | {:g} | {} | — | — | — | 0 | — | — |".format(mon, lam, ood))
                    continue
                m = c["mahalanobis_classcond"]
                sd = " ± {:.3f}".format(m["sd"]) if m.get("sd") is not None else ""
                f = c.get("fpr95")
                fpr = "—" if not f else "{:.3f}".format(f["fpr95tpr_mean"]) + (
                    " ± {:.3f}".format(f["fpr95tpr_sd"]) if f.get("fpr95tpr_sd") is not None else "")
                L.append(
                    "| {} | {:g} | {} | {:.3f}{} | [{:.3f}, {:.3f}] | {} | {} | {} | {} |".format(
                        mon, lam, ood, m["mean"], sd, m["ci_lo"], m["ci_hi"],
                        m["k_below"], m["n_seeds"], c["rule_A"], fpr))
    L += ["", "## Avoids inversion? (λ=2 vs λ=0, labelled sets)", ""]
    for mon in ("M1", "M2", "M3"):
        rule_hits = []
        within = []
        for ood in LABELLED:
            c2 = cells.get("{}|lam2|{}".format(mon, ood))
            c0 = cells.get("{}|lam0|{}".format(mon, ood))
            if c2 and c2.get("rule_A"):
                rule_hits.append(ood)
            if mon == "M3":
                within.append((ood, True, None))  # λ-independent
            elif c2 and c0:
                d = abs(c2["mahalanobis_classcond"]["mean"] - c0["mahalanobis_classcond"]["mean"])
                within.append((ood, d <= 0.03, d))
        if mon == "M3":
            avoids = (len(rule_hits) == 0)
            L.append("- **M3**: Rule A on labelled = {}; avoids_inversion = {} "
                     "(λ-independent by construction).".format(rule_hits or "none", avoids))
        else:
            ok_d = all(w for _, w, _ in within) if within else False
            avoids = (len(rule_hits) == 0) and ok_d
            diffs = ", ".join("{} Δ={:.3f}".format(o, d) for o, _, d in within if d is not None)
            L.append("- **{}**: Rule A hits = {}; |AUROC_λ2−λ0|≤0.03 on labelled = {}; "
                     "avoids_inversion = {}. ({})".format(
                         mon, rule_hits or "none", ok_d, avoids, diffs))
    L += ["", "## Clinical: λ=2 classifier, MEL → pred NV conf>0.9, share flagged at 95% TPR", ""]
    L += ["| monitor | OOD | n_nv09 (mean) | share flagged mean±sd | n_seeds |",
          "|---|---|---:|---|---:|"]
    for mon in ("M1", "M2", "M3"):
        for ood in CLINICAL_OODS:
            c = clinical.get("{}|{}".format(mon, ood))
            if not c:
                L.append("| {} | {} | — | — | 0 |".format(mon, ood))
                continue
            sd = " ± {:.3f}".format(c["share_sd"]) if c.get("share_sd") is not None else ""
            share = "—" if c["share_mean"] is None else "{:.3f}{}".format(c["share_mean"], sd)
            L.append("| {} | {} | {:.1f} | {} | {} |".format(
                mon, ood, c["n_nv09_mean"], share, c["n_seeds"]))
    L += ["", "## Degraded ID bal_acc < 0.25", ""]
    deg = [(lam, s, v) for (lam, s), v in sorted(id_bal.items()) if v < DEG]
    L.append("None." if not deg else "; ".join("λ={:g} s{}={:.3f}".format(l, s, v) for l, s, v in deg))
    L += ["", "## Notes", "",
          "- λ=0.5 × M2: only seed 42 pairs with erm_ref.",
          "- M3: single feature bank; n_seeds=1 for AUROC cells; MSP/Energy omitted.",
          "- SCIN included for completeness; X1 = Not label space.",
          ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "REPORT.md", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = json.loads(R.PREREG.read_text())
    n_boot, base_seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    id_groups = R.idx_groups(lesion_g)
    ood_groups, ood_units = {}, {}
    for n in OODS:
        g, unit = R.ood_groups(n, data)  # already idx_groups lists
        ood_groups[n], ood_units[n] = g, unit
    print("units", ood_units, flush=True)

    id_bal = id_bal_map()
    y_ood = PH.ood_label_arrays()
    # pad labels from any dual pack
    cells = {}
    per_raw = {}  # keep sid/sood for clinical / fpr

    m3_cache = {}  # ood -> pooled cell (λ-independent)

    for mon in ("M1", "M2", "M3"):
        dets = M3_DETS if mon == "M3" else DETS
        for lam, (seeds, _) in CLS.items():
            for ood in OODS:
                key = "{}|lam{:g}|{}".format(mon, lam, ood)
                if mon == "M3" and ood in m3_cache:
                    cells[key] = m3_cache[ood]
                    continue
                print("cell", key, flush=True)
                per = {}
                degraded = []
                for seed in seeds:
                    bal = id_bal.get((lam, seed))
                    if bal is not None and bal < DEG:
                        degraded.append(seed)
                        continue
                    fd = monitor_feat_dir(mon, lam, seed)
                    if fd is None or not (fd / "isic_train.npz").exists():
                        print("  skip unpaired/missing", mon, lam, seed, fd, flush=True)
                        continue
                    if not (fd / "{}.npz".format(ood)).exists():
                        print("  missing ood feat", fd / "{}.npz".format(ood), flush=True)
                        continue
                    # M3: single feature bank — score once under seed 0
                    per[0 if mon == "M3" else seed] = (fd, bal)
                    if mon == "M3":
                        break
                if not per:
                    cells[key] = None
                    continue
                # shared bootstrap draws across seeds in cell
                rng = np.random.default_rng(base_seed)
                id_draws = [np.concatenate([id_groups[j] for j in rng.integers(0, len(id_groups), len(id_groups))])
                            for _ in range(n_boot)]
                og = ood_groups[ood]
                oo_draws = [np.concatenate([og[j] for j in rng.integers(0, len(og), len(og))])
                            for _ in range(n_boot)]
                per_seed = {}
                for seed, (fd, bal) in per.items():
                    Fs = R.load_feats(fd, ("isic_train", "isic_test", ood))
                    S = R.Scorer(Fs["isic_train"])
                    sid, so = S.scores(Fs["isic_test"]), S.scores(Fs[ood])
                    blk = {}
                    for det in dets:
                        a, b = sid[det], so[det]
                        draws = np.array([R.auroc(a[id_draws[i]], b[oo_draws[i]]) for i in range(n_boot)])
                        lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
                        blk[det] = {
                            "point": R.auroc(a, b), "ci_lo": lo, "ci_hi": hi,
                            "below_chance_entire_ci": hi < 0.5, "draws": draws,
                            "sid": a, "sood": b,
                        }
                    per_seed[seed] = blk
                pooled = pool_cell(per_seed)
                if pooled:
                    pooled["fpr95"] = fpr95_cell(per_seed)
                    pooled["degraded_seeds"] = degraded
                    if mon == "M3":
                        pooled["excluded_unpaired"] = []
                        pooled["note"] = "seed-independent ImageNet EffB3; n_seeds=1"
                    else:
                        pooled["excluded_unpaired"] = [s for s in seeds if s not in per_seed and s not in degraded]
                cells[key] = pooled
                per_raw[key] = per_seed
                if mon == "M3":
                    m3_cache[ood] = pooled

    # Clinical: λ=2 classifier logits × each monitor
    clinical = {}
    lam = 2.0
    seeds = CLS[lam][0]
    for mon in ("M1", "M2", "M3"):
        for ood in CLINICAL_OODS:
            shares, ns = [], []
            for seed in seeds:
                bal = id_bal.get((lam, seed))
                if bal is not None and bal < DEG:
                    continue
                cls_fd = DUAL_FEAT / dual_tag(lam, seed)
                mon_fd = monitor_feat_dir(mon, lam, seed)
                if mon_fd is None or not (cls_fd / "{}.npz".format(ood)).exists():
                    continue
                if not (mon_fd / "{}.npz".format(ood)).exists():
                    continue
                cls_pack = load_pack(cls_fd, ood)
                mon_tr = load_pack(mon_fd, "isic_train")
                mon_te = load_pack(mon_fd, "isic_test")
                mon_ood = load_pack(mon_fd, ood)
                S = R.Scorer(mon_tr)
                sid = S.scores(mon_te)["mahalanobis_classcond"]
                sood = S.scores(mon_ood)["mahalanobis_classcond"]
                y = y_ood[ood] if y_ood[ood] is not None else mon_ood["labels"].astype(int)
                rec = clinical_mel_nv09(cls_pack["logits"], y, sid, sood)
                if rec["share_flagged"] is not None:
                    shares.append(rec["share_flagged"])
                ns.append(rec["n_nv09"])
                print("clinical", mon, ood, seed, rec, flush=True)
            if not ns:
                clinical["{}|{}".format(mon, ood)] = None
            else:
                clinical["{}|{}".format(mon, ood)] = {
                    "n_nv09_mean": float(np.mean(ns)),
                    "share_mean": float(np.mean(shares)) if shares else None,
                    "share_sd": float(np.std(shares, ddof=1)) if len(shares) > 1 else None,
                    "n_seeds": len(ns),
                    "shares": shares,
                }

    # Drop draws from JSON (huge)
    cells_out = {}
    for k, v in cells.items():
        if v is None:
            cells_out[k] = None
            continue
        cells_out[k] = {kk: vv for kk, vv in v.items() if kk != "per_seed"}
        # keep per_seed without draws/sid/sood
        ps = {}
        for s, blk in v.get("per_seed", {}).items():
            ps[s] = {det: {x: blk[det][x] for x in ("point", "ci_lo", "ci_hi", "below_chance_entire_ci")
                           if x in blk[det]} for det in blk}
        cells_out[k]["per_seed"] = ps

    payload = {
        "design": "decisions/paperB_r11_x2.md",
        "commit": R.git_head(),
        "cells": cells_out,
        "clinical": clinical,
        "id_bal": { "{}|{}".format(l, s): v for (l, s), v in id_bal.items()},
        "ood_units": ood_units,
    }
    (OUT / "RESULTS.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    write_report(cells, clinical, id_bal)
    print("DONE_X2_SCORE", flush=True)


if __name__ == "__main__":
    main()
