#!/usr/bin/env python3
"""Paper B R12 scoring: stratified AUROC, FPR@95TPR, leakage, paired contrasts, historical rescore.

Run AFTER training. Does not change seeds/priors/schedules.
Decisions: decisions/paperB_r12_reweight.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r12_reweight_common as C  # noqa: E402

DET = "mahalanobis_classcond"
PRIMARY_UNSEEN = C.OOD_PRIMARY_UNSEEN
ALL_OOD = C.OOD_ALL
# Historical Fitzpatrick baselines attached to their series (do not swap):
# R1 shift100 λ2 mean Fitz = 0.340; R1 shift0 λ2 Fitz mean in recovery table context = 0.405 (R1 REPORT).
HIST_FITZ_R1_SHIFT100_L2 = 0.340
HIST_FITZ_R1_SHIFT0_L2 = 0.405


def rates_at_id95(sid, sood):
    """FPR@95TPR = fraction of OOD accepted as ID (score ≤ τ); higher score = more OOD."""
    tau = float(np.quantile(sid, 0.95))
    return float(np.mean(sood <= tau)), float(np.mean(sood >= tau)), tau


def stratified_auroc(sid, yid, sood):
    """Each ID class vs SAME OOD panel; macro over fixed 8-class list (nan if class absent)."""
    yid = np.asarray(yid, int)
    per = {}
    for c, name in enumerate(R.LABELS):
        m = yid == c
        if m.sum() == 0:
            per[name] = None
            continue
        y = np.r_[np.zeros(int(m.sum())), np.ones(len(sood))]
        s = np.r_[sid[m], sood]
        per[name] = float(roc_auc_score(y, s))
    vals = [v for v in per.values() if v is not None]
    return {"per_class": per, "macro": float(np.mean(vals)) if vals else None,
            "n_classes_used": len(vals), "eligible_list": list(R.LABELS)}


def load_pack(feat_dir, name):
    p = Path(feat_dir) / "{}.npz".format(name)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def score_features(feat_dir, seed, tag, extra=None):
    feat_dir = Path(feat_dir)
    tr = load_pack(feat_dir, "isic_train")
    te = load_pack(feat_dir, "isic_test")
    if tr is None or te is None:
        return {"tag": tag, "seed": seed, "status": "missing_features", "feat_dir": str(feat_dir)}
    sc = R.Scorer(tr)
    sid = sc.scores(te)
    id_bal = float(R.run_metrics({"isic_train": tr, "isic_test": te})["id"]["bal_acc"])
    # leakage if pad available
    Fs = {"isic_train": tr, "isic_test": te}
    for n in ("pad_full", "pad_heldout", "pad_adv", "fitzpatrick17k") + R.P7A_NAMES:
        pk = load_pack(feat_dir, n)
        if pk is not None:
            Fs[n] = pk
    full = R.run_metrics(Fs)
    out = {
        "tag": tag, "seed": int(seed), "status": "ok", "feat_dir": str(feat_dir),
        "id_bal": id_bal, "degraded": bool(id_bal < R.DEGRADED_ID_BAL),
        "leakage_v1": full.get("leakage"),
        "leakage_classmatched": full.get("leakage_classmatched"),
        "ood": {},
    }
    for ood in ALL_OOD:
        if ood not in Fs:
            out["ood"][ood] = {"status": "missing"}
            continue
        so = sc.scores(Fs[ood])
        pooled = float(R.auroc(sid[DET], so[DET]))
        fpr, tpr_flag, tau = rates_at_id95(sid[DET], so[DET])
        strat = stratified_auroc(sid[DET], te["labels"], so[DET])
        # save per-image scores
        score_path = C.OUT / "scores" / tag / "{}_{}.npz".format(ood, DET)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            sid=sid[DET], sood=so[DET], id_labels=te["labels"].astype(int),
            id_groups=te["labels"].astype(int),  # placeholder; groups filled if present
            ood_n=len(so[DET]), seed=int(seed), tag=tag, detector=DET,
            tau95=tau, fpr95tpr_ood_accepted=fpr,
        )
        cell = {
            "status": "ok", "pooled_auroc": pooled,
            "stratified": strat,
            "fpr95tpr_ood_accepted": fpr, "ood_flagged_at_95id": tpr_flag, "tau95": tau,
            "n_ood": int(len(so[DET])),
            "nv_nearest_frac": full["ood"][ood].get("nv_nearest_frac"),
            "nearest_class_share": full["ood"][ood].get("nearest_class_share"),
            "score_path": str(score_path),
        }
        if "decomposition_maha" in full["ood"][ood]:
            cell["decomposition_maha"] = full["ood"][ood]["decomposition_maha"]
        out["ood"][ood] = cell
    if extra:
        out.update(extra)
    return out


def cell_key(block, lam, prior, init="scratch"):
    return "{}_ladv{}_{}_init{}".format(block, R.lam_tag(lam), prior, init)


def aggregate_seeds(per_seed_dicts, ood):
    rows = [d for d in per_seed_dicts if d.get("status") == "ok" and d["ood"].get(ood, {}).get("status") == "ok"]
    if not rows:
        return {"status": "missing", "n_seeds": 0}
    pooled = [r["ood"][ood]["pooled_auroc"] for r in rows]
    macro = [r["ood"][ood]["stratified"]["macro"] for r in rows]
    fpr = [r["ood"][ood]["fpr95tpr_ood_accepted"] for r in rows]
    idb = [r["id_bal"] for r in rows]
    seeds = [r["seed"] for r in rows]

    def ms(xs):
        xs = np.asarray(xs, float)
        return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
                "per_seed": {int(s): float(v) for s, v in zip(seeds, xs)}}

    # bootstrap: feat_dirs must be {seed: Path}; units = patient/P13/group — not between-seed CI
    boots = {}
    try:
        data = R.load_data()
        lesion_g, _ = R.T.lesion_groups(data["test"])
        id_g = R.idx_groups(lesion_g)
        id_unit = "lesion"
        o_g, o_unit = R.ood_groups(ood, data)
        feat_map = {int(r["seed"]): Path(r["feat_dir"]) for r in rows}
        boots = R.bootstrap_cell(feat_map, ood, id_g, o_g) or {}
        boots["id_unit"] = id_unit
        boots["ood_unit"] = o_unit
    except Exception as exc:  # noqa: BLE001
        boots = {"error": repr(exc)}

    rule_a = False
    if isinstance(boots, dict) and boots.get("rule_A") is not None:
        rule_a = bool(boots["rule_A"])
    elif isinstance(boots, dict) and DET in boots:
        b = boots[DET]
        hi = b.get("ci_hi")
        k = b.get("k_below")
        rule_a = bool(hi is not None and hi < 0.5 and k is not None and k >= 3)

    return {
        "status": "ok", "n_seeds": len(rows), "seeds": seeds,
        "pooled": ms(pooled), "strat_macro": ms(macro), "fpr95tpr": ms(fpr), "id_bal": ms(idb),
        "bootstrap": {DET: boots.get(DET) if isinstance(boots, dict) else None,
                      "units_note": "cluster/patient bootstrap over images within seeds pooled; NOT between-seed CI",
                      "error": boots.get("error") if isinstance(boots, dict) else None},
        "rule_A": rule_a,
        "leakage_v1_mean": float(np.mean([r["leakage_v1"]["bal_acc_mean"] for r in rows if r.get("leakage_v1")]))
        if any(r.get("leakage_v1") for r in rows) else None,
        "leakage_classmatched_mean": float(np.mean(
            [r["leakage_classmatched"]["bal_acc_mean"] for r in rows if r.get("leakage_classmatched")]))
        if any(r.get("leakage_classmatched") for r in rows) else None,
    }


def paired_contrast(a_rows, b_rows, ood, label):
    """a − b per matched seed (e.g. prior0 − prior100)."""
    by_a = {r["seed"]: r for r in a_rows if r.get("status") == "ok"}
    by_b = {r["seed"]: r for r in b_rows if r.get("status") == "ok"}
    seeds = sorted(set(by_a) & set(by_b))
    diffs = []
    for s in seeds:
        oa, ob = by_a[s]["ood"].get(ood, {}), by_b[s]["ood"].get(ood, {})
        if oa.get("status") != "ok" or ob.get("status") != "ok":
            continue
        diffs.append(oa["pooled_auroc"] - ob["pooled_auroc"])
    if not diffs:
        return {"label": label, "ood": ood, "status": "missing", "n": 0}
    d = np.asarray(diffs, float)
    return {
        "label": label, "ood": ood, "status": "ok", "n": len(diffs), "seeds": seeds,
        "diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "per_seed": {int(s): float(v) for s, v in zip(seeds, d)},
    }


def resolve_run(job, reference_map):
    """Return (feat_dir, status, note). Reference rows point at baseline features."""
    if job.get("role") == "reference":
        ref = job["reference_of"]
        feat = C.OUT / "features" / ref
        summ = C.OUT / "runs" / ref / "summary.json"
        if not summ.exists():
            return None, "blocked", "baseline missing: " + ref
        return feat, "reference", "features from " + ref
    feat = C.OUT / "features" / job["tag"]
    summ = C.OUT / "runs" / job["tag"] / "summary.json"
    if not summ.exists():
        return None, "failed" if (C.CKPT / job["tag"]).exists() else "pending", "no summary"
    s = json.loads(summ.read_text())
    if s.get("status") == "invalid":
        return None, "invalid", "zero source support"
    return feat, "ok", None


def historical_rescore():
    """Rescore R1 (resampling), R2 IWDAN/oracle, and Table-1 dual where features exist. No imputation."""
    out = {"r1_resampling": {}, "phase3_main": {}, "iwdan": {}, "iwdan_oracle": {}, "missing": []}
    r1_root = R.RES / "r6" / "r1" / "cur" / "features"
    for prior_pct, prior in ((100, "prior100"), (50, "prior50"), (0, "prior0")):
        for lam in (0.0, 0.5, 2.0):
            rows = []
            for seed in C.SEEDS:
                tag = "r1_cur_shift{}_ladv{}_s{}".format(prior_pct, R.lam_tag(lam), seed)
                fd = r1_root / tag
                if not fd.exists():
                    out["missing"].append(str(fd))
                    continue
                rows.append(score_features(fd, seed, "hist_" + tag,
                                           extra={"series": "r1_resample", "prior": prior}))
            key = "shift{}_ladv{}".format(prior_pct, R.lam_tag(lam))
            out["r1_resampling"][key] = {
                "per_seed_n": len(rows),
                "agg": {ood: aggregate_seeds(rows, ood) for ood in ALL_OOD},
            }

    # Table-1 / phase3 dual (main) features under r2/item4 when present
    item4 = R.RES / "r2" / "item4" / "features"
    for lam in (0.0, 2.0):
        rows = []
        for seed in C.SEEDS:
            tag = "runB_orth1_ladv{}_s{}".format(R.lam_tag(lam), seed)
            fd = item4 / tag
            if not fd.exists():
                out["missing"].append(str(fd))
                continue
            rows.append(score_features(fd, seed, "hist_" + tag, extra={"series": "table1_dual"}))
        out["phase3_main"]["ladv{}".format(R.lam_tag(lam))] = {
            "per_seed_n": len(rows),
            "agg": {ood: aggregate_seeds(rows, ood) for ood in ALL_OOD},
        }

    iw_root = R.RES / "r6" / "r2" / "cur" / "features"
    for align, bucket in (("iwdan", "iwdan"), ("iwdan_oracle", "iwdan_oracle")):
        for lam in (0.5, 2.0):
            rows = []
            for seed in C.SEEDS:
                tag = "r2_cur_{}_ladv{}_s{}".format(align, R.lam_tag(lam), seed)
                fd = iw_root / tag
                if not fd.exists():
                    out["missing"].append(str(fd))
                    continue
                rows.append(score_features(fd, seed, "hist_" + tag,
                                           extra={"series": align, "lambda_adv": lam}))
            out[bucket]["ladv{}".format(R.lam_tag(lam))] = {
                "per_seed_n": len(rows),
                "agg": {ood: aggregate_seeds(rows, ood) for ood in ALL_OOD},
            }
    out["missing"] = sorted(set(out["missing"]))
    return out


def build_tables():
    manifest = json.loads((C.OUT / "MANIFEST.json").read_text())
    cells = {}
    status_rows = []

    def collect(jobs, init_default="scratch"):
        grouped = {}
        for j in jobs:
            init = j.get("init", init_default)
            key = cell_key(j["block"], j["lam"], j["prior"], init)
            grouped.setdefault(key, []).append(j)
        return grouped

    all_jobs = list(manifest["block_a_jobs"]) + list(manifest["block_a_reference_rows"]) + list(manifest["block_b_jobs"])
    for j in all_jobs:
        feat, st, note = resolve_run(j, {})
        status_rows.append({"tag": j["tag"], "block": j["block"], "lam": j["lam"], "prior": j["prior"],
                            "seed": j["seed"], "role": j.get("role"), "init": j.get("init", "scratch"),
                            "status": st, "note": note, "feat": str(feat) if feat else None})

    # score each unique feature dir once
    scored = {}
    for row in status_rows:
        if row["status"] not in ("ok", "reference") or not row["feat"]:
            continue
        key = (row["feat"], row["seed"], row["tag"] if row["status"] == "ok" else row["note"])
        cache_key = row["feat"]
        if cache_key not in scored:
            scored[cache_key] = score_features(row["feat"], row["seed"], Path(row["feat"]).name)

    # cell aggregates for Block A train+reference
    for lam in (0.0, 0.5, 2.0):
        for prior in C.PRIORS:
            rows = []
            for seed in C.SEEDS:
                if lam == 0.0:
                    tag = C.tag_a0(seed)  # reference shares baseline
                    feat = C.OUT / "features" / tag
                else:
                    tag = C.tag_a(lam, prior, seed)
                    feat = C.OUT / "features" / tag
                if str(feat) in scored:
                    r = dict(scored[str(feat)])
                    r["prior"] = prior
                    r["lambda_adv"] = lam
                    rows.append(r)
            key = cell_key("A", lam, prior, "scratch")
            cells[key] = {
                "block": "A", "lam": lam, "prior": prior, "init": "scratch",
                "role": "reference" if lam == 0.0 and prior != "prior100" else "train",
                "per_seed": rows,
                "agg": {ood: aggregate_seeds(rows, ood) for ood in ALL_OOD},
            }

    for lam in (0.0, 2.0):
        rows = []
        for seed in C.SEEDS:
            tag = C.tag_b(lam, seed, "erm")
            feat = C.OUT / "features" / tag
            if str(feat) in scored:
                rows.append(scored[str(feat)])
        key = cell_key("B", lam, "prior100", "erm")
        cells[key] = {
            "block": "B", "lam": lam, "prior": "prior100", "init": "erm",
            "per_seed": rows,
            "agg": {ood: aggregate_seeds(rows, ood) for ood in ALL_OOD},
        }

    # paired prior0 - prior100 for each λ in {0.5, 2}
    contrasts = []
    for lam in (0.5, 2.0):
        a0 = cells[cell_key("A", lam, "prior0", "scratch")]["per_seed"]
        a100 = cells[cell_key("A", lam, "prior100", "scratch")]["per_seed"]
        for ood in list(PRIMARY_UNSEEN) + ["pad_heldout"]:
            contrasts.append(paired_contrast(a0, a100, ood, "prior0_minus_prior100_ladv{}".format(R.lam_tag(lam))))

    hist = historical_rescore()

    # recovery vs R1 series (cross-protocol reference)
    recovery_note = {
        "hist_fitz_r1_shift100_l2": HIST_FITZ_R1_SHIFT100_L2,
        "hist_fitz_r1_shift0_l2": HIST_FITZ_R1_SHIFT0_L2,
        "note": "Reconcile recovery % only within matched series; do not swap 0.340/0.405 across protocols.",
    }

    doc = {
        "commit": C.git_head(),
        "status_rows": status_rows,
        "cells": cells,
        "paired_contrasts": contrasts,
        "historical": hist,
        "recovery_anchors": recovery_note,
        "far_ood": {"status": "unavailable", "reason": "no permitted non-skin panel pre-staged; no download"},
        "endpoints": manifest["endpoints"],
    }
    (C.OUT / "CELLS.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return doc


def write_report(doc):
    L = ["# Paper B R12 — fixed-image reweight + ERM-init REPORT", "",
         "Locked design: `decisions/paperB_r12_reweight.md`. Not manuscript edits.",
         "Protocol: fixed ISIC/PAD adversary image schedules; source-adv domain-loss weights "
         "`w=π_req/π_src` (mean 1 over stream). π_pad_adv from pad_adv labels (disclosed; not label-free).",
         "Implementation: `align=iwdan_oracle` + `iw_cls=False` for λ>0; λ=0 is plain `dann`. "
         "Cross-protocol vs R1 resampling.",
         ""]

    # status table
    L += ["## Cell status", "",
          "| tag | block | λ | prior | seed | init | status | note |",
          "|---|---|---|---|---|---|---|---|"]
    for r in doc["status_rows"]:
        L.append("| {} | {} | {:g} | {} | {} | {} | {} | {} |".format(
            r["tag"], r["block"], r["lam"], r["prior"], r["seed"], r["init"], r["status"], r.get("note") or ""))

    L += ["", "## Block A pooled Mahalanobis (seed mean ± sd)", "",
          "| λ | prior | role | pad_heldout | Fitz | MED | SD | RuleA Fitz |",
          "|---|---|---|---|---|---|---|---|"]
    for lam in (0.0, 0.5, 2.0):
        for prior in C.PRIORS:
            key = cell_key("A", lam, prior, "scratch")
            cell = doc["cells"].get(key, {})
            agg = cell.get("agg", {})
            def fmt(ood):
                a = agg.get(ood, {})
                if a.get("status") != "ok":
                    return a.get("status", "—")
                return "{:.3f}±{:.3f}".format(a["pooled"]["mean"], a["pooled"]["sd"])
            ra = agg.get("fitzpatrick17k", {}).get("rule_A")
            L.append("| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam, prior, cell.get("role"), fmt("pad_heldout"), fmt("fitzpatrick17k"),
                fmt("mednode"), fmt("sd198"), ra))

    L += ["", "## Paired contrasts prior0 − prior100 (pooled AUROC)", ""]
    for c in doc["paired_contrasts"]:
        if c["status"] != "ok":
            L.append("- {}: {}".format(c["label"], c["status"]))
        else:
            L.append("- {} / {}: Δ={:.3f}±{:.3f} (n={})".format(
                c["label"], c["ood"], c["diff_mean"], c["diff_sd"], c["n"]))

    L += ["", "## Block B (ERM-init trainable dual)", "",
          "| λ | pad_heldout | Fitz | ID bal | leakage V1 | RuleA Fitz |",
          "|---|---|---|---|---|---|"]
    for lam in (0.0, 2.0):
        key = cell_key("B", lam, "prior100", "erm")
        agg = doc["cells"].get(key, {}).get("agg", {})
        def fmt(ood):
            a = agg.get(ood, {})
            if a.get("status") != "ok":
                return "—"
            return "{:.3f}±{:.3f}".format(a["pooled"]["mean"], a["pooled"]["sd"])
        idb = agg.get("fitzpatrick17k", {}).get("id_bal", {})
        idm = "—" if not idb else "{:.3f}±{:.3f}".format(idb["mean"], idb["sd"])
        leak = agg.get("fitzpatrick17k", {}).get("leakage_v1_mean")
        L.append("| {:g} | {} | {} | {} | {} | {} |".format(
            lam, fmt("pad_heldout"), fmt("fitzpatrick17k"), idm,
            "—" if leak is None else "{:.3f}".format(leak),
            agg.get("fitzpatrick17k", {}).get("rule_A")))

    L += ["", "## Reading (post-hoc vs locked questions)", "",
          "- Recovery on unseen under fixed exposure → strengthens prior-weighting (not exposure) account.",
          "- Recovery on unseen but not pad_heldout → limited for aligned-domain headline.",
          "- No recovery → weakens prior-only explanation (retain).",
          "- Block B: if ID↑ without invariance, does not answer competent-invariant inversion; "
          "if invariant without inversion, retain negative.",
          "- R1 resampling is cross-protocol; gradient variance / optimization differ.",
          "",
          "## Far-OOD", "",
          "- Status: **unavailable** (no permitted non-skin panel on disk; no download).",
          "",
          "## Historical anchors", "",
          "- R1 Fitzpatrick λ2 shift100 mean **{:.3f}**; shift0 series **{:.3f}** (do not swap).".format(
              HIST_FITZ_R1_SHIFT100_L2, HIST_FITZ_R1_SHIFT0_L2),
          "- Verify reported ~45%/43%/20% recovery against R1 source tables before any corrected summary.",
          "",
          "## Negative / unresolved", "",
          "- See paired contrasts and Rule A columns; do not expand grid to hunt inversion.",
          ""]
    (C.OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote", C.OUT / "REPORT.md", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write_report", action="store_true")
    a = p.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)
    doc = build_tables()
    n_ok = sum(1 for r in doc["status_rows"] if r["status"] in ("ok", "reference"))
    print("scored status ok/ref", n_ok, "/", len(doc["status_rows"]), flush=True)
    if a.write_report:
        write_report(doc)


if __name__ == "__main__":
    main()
