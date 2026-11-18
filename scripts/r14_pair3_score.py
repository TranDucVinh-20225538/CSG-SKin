#!/usr/bin/env python3
"""Paper B R14 full scoring → CELLS.json, RESULTS.json, REPORT.md.

Uses r6 bootstrap_cell (2000 draws, rng 20261005). Reading rules from decisions/paperB_r14_pair3.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r14_pair3_common as C  # noqa: E402
import r6_class_aware_heldout_probes as CA  # noqa: E402

DET = "mahalanobis_classcond"
DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max", "MSP", "energy")
PSEEDS = (42, 52, 62)


def load_pack(feat_dir, name):
    p = Path(feat_dir) / "{}.npz".format(name)
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def v1_group(Xi, Xh, gi, gh):
    X = np.r_[Xi, Xh]
    d = np.r_[np.zeros(len(Xi)), np.ones(len(Xh))]
    g = np.r_[np.char.add("I", np.asarray(gi).astype(str)),
              np.char.add("M", np.asarray(gh).astype(str))]
    bals = []
    for s in PSEEDS:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=s))
        clf.fit(X[a], d[a])
        bals.append(float(balanced_accuracy_score(d[b], clf.predict(X[b]))))
    return {"bal_acc_mean": float(np.mean(bals)),
            "bal_acc_sd": float(np.std(bals, ddof=1)) if len(bals) > 1 else 0.0}


def class_aware_probes(te, mh, gi, gh):
    """Reuse R6 probes (a)/(b)/(c) on ISIC test ∪ milk_heldout."""
    try:
        a = v1_group(te["z_lesion_norm"], mh["z_lesion_norm"], gi, gh)
        # (b) class-reweighted like CA
        y_i, y_m = te["labels"].astype(int), mh["labels"].astype(int)
        w = CA.class_weights(y_i, y_m)
        bals_b = []
        X = np.r_[te["z_lesion_norm"], mh["z_lesion_norm"]]
        d = np.r_[np.zeros(len(y_i)), np.ones(len(y_m))]
        g = np.r_[np.char.add("I", np.asarray(gi).astype(str)),
                  np.char.add("M", np.asarray(gh).astype(str))]
        wtr_full = np.r_[w, np.ones(len(y_m))]
        for s in PSEEDS:
            tr, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
            bals_b.append(CA.fit_eval(X[tr], d[tr], X[te_idx], d[te_idx], s, wtr=wtr_full[tr]))
        b = {"bal_acc_mean": float(np.mean(bals_b)),
             "bal_acc_sd": float(np.std(bals_b, ddof=1)) if len(bals_b) > 1 else 0.0}
        # (c) within-class
        per_c = []
        for c in range(C.NCLS):
            mi = y_i == c
            mm = y_m == c
            if mi.sum() < 20 or mm.sum() < 20:
                continue
            vc = v1_group(te["z_lesion_norm"][mi], mh["z_lesion_norm"][mm],
                          np.asarray(gi)[mi], np.asarray(gh)[mm])
            per_c.append((c, mi.sum() + mm.sum(), vc["bal_acc_mean"]))
        if per_c:
            wts = np.array([n for _, n, _ in per_c], float)
            vals = np.array([v for _, _, v in per_c], float)
            cmean = float(np.average(vals, weights=wts))
        else:
            cmean = None
        return {"a": a, "b": b, "c": {"bal_acc_mean": cmean, "per_class": [
            {"class": C.LABELS[c], "n": int(n), "bal_acc": float(v)} for c, n, v in per_c
        ]}}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def isic_test_groups(data):
    lid_map = C.isic_lesion_map()
    stems = data["test"]["path"].map(lambda p: Path(p).stem)
    return R.idx_groups(stems.map(lambda s: lid_map.get(s, s)).astype(str))


def ood_unit_groups(name, data):
    if name == "milk_heldout":
        return R.idx_groups(data["milk_heldout"]["lesion_id"].astype(str)), "lesion"
    if name == "pad_ufes20":
        return R.ood_groups("pad_full", {"pad_full": data["pad_ufes20"]})[0], "patient"
    if name == "fitzpatrick17k":
        g, u = R.ood_groups("fitzpatrick17k", data)
        return g, u
    if name == "mednode":
        meta = pd.read_csv(R.P7A_SETS / "mednode.csv", keep_default_na=False)
        return [np.array([i]) for i in range(len(meta))], "image"
    if name == "sd198":
        g, _ = R.ood_groups("sd198", data)
        return g, "cluster_union"
    if name == "scin":
        g, _ = R.ood_groups("scin", data)
        return g, "case"
    return [np.array([i]) for i in range(1)], "image"


def score_run(tag, data, id_groups):
    feat_dir = C.OUT / "features" / tag
    summary_path = C.OUT / "runs" / tag / "summary.json"
    if not summary_path.exists():
        return {"tag": tag, "status": "missing_summary"}
    summary = json.loads(summary_path.read_text())
    if summary.get("degenerate"):
        return {"tag": tag, "status": "degenerate", "summary": summary,
                "seed": summary.get("seed"), "lam": summary.get("lam"), "prior": summary.get("prior")}
    tr = load_pack(feat_dir, "isic_train")
    te = load_pack(feat_dir, "isic_test")
    if tr is None or te is None:
        return {"tag": tag, "status": "missing_features", "seed": summary.get("seed")}
    # ensure full OOD features exist (re-extract if train was --light)
    needed = list(C.OOD_ALL) + ["isic_train", "isic_test", "milk_adv"]
    missing = [n for n in needed if not (feat_dir / "{}.npz".format(n)).exists()]
    if missing:
        return {"tag": tag, "status": "incomplete_features", "missing": missing, "seed": summary.get("seed")}

    Fs = R.load_feats(feat_dir)
    m = R.run_metrics(Fs, ood_names=[n for n in C.OOD_ALL if n in Fs])
    # milk_heldout block
    sc = R.Scorer(Fs["isic_train"])
    sid = sc.scores(Fs["isic_test"])
    for n in C.OOD_ALL:
        if n not in Fs:
            continue
        if n in m.get("ood", {}):
            continue
        so = sc.scores(Fs[n])
        blk = {d: R.auroc(sid[d], so[d]) for d in R.DETS}
        blk["n"] = int(len(so["MSP"]))
        nc = sc.nearest_class(Fs[n]["z_lesion_norm"])
        blk["nv_nearest_frac"] = float(np.mean(nc == R.NV))
        blk["nearest_class_share"] = {C.LABELS[c]: float(np.mean(nc == c)) for c in range(C.NCLS)}
        m.setdefault("ood", {})[n] = blk

    gi = C.isic_lesion_map()
    stems = data["test"]["path"].map(lambda p: Path(p).stem)
    g_isic = stems.map(lambda s: gi.get(s, s)).astype(str).to_numpy()
    g_milk = data["milk_heldout"]["lesion_id"].astype(str).to_numpy()
    v1 = v1_group(Fs["isic_test"]["z_lesion_norm"], Fs["milk_heldout"]["z_lesion_norm"], g_isic, g_milk)
    ca = class_aware_probes(Fs["isic_test"], Fs["milk_heldout"], g_isic, g_milk)

    pred = Fs["milk_heldout"]["logits"].argmax(1)
    maj = int(np.bincount(pred, minlength=C.NCLS).argmax())

    out = {
        "tag": tag, "status": "ok",
        "seed": int(summary["seed"]), "lam": float(summary["lam"]), "prior": summary["prior"],
        "degenerate": False,
        "id_balanced_acc": float(m["id"]["bal_acc"]),
        "degraded": bool(m["id"]["bal_acc"] < 0.25),
        "heldout_leakage_V1": float(v1["bal_acc_mean"]),
        "heldout_leakage_V1_sd": float(v1["bal_acc_sd"]),
        "class_aware_leakage": ca,
        "milk_heldout_pred_majority_label": C.LABELS[maj],
        "milk_heldout_pred_majority_share": float(np.mean(pred == maj)),
        "ood": m.get("ood", {}),
        "feat_dir": str(feat_dir),
        "commit": C.git_head(),
    }
    # also score best.ckpt if present and features missing for best — report path only
    out["best_checkpoint"] = summary.get("best_checkpoint")
    out["primary_checkpoint"] = summary.get("primary_checkpoint")
    return out


def ms(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "n": len(xs)}


def aggregate(per_run, data, id_groups):
    cells = {}
    for prior in C.PRIORS:
        for lam in (0.0, 2.0):
            if lam == 0.0 and prior != "prior100":
                continue
            key = "ladv{}_{}".format(C.lam_tag(lam), prior)
            rows = [r for r in per_run if r.get("status") == "ok"
                    and float(r["lam"]) == float(lam) and r["prior"] == prior]
            deg = [r for r in per_run if r.get("status") == "degenerate"
                   and float(r.get("lam", -1)) == float(lam) and r.get("prior") == prior]
            kept = [r for r in rows if not r.get("degraded")]
            excl = [r["seed"] for r in rows if r.get("degraded")]
            feat_map = {r["seed"]: Path(r["feat_dir"]) for r in kept}
            cell = {
                "key": key, "lam": lam, "prior": prior,
                "n_ok": len(rows), "n_kept": len(kept), "n_degenerate": len(deg),
                "excluded_degraded_seeds": excl,
                "id_bal": ms([r["id_balanced_acc"] for r in kept]),
                "V1": ms([r["heldout_leakage_V1"] for r in kept]),
                "per_seed": {str(r["seed"]): {
                    "id_bal": r["id_balanced_acc"], "V1": r["heldout_leakage_V1"],
                    "degraded": r.get("degraded"), "tag": r["tag"],
                    "ood_maha": {o: r["ood"].get(o, {}).get(DET) for o in C.OOD_ALL},
                } for r in rows},
                "ood": {},
            }
            for ood in C.OOD_ALL:
                pts = []
                for r in kept:
                    v = r["ood"].get(ood, {}).get(DET)
                    if v is not None:
                        pts.append(v)
                o_groups, unit = ood_unit_groups(ood, data)
                # symlink milk_heldout as needed for bootstrap_cell load_feats
                boot = None
                if feat_map:
                    # bootstrap_cell expects ood_name file under feat dir
                    boot = R.bootstrap_cell(feat_map, ood, id_groups, o_groups,
                                            exclude_seeds=set(excl))
                cell["ood"][ood] = {
                    "maha_seed": ms(pts),
                    "unit": unit,
                    "bootstrap": boot,
                    "rule_A": None if boot is None else boot.get("rule_A"),
                    "counted": ood in C.OOD_LABELLED_UNSEEN,
                }
            cells[key] = cell
    return cells


def verdict(cells):
    """Apply §6 reading rules for λ=2."""
    c100 = cells.get("ladv2_prior100")
    c0 = cells.get("ladv2_prior0")
    c_l0 = cells.get("ladv0_prior100")
    out = {"small_gap": False, "invariant_prior100": None, "competent": None, "case": None, "text": None}

    if not c100 or not c100.get("V1"):
        out["text"] = "missing λ=2 prior100 cell"
        return out

    v100 = c100["V1"]["mean"]
    out["invariant_prior100"] = bool(v100 <= C.INVARIANCE_V1_MAX)

    if c_l0 and c_l0.get("V1") and c_l0["V1"]["mean"] <= C.INVARIANCE_V1_MAX:
        out["small_gap"] = True

    if c_l0 and c_l0.get("id_bal") and c100.get("id_bal"):
        out["competent"] = bool(abs(c100["id_bal"]["mean"] - c_l0["id_bal"]["mean"]) <= 0.03)
        out["id_bal_vs_erm"] = {
            "cell": c100["id_bal"]["mean"], "erm": C.ERM_ID_BAL,
            "delta": c100["id_bal"]["mean"] - C.ERM_ID_BAL,
        }

    if not out["invariant_prior100"]:
        out["case"] = "C"
        out["text"] = "Not invariant (V1={:.3f} > 0.60 at λ=2 prior100). Case C, settles nothing.".format(v100)
        return out

    def hits(cell):
        if not cell:
            return 0
        n = 0
        for ood in C.OOD_LABELLED_UNSEEN:
            if cell["ood"].get(ood, {}).get("rule_A"):
                n += 1
        return n

    h100 = hits(c100)
    h0 = hits(c0)
    out["rule_a_hits_prior100"] = h100
    out["rule_a_hits_prior0"] = h0

    v0_ok = bool(c0 and c0.get("V1") and c0["V1"]["mean"] <= C.INVARIANCE_V1_MAX)
    out["invariant_prior0"] = v0_ok

    if h100 >= 2 and h0 == 0:
        if v0_ok:
            out["text"] = ("inversion reproduced, label shift necessary in this pair "
                           "(Rule A ≥2 at prior100, 0 at prior0; prior0 also invariant).")
        else:
            out["text"] = ("reproduced; necessity not separated from invariance depth "
                           "(Rule A ≥2 at prior100 and 0 at prior0, but prior0 V1 > 0.60).")
    elif h100 >= 2 and h0 >= 1:
        # "Met at prior100 and also at prior0"
        if h0 >= 2:
            out["text"] = "reproduced, label shift not necessary here."
        else:
            out["text"] = ("reproduced at prior100; prior0 has Rule A on {} of 3 "
                           "(not 'none' and not full).".format(h0))
    elif h100 < 2:
        out["text"] = "not reproduced in this pair (Rule A on {} of 3 at prior100).".format(h100)
    else:
        out["text"] = "see hits prior100={} prior0={}".format(h100, h0)
    return out


def write_report(cells, per_run, verd, job_meta):
    L = [
        "# REPORT — Paper B R14 ISIC 2019 → MILK10k dermoscopic",
        "",
        "Generated: {}".format(datetime.now(timezone.utc).isoformat()),
        "Commit: `{}`".format(C.git_head()),
        "",
    ]
    if verd.get("small_gap"):
        L += [
            "## Small-gap check",
            "",
            "V1 at λ = 0 (seed mean) is already ≤ 0.60. The invariance gate says nothing about what "
            "the adversary did in this pair; the verdict below is still computed.",
            "",
        ]
    L += ["## Verdict", "", verd.get("text") or "—", "",
          "```json", json.dumps(verd, indent=2, default=float), "```", ""]

    L += ["## Cells (λ=2 Mahalanobis / V1)", "",
          "| cell | ID bal | V1 | Fitz Rule A | MED Rule A | SD Rule A |",
          "|---|---:|---:|---|---|---|"]
    for key in ("ladv0_prior100", "ladv2_prior100", "ladv2_prior50", "ladv2_prior0"):
        c = cells.get(key)
        if not c:
            L.append("| {} | — | — | — | — | — |".format(key))
            continue
        def ra(o):
            v = c["ood"].get(o, {}).get("rule_A")
            return "—" if v is None else ("met" if v else "not")
        ib = c["id_bal"]
        v1 = c["V1"]
        L.append("| {} | {} | {} | {} | {} | {} |".format(
            key,
            "—" if not ib else "{:.3f}±{:.3f}".format(ib["mean"], ib["sd"] or 0),
            "—" if not v1 else "{:.3f}±{:.3f}".format(v1["mean"], v1["sd"] or 0),
            ra("fitzpatrick17k"), ra("mednode"), ra("sd198"),
        ))

    L += ["", "## Per-run rows", "",
          "| tag | status | ID bal | V1 | Fitz Maha |",
          "|---|---|---:|---:|---:|"]
    for r in sorted(per_run, key=lambda x: (x.get("lam", 99), x.get("prior", ""), x.get("seed", 0))):
        if r.get("status") == "ok":
            L.append("| {} | ok | {:.3f} | {:.3f} | {} |".format(
                r["tag"], r["id_balanced_acc"], r["heldout_leakage_V1"],
                "{:.3f}".format(r["ood"].get("fitzpatrick17k", {}).get(DET, float("nan")))))
        else:
            L.append("| {} | {} | — | — | — |".format(r.get("tag"), r.get("status")))

    L += ["", "## Job / compute meta", "",
          "```json", json.dumps(job_meta, indent=2, default=float), "```", "",
          "milk_heldout, PAD-UFES-20 and SCIN reported, not counted. prior50 = dose step.",
          "No wording beyond reading rules in decisions/paperB_r14_pair3.md.",
          ""]
    (C.OUT / "REPORT.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    a = ap.parse_args()
    data = C.load_data()
    id_groups = isic_test_groups(data)
    jobs = C.jobs_main()
    tags = a.tags or [j["tag"] for j in jobs]
    per_run = []
    for tag in tags:
        print("score", tag, flush=True)
        per_run.append(score_run(tag, data, id_groups))
    (C.OUT / "RESULTS.json").write_text(json.dumps({"runs": per_run, "commit": C.git_head()},
                                                    indent=1, default=float) + "\n")
    cells = aggregate(per_run, data, id_groups)
    verd = verdict(cells)
    (C.OUT / "CELLS.json").write_text(json.dumps({"cells": cells, "verdict": verd, "commit": C.git_head()},
                                                  indent=1, default=float) + "\n")
    job_meta = {
        "commit": C.git_head(),
        "n_runs_scored": len(per_run),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_report(cells, per_run, verd, job_meta)
    C.write_status("# R14 STATUS\n\nphase: scored\nverdict: {}\n".format(verd.get("text")))
    print("VERDICT:", verd.get("text"), flush=True)


if __name__ == "__main__":
    main()
