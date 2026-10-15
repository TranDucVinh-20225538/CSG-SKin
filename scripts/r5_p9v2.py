#!/usr/bin/env python3
"""Paper B R5 P9 v2 (decisions/paperB_r5_p9.md, amendment 2026-10-15): single-encoder EfficientNet-B3 toolbox grid,
held-out leakage probe (P6 V1), target from the dual encoder, selection steps and report.

Subcommands (all CPU):
  target                write v2/TARGET.json from the P6 per-run JSONs (refuses if a P9 v2 summary exists)
  plan PHASE TASK       print "CONFIG LAMBDA SEED" for a GPU task (PHASE r1 | r2 | s3 | s5); exit 3 = skip
  round2                write v2/ROUND2.json (per λ, the round-1 LR with the lowest V1; depth 1 and 3)
  best                  write v2/BEST.json (lowest V1 among unflagged pilot runs)
  go5                   write v2/GO5.json if the 3-seed mean V1 of the best config ≤ T, else v2/NO_GO5.json
  report                probes, bootstrap, verdict, outcome -> summary.csv, BOOTSTRAP.json, REPORT.md
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
from r5_p6_leakage_probe import fit_eval  # noqa: E402

V2 = R.R5 / "p9" / "v2"
P6_RUNS = R.R5 / "p6" / "per_run"
LAMS, LRS = (2.0, 4.0, 8.0), (10, 30, 100)
PSEEDS = (42, 52, 62)
DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max", "MSP", "Energy_T1")
FLAG_DROP = 0.05
CONTROL = "baseline"


def cfg_name(lr, depth):
    return "gw_lr{}_d{}".format(int(lr), int(depth))


def parse_cfg(name):
    if not name.startswith("gw_lr"):
        return None, None
    lr, d = name[len("gw_lr"):].split("_d")
    return int(float(lr)), int(d)


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def rjson(name):
    p = V2 / name
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------------------------------------------------
# Held-out probe (P6 V1) on a P9 run
# ----------------------------------------------------------------------------------------------------------------------

_GROUPS = {}


def groups():
    if not _GROUPS:
        _rec, (_tr, _va, test) = R.T4.split_record()
        gi, _ = R.T4.lesion_groups(test)
        import r2_common as C

        meta = C.pad_heldout_meta()
        _GROUPS.update(gi=np.asarray(gi).astype(str), yi=test.label_idx.astype(int).to_numpy(),
                       gp=np.asarray(meta.patient_id.astype(str).tolist(), dtype=str), pp=np.asarray(meta.path.astype(str).tolist(), dtype=str))
        hold = R.isic_lesion_data()["pad_hold"]
        assert np.array_equal(hold.path.astype(str).to_numpy(), _GROUPS["pp"]), "pad_heldout order vs patient meta"
    return _GROUPS


def v1_probe(Xi, Xh, gi, gp):
    X = np.r_[Xi, Xh]
    d = np.r_[np.zeros(len(Xi)), np.ones(len(Xh))]
    g = np.r_[np.char.add("I", gi), np.char.add("P", gp)]
    res = []
    for s in PSEEDS:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
        res.append(fit_eval(X[a], d[a], X[b], d[b], s))
    ba = [r["bal_acc"] for r in res]
    return {"bal_acc_mean": float(np.mean(ba)), "bal_acc_sd": float(np.std(ba, ddof=1)),
            "auroc_mean": float(np.mean([r["auroc"] for r in res])), "per_probe_seed": res}


def heldout_probe(run_dir):
    out = run_dir / "heldout_probe.json"
    if out.exists():
        return json.loads(out.read_text())
    G = groups()
    F = {n: np.load(run_dir / "features" / "{}.npz".format(n)) for n in ("isic_train", "isic_test", "pad_heldout")}
    te, ho = F["isic_test"], F["pad_heldout"]
    assert np.array_equal(te["labels"].astype(int), G["yi"]), "isic_test row order"
    assert len(ho["labels"]) == len(G["gp"]) == 716
    pca = PCA(n_components=16, random_state=0).fit(F["isic_train"]["z"])
    rec = {"V1": v1_probe(te["z"], ho["z"], G["gi"], G["gp"]),
           "V1_pca16": v1_probe(pca.transform(te["z"]), pca.transform(ho["z"]), G["gi"], G["gp"]),
           "z_dim": int(te["z"].shape[1]),
           "protocol": "P6 V1: ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit (lesion / patient), probe seeds 42/52/62"}
    wjson(out, rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------------------------------


def runs(with_probe=True):
    rows = []
    for f in sorted(V2.glob("*/summary.json")):
        s = json.loads(f.read_text())
        lr, depth = parse_cfg(s["config"])
        r = {"tag": s["tag"], "dir": f.parent, "config": s["config"], "lam": float(s["lambda_adv"]),
             "seed": int(s["seed"]), "lr": lr, "depth": depth, "id_bal_acc": s["id_balanced_acc"],
             "xdom_bal_acc": s["xfer_pad_heldout_6class"]["balanced_accuracy"],
             "leak_padfull": s["leakage"]["test_vs_pad_full"]["bal_acc_mean"],
             "maha_pad_heldout": s["ood"]["pad_heldout"]["mahalanobis_classcond"],
             "maha_fitz": s["ood"]["fitzpatrick17k"]["mahalanobis_classcond"],
             "gate": s.get("gate", {}).get("status") if isinstance(s.get("gate"), dict) else s.get("gate"),
             "gpu_h": float(s.get("gpu_hours", 0.0))}
        if with_probe:
            p = heldout_probe(f.parent)
            r.update(v1=p["V1"]["bal_acc_mean"], v1_sd=p["V1"]["bal_acc_sd"], v1_pca16=p["V1_pca16"]["bal_acc_mean"])
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        ctrl = df[(df.config == CONTROL) & (df.lam == 0)]
        base = float(ctrl.id_bal_acc.mean()) if len(ctrl) else np.nan
        df["control_id_bal_acc"] = base
        df["degraded"] = (df.lam > 0) & (df.id_bal_acc < base - FLAG_DROP)
    return df


def skipped():
    return [json.loads(f.read_text()) for f in sorted(V2.glob("*/skipped.json"))]


# ----------------------------------------------------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------------------------------------------------


def cmd_target():
    if list(V2.glob("*/summary.json")):
        sys.exit("TARGET.json must be written before any P9 v2 summary exists")
    if (V2 / "TARGET.json").exists():
        print("TARGET.json exists:", rjson("TARGET.json"))
        return
    recs = [json.loads(p.read_text()) for p in sorted(P6_RUNS.glob("*.json"))]
    l2 = [r for r in recs if float(r["lambda"]) == 2.0]
    if len(l2) != 5:
        sys.exit("P6 per-run JSONs at λ = 2: {} of 5; target not written".format(len(l2)))
    v = [r["z_lesion_norm"]["V1"]["bal_acc_mean"] for r in l2]
    t = {"dual_lambda2_V1_z_lesion_norm_per_seed": {str(r["seed"]): x for r, x in zip(l2, v)},
         "dual_lambda2_V1_mean": float(np.mean(v)), "margin": 0.05, "target": float(np.mean(v)) + 0.05,
         "source": str(P6_RUNS), "commit": R.git_head()}
    wjson(V2 / "TARGET.json", t)
    print(json.dumps(t, indent=2))


def cmd_plan(phase, task):
    task = int(task)
    if phase == "r1":
        if task < 9:
            print(cfg_name(LRS[task % 3], 2), LAMS[task // 3], 42)
        else:
            print(CONTROL, 0, 42)
        return
    if phase == "r2":
        p = rjson("ROUND2.json")
        if p is None:
            print("SKIP no ROUND2.json")
            sys.exit(3)
        t = p["tasks"][task]
        print(t["config"], t["lambda"], 42)
        return
    b = rjson("BEST.json")
    if b is None:
        print("SKIP no BEST.json")
        sys.exit(3)
    if phase == "s3":
        print(*[(b["config"], b["lambda"], 52), (b["config"], b["lambda"], 62), (CONTROL, 0, 52), (CONTROL, 0, 62)][task])
        return
    if phase == "s5":
        if rjson("GO5.json") is None:
            print("SKIP no GO5.json")
            sys.exit(3)
        print(b["config"], b["lambda"], (72, 82)[task])
        return
    sys.exit("unknown phase " + phase)


def cmd_round2():
    if (V2 / "ROUND2.json").exists():
        print("ROUND2.json exists")
        return
    df = runs()
    r1 = df[(df.seed == 42) & (df.lam > 0) & (df.depth == 2)]
    tasks, basis = [], {}
    for lam in LAMS:
        c = r1[r1.lam == lam]
        if not len(c):
            basis[str(lam)] = "no round-1 run"
            continue
        pool = c[~c.degraded] if (~c.degraded).any() else c
        pick = pool.sort_values(["v1", "lr"]).iloc[0]
        basis[str(lam)] = {"lr": int(pick.lr), "v1": float(pick.v1), "all_flagged": bool(not (~c.degraded).any()),
                           "candidates": c[["config", "v1", "id_bal_acc", "degraded"]].to_dict("records")}
        tasks += [{"config": cfg_name(pick.lr, d), "lambda": lam} for d in (1, 3)]
    wjson(V2 / "ROUND2.json", {"tasks": tasks, "basis": basis, "commit": R.git_head()})
    print(json.dumps(tasks, indent=2))


def cmd_best():
    if (V2 / "BEST.json").exists():
        print("BEST.json exists")
        return
    df = runs()
    pilot = df[(df.seed == 42) & (df.lam > 0)]
    if not len(pilot):
        sys.exit("no pilot run")
    pool = pilot[~pilot.degraded] if (~pilot.degraded).any() else pilot
    pick = pool.sort_values(["v1", "lam", "lr", "depth"]).iloc[0]
    b = {"config": pick.config, "lambda": float(pick.lam), "v1": float(pick.v1), "degraded": bool(pick.degraded),
         "all_flagged": bool(not (~pilot.degraded).any()), "n_pilot": int(len(pilot)), "commit": R.git_head()}
    wjson(V2 / "BEST.json", b)
    print(json.dumps(b, indent=2))


def cmd_go5():
    if (V2 / "GO5.json").exists() or (V2 / "NO_GO5.json").exists():
        print("go5 already decided")
        return
    b, t = rjson("BEST.json"), rjson("TARGET.json")
    df = runs()
    c = df[(df.config == b["config"]) & (df.lam == b["lambda"]) & df.seed.isin([42, 52, 62])]
    rec = {"best": b, "target": t["target"], "seeds": sorted(c.seed.tolist()), "v1_per_seed": c.set_index("seed").v1.to_dict(),
           "mean_v1": float(c.v1.mean()) if len(c) else None, "commit": R.git_head()}
    if len(c) == 3 and rec["mean_v1"] <= t["target"]:
        wjson(V2 / "GO5.json", rec)
    else:
        rec["reason"] = "fewer than 3 seeds (cap or failure)" if len(c) < 3 else "3-seed mean V1 above the target"
        wjson(V2 / "NO_GO5.json", rec)
    print(json.dumps(rec, indent=2, default=str))


# ----------------------------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------------------------


def verdict(blk):
    """Primary (amendment 16:18): the code's rule A on Mahalanobis pad_heldout (r5_common.bootstrap_cell 'rule')."""
    m = blk["mahalanobis_classcond"]
    if m["n_seeds"] < 5:
        return "not assigned ({} seeds; rule {})".format(m["n_seeds"], m["rule"])
    if m["rule"] == "A":
        return "inverts (rule A)"
    return "rule {}; {}".format(m["rule"], "collapse" if 0.45 <= m["mean"] <= 0.55 else "no collapse")


def strict(maha_hi, k_seeds, n_seeds, det_his):
    """Secondary: Maha mean CI hi < 0.5, >= 4/5 seeds' CI hi < 0.5, >= 3 of 5 detectors' mean CI hi < 0.5."""
    if n_seeds < 5:
        return "not assigned ({} seeds)".format(n_seeds)
    ok = maha_hi < 0.5 and k_seeds >= 4 and sum(h < 0.5 for h in det_his) >= 3
    return "inverts (strict)" if ok else "not strict"


def strict_p9(blk):
    m = blk["mahalanobis_classcond"]
    return strict(m["pooled_ci"][1], m["k_below"], m["n_seeds"], [blk[d]["pooled_ci"][1] for d in DETS])


def dual_lambda2_strict():
    """Dual encoder λ = 2, 5 seeds, five detectors (z_lesion_norm, fit on ISIC train), pad_heldout, PREREGISTER
    patient-clustered bootstrap (r5cpu_common.pooled_bootstrap, the main-results path). Cached."""
    out = V2 / "DUAL_L2_STRICT.json"
    if out.exists():
        return json.loads(out.read_text())
    import r5cpu_common as RC

    pre = RC.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    rng = np.random.default_rng(seed)
    di, do = RC.draws(RC.id_test_groups(), n_boot, rng), RC.draws(RC.pad_heldout_patients(), n_boot, rng)
    per = {d: {} for d in DETS}
    for sd in (42, 52, 62, 72, 82):
        t = RC.run_tag(2.0, sd)
        sc, _ = RC.detector_scores(RC.load(t, "isic_train"), RC.load(t, "isic_test"), {"pad_heldout": RC.load(t, "pad_heldout")})
        for d in DETS:
            sid, so = sc[d]
            per[d][sd] = (sid, so["pad_heldout"])
    B = {d: RC.pooled_bootstrap(per[d], di, do) for d in DETS}
    m = B["mahalanobis_classcond"]
    res = {"per_detector": {d: {k: v for k, v in b.items() if k != "per_seed"} for d, b in B.items()},
           "maha_per_seed": m["per_seed"], "rule_A": m["rule_A"],
           "strict": strict(m["pooled_ci_hi"], m["k_seeds_ci_below_0.5"], m["n_seeds"],
                            [B[d]["pooled_ci_hi"] for d in DETS]), "commit": R.git_head()}
    wjson(out, res)
    return res


def dual_reference():
    rows = []
    for p in sorted(P6_RUNS.glob("*.json")):
        r = json.loads(p.read_text())
        if float(r["lambda"]) in (0.0, 2.0):
            rows.append({"lambda": float(r["lambda"]), "seed": r["seed"],
                         "v1_z_lesion_norm": r["z_lesion_norm"]["V1"]["bal_acc_mean"],
                         "v1_backbone_raw": r["backbone_raw_lesion"]["V1"]["bal_acc_mean"],
                         "v0_z_lesion_norm": r["z_lesion_norm"]["V0"]["bal_acc_mean"]})
    return pd.DataFrame(rows)


def ci(b):
    return "[{:.3f}, {:.3f}]".format(*b["pooled_ci"])


def cmd_report():
    df = runs()
    t, b = rjson("TARGET.json"), rjson("BEST.json")
    sk = skipped()
    L = ["# Paper B R5 P9 v2: single-encoder EfficientNet-B3 invariance toolbox", "",
         "Decisions: `decisions/paperB_r5_p9.md` (amendment 2026-10-15). Commit {}.".format(R.git_head()), ""]
    if t:
        L += ["Target T = dual-encoder λ = 2 held-out probe (V1, z_lesion_norm, 5-seed mean) {:.3f} + 0.05 = **{:.3f}**."
              .format(t["dual_lambda2_V1_mean"], t["target"]), ""]
    if not len(df):
        L.append("No P9 v2 run has finished.")
        (V2 / "REPORT.md").write_text("\n".join(L) + "\n")
        return
    df.drop(columns=["dir"]).to_csv(V2 / "summary.csv", index=False)
    pilot = df[df.seed == 42].sort_values(["lam", "lr", "depth"], na_position="first")
    B1 = R.isic_bootstrap({r.tag: {42: str(r.dir)} for r in pilot.itertuples()}, ood_sets=("pad_heldout",), dets=DETS)
    L += ["## Pilot (seed 42; CI = single-seed patient-clustered bootstrap, ID by lesion, pad_heldout by patient)", "",
          "| config | λ | LR × | depth | V1 held-out (± probe SD) | V1 PCA-16 | pad_full probe | ID bal acc | X-dom bal acc "
          "| degraded | gate | Maha pad_heldout | kNN | cosine | MSP | Energy | Maha Fitz | GPU-h |",
          "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---:|---:|"]
    for r in pilot.itertuples():
        bb = B1.get(r.tag, {}).get("pad_heldout", {})
        cells = ["{:.3f} {}".format(bb[d]["mean"], ci(bb[d])) if d in bb else "—" for d in DETS]
        L.append("| {} | {:g} | {} | {} | {:.3f} ± {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {:.3f} | {:.2f} |".format(
            r.config, r.lam, "—" if pd.isna(r.lr) else int(r.lr), "—" if pd.isna(r.depth) else int(r.depth), r.v1, r.v1_sd,
            r.v1_pca16, r.leak_padfull, r.id_bal_acc, r.xdom_bal_acc, "yes" if r.degraded else "no", r.gate,
            " | ".join(cells), r.maha_fitz, r.gpu_h))
    L += ["", "λ = 0 control ID balanced accuracy (mean over available seeds): {:.3f}; flag threshold {:.3f}.".format(
        df.control_id_bal_acc.iloc[0], df.control_id_bal_acc.iloc[0] - FLAG_DROP), ""]
    for name in ("ROUND2.json", "BEST.json", "GO5.json", "NO_GO5.json"):
        if rjson(name) is not None:
            L += ["- {}: `{}`".format(name, json.dumps({k: v for k, v in rjson(name).items() if k not in ("basis", "commit")},
                                                      default=str))]
    L.append("")

    BS = {}
    outcome = "untested: no best config"
    if b:
        cells = {"best": df[(df.config == b["config"]) & (df.lam == b["lambda"])],
                 "control_lambda0": df[(df.config == CONTROL) & (df.lam == 0)]}
        BS = R.isic_bootstrap({k: {int(r.seed): str(r.dir) for r in c.itertuples()} for k, c in cells.items() if len(c)},
                              ood_sets=("pad_heldout", "fitzpatrick17k"), dets=DETS)
        wjson(V2 / "BOOTSTRAP.json", BS)
        L += ["## Seed cells (pooled mean-over-seeds CI; pad_heldout by patient)", "",
              "| cell | config | λ | seeds | V1 mean (per seed) | ID bal acc | X-dom | degraded (any) | "
              "Maha | kNN | cosine | MSP | Energy | Maha seeds with CI hi < 0.5 | primary (rule A) | secondary (strict) |",
              "|---|---|---:|---:|---|---:|---:|---|---|---|---|---|---|---:|---|---|"]
        verdicts = {}
        for k, c in cells.items():
            if not len(c) or k not in BS:
                continue
            ph = BS[k]["pad_heldout"]
            verdicts[k] = verdict(ph)
            L.append("| {} | {} | {:g} | {} | {:.3f} ({}) | {:.3f} | {:.3f} | {} | {} | {}/{} | {} | {} |".format(
                k, c.config.iloc[0], c.lam.iloc[0], len(c), c.v1.mean(),
                ", ".join("{:.3f}".format(x) for x in c.sort_values("seed").v1), c.id_bal_acc.mean(), c.xdom_bal_acc.mean(),
                "yes" if c.degraded.any() else "no",
                " | ".join("{:.3f} {}".format(ph[d]["mean"], ci(ph[d])) for d in DETS),
                ph["mahalanobis_classcond"]["k_below"], ph["mahalanobis_classcond"]["n_seeds"], verdicts[k],
                strict_p9(ph)))
        dl = dual_lambda2_strict()
        pdd = dl["per_detector"]
        L.append("| dual encoder λ = 2 (Table 1 runs) | — | 2 | 5 | (P6 table below) | — | — | — | {} | {}/5 | {} | {} |".format(
            " | ".join("{:.3f} [{:.3f}, {:.3f}]".format(pdd[d]["mean"], pdd[d]["pooled_ci_lo"], pdd[d]["pooled_ci_hi"])
                       for d in DETS), pdd["mahalanobis_classcond"]["k_seeds_ci_below_0.5"],
            "inverts (rule A)" if dl["rule_A"] else "not rule A", dl["strict"]))
        L.append("")
        bc = cells["best"]
        if t and len(bc) == 5:
            reached = bc.v1.mean() <= t["target"]
            degraded = bool(bc.id_bal_acc.mean() < bc.control_id_bal_acc.iloc[0] - FLAG_DROP)
            if not reached:
                outcome = "untested: 5-seed mean V1 {:.3f} > T {:.3f}".format(bc.v1.mean(), t["target"])
            elif degraded:
                outcome = "target reached with a degraded classifier: neither supports nor refutes"
            elif verdicts.get("best") == "inverts (rule A)":
                outcome = "target reached and the model inverts (rule A): consistent with the hypothesis"
            else:
                outcome = "target reached and the model does not meet rule A ({}): hypothesis refuted".format(verdicts.get("best"))
        elif rjson("NO_GO5.json"):
            outcome = "untested: " + rjson("NO_GO5.json")["reason"]
        else:
            outcome = "pending: {} of 5 seeds of the best config".format(len(bc))
    L += ["**Outcome:** " + outcome, ""]

    dr = dual_reference()
    if len(dr):
        L += ["## Dual encoder under the same probes (P6, lesion-level runs)", "",
              "| λ | seeds | V1 z_lesion_norm | V1 backbone_raw (1,536-d) | V0 (Table 1 protocol) |", "|---:|---:|---|---|---|"]
        for lam, g in dr.groupby("lambda"):
            L.append("| {:g} | {} | {} | {} | {} |".format(lam, len(g), R.ms(g.v1_z_lesion_norm), R.ms(g.v1_backbone_raw),
                                                         R.ms(g.v0_z_lesion_norm)))
        L.append("")
    used = float(df.gpu_h.sum())
    L += ["GPU-h used by P9 v2 runs: {:.2f} of the 30 cap; skipped by the cap: {}.".format(
        used, ", ".join(s["tag"] for s in sk) or "none"), ""]
    L += ["## LaTeX rows (config & λ & V1 & ID bal acc & X-dom & Maha pad_heldout [CI])", "", "```latex"]
    for r in pilot.itertuples():
        m = B1.get(r.tag, {}).get("pad_heldout", {}).get("mahalanobis_classcond")
        if m:
            L.append("{} & {:g} & {:.3f} & {:.3f} & {:.3f} & {:.3f} [{:.3f}, {:.3f}] \\\\".format(
                r.config.replace("_", "\\_"), r.lam, r.v1, r.id_bal_acc, r.xdom_bal_acc, m["mean"], *m["pooled_ci"]))
    for k, blk in BS.items():
        m = blk["pad_heldout"]["mahalanobis_classcond"]
        L.append("{} ({} seeds) & & & & & {:.3f} [{:.3f}, {:.3f}] \\\\".format(k.replace("_", "\\_"), m["n_seeds"], m["mean"],
                                                                              *m["pooled_ci"]))
    L += ["```", ""]
    (V2 / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    cmd = sys.argv[1]
    {"target": cmd_target, "round2": cmd_round2, "best": cmd_best, "go5": cmd_go5, "report": cmd_report}.get(
        cmd, lambda: cmd_plan(*sys.argv[2:4]))()


if __name__ == "__main__":
    main()
