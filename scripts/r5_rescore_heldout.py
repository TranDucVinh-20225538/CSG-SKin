#!/usr/bin/env python3
"""Paper B R5: rescore tab:single and tab:obj on pad_heldout with the patient-clustered CI
(decisions/paperB_r5_rescore_heldout.md). CPU.

  r5_rescore_heldout.py run TASK   re-forward run RUNS[TASK] (r4_mmd_pad_heldout path), save per-image scores
  r5_rescore_heldout.py report     pooled bootstrap per cell, tables, LaTeX rows
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase3_sweep as p3  # noqa: E402,F401  (transformers stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB")
OUT = Path(os.environ.get("RESCORE_SMOKE", RES / "r5" / "rescore_heldout"))
DETS = ("mahalanobis_classcond", "knn_k50", "cosine_max", "MSP", "Energy_T1")
DET_LABEL = {"mahalanobis_classcond": "Maha", "knn_k50": "kNN", "cosine_max": "Cosine", "MSP": "MSP",
             "Energy_T1": "Energy"}
TOL = 2e-3


def run_list():
    single = sorted((RES / "phase15" / "single_dann").glob("single_dann_ladv*_s*"))
    obj = sorted((RES / "phase15" / "objectives").glob("*_w*_s*"))
    mmd = sorted((RES / "phase15b" / "mmd").glob("mmd_w*_s*"))
    runs = [("single", d) for d in single] + [("obj", d) for d in obj] + [("obj", d) for d in mmd]
    runs = [(k, d) for k, d in runs if d.is_dir() and (d / "summary.json").exists()]
    assert len(runs) == 76, len(runs)
    return runs


def scores(ztr, ytr, zid, zood, lid, lood):
    import torch
    from phase15_common import cosine_ood_scores, fit_class_means, fit_knn, knn_scores
    from src.utils import ood_metrics

    means, prec = ood_metrics.compute_mahalanobis_params_from_arrays(ztr, ytr, num_classes=8, reg_eps=1e-3)
    cm, present = fit_class_means(ztr, ytr, 8)
    knn = fit_knn(ztr, k=50)
    out = {}
    for side, z, lg in (("id", zid, lid), ("ood", zood, lood)):
        p = torch.softmax(torch.from_numpy(lg), dim=1).numpy()
        e = torch.logsumexp(torch.from_numpy(lg).float(), dim=1).numpy()
        out[side] = {"mahalanobis_classcond": ood_metrics.mahalanobis_min_squared_distances(z, means, prec),
                     "cosine_max": cosine_ood_scores(z, cm, present), "knn_k50": knn_scores(knn, z),
                     "MSP": -p.max(axis=1), "Energy_T1": -e}
    return out


def heldout_probe(zid, gid, zh, gh):
    from sklearn.model_selection import GroupShuffleSplit

    from r5_p6_leakage_probe import fit_eval

    X, d = np.r_[zid, zh], np.r_[np.zeros(len(zid)), np.ones(len(zh))]
    g = np.r_[np.char.add("I", gid), np.char.add("P", gh)]
    ba = []
    for s in (42, 52, 62):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s).split(X, d, g))
        ba.append(fit_eval(X[a], d[a], X[b], d[b], s)["bal_acc"])
    return {"bal_acc_mean": float(np.mean(ba)), "bal_acc_sd": float(np.std(ba, ddof=1)), "per_probe_seed": ba}


def do_run(task):
    import torch

    import r2_common as C
    import train_phase2_pad_holdout as p2
    import train_r2_item4_lesion_split as T4
    from phase15_common import auroc_pair
    from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust
    from src.utils.seed import seed_everything

    kind, d = run_list()[task]
    out = OUT / "per_run" / d.name
    if (out / "record.json").exists():
        print("skip_done", d.name)
        return
    torch.set_num_threads(8)
    summ = json.loads((d / "summary.json").read_text())
    seed_everything(int(summ["seed"]))
    spec = json.loads(p3.P2_SPLIT.read_text())
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy()
    pad = df[df.domain == "pad_ufes"].copy().reset_index(drop=True)
    train, _val, test = p2.isic_splits(isic)
    if os.environ.get("RESCORE_SMOKE"):
        train, test = train.sample(600, random_state=0), test.reset_index(drop=True).head(300)
    hmask = pad.path.isin(set(spec["pad_heldout_paths"])).to_numpy()
    hold_paths = pad.path[hmask].astype(str).to_numpy()
    meta = C.pad_heldout_meta()
    assert hmask.sum() == 716 and np.array_equal(hold_paths, meta.path.astype(str).to_numpy()), "pad_heldout order"
    if kind == "single":
        from train_phase15_1_single_dann import SingleDannLightning, collect_all

        lit = SingleDannLightning.load_from_checkpoint(summ["best_checkpoint"], strict=False, map_location="cpu")
    else:
        from train_phase15_2_objectives import InvariantLightning, collect_all

        lit = InvariantLightning.load_from_checkpoint(summ["best_checkpoint"], strict=False, map_location="cpu")
    net, dev, tf = lit.net.eval(), torch.device("cpu"), build_val_transform_robust()

    def fwd(frame):
        with torch.no_grad():
            return collect_all(net, p2.make_loader(SkinDataset(frame, transform=tf), 64, 6, False), dev)

    tr, te, full = fwd(train), fwd(test), fwd(pad)
    ytr = np.asarray(tr["labels"]).astype(int)
    hold = {k: np.asarray(v)[hmask] for k, v in full.items()}
    S_full = scores(tr["z"], ytr, te["z"], full["z"], te["logits"], full["logits"])
    S_hold = {"id": S_full["id"], "ood": {k: v[hmask] for k, v in S_full["ood"].items()}}
    au = {o: {det: float(auroc_pair(S["id"][det], S["ood"][det])) for det in DETS}
          for o, S in (("pad_full", S_full), ("pad_heldout", S_hold))}
    stored_full = summ["ood"]["pad_full"] if kind == "single" else summ["ood_pad_full"]
    check = {"pad_full": {det: au["pad_full"][det] - float(stored_full[det]) for det in DETS}}
    if kind == "single":
        check["pad_heldout"] = {det: au["pad_heldout"][det] - float(summ["ood"]["pad_heldout"][det]) for det in DETS}
    maxdev = max(abs(v) for blk in check.values() for v in blk.values())
    gid, _ = T4.lesion_groups(test)
    probe = heldout_probe(np.asarray(te["z"]), np.asarray(gid).astype(str), hold["z"],
                          np.asarray(meta.patient_id.astype(str).tolist(), dtype=str))
    lk = summ["leakage"]
    stored_leak = (lk.get("pad_full", lk) or {}).get("bal_acc_mean") if isinstance(lk, dict) else None
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", **{"id__" + k: v for k, v in S_full["id"].items()},
                        **{"pad_heldout__" + k: v for k, v in S_hold["ood"].items()},
                        **{"pad_full__" + k: v for k, v in S_full["ood"].items()},
                        id_groups=np.asarray(gid).astype(str), id_labels=np.asarray(te["labels"]).astype(int))
    if kind == "single":
        m = re.fullmatch(r"single_dann_ladv([0-9p]+)_s(\d+)", d.name)
        table, cell, weight = "single", "lambda", float(m.group(1).replace("p", "."))
    else:
        table, cell, weight = "obj", summ["objective"], float(summ["weight"])
    rec = {"run": d.name, "source": str(d), "table": table, "objective": cell, "weight": weight,
           "seed": int(summ["seed"]), "checkpoint": summ["best_checkpoint"], "auroc": au, "check_minus_stored": check,
           "max_abs_check": float(maxdev), "check_flag": bool(maxdev > TOL), "heldout_probe": probe,
           "stored_leakage_pad_full": stored_leak, "id_balanced_acc": summ.get("id_balanced_acc"),
           "n": {"isic_train": len(ytr), "isic_test": len(te["labels"]), "pad_full": int(len(pad)), "pad_heldout": 716}}
    (out / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("saved", d.name, json.dumps(au["pad_heldout"]), "maxdev {:.2e}".format(maxdev), flush=True)


def report():
    import r5cpu_common as R

    recs = [json.loads(p.read_text()) for p in sorted((OUT / "per_run").glob("*/record.json"))]
    pre = R.prereg()
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    gp = R.pad_heldout_patients()
    cells = {}
    for r in recs:
        cells.setdefault((r["table"], r["objective"], r["weight"]), []).append(r)
    BS, rows = {}, []
    for key in sorted(cells, key=lambda k: (k[0] != "single", k[1], k[2])):
        rs = sorted(cells[key], key=lambda r: r["seed"])
        S = {r["seed"]: dict(np.load(OUT / "per_run" / r["run"] / "scores.npz")) for r in rs}
        gid = next(iter(S.values()))["id_groups"]
        assert all(np.array_equal(v["id_groups"], gid) for v in S.values())
        rng = np.random.default_rng(seed)
        di, do = R.draws(gid, n_boot, rng), R.draws(gp, n_boot, rng)
        blk = {}
        for det in DETS:
            b = R.pooled_bootstrap({s: (v["id__" + det], v["pad_heldout__" + det]) for s, v in S.items()}, di, do)
            b["pad_full_mean"] = float(np.mean([r["auroc"]["pad_full"][det] for r in rs]))
            blk[det] = b
        name = "{}|{}|{:g}".format(*key)
        BS[name] = blk
        row = {"table": key[0], "objective": key[1], "weight": key[2], "n_seeds": len(rs),
               "seeds": ",".join(str(r["seed"]) for r in rs),
               "heldout_probe": float(np.mean([r["heldout_probe"]["bal_acc_mean"] for r in rs])),
               "stored_leakage_pad_full": float(np.mean([r["stored_leakage_pad_full"] for r in rs]))
               if all(r["stored_leakage_pad_full"] is not None for r in rs) else np.nan,
               "max_abs_check": max(r["max_abs_check"] for r in rs), "check_flag": any(r["check_flag"] for r in rs)}
        for det in DETS:
            b = blk[det]
            row.update({det: b["mean"], det + "_sd": b["sd"], det + "_lo": b["pooled_ci_lo"], det + "_hi": b["pooled_ci_hi"],
                        det + "_rule": "A" if b["rule_A"] else ("B" if b["pooled_ci_hi"] < 0.5 else
                                                                 ("C" if b["pooled_ci_lo"] < 0.5 else "above")),
                        det + "_k_below": b["k_seeds_ci_below_0.5"], det + "_pad_full": b["pad_full_mean"]})
        rows.append(row)
    R.write_json(OUT / "BOOTSTRAP.json", BS)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "summary.csv", index=False)
    L = ["# Paper B R5: tab:single and tab:obj rescored on pad_heldout", "",
         "Decisions: `decisions/paperB_r5_rescore_heldout.md`. Commit {}. {} runs. Bootstrap {} draws, rng seed {}: ID by "
         "lesion group (image-level ISIC test), pad_heldout (716 images / 412 patients) by patient, draws shared across "
         "seeds; CI = percentile 95% of the seed mean; rule as in the main results (A: CI < 0.5 and ≥ 3 seeds' CI < 0.5; "
         "B: CI < 0.5 otherwise; C: CI straddles 0.5).".format(R.git_head(), len(recs), n_boot, seed), "",
         "Pipeline check: max |recomputed − stored| over runs = {:.2e} (tolerance {:g}); flagged runs: {}.".format(
             max(r["max_abs_check"] for r in recs), TOL, ", ".join(r["run"] for r in recs if r["check_flag"]) or "none"), ""]
    for table, title in (("single", "tab:single (single-encoder DANN, λ)"), ("obj", "tab:obj (objectives × weight)")):
        T = D[D.table == table]
        L += ["## " + title, "", "| cell | seeds | " + " | ".join("{} pad_heldout [CI] (pad_full)".format(DET_LABEL[d])
                                                             for d in DETS) + " | held-out probe | pad_full leakage |",
              "|---|---:|" + "---|" * len(DETS) + "---:|---:|"]
        for r in T.itertuples(index=False):
            r = r._asdict()
            cell = "λ = {:g}".format(r["weight"]) if table == "single" else "{} w{:g}".format(r["objective"], r["weight"])
            L.append("| {} | {} | {} | {:.3f} | {} |".format(cell, r["n_seeds"], " | ".join(
                "{:.3f} ± {:.3f} [{:.3f}, {:.3f}] {} ({:.3f})".format(r[d], r[d + "_sd"], r[d + "_lo"], r[d + "_hi"],
                                                                      r[d + "_rule"], r[d + "_pad_full"]) for d in DETS),
                r["heldout_probe"], "—" if pd.isna(r["stored_leakage_pad_full"]) else "{:.3f}".format(r["stored_leakage_pad_full"])))
        L += ["", "```latex"]
        for r in T.itertuples(index=False):
            r = r._asdict()
            cell = "{:g}".format(r["weight"]) if table == "single" else "{} & {:g}".format(r["objective"], r["weight"])
            L.append("{} & {} \\\\".format(cell, " & ".join("{:.3f} [{:.3f}, {:.3f}]".format(r[d], r[d + "_lo"], r[d + "_hi"])
                                                           for d in DETS)))
        L += ["```", ""]
    M = D[(D.table == "obj") & (D.objective.isin(["mmd", "erm"]))].sort_values("weight")
    L += ["## MMD text values (Mahalanobis, pad_heldout)", ""]
    L += ["- {} w{:g}: {:.3f} [{:.3f}, {:.3f}] (pad_full {:.3f})".format(r.objective, r.weight, r.mahalanobis_classcond,
                                                                      r.mahalanobis_classcond_lo, r.mahalanobis_classcond_hi,
                                                                      r.mahalanobis_classcond_pad_full) for r in M.itertuples()]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        do_run(int(sys.argv[2]))
    else:
        report()
