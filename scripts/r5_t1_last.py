#!/usr/bin/env python3
"""Paper B R5: Table 1 (R2 item 4, 35 runs) at last.ckpt next to the best-val checkpoint (decisions/paperB_r5_t1last.md).

Subcommands (CPU):
  run TASK   run TASK (0..34, sorted run names): re-forward last.ckpt -> features_last/<run>/, metrics + held-out probe
             for both checkpoints -> per_run/<run>.json
  report     summary.csv, BOOTSTRAP.json, REPORT.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import r5_ckpt_audit as A  # noqa: E402
import r5_p9v2 as V  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

OUT = R.RES / "r5" / "t1last"
BEST_FEAT = R.ITEM4_FEAT
SETS = ("isic_train", "isic_test", "pad_full", "pad_heldout", "fitzpatrick17k")
OODS = ("pad_heldout", "fitzpatrick17k")
TOL = 2e-3


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def extract_last(tag, data):
    out = OUT / "features_last" / tag
    out.mkdir(parents=True, exist_ok=True)
    tf = build_val_transform_robust()
    frames = {"isic_train": SkinDataset(data["train"], transform=tf), "isic_test": SkinDataset(data["test"], transform=tf),
              "pad_full": SkinDataset(data["pad_full"], transform=tf), "pad_heldout": SkinDataset(data["pad_heldout"], transform=tf),
              "fitzpatrick17k": FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf)}
    net = None
    for name, ds in frames.items():
        f = out / "{}.npz".format(name)
        if f.exists():
            continue
        net = net or fin.load_csg(A.ITEM4_CKPT / tag / "last.ckpt", torch.device("cpu"))
        pack = x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=4), torch.device("cpu"))
        np.savez_compressed(f, **{k: pack[k] for k in R.FEAT_KEYS})
        print("features", tag, name, pack["labels"].shape, flush=True)
    return out


def evaluate(feat_dir):
    Fs = R.load_feats(feat_dir, SETS)
    m = R.run_metrics(Fs, ood_names=list(OODS))
    G = V.groups()
    assert np.array_equal(Fs["isic_test"]["labels"].astype(int), G["yi"]), "isic_test row order"
    v1 = V.v1_probe(Fs["isic_test"]["z_lesion_norm"], Fs["pad_heldout"]["z_lesion_norm"], G["gi"], G["gp"])
    return {"heldout_leakage_V1": v1["bal_acc_mean"], "heldout_leakage_V1_sd": v1["bal_acc_sd"],
            "padfull_leakage": m["leakage"]["bal_acc_mean"], "id_bal_acc": m["id"]["bal_acc"],
            "ood": {o: {d: m["ood"][o][d] for d in R.DETS} for o in OODS}}


def run(task):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    tag = A.table1_runs()[task]
    out = OUT / "per_run" / "{}.json".format(tag)
    if out.exists():
        print("done", out)
        return
    data = R.load_data()
    fl = extract_last(tag, data)
    ep_best, best = A.best_epoch(tag)
    last = A.ITEM4_CKPT / tag / "last.ckpt"
    rec = {"run": tag, "lambda_adv": A.lam_of(tag), "seed": int(tag.rsplit("_s", 1)[1]),
           "best_checkpoint": str(best), "best_epoch": ep_best, "last_checkpoint": str(last), "last_epoch": R.ckpt_epoch(last),
           "best": evaluate(BEST_FEAT / tag), "last": evaluate(fl)}
    stored = json.loads((A.ITEM4_RUNS / tag / "summary.json").read_text())["ood"]["pad_heldout"]["z_lesion"]
    stored = stored["mahalanobis_classcond"]["unrestricted"]
    rec["stored_best_maha_pad_heldout"] = stored
    rec["check_best_vs_stored"] = abs(rec["best"]["ood"]["pad_heldout"]["mahalanobis_classcond"] - stored) <= TOL
    wjson(out, rec)
    print(json.dumps({k: rec[k] for k in ("run", "best_epoch", "last_epoch", "check_best_vs_stored")}), flush=True)


def fitz_groups():
    paths = sorted(R.p3.FITZ_DIR.glob("*.jpg"))
    cl = pd.read_csv(R.RES / "r5" / "p13" / "fitz_clusters.csv").set_index("md5hash")["cluster_union"]
    g = np.array([str(cl[p.stem]) for p in paths])
    return R.idx_groups(g), "P13 cluster_union"


COLS = (("heldout_leakage_V1", "Leakage (held-out V1)"), ("id_bal_acc", "ID bal acc"),
        ("maha_pad_heldout", "Maha pad_heldout"), ("maha_fitz", "Maha Fitzpatrick"))


def report():
    rows = []
    for f in sorted((OUT / "per_run").glob("*.json")):
        j = json.loads(f.read_text())
        for ck in ("best", "last"):
            e = j[ck]
            rows.append({"run": j["run"], "lambda": j["lambda_adv"], "seed": j["seed"], "checkpoint": ck,
                         "epoch": j["{}_epoch".format(ck)], "heldout_leakage_V1": e["heldout_leakage_V1"],
                         "padfull_leakage": e["padfull_leakage"], "id_bal_acc": e["id_bal_acc"],
                         "maha_pad_heldout": e["ood"]["pad_heldout"]["mahalanobis_classcond"],
                         "maha_fitz": e["ood"]["fitzpatrick17k"]["mahalanobis_classcond"],
                         **{"{}_{}".format(o, d): e["ood"][o][d] for o in OODS for d in R.DETS},
                         "check_best_vs_stored": j["check_best_vs_stored"]})
    df = pd.DataFrame(rows).sort_values(["lambda", "seed", "checkpoint"])
    df.to_csv(OUT / "summary.csv", index=False)

    data = R.load_data()
    id_g, _ = R.T.lesion_groups(data["test"])
    id_groups = R.idx_groups(id_g)
    og = {"pad_heldout": R.ood_groups("pad_heldout", data), "fitzpatrick17k": fitz_groups()}
    boots = {}
    for lam, d in df[df.checkpoint == "last"].groupby("lambda"):
        seeds = sorted(d.seed.unique())
        for ck, root in (("best", BEST_FEAT), ("last", OUT / "features_last")):
            runs = {s: root / d[d.seed == s].run.iloc[0] for s in seeds}
            for o in OODS:
                b = R.bootstrap_cell(runs, o, id_groups, og[o][0])
                boots["{:g}|{}|{}".format(lam, ck, o)] = {"unit_ood": og[o][1], "unit_id": "lesion",
                                                          "mahalanobis_classcond": b["mahalanobis_classcond"],
                                                          "rule_A": b["rule_A"],
                                                          **{k: {x: b[k][x] for x in ("mean", "ci_lo", "ci_hi", "k_below", "n_seeds")}
                                                             for k in R.BOOT_DETS if k != "mahalanobis_classcond"}}
    wjson(OUT / "BOOTSTRAP.json", boots)

    L = ["# Table 1 at last.ckpt vs best-val (decisions/paperB_r5_t1last.md)", "",
         "Runs with results: {} / 35. Best-val vs stored item-4 pad_heldout Mahalanobis within {}: {} / {}.".format(
             df.run.nunique(), TOL, int(df[df.checkpoint == "best"].check_best_vs_stored.sum()), df.run.nunique()), "",
         "Bootstrap units: ID by lesion; pad_heldout by patient; Fitzpatrick by P13 cluster (cluster_union). "
         "PREREGISTER 2000 draws, seed 20261005; rule A = pooled CI upper bound < 0.5 and ≥ 3 seeds with CI upper bound < 0.5.", "",
         "## Mean ± SD over seeds", "",
         "| λ | n | epochs best (min–max) | " + " | ".join("{} best | {} last".format(h, h) for _c, h in COLS) + " |",
         "|---:|---:|---|" + "---:|---:|" * len(COLS)]
    tex = []
    for lam, d in df.groupby("lambda"):
        b, l = d[d.checkpoint == "best"], d[d.checkpoint == "last"]
        cells = []
        for c, _h in COLS:
            cells += [R.pm(R.ms(b[c])), R.pm(R.ms(l[c]))]
        L.append("| {:g} | {} | {}–{} | ".format(lam, len(b), b.epoch.min(), b.epoch.max()) + " | ".join(cells) + " |")
        tex.append("{:g} & ".format(lam) + " & ".join(
            "{} & {}".format(R.tex_pm(R.ms(b[c])), R.tex_pm(R.ms(l[c]))) for c, _h in COLS) + " \\\\")
    L += ["", "## Paired difference last − best (mean ± SD over seeds)", "",
          "| λ | " + " | ".join(h for _c, h in COLS) + " |", "|---:|" + "---:|" * len(COLS)]
    for lam, d in df.groupby("lambda"):
        p = d.pivot(index="seed", columns="checkpoint")
        L.append("| {:g} | ".format(lam) + " | ".join(R.pm(R.ms((p[c]["last"] - p[c]["best"]).tolist())) for c, _h in COLS) + " |")
    L += ["", "## Mahalanobis bootstrap (pooled CI, k seeds with CI < 0.5, rule A)", "",
          "| λ | pad_heldout best | pad_heldout last | Fitzpatrick best | Fitzpatrick last |", "|---:|---|---|---|---|"]
    for lam in sorted(df["lambda"].unique()):
        cells = []
        for o in OODS:
            for ck in ("best", "last"):
                b = boots.get("{:g}|{}|{}".format(lam, ck, o))
                m = b["mahalanobis_classcond"] if b else None
                cells.append("—" if b is None else "{:.3f} [{:.3f}, {:.3f}], {}/{}, rule A {}".format(
                    m["mean"], m["ci_lo"], m["ci_hi"], m["k_below"], m["n_seeds"], "met" if b["rule_A"] else "not met"))
        L.append("| {:g} | ".format(lam) + " | ".join(cells) + " |")
    L += ["", "## LaTeX rows (λ & leakage best & last & ID bal acc best & last & Maha pad_heldout best & last & Maha Fitz best & last)",
          "", "```"] + tex + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "report":
        report()
    else:
        raise SystemExit("unknown " + cmd)
