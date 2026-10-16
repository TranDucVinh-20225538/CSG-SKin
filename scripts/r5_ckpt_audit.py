#!/usr/bin/env python3
"""Paper B R5 checkpoint audit (decisions/paperB_r5_p8.md, deviation 2026-10-15 21:57, items 3 and 4).

Subcommands:
  epochs        Table 1 (R2 item 4, 35 runs): selected (best-val) epoch per run and λ, with the GRL α and the adversary
                train statistics of that epoch from lightning_logs; lists λ > 0 runs selected before α ≥ ALPHA_ON
  compete TASK  λ = 0 Table 1 run TASK (0..4, seeds 42..82), CPU re-forward of best-val and last.ckpt on the lesion-level
                ISIC val and test sets -> per-run JSON
  report        summary.csv + REPORT.md: epoch audit, competence (Table 1 recipe vs A4, both checkpoints)

Outputs: results/paperB/r5/ckpt_audit/
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

OUT = R.RES / "r5" / "ckpt_audit"
ITEM4_RUNS = T.RUNS
ITEM4_CKPT = T.CKPT
P8_MAIN = R.P8_DIR / "main" / "runs"
ALPHA_ON = 0.99
TOL = 2e-3


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def table1_runs():
    return sorted(d.name for d in ITEM4_CKPT.iterdir() if d.is_dir() and d.name.startswith("runB_orth1_ladv"))


def lam_of(tag):
    return float(re.search(r"ladv([0-9p]+)_s", tag).group(1).replace("p", "."))


def best_epoch(tag):
    (b,) = list((ITEM4_CKPT / tag).glob("best-*.ckpt"))
    return int(b.stem.split("-")[1]), b


def epochs():
    rows = []
    for tag in table1_runs():
        ep, _b = best_epoch(tag)
        m = pd.read_csv(ITEM4_RUNS / tag / "lightning_logs" / "version_0" / "metrics.csv")
        e = m.groupby("epoch").mean(numeric_only=True)
        r = e.loc[ep]
        rows.append({"run": tag, "lambda_adv": lam_of(tag), "seed": int(tag.rsplit("_s", 1)[1]), "selected_epoch": ep,
                     "last_epoch": int(e.index.max()), "grl_alpha_epoch_mean": float(r["train/grl_a"]),
                     "adv_train_acc": float(r["train/acc_adv"]), "adv_train_loss": float(r["train/loss_adv"]),
                     "val_acc_logged": float(r["val/acc"]),
                     "before_alpha_on": bool(lam_of(tag) > 0 and r["train/grl_a"] < ALPHA_ON)})
    df = pd.DataFrame(rows).sort_values(["lambda_adv", "seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "table1_selected_epochs.csv", index=False)
    wjson(OUT / "LAST_NEEDED.json", {"alpha_on": ALPHA_ON, "runs": df[df.before_alpha_on].run.tolist()})
    print(df.to_string(index=False))


def forward(ckpt, frame):
    net = fin.load_csg(Path(ckpt), torch.device("cpu"))
    ds = SkinDataset(frame, transform=build_val_transform_robust())
    pack = x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=4), torch.device("cpu"))
    return pack["logits"], pack["labels"].astype(int)


def compete(task):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    tag = "runB_orth1_ladv0_s{}".format(R.SEEDS[task])
    out = OUT / "compete" / "{}.json".format(tag)
    if out.exists():
        print("done", out)
        return
    _rec, (_tr, val, test) = T.split_record()
    ep, best = best_epoch(tag)
    last = ITEM4_CKPT / tag / "last.ckpt"
    rec = {"run": tag, "seed": R.SEEDS[task], "best_checkpoint": str(best), "best_epoch": ep,
           "last_checkpoint": str(last), "last_epoch": R.ckpt_epoch(last)}
    for ck_name, ck in (("best", best), ("last", last)):
        for split, frame in (("val", val), ("test", test)):
            lg, y = forward(ck, frame)
            rec["{}_{}_bal_acc".format(ck_name, split)] = float(balanced_accuracy_score(y, lg.argmax(1)))
            print(tag, ck_name, split, rec["{}_{}_bal_acc".format(ck_name, split)], flush=True)
    stored = json.loads((ITEM4_RUNS / tag / "summary.json").read_text())["id_balanced_acc"]
    rec["stored_test_bal_acc_best"] = stored
    rec["check_best_test_vs_stored"] = abs(rec["best_test_bal_acc"] - stored) <= TOL
    wjson(out, rec)


ERM_REF = R.P8_DIR / "erm_ref"
ERM_CKPT = R.CKPT / "r5_p8" / "erm_ref"


def erm(task):
    """Lesion-level single-encoder ERM reference (seeds 42/52/62/72/82): val/test bal acc at best-val and last.ckpt."""
    import train_phase15_1_single_dann as sd

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    seed = R.SEEDS[task]
    tag = "single_dann_ladv0_s{}".format(seed)
    out = OUT / "compete" / "erm_{}.json".format(tag)
    if out.exists():
        print("done", out)
        return
    _rec, (_tr, val, test) = T.split_record()
    (best,) = list((ERM_CKPT / tag).glob("best-*.ckpt"))
    last = ERM_CKPT / tag / "last.ckpt"
    rec = {"run": tag, "seed": seed, "best_checkpoint": str(best), "best_epoch": int(best.stem.split("-")[1]),
           "last_checkpoint": str(last), "last_epoch": R.ckpt_epoch(last)}
    tf = build_val_transform_robust()
    for ck_name, ck in (("best", best), ("last", last)):
        net = sd.SingleDannLightning.load_from_checkpoint(str(ck), strict=False, map_location="cpu").net.eval()
        for split, frame in (("val", val), ("test", test)):
            with torch.no_grad():
                p = sd.collect_all(net, DataLoader(SkinDataset(frame, transform=tf), batch_size=64, num_workers=4), torch.device("cpu"))
            rec["{}_{}_bal_acc".format(ck_name, split)] = float(balanced_accuracy_score(p["labels"], p["logits"].argmax(1)))
            print(tag, ck_name, split, rec["{}_{}_bal_acc".format(ck_name, split)], flush=True)
    stored = json.loads((ERM_REF / tag / "summary.json").read_text())["id_balanced_acc"]
    rec["stored_test_bal_acc_best"] = stored
    rec["check_best_test_vs_stored"] = abs(rec["best_test_bal_acc"] - stored) <= TOL
    wjson(out, rec)


def p8_a4_rows():
    rows = []
    for s in R.SEEDS:
        p = P8_MAIN / "main_ladv0_s{}".format(s) / "summary.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        if j.get("checkpoint_policy") != R.CKPT_POLICY:
            continue
        b = j["secondary_bestval"]
        rows.append({"recipe": "A4 (P8)", "seed": s, "best_epoch": b["epoch"], "last_epoch": j["primary_epoch"],
                     "best_val_bal_acc": b["metrics"]["val"]["bal_acc"], "best_test_bal_acc": b["metrics"]["id"]["bal_acc"],
                     "last_val_bal_acc": j["metrics"]["val"]["bal_acc"], "last_test_bal_acc": j["metrics"]["id"]["bal_acc"]})
    return rows


def report():
    ep = pd.read_csv(OUT / "table1_selected_epochs.csv")
    old = []
    for f in sorted((OUT / "compete").glob("*.json")):
        j = json.loads(f.read_text())
        name = "single-encoder ERM (lesion-level)" if f.stem.startswith("erm_") else "Table 1 (current)"
        old.append({"recipe": name, "seed": j["seed"], "best_epoch": j["best_epoch"], "last_epoch": j["last_epoch"],
                    **{k: j[k] for k in ("best_val_bal_acc", "best_test_bal_acc", "last_val_bal_acc", "last_test_bal_acc")},
                    "check_best_test_vs_stored": j["check_best_test_vs_stored"]})
    comp = pd.DataFrame(old + p8_a4_rows())
    comp.to_csv(OUT / "summary.csv", index=False)
    cols = ["best_val_bal_acc", "best_test_bal_acc", "last_val_bal_acc", "last_test_bal_acc"]
    L = ["# Checkpoint audit (P8 deviation 2026-10-15 21:57, items 3 and 4)", "",
         "Table 1 GRL schedule (csg_lightning, also used by the P8 recipe): α = max(0.2, 2/(1+exp(-20 p^0.5)) - 1), "
         "p = global step / total steps; no adversary warm-up epochs (the adversary is trained and the reversed gradient "
         "is applied from step 0). `grl_alpha_epoch_mean` is the epoch mean of the logged α. Flag: λ > 0 and α < {} at "
         "the selected epoch (criterion fixed after the selected epochs were listed).".format(ALPHA_ON), "",
         "## Item 4: Table 1 selected epochs (35 runs)", "",
         ep.to_markdown(index=False, floatfmt=".3f"), "",
         "λ > 0 runs flagged: {}".format(int(ep.before_alpha_on.sum())), "",
         "## Item 3: competence, λ = 0, lesion-level split (5 seeds; single-encoder ERM: 5 seeds)", "",
         "Image-level references (not like-for-like; leakage-inflated): paper single-encoder ERM {:.3f} "
         "(the 0.70 competence threshold was set from this figure; original ≥ 0.70 verdicts stay visible). "
         "Phase 15 image-level single-encoder λ = 0 ID test bal acc {} (5 seeds, best-val). "
         "Lesion-level ERM competence reference: `results/paperB/r5/p8/erm_ref/REPORT.md`.".format(
             R.ERM_SINGLE_ENCODER_IMAGE_LEVEL,
             R.pm(R.ms([json.loads(p.read_text())["id_balanced_acc"]
                        for p in sorted((R.RES / "phase15" / "single_dann").glob("single_dann_ladv0_s*/summary.json"))]))), "",
         comp.to_markdown(index=False, floatfmt=".3f") if len(comp) else "(no rows yet)", ""]
    if len(comp):
        g = comp.groupby("recipe")[cols].agg(["mean", "std", "count"])
        L += ["Mean ± SD (n):", "", g.to_markdown(floatfmt=".3f"), "", "LaTeX rows:", "", "```"]
        for rc, d in comp.groupby("recipe"):
            L.append("{} & ".format(rc) + " & ".join("{:.3f} $\\pm$ {:.3f}".format(d[c].mean(), d[c].std()) for c in cols)
                     + " \\\\")
        L += ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "epochs":
        epochs()
    elif cmd == "compete":
        compete(int(sys.argv[2]))
    elif cmd == "erm":
        erm(int(sys.argv[2]))
    elif cmd == "report":
        report()
    else:
        raise SystemExit("unknown " + cmd)
