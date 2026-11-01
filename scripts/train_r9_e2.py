#!/usr/bin/env python3
"""Paper B R9 E2: stronger λ=0 recipe / 80-epoch schedule (decisions/paperB_r9_e2.md).

Stages:
  S     A4_e80 and A7_e80 at λ=0, seeds 42/52, eval=id (blind to OOD) + ERM EffB3 80ep × 2
  selS  select by mean val bal acc (tie ≤0.005 → A4); Gate S on ID test vs ERM reference
  C     selected recipe λ=0 seeds 62/72/82 (eval=id); Gate C five-seed
  M     selected recipe λ=2 × 5 seeds, then full eval of λ=0 and λ=2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
OUT = R.RES / "r9" / "e2"
CK = R.CKPT / "r9_e2"
SCREEN_SEEDS = (42, 52)
CONFIRM_SEEDS = (62, 72, 82)
CANDIDATES = ("A4", "A7")
TIE = 0.005
ERM_40_REF = 0.558  # five-seed 40-epoch ERM (work order)
SEL = OUT / "SELECTED_RECIPE.json"
GATE_S = OUT / "GATE_S.json"
GATE_C = OUT / "GATE_C.json"
ERM_DIR = OUT / "erm_ref"
ERM_CK = CK / "erm_ref"


def dual_spec(cid, seed, lam, stage, eval_mode):
    r = R.grid_recipe(cid, 80)
    tag = "{}_e80_ladv{}_s{}".format(cid, R.lam_tag(lam), seed)
    return {
        "item": "r9_e2",
        "out_root": str(OUT / stage),
        "ckpt_root": str(CK / stage),
        "tag": tag,
        "recipe": r,
        "lam": float(lam),
        "seed": int(seed),
        "mode": R.default_mode(r),  # two-stream for A4/A7 (loss balance)
        "align": "dann",
        "eval": eval_mode,
        "reuse_ckpt": None,
        "extra": {"stage": stage, "grid_id": cid, "epochs": 80},
    }


def jobs(stage):
    if stage == "S":
        out = [dual_spec(cid, s, 0.0, "S", "id") for cid in CANDIDATES for s in SCREEN_SEEDS]
        out += [{"kind": "erm80", "tag": "erm80_s{}".format(s), "seed": s, "lam": 0.0,
                 "mode": "single-encoder", "align": "erm",
                 "recipe": {"name": "erm_effb3_e80"}} for s in SCREEN_SEEDS]
        return out
    sel = json.loads(SEL.read_text())
    cid = sel["winner"]
    if stage == "C":
        return [dual_spec(cid, s, 0.0, "C", "id") for s in CONFIRM_SEEDS]
    if stage == "M":
        # train λ=2 for all 5 seeds; λ=0 reuse S/C checkpoints then full eval
        out = []
        for s in R.SEEDS:
            reuse = None
            if s in SCREEN_SEEDS:
                reuse = str(CK / "S" / "{}_e80_ladv0_s{}".format(cid, s))
            elif s in CONFIRM_SEEDS:
                reuse = str(CK / "C" / "{}_e80_ladv0_s{}".format(cid, s))
            out.append(dual_spec(cid, s, 2.0, "M", "full"))
            j0 = dual_spec(cid, s, 0.0, "M", "full")
            j0["reuse_ckpt"] = reuse
            j0["tag"] = "{}_e80_ladv0_s{}".format(cid, s)
            out.append(j0)
        # λ=2 first
        out = [j for j in out if j["lam"] == 2.0] + [j for j in out if j["lam"] == 0.0]
        return out
    raise SystemExit(stage)


def _summary_id(path):
    s = json.loads(Path(path).read_text())
    m = s["metrics"]
    return {
        "tag": s.get("tag"),
        "seed": int(s["seed"]),
        "val_bal_acc": float(m["val"]["bal_acc"]),
        "id_bal_acc": float(m["id"]["bal_acc"]),
        "degraded": bool(m.get("degraded", m["id"]["bal_acc"] < R.DEGRADED_ID_BAL)),
    }


def erm_reference():
    """Higher of 0.558 (40ep, 5 seeds) and 80-epoch two-seed mean ID test."""
    rows = []
    for s in SCREEN_SEEDS:
        p = ERM_DIR / "single_dann_ladv0_s{}".format(s) / "summary.json"
        if not p.exists():
            raise SystemExit("ERM 80ep summary missing for seed {}: {}".format(s, p))
        sjs = json.loads(p.read_text())
        rows.append(float(sjs["metrics"]["id"]["bal_acc"]))
    mean80 = float(np.mean(rows))
    ref = float(max(ERM_40_REF, mean80))
    return {"erm_40_ref": ERM_40_REF, "erm_80_seed_mean": mean80, "erm_80_per_seed": rows,
            "erm_reference": ref, "source": "max(0.558, mean of 80ep seeds 42/52)"}


def select_s():
    per = {}
    rows = []
    for cid in CANDIDATES:
        vals, tests = [], []
        for s in SCREEN_SEEDS:
            p = OUT / "S" / "runs" / "{}_e80_ladv0_s{}".format(cid, s) / "summary.json"
            if not p.exists():
                raise SystemExit("stage S incomplete: {}".format(p))
            r = _summary_id(p)
            vals.append(r["val_bal_acc"])
            tests.append(r["id_bal_acc"])
            rows.append({"recipe": cid, **r})
        per[cid] = {"val_mean": float(np.mean(vals)), "val": vals,
                    "id_test_mean": float(np.mean(tests)), "id_test": tests}
    top = max(per[c]["val_mean"] for c in CANDIDATES)
    near = [c for c in CANDIDATES if per[c]["val_mean"] >= top - TIE]
    # ties within 0.005 → A4
    winner = "A4" if "A4" in near else near[0]
    erm = erm_reference()
    gate_thr = erm["erm_reference"] - 0.03
    id_mean = per[winner]["id_test_mean"]
    gate_ok = bool(id_mean >= gate_thr)
    sel = {
        "rule": "higher mean val bal acc over seeds 42/52; ties within {} → A4".format(TIE),
        "per_recipe": per,
        "within_tie": near,
        "winner": winner,
        "winner_recipe": R.grid_recipe(winner, 80),
        "rows": rows,
        "erm": erm,
        "gate_S_threshold": gate_thr,
        "gate_S_id_test_mean": id_mean,
        "gate_S_met": gate_ok,
        "stop_message": None if gate_ok else "competence gap not closed by an 80-epoch schedule",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps(sel, indent=1) + "\n")
    GATE_S.write_text(json.dumps({
        "met": gate_ok, "id_test_mean": id_mean, "threshold": gate_thr,
        "erm_reference": erm["erm_reference"], "winner": winner,
        "message": sel["stop_message"],
    }, indent=1) + "\n")
    print("selected", winner, "val_mean", per[winner]["val_mean"],
          "gate_S", gate_ok, "id_test", id_mean, "thr", gate_thr, flush=True)
    if not gate_ok:
        raise SystemExit(2)


def gate_c():
    sel = json.loads(SEL.read_text())
    cid = sel["winner"]
    erm = sel["erm"]
    tests = []
    for s in R.SEEDS:
        if s in SCREEN_SEEDS:
            p = OUT / "S" / "runs" / "{}_e80_ladv0_s{}".format(cid, s) / "summary.json"
        else:
            p = OUT / "C" / "runs" / "{}_e80_ladv0_s{}".format(cid, s) / "summary.json"
        if not p.exists():
            raise SystemExit("stage C incomplete: {}".format(p))
        tests.append(_summary_id(p)["id_bal_acc"])
    mean5 = float(np.mean(tests))
    thr = erm["erm_reference"] - 0.03
    ok = bool(mean5 >= thr)
    rec = {"met": ok, "id_test_mean_5seed": mean5, "per_seed": dict(zip(R.SEEDS, tests)),
           "threshold": thr, "erm_reference": erm["erm_reference"], "winner": cid,
           "message": None if ok else "competence gap not closed by an 80-epoch schedule"}
    GATE_C.write_text(json.dumps(rec, indent=1) + "\n")
    print("gate_C", ok, mean5, thr, flush=True)
    if not ok:
        raise SystemExit(2)


def _erm80_id_eval(seed, a):
    """last.ckpt ID val/test only into erm_ref summary (blind to OOD for stage S)."""
    import torch
    import train_phase15_1_single_dann as sd
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust

    cd = ERM_CK / "single_dann_ladv0_s{}".format(seed)
    rd = ERM_DIR / "single_dann_ladv0_s{}".format(seed)
    last = cd / "last.ckpt"
    if not last.exists():
        raise FileNotFoundError(last)
    if a.dry_run:
        print("dry_run erm80 id-eval", seed, last, flush=True)
        return
    data = R.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = sd.SingleDannLightning.load_from_checkpoint(str(last), strict=False, map_location=device).net.to(device).eval()
    tf = build_val_transform_robust()
    packs = {}
    for name, frame in (("isic_val", data["val"]), ("isic_test", data["test"])):
        loader = sd.p2.make_loader(SkinDataset(frame, transform=tf), 64, a.num_workers, False)
        packs[name] = sd.collect_all(net, loader, device)
    pv, yv = packs["isic_val"]["logits"].argmax(1), packs["isic_val"]["labels"]
    pt, yt = packs["isic_test"]["logits"].argmax(1), packs["isic_test"]["labels"]
    id_bal = float(balanced_accuracy_score(yt, pt))
    summary = {
        "tag": "erm80_s{}".format(seed),
        "seed": int(seed),
        "lambda_adv": 0.0,
        "max_epochs": 80,
        "checkpoint_policy": "last",
        "primary_checkpoint": str(last),
        "primary_epoch": R.ckpt_epoch(last),
        "metrics": {
            "val": {"bal_acc": float(balanced_accuracy_score(yv, pv)),
                    "acc": float(accuracy_score(yv, pv))},
            "id": {"bal_acc": id_bal, "acc": float(accuracy_score(yt, pt))},
            "degraded": bool(id_bal < R.DEGRADED_ID_BAL),
        },
        "id_balanced_acc": id_bal,
        "note": "R9 E2 stage S: ID only (no leakage / OOD)",
    }
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("saved", rd / "summary.json", "id_bal", id_bal, flush=True)


def run_erm80(seed, a):
    """Train/eval single-encoder ERM at 80 epochs into r9/e2/erm_ref (do not overwrite P8 40ep)."""
    import train_phase15_1_single_dann as sd

    last = ERM_CK / "single_dann_ladv0_s{}".format(seed) / "last.ckpt"
    run_out = ERM_DIR / "single_dann_ladv0_s{}".format(seed)
    summ = run_out / "summary.json"
    if a.skip_done and summ.exists() and last.exists():
        s = json.loads(summ.read_text())
        if "metrics" in s and "id" in s["metrics"] and int(s.get("max_epochs", 0)) == 80:
            print("skip_done erm80", run_out, flush=True)
            return
    if not (a.skip_done and last.exists() and R.ckpt_epoch(last) >= 79):
        sd.p2.isic_splits = R.T.isic_splits_by_lesion
        sys.argv = [sys.argv[0], "--lambda_adv", "0", "--seed", str(seed),
                    "--output_root", str(ERM_DIR), "--ckpt_root", str(ERM_CK),
                    "--num_workers", str(a.num_workers), "--max_epochs", "80"]
        if a.resume:
            sys.argv.append("--resume")
        if a.dry_run:
            sys.argv.append("--dry_run")
        sd.main()
    _erm80_id_eval(seed, a)


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--stage", required=True, choices=("S", "selS", "C", "gateC", "M"))
    a = p.parse_args()
    if a.stage == "selS":
        return select_s()
    if a.stage == "gateC":
        return gate_c()
    js = jobs(a.stage)
    if a.list:
        return R.print_jobs(js)
    j = js[a.task_id]
    if j.get("kind") == "erm80":
        return run_erm80(j["seed"], a)
    R.run(j, a)


if __name__ == "__main__":
    main()
