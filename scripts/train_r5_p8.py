#!/usr/bin/env python3
"""Paper B R5 P8: stronger dual-encoder base model (decisions/paperB_r5_p8.md).

Stages (each a SLURM array over --task_id; see slurm/r5_p8_*.sbatch and scripts/launch_r5_p8.sh):
  A     8 recipes x seeds 42, 52 at λ = 0, 40 epochs, validation-only evaluation  (+ 2 lesion-level single-encoder
        ERM reference runs, tasks 16-17)
  selA  pick the stage-A recipe on mean validation balanced accuracy (rule in the decisions file)
  B     the stage-A winner at 80 epochs, seeds 42, 52, validation-only
  selB  keep 80 epochs only if mean val bal acc gains >= 0.02; writes results/paperB/r5/p8/selection_stageB.json
  main  selected recipe, λ ∈ {0, 0.5, 2} x seeds 42/52/62/72/82, full evaluation (λ = 0 seeds 42/52 reuse the
        stage A/B checkpoint of the selected recipe: identical recipe, λ and seed)
  ext   λ ∈ {0.25, 1, 4, 8} x 5 seeds; runs only if RECIPE_DECISION.json says the inversion survives (rule A at λ=2)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

GRID = R.P8_DIR / "grid"
MAIN = R.P8_DIR / "main"
TRAJ = R.P8_DIR / "traj"
ERM = R.P8_DIR / "erm_ref"
CK = R.CKPT / "r5_p8"
SEL_A = R.P8_DIR / "selection_stageA.json"
SEL = R.P8_DIR / "selection_stageB.json"
# read by r5_recipe.py (--recipe p8 for Q1/P9/P10/Q3/P11): flat dict, keys must be a subset of r5_recipe.CURRENT
HOOK = R.P8_DIR / "SELECTED_RECIPE.json"
HOOK_BALANCE = {"none": "none", "loss": "weighted_ce", "sampling": "sampler"}
DECISION = R.P8_DIR / "RECIPE_DECISION.json"
GRID_SEEDS = (42, 52)
MAIN_LAMS = (0.0, 0.5, 2.0)
EXT_LAMS = (0.25, 1.0, 4.0, 8.0)
TIE = 0.01
LONGER_GAIN = 0.02
LOCK_40_EPOCHS = True  # user decision 2026-10-15: no 80-epoch schedule (keeps R1/R2/P7b at 40-epoch cost)


def n_changed(cid):
    g = R.P8_GRID[cid]
    return int(g["latent"] != 16) + int(g["cls_lr_mult"] != 0.2) + int(g["balance"] != "none")


def grid_spec(cid, epochs, seed):
    r = R.grid_recipe(cid, epochs)
    return {"item": "p8", "out_root": str(GRID), "ckpt_root": str(CK / "grid"), "tag": "{}_ladv0_s{}".format(r["name"], seed),
            "recipe": r, "lam": 0.0, "seed": seed, "mode": R.default_mode(r), "align": "dann", "eval": "val"}


def jobs(stage):
    if stage == "A":
        out = [grid_spec(cid, 40, s) for cid in R.P8_GRID for s in GRID_SEEDS]
        out += [{"kind": "erm_ref", "tag": "erm_ref_s{}".format(s), "seed": s, "lam": 0.0, "mode": "single-encoder",
                 "recipe": {"name": "single_encoder_erm"}} for s in GRID_SEEDS]
        return out
    if stage == "B":
        cid = json.loads(SEL_A.read_text())["winner"]
        return [grid_spec(cid, 80, s) for s in GRID_SEEDS]
    if stage in ("main", "ext", "traj"):
        sel = json.loads(SEL.read_text())
        r = R.resolve_recipe("p8")
        lams = EXT_LAMS if stage == "ext" else MAIN_LAMS
        out = []
        for lam in lams:
            for s in R.SEEDS:
                reuse = None
                if lam == 0.0 and s in GRID_SEEDS:
                    reuse = str(CK / "grid" / "{}_ladv0_s{}".format(sel["recipe"]["grid_name"], s))
                out.append({"item": "p8", "out_root": str(MAIN), "ckpt_root": str(CK / "main"),
                            "tag": "main_ladv{}_s{}".format(R.lam_tag(lam), s), "recipe": r, "lam": lam, "seed": s,
                            "mode": R.default_mode(r), "align": "dann", "eval": "full", "reuse_ckpt": reuse,
                            "save_every_epoch": stage == "main" and lam in (0.0, 2.0) and reuse is None})
        if stage == "traj":
            return [dict(j, tag=j["tag"].replace("main_", "traj_"), out_root=str(TRAJ), ckpt_root=str(CK / "traj"),
                         reuse_ckpt=None, save_every_epoch=True)
                    for j in out if j["seed"] == 42 and j["lam"] in (0.0, 2.0)]
        if stage == "main":
            first = ("main_ladv0_s42", "main_ladv2_s42")
            out = [j for t in first for j in out if j["tag"] == t] + [j for j in out if j["tag"] not in first]
        return out
    raise SystemExit(stage)


def val_scores(epochs):
    """{cid: [val bal acc per seed]} for grid runs at the given epoch count."""
    out = {}
    for cid in R.P8_GRID:
        for s in GRID_SEEDS:
            p = GRID / "runs" / "{}_e{}_ladv0_s{}".format(cid, epochs, s) / "summary.json"
            if p.exists():
                out.setdefault(cid, []).append(json.loads(p.read_text())["metrics"]["val"]["bal_acc"])
    return out


def select_a():
    sc = val_scores(40)
    missing = [c for c in R.P8_GRID if len(sc.get(c, [])) < len(GRID_SEEDS)]
    if missing:
        raise SystemExit("stage A incomplete for {}; selection not made".format(missing))
    mean = {c: sum(v) / len(v) for c, v in sc.items()}
    top = max(mean.values())
    near = [c for c in mean if mean[c] >= top - TIE]
    winner = sorted(near, key=lambda c: (n_changed(c), R.P8_GRID[c]["latent"], list(R.P8_GRID).index(c)))[0]
    rec = {"rule": "highest mean val bal acc (seeds 42, 52); among recipes within {} of the top: fewest changes from "
                   "the current recipe, then smaller latent, then grid order".format(TIE),
           "mean_val_bal_acc": mean, "per_seed": sc, "within_tie": near, "winner": winner,
           "winner_recipe": R.P8_GRID[winner]}
    SEL_A.parent.mkdir(parents=True, exist_ok=True)
    SEL_A.write_text(json.dumps(rec, indent=1) + "\n")
    print("stage A winner", winner, mean[winner])


def select_b():
    a = json.loads(SEL_A.read_text())
    cid = a["winner"]
    v40 = a["per_seed"][cid]
    if LOCK_40_EPOCHS:
        v80, gain, epochs = [], None, 40
        rule = "40 epochs fixed by the user (2026-10-15); stage B not run"
    else:
        v80 = val_scores(80).get(cid, [])
        if len(v80) < len(GRID_SEEDS):
            raise SystemExit("stage B incomplete; selection not made")
        gain = sum(v80) / len(v80) - sum(v40) / len(v40)
        epochs = 80 if gain >= LONGER_GAIN else 40
        rule = "80 epochs only if gain >= {}".format(LONGER_GAIN)
    r = dict(R.P8_GRID[cid], max_epochs=epochs, grid_name="{}_e{}".format(cid, epochs), grid_id=cid)
    SEL.write_text(json.dumps({"recipe": r, "stageA_winner": cid, "val_bal_acc_e40": v40, "val_bal_acc_e80": v80,
                               "gain_e80_minus_e40": gain, "rule": rule}, indent=1) + "\n")
    hook = {"lesion_latent_dim": int(r["latent"]), "cls_lr_multiplier": float(r["cls_lr_mult"]),
            "max_epochs": int(epochs), "class_balance": HOOK_BALANCE[r["balance"]],
            "_source": str(SEL), "_grid_name": r["grid_name"]}
    HOOK.write_text(json.dumps(hook, indent=1) + "\n")
    print("selected", r)


def run_erm_ref(seed, a):
    import train_phase15_1_single_dann as sd

    sd.p2.isic_splits = R.T.isic_splits_by_lesion
    sys.argv = [sys.argv[0], "--lambda_adv", "0", "--seed", str(seed), "--output_root", str(ERM),
                "--ckpt_root", str(CK / "erm_ref"), "--num_workers", str(a.num_workers)]
    sys.argv += [f for f, on in (("--resume", a.resume), ("--skip_done", a.skip_done), ("--dry_run", a.dry_run)) if on]
    sd.main()


def main():
    p = R.common_args(argparse.ArgumentParser())
    p.add_argument("--stage", required=True, choices=("A", "selA", "B", "selB", "main", "ext", "traj"))
    p.add_argument("--force_ext", action="store_true", help="run ext even if the gate file says no")
    a = p.parse_args()
    if a.stage == "selA":
        return select_a()
    if a.stage == "selB":
        return select_b()
    if a.stage == "ext" and not a.force_ext:
        d = json.loads(DECISION.read_text()) if DECISION.exists() else {}
        if not d.get("inversion_survives"):
            print("ext gate: inversion_survives =", d.get("inversion_survives"), "-> no extension run", flush=True)
            return
    js = jobs(a.stage)
    if a.list:
        return R.print_jobs(js)
    j = js[a.task_id]
    if j.get("kind") == "erm_ref":
        return run_erm_ref(j["seed"], a)
    R.run(j, a)


if __name__ == "__main__":
    main()
