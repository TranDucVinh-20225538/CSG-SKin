#!/usr/bin/env python3
"""Lesion-level single-encoder ERM λ = 0 reference (Paper B R5 P8).

Seeds 42/52/62/72/82. last.ckpt primary, best-val secondary. Writes features for patient /
P13 Fitzpatrick bootstrap of all five detectors, plus erm_ref/REPORT.md and summary.csv.

Usage:
  r5_erm_lesion_ref.py train --task_id N   # train seed SEEDS[N] if last.ckpt missing
  r5_erm_lesion_ref.py eval  --task_id N   # dual-ckpt eval + features for SEEDS[N]
  r5_erm_lesion_ref.py report             # mean±SD + bootstrap CIs + REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import train_phase15_1_single_dann as sd  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

ERM = R.P8_DIR / "erm_ref"
CK = R.CKPT / "r5_p8" / "erm_ref"
SEEDS = R.SEEDS
DETS = R.DETS


def tag(seed):
    return "single_dann_ladv0_s{}".format(seed)


def run_dir(seed):
    return ERM / tag(seed)


def ckpt_dir(seed):
    return CK / tag(seed)


def train_seed(seed, a):
    """Train via phase15 single-encoder ERM on the lesion-level split; skip if last.ckpt exists."""
    last = ckpt_dir(seed) / "last.ckpt"
    if a.skip_done and last.exists():
        print("skip_train", last, flush=True)
        return
    sd.p2.isic_splits = R.T.isic_splits_by_lesion
    sys.argv = [sys.argv[0], "--lambda_adv", "0", "--seed", str(seed), "--output_root", str(ERM),
                "--ckpt_root", str(CK), "--num_workers", str(a.num_workers)]
    if a.resume:
        sys.argv.append("--resume")
    if a.dry_run:
        sys.argv.append("--dry_run")
    # Do not pass --skip_done: summary may exist from a best-only eval while we still need a train.
    sd.main()


def _save_feats(net, data, feat_dir, num_workers, device):
    """Write feature npz in the dual-encoder Scorer layout (z_lesion_norm = single-encoder z_norm)."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    tf = build_val_transform_robust()
    sets = {
        "isic_train": SkinDataset(data["train"], transform=tf),
        "isic_val": SkinDataset(data["val"], transform=tf),
        "isic_test": SkinDataset(data["test"], transform=tf),
        "pad_full": SkinDataset(data["pad_full"], transform=tf),
        "pad_adv": SkinDataset(data["pad_adv"], transform=tf),
        "pad_heldout": SkinDataset(data["pad_heldout"], transform=tf),
        "fitzpatrick17k": R.FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf),
    }
    for name, ds in sets.items():
        out = feat_dir / "{}.npz".format(name)
        if out.exists():
            continue
        pack = sd.collect_all(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers), device)
        z = pack["z"]
        np.savez_compressed(
            out,
            z_lesion=z,
            z_lesion_norm=z,
            z_context=np.zeros_like(z),
            logits=pack["logits"],
            labels=pack["labels"],
        )
        print("features", feat_dir.name, name, pack["labels"].shape, flush=True)


def eval_seed(seed, a):
    """last.ckpt primary + best-val secondary; features for bootstrap; summary with checkpoint_policy."""
    rd, cd = run_dir(seed), ckpt_dir(seed)
    summary_path = rd / "summary.json"
    legacy_best = None
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("checkpoint_policy") != R.CKPT_POLICY:
            legacy_best = old.get("id_balanced_acc")
        elif a.skip_done and (ERM / "features" / tag(seed) / "isic_test.npz").exists():
            print("skip_eval", summary_path, flush=True)
            return
        else:
            legacy_best = old.get("legacy_bestval_only_id_balanced_acc")
    last = cd / "last.ckpt"
    best = R.find_best(cd)
    if not last.exists() or best is None:
        raise FileNotFoundError("missing checkpoint for seed {}: last={} best={}".format(seed, last, best))
    if a.dry_run:
        print("dry_run eval", seed, last, best, flush=True)
        return
    data = R.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary = last if R.CKPT_POLICY == "last" else best
    feat_dir = ERM / "features" / tag(seed)
    fb = ERM / "features_bestval" / tag(seed)
    if not a.skip_done:
        for d in (feat_dir, fb):
            if d.exists():
                for p in d.glob("*.npz"):
                    p.unlink()
    net = sd.SingleDannLightning.load_from_checkpoint(str(primary), strict=False, map_location=device).net.to(device).eval()
    _save_feats(net, data, feat_dir, a.num_workers, device)
    m = R.run_metrics(R.load_feats(feat_dir))
    net = sd.SingleDannLightning.load_from_checkpoint(str(best), strict=False, map_location=device).net.to(device).eval()
    _save_feats(net, data, fb, a.num_workers, device)
    secondary = {
        "checkpoint": str(best),
        "epoch": R.ckpt_epoch(best),
        "metrics": R.run_metrics(R.load_feats(fb)),
        "features": str(fb),
    }
    id_bal = float(m["id"]["bal_acc"])
    summary = {
        "run_name": tag(seed),
        "tag": tag(seed),
        "seed": seed,
        "lambda_adv": 0.0,
        "architecture": "single_encoder_dann_effb3",
        "mode": "single-encoder",
        "split": "lesion-level",
        "checkpoint_policy": R.CKPT_POLICY,
        "primary_checkpoint": str(primary),
        "primary_epoch": R.ckpt_epoch(primary),
        "best_checkpoint": str(best),
        "best_epoch": R.ckpt_epoch(best),
        "secondary_bestval": secondary,
        "metrics": m,
        "id_balanced_acc": id_bal,
        "id_acc": float(m["id"]["acc"]),
        "id_ece": float(m["id"]["ece"]),
        "ood": m["ood"],
        "legacy_bestval_only_id_balanced_acc": legacy_best,
    }
    rd.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("saved", summary_path, "id_bal_acc(last)", id_bal,
          "bestval", secondary["metrics"]["id"]["bal_acc"], flush=True)


def _fmt_boot(b, det):
    if b is None or det not in b:
        return "—"
    d = b[det]
    sd = " ± {:.3f}".format(d["sd"]) if d.get("sd") is not None else ""
    return "{:.3f}{} [{:.3f}, {:.3f}]".format(d["mean"], sd, d["ci_lo"], d["ci_hi"])


def report(a):
    rows = []
    for s in SEEDS:
        p = run_dir(s) / "summary.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        if j.get("checkpoint_policy") != R.CKPT_POLICY:
            continue
        m, b = j["metrics"], j["secondary_bestval"]["metrics"]
        rows.append({
            "seed": s,
            "last_test_bal_acc": m["id"]["bal_acc"],
            "bestval_test_bal_acc": b["id"]["bal_acc"],
            "last_val_bal_acc": m.get("val", {}).get("bal_acc"),
            "bestval_val_bal_acc": b.get("val", {}).get("bal_acc"),
            "last_epoch": j["primary_epoch"],
            "best_epoch": j["best_epoch"],
        })
    if len(rows) < 5:
        print("warning: only {}/5 seeds with last.ckpt policy summary".format(len(rows)), flush=True)
    last_ms = R.ms([r["last_test_bal_acc"] for r in rows])
    best_ms = R.ms([r["bestval_test_bal_acc"] for r in rows])
    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    idg = R.idx_groups(lesion_g)
    feat = {r["seed"]: ERM / "features" / tag(r["seed"]) for r in rows}
    boots = {}
    if not a.no_boot and feat:
        for ood in ("pad_heldout", "fitzpatrick17k"):
            og, unit = R.ood_groups(ood, data)
            b = R.bootstrap_cell(feat, ood, idg, og, dets=DETS)
            if b is not None:
                b["unit_ood"] = unit
                boots[ood] = b
    out = {
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "last_ckpt_test_bal_acc": last_ms,
        "bestval_test_bal_acc": best_ms,
        "per_seed": rows,
        "image_level_erm_paper": R.ERM_SINGLE_ENCODER_IMAGE_LEVEL,
        "image_level_note": "leakage-inflated on the image-level split; not the competence reference",
        "competence_target": R.COMPETENCE_TARGET,
        "competence_target_note": (
            "0.70 was set from the image-level single-encoder ERM {:.3f}, which is leakage-inflated; "
            "the lesion-level ERM reference (last.ckpt, this report) is reported alongside; "
            "original ≥ 0.70 verdicts are kept visible".format(R.ERM_SINGLE_ENCODER_IMAGE_LEVEL)
        ),
        "bootstrap": boots,
        "commit": R.C.git_head(),
    }
    ERM.mkdir(parents=True, exist_ok=True)
    (ERM / "REFERENCE.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    import csv
    fields = ["seed", "last_test_bal_acc", "bestval_test_bal_acc", "last_val_bal_acc",
              "bestval_val_bal_acc", "last_epoch", "best_epoch"]
    with (ERM / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    L = [
        "# Lesion-level single-encoder ERM reference (λ = 0)",
        "",
        "Design deviation: `decisions/paperB_r5_p8.md` (2026-10-16). Primary checkpoint: last.ckpt; "
        "best-val secondary. Seeds: {}.".format("/".join(str(s) for s in SEEDS)),
        "",
        "## ID test balanced accuracy (competence reference)",
        "",
        "| checkpoint | n | test bal acc mean ± SD |",
        "|---|---:|---|",
        "| last.ckpt (primary) | {} | {} |".format(last_ms["n"] if last_ms else 0, R.pm(last_ms)),
        "| best-val (secondary) | {} | {} |".format(best_ms["n"] if best_ms else 0, R.pm(best_ms)),
        "",
        "Image-level single-encoder ERM in the paper: {:.3f} (leakage-inflated on the image-level split; not like-for-like). "
        "The competence threshold 0.70 was set from that 0.779 figure; original ≥ 0.70 verdicts stay visible; "
        "this lesion-level last.ckpt mean ± SD is the paper's competence reference.".format(
            R.ERM_SINGLE_ENCODER_IMAGE_LEVEL),
        "",
        "## Per seed",
        "",
        "| seed | last test bal acc | best-val test bal acc | last val bal acc | best-val val bal acc | last epoch | best epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append("| {} | {:.3f} | {:.3f} | {} | {} | {} | {} |".format(
            r["seed"], r["last_test_bal_acc"], r["bestval_test_bal_acc"],
            "—" if r["last_val_bal_acc"] is None else "{:.3f}".format(r["last_val_bal_acc"]),
            "—" if r["bestval_val_bal_acc"] is None else "{:.3f}".format(r["bestval_val_bal_acc"]),
            r["last_epoch"], r["best_epoch"]))
    L += ["", "## Detector AUROC (last.ckpt features; patient / P13 cluster bootstrap)", ""]
    for ood in ("pad_heldout", "fitzpatrick17k"):
        b = boots.get(ood)
        L += ["### {} (unit: {})".format(ood, b.get("unit_ood", "—") if b else "—"), "",
              "| detector | mean ± SD [95% CI] | k seed CIs < 0.5 |",
              "|---|---|---:|"]
        if b is None:
            L.append("| — | — | — |")
        else:
            for d in DETS:
                L.append("| {} | {} | {}/{} |".format(
                    d, _fmt_boot(b, d), b[d]["k_below"], b[d]["n_seeds"]))
        L.append("")
    L += [
        "## LaTeX",
        "",
        "```latex",
        "Lesion-level ERM (last.ckpt) & {} & {} \\\\".format(
            last_ms["n"] if last_ms else 0,
            "{:.3f}\\,$\\pm$\\,{:.3f}".format(last_ms["mean"], last_ms["sd"]) if last_ms and last_ms.get("sd") is not None
            else "—"),
        "```",
        "",
    ]
    (ERM / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", ERM / "REPORT.md", ERM / "REFERENCE.json", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("train", "eval", "report", "train_eval"))
    p.add_argument("--task_id", type=int)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_boot", action="store_true")
    a = p.parse_args()
    if a.cmd == "report":
        return report(a)
    if a.task_id is None or not (0 <= a.task_id < len(SEEDS)):
        raise SystemExit("--task_id in 0..{}".format(len(SEEDS) - 1))
    seed = SEEDS[a.task_id]
    if a.cmd in ("train", "train_eval"):
        train_seed(seed, a)
    if a.cmd in ("eval", "train_eval"):
        eval_seed(seed, a)


if __name__ == "__main__":
    main()
