#!/usr/bin/env python3
"""CPU re-score of existing P10 shift-arms checkpoints on NCT-CRC-HE (bug fix).

Root cause: train_r5_p10_camelyon_toolbox.py wrote semantic_ood into config.json but never
loaded NCT into the ood dict. This script attaches ood.nct_crc_he detectors for each existing
shift-arms best.pt without retraining. Gate-failed runs: numbers are descriptive only.

NFS / DataLoader resilience: --num_workers 0 (default), optional --local_scratch copy of the
NCT eval list into $SLURM_TMPDIR|/tmp, and retries on OSError/RuntimeError.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_r2_item5_camelyon2 as c2  # noqa: E402
import train_r5_p10_camelyon_toolbox as P10  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

p12 = c2.p12
DET = "mahalanobis_classcond_sharedcov"
ARMS = ("natural_l2", "induced_l2")
SEEDS = (42, 52, 62, 72, 82)


def parse_only(specs):
    """Parse 'induced_l2:72,induced_l2:82' or 'induced_l2:72'."""
    out = set()
    for spec in specs or []:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            arm, seed = part.split(":")
            out.add((arm.strip(), int(seed)))
    return out


def stage_nct_local(scratch: Path):
    """Copy NCT eval paths into scratch; return rewritten path list. Flat names to avoid deep trees."""
    src_paths = P10.nct_path_list()
    dest = scratch / "nct_eval"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / "paths.txt"
    if manifest.exists() and manifest.stat().st_size > 0:
        local = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
        if len(local) == len(src_paths) and all(Path(p).exists() for p in local[:5]):
            print("reuse local NCT staging", dest, "n=", len(local), flush=True)
            return local
    local = []
    for i, src in enumerate(src_paths):
        # preserve extension; unique index avoids collisions across classes
        ext = Path(src).suffix or ".tif"
        dst = dest / "{:05d}{}".format(i, ext)
        if not dst.exists() or dst.stat().st_size == 0:
            shutil.copy2(src, dst)
        local.append(str(dst))
        if (i + 1) % 500 == 0:
            print("staged NCT", i + 1, "/", len(src_paths), flush=True)
    manifest.write_text("\n".join(local) + "\n")
    print("staged NCT to", dest, "n=", len(local), flush=True)
    return local


def collect_nct_paths(paths, ev, device, num_workers, batch=128):
    ds = P10._PathImageDataset(paths, p12.eval_tf())
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                        pin_memory=False)
    return p12.collect(ev, loader, device)


def score_one(tag, arm, seed, device, num_workers, nct_paths, dry_run, retries, retry_sleep):
    run_dir = P10.OUT / "shift_arms" / tag
    ckpt = P10.CKPT / "shift_arms" / tag / "best.pt"
    summary_p = run_dir / "summary.json"
    if not summary_p.exists() or not ckpt.exists():
        print("missing", tag, flush=True)
        return None
    summary = json.loads(summary_p.read_text())
    if P10.NCT_KEY in summary.get("ood", {}) and summary["ood"][P10.NCT_KEY].get("detectors"):
        maha = summary["ood"][P10.NCT_KEY]["detectors"][DET]
        print("already", tag, "maha", maha, flush=True)
        return {"arm": arm, "seed": seed, "maha": maha, "gate_met": summary.get("gate_met"),
                "note": "gate failed, descriptive only", "status": "already"}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            tb = summary["toolbox"]
            model = P10.Net(tb["head_specs"]).to(device)
            blob = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(blob["state_dict"])
            model.eval()
            ev = P10._EvalWrap(model)
            # Camelyon ID packs: workers 0 to avoid NFS abort on a_train/a_test too
            packs = {k: p12.collect(ev, p12.loader_for(ds_global, sp_global[k], p12.eval_tf(), 256,
                                                       0, False), device)
                     for k in ("a_train", "a_test")}
            if dry_run:
                print("dry_run would score", tag, "n_nct", len(nct_paths), flush=True)
                return {"arm": arm, "seed": seed, "maha": None, "status": "dry_run"}
            nct = collect_nct_paths(nct_paths, ev, device, num_workers=num_workers)
            tr, idp = packs["a_train"], packs["a_test"]
            block = p12.detector_block(tr["z"], tr["y"], idp["z"], idp["logits"], nct["z"], nct["logits"],
                                       p12.KNN_BANK, seed)
            ood = dict(summary.get("ood") or {})
            ood[P10.NCT_KEY] = {"n": int(len(nct["y"])), "detectors": block,
                                "note": "semantic OOD NCT-CRC-HE-100K; AUROC ID(a_test) vs OOD; "
                                        "no label map; gate failed, descriptive only"}
            summary["ood"] = ood
            summary["nct_rescore"] = {
                "bug_fix": True,
                "note": "NCT was missing from ood; re-scored from best.pt on CPU",
                "gate_failed_descriptive_only": not bool(summary.get("gate_met")),
                "num_workers": num_workers,
                "attempt": attempt,
                "local_scratch": bool(os.environ.get("NCT_LOCAL_SCRATCH")),
            }
            summary_p.write_text(json.dumps(summary, indent=2) + "\n")
            maha = block[DET]
            print("scored", tag, "maha", maha, "attempt", attempt, flush=True)
            return {"arm": arm, "seed": seed, "maha": maha, "gate_met": summary.get("gate_met"),
                    "note": "gate failed, descriptive only", "status": "scored", "attempt": attempt}
        except (OSError, RuntimeError) as e:
            last_err = e
            print("retryable error", tag, "attempt", attempt, type(e).__name__, e, flush=True)
            time.sleep(retry_sleep * attempt)
    print("FAILED", tag, last_err, flush=True)
    return {"arm": arm, "seed": seed, "maha": None, "status": "failed", "error": str(last_err)}


# filled in main for pack collection
ds_global = None
sp_global = None


def main():
    global ds_global, sp_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers for NCT (0 = safest on NFS)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only", action="append", default=[],
                    help="Restrict to arm:seed (repeatable / comma-separated), e.g. induced_l2:72")
    ap.add_argument("--local_scratch", action="store_true",
                    help="Copy NCT eval images to $SLURM_TMPDIR or /tmp before scoring")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry_sleep", type=float, default=5.0)
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    only = parse_only(args.only)

    from wilds import get_dataset
    ds_global = get_dataset(dataset="camelyon17", root_dir=str(p12.DATA_ROOT), download=False)
    sp_global = dict(np.load(c2.split_path("primary")))

    if args.local_scratch:
        # Prefer Slurm per-job tmp; else user-private dir under $TMPDIR|/tmp (shared /tmp/p10_nct may be denied).
        if os.environ.get("SLURM_TMPDIR"):
            scratch = Path(os.environ["SLURM_TMPDIR"]) / "p10_nct"
        else:
            scratch = Path(os.environ.get("TMPDIR") or "/tmp") / "toandq_p10_nct_{}".format(os.getuid())
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ["NCT_LOCAL_SCRATCH"] = str(scratch)
        nct_paths = stage_nct_local(scratch)
    else:
        nct_paths = P10.nct_path_list()

    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            if only and (arm, seed) not in only:
                continue
            tag = "camelyon2_{}_ladv2_s{}".format(arm, seed)
            row = score_one(tag, arm, seed, device, args.num_workers, nct_paths,
                            args.dry_run, args.retries, args.retry_sleep)
            if row:
                rows.append(row)

    # merge with any previous nct_rescore.json
    out = P10.OUT / "shift_arms" / "nct_rescore.json"
    prev = {}
    if out.exists():
        try:
            prev = {(r["arm"], r["seed"]): r for r in json.loads(out.read_text()).get("rows", [])}
        except Exception:
            prev = {}
    for r in rows:
        if r.get("maha") is not None:
            prev[(r["arm"], r["seed"])] = r
    # also pick up already-scored from summaries not in this run
    for arm in ARMS:
        for seed in SEEDS:
            if (arm, seed) in prev:
                continue
            tag = "camelyon2_{}_ladv2_s{}".format(arm, seed)
            sp = P10.OUT / "shift_arms" / tag / "summary.json"
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            if P10.NCT_KEY in s.get("ood", {}) and s["ood"][P10.NCT_KEY].get("detectors"):
                prev[(arm, seed)] = {
                    "arm": arm, "seed": seed,
                    "maha": s["ood"][P10.NCT_KEY]["detectors"][DET],
                    "gate_met": s.get("gate_met"),
                    "note": "gate failed, descriptive only", "status": "from_summary",
                }
    all_rows = [prev[k] for k in sorted(prev)]
    out.write_text(json.dumps({"label": "gate failed, descriptive only", "detector": DET,
                               "rows": all_rows}, indent=2) + "\n")
    print("wrote", out, flush=True)
    print("\nNCT Maha AUROC (gate failed, descriptive only)")
    print(f"{'arm':12s} {'seed':>4} {'Maha':>7}  status")
    for r in all_rows:
        m = "—" if r.get("maha") is None else "{:7.3f}".format(r["maha"])
        print(f"{r['arm']:12s} {r['seed']:4d} {m}  {r.get('status','')}")


if __name__ == "__main__":
    main()
