#!/usr/bin/env python3
"""CPU re-score of existing P10 shift-arms checkpoints on NCT-CRC-HE (bug fix).

Root cause: train_r5_p10_camelyon_toolbox.py wrote semantic_ood into config.json but never
loaded NCT into the ood dict. This script attaches ood.nct_crc_he detectors for each existing
shift-arms best.pt without retraining. Gate-failed runs: numbers are descriptive only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_r2_item5_camelyon2 as c2  # noqa: E402
import train_r5_p10_camelyon_toolbox as P10  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

p12 = c2.p12
DET = "mahalanobis_classcond_sharedcov"
ARMS = ("natural_l2", "induced_l2")
SEEDS = (42, 52, 62, 72, 82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    from wilds import get_dataset
    ds = get_dataset(dataset="camelyon17", root_dir=str(p12.DATA_ROOT), download=False)
    sp = dict(np.load(c2.split_path("primary")))

    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            tag = "camelyon2_{}_ladv2_s{}".format(arm, seed)
            run_dir = P10.OUT / "shift_arms" / tag
            ckpt = P10.CKPT / "shift_arms" / tag / "best.pt"
            summary_p = run_dir / "summary.json"
            if not summary_p.exists() or not ckpt.exists():
                print("missing", tag, flush=True)
                continue
            summary = json.loads(summary_p.read_text())
            if P10.NCT_KEY in summary.get("ood", {}) and summary["ood"][P10.NCT_KEY].get("detectors"):
                maha = summary["ood"][P10.NCT_KEY]["detectors"][DET]
                print("already", tag, "maha", maha, flush=True)
                rows.append({"arm": arm, "seed": seed, "maha": maha, "gate_met": summary.get("gate_met"),
                             "note": "gate failed, descriptive only"})
                continue
            tb = summary["toolbox"]
            model = P10.Net(tb["head_specs"]).to(device)
            blob = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(blob["state_dict"])
            model.eval()
            ev = P10._EvalWrap(model)
            packs = {k: p12.collect(ev, p12.loader_for(ds, sp[k], p12.eval_tf(), 256, args.num_workers, False),
                                    device)
                     for k in ("a_train", "a_test")}
            if args.dry_run:
                print("dry_run would score", tag, "n_nct", len(P10.nct_path_list()), flush=True)
                continue
            ood = dict(summary.get("ood") or {})
            P10.score_nct_into_ood(ood, packs, ev, device, seed, num_workers=args.num_workers)
            summary["ood"] = ood
            summary["nct_rescore"] = {
                "bug_fix": True,
                "note": "NCT was missing from ood because eval loop omitted it; re-scored from best.pt on CPU",
                "gate_failed_descriptive_only": not bool(summary.get("gate_met")),
            }
            summary_p.write_text(json.dumps(summary, indent=2) + "\n")
            maha = ood[P10.NCT_KEY]["detectors"][DET]
            print("scored", tag, "maha", maha, flush=True)
            rows.append({"arm": arm, "seed": seed, "maha": maha, "gate_met": summary.get("gate_met"),
                         "note": "gate failed, descriptive only"})

    out = P10.OUT / "shift_arms" / "nct_rescore.json"
    out.write_text(json.dumps({"label": "gate failed, descriptive only", "detector": DET,
                               "rows": rows}, indent=2) + "\n")
    print("wrote", out, flush=True)
    print("\nNCT Maha AUROC (gate failed, descriptive only)")
    print(f"{'arm':12s} {'seed':>4} {'Maha':>7}")
    for r in rows:
        print(f"{r['arm']:12s} {r['seed']:4d} {r['maha']:7.3f}")


if __name__ == "__main__":
    main()
