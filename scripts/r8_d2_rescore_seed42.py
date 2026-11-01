#!/usr/bin/env python3
"""R8 D2: reuse D1 seed-42 checkpoint; run full Table-1 eval into results/paperB/r8/d2/."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_common as R  # noqa: E402
import r5_models as M  # noqa: E402
import r5_recipe  # noqa: E402
import torch  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

SEL = R.R8 / "d1" / "SELECTED_dinov2_vits14.json"
D1_TAG = "r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_mkmmd30"
D1_RUN = R.R8 / "d1" / "runs" / D1_TAG
D1_CKPT = R.PAPERB / "checkpoints" / "r8_d1" / "runs" / D1_TAG / "last.ckpt"
OUT = R.R8 / "d2" / "runs"


def main():
    sel = json.loads(SEL.read_text())
    d1 = json.loads((D1_RUN / "summary.json").read_text())
    cfg = json.loads((D1_RUN / "config.json").read_text())
    tag = "r8_d2_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_mkmmd30"
    run_dir = OUT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        print("skip_done", tag)
        return
    if not D1_CKPT.exists():
        # fallback: search
        alts = list((R.PAPERB / "checkpoints" / "r8_d1").rglob("**/{}/**/last.ckpt".format(D1_TAG)))
        alts = list((R.PAPERB / "checkpoints" / "r8_d1").rglob("last.ckpt"))
        ckpt = None
        for p in alts:
            if D1_TAG in str(p):
                ckpt = p
                break
        if ckpt is None:
            raise SystemExit("missing D1 last.ckpt for " + D1_TAG)
    else:
        ckpt = D1_CKPT
    seed_everything(42)
    rec, _ = r5_recipe.resolve("current")
    tb = dict(M.TOOLBOX_DEFAULT)
    tb.update(cfg.get("toolbox") or {})
    kw = dict(backbone="dinov2_vits14", n_classes=8, n_domains=2, lr=float(rec["lr"]),
              weight_decay=float(rec["weight_decay"]), cls_lr_multiplier=float(rec["cls_lr_multiplier"]),
              lambda_adv=float(sel["lambda_adv"]), toolbox=tb, cls_weight=None,
              penalty=sel["penalty"], penalty_weight=float(sel["penalty_weight"]),
              total_steps=1, backbone_lr_mult=0.1, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = M.SingleEncToolboxLit(**kw)
    lit.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"])
    net = lit.net.to(device).eval()
    D = R.isic_lesion_data()
    summ = R.eval_isic(lambda x: net(x), D, device, run_dir, workers=6, mode="full")
    # V1
    import r5_p9v2 as V
    probe = V.heldout_probe(run_dir)
    summ.update({
        "tag": tag, "seed": 42, "lambda_adv": sel["lambda_adv"], "backbone": "dinov2_vits14",
        "config": sel["config"], "penalty": sel["penalty"], "penalty_weight": sel["penalty_weight"],
        "backbone_lr_mult": 0.1, "checkpoint_policy": "last", "primary_checkpoint": str(ckpt),
        "v1": probe["V1"]["bal_acc_mean"], "v1_sd": probe["V1"]["bal_acc_sd"],
        "degraded": bool(summ["id_balanced_acc"] < 0.25),
        "gpu_hours": float(d1.get("gpu_hours") or 0), "reused_from_d1": str(D1_RUN),
        "commit": R.git_head(), "recipe": "current",
    })
    shutil.copy2(D1_RUN / "config.json", run_dir / "config.json")
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print("saved", run_dir / "summary.json", "V1", summ["v1"], "id", summ["id_balanced_acc"])


if __name__ == "__main__":
    main()
