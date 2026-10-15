#!/usr/bin/env python3
"""Paper B R5 P12: stronger frozen encoders as the monitor, against the pre-registered mitigation target
(Mahalanobis ≥ 0.90 on pad_heldout and on Fitzpatrick17k) — decisions/paperB_r5_p12.md. CPU scoring.

Features come from r5_frozen_extract.py (GPU-light) on the lesion-level frames: results/paperB/r5/frozen/features/<enc>/.
Encoders scored when present: imagenet_r50 (reference), dinov2_b14, dinov2_l14 (CLS; patch-mean as sensitivity),
monet, panderm_l. Same detectors / CIs / report as P3 (r5_p3_frozen_monitor.score_frozen, .report).
Usage: r5_p12_frozen_encoders.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5cpu_common as R  # noqa: E402
import r5_p3_frozen_monitor as P3  # noqa: E402

OUT = R.R5 / "p12"
ENCODERS = ("imagenet_r50", "dinov2_b14", "dinov2_l14", "monet", "panderm_l")
SPLITS = ("isic_train", "isic_test") + P3.OODS


def encoder_pack(enc, key="z"):
    d = P3.FROZEN / enc
    f = {s: dict(np.load(d / "{}.npz".format(s), allow_pickle=True)) for s in SPLITS}
    if key not in f["isic_test"]:
        return None
    n_te = len(f["isic_test"][key])
    assert n_te == 5067 and len(f["pad_heldout"][key]) == 716 and len(f["fitzpatrick17k"][key]) == 3887, enc
    return {"ztr": f["isic_train"][key].astype(np.float64), "ytr": f["isic_train"]["labels"].astype(int),
            "keep_tr": np.ones(len(f["isic_train"][key]), bool), "zte": f["isic_test"][key].astype(np.float64),
            "yte": f["isic_test"]["labels"].astype(int), "keep_te": np.ones(n_te, bool),
            "oods": {o: f[o][key].astype(np.float64) for o in P3.OODS}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = []
    for enc in ENCODERS:
        if not all((P3.FROZEN / enc / "{}.npz".format(s)).exists() for s in SPLITS):
            print("not extracted:", enc)
            continue
        for key, label in (("z", enc), ("z_patchmean", enc + "_patchmean")):
            pack = encoder_pack(enc, key)
            if pack is not None:
                frozen.append(P3.score_frozen(pack, label))
                print("scored", label, flush=True)
    if not frozen:
        raise SystemExit("no frozen features under {}".format(P3.FROZEN))
    P3.report(frozen, P3.score_model(None), out=OUT,
              title="Paper B R5 P12 — stronger frozen encoders vs the mitigation target", decisions="decisions/paperB_r5_p12.md")


if __name__ == "__main__":
    main()
