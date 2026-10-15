#!/usr/bin/env python3
"""Paper B R5 P8: validation balanced-accuracy noise for A4 λ = 0 seed 42 (decisions/paperB_r5_p8.md, 2026-10-15 23:30).

(a) Lesion-clustered bootstrap SE of val bal acc (lesion-level ISIC val split; PREREGISTER 2000 draws, seed 20261005)
    at the best-val (epoch 1) and last (epoch 39) checkpoints, from the P8 main features.
(b) Epoch-to-epoch variation of the logged val bal acc (val/acc = macro recall) after epoch 1 (epochs 2–39), from the
    grid run's lightning log (the main λ = 0 s42 run reuses that checkpoint).
Outputs: results/paperB/r5/p8/val_se/{val_se.json, summary.csv, REPORT.md}
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

OUT = R.P8_DIR / "val_se"
TAG = "main_ladv0_s42"
LOG = R.P8_DIR / "grid" / "runs" / "A4_e40_ladv0_s42" / "lightning_logs" / "version_0" / "metrics.csv"


def main():
    pre = json.loads(R.PREREG.read_text())
    n_boot, seed = int(pre["n_resamples"]), int(pre["rng_seed"])
    val = R.load_data()["val"]
    g, _ = R.T.lesion_groups(val)
    groups = R.idx_groups(g)
    s = json.loads((R.P8_DIR / "main" / "runs" / TAG / "summary.json").read_text())
    rec = {"run": TAG, "n_val_images": len(val), "n_val_lesions": len(groups), "n_boot": n_boot, "rng_seed": seed}
    for ck, d, ep in (("best-val", R.P8_DIR / "main" / "features_bestval" / TAG, s["best_epoch"]),
                      ("last", R.P8_DIR / "main" / "features" / TAG, s["primary_epoch"])):
        v = np.load(d / "isic_val.npz")
        y, p = v["labels"].astype(int), v["logits"].argmax(1)
        assert len(y) == len(val)
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(n_boot):
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            bs.append(balanced_accuracy_score(y[idx], p[idx]))
        bs = np.asarray(bs)
        rec[ck] = {"epoch": int(ep), "val_bal_acc": float(balanced_accuracy_score(y, p)), "boot_se": float(bs.std(ddof=1)),
                   "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    m = pd.read_csv(LOG).groupby("epoch")["val/acc"].mean().dropna()
    after = m[m.index >= 2]
    d = np.abs(np.diff(after.to_numpy()))
    rec["logged_val_bal_acc"] = {"per_epoch": {int(k): float(x) for k, x in m.items()},
                                 "epoch1": float(m.loc[1]), "after_epoch1_min": float(after.min()),
                                 "after_epoch1_max": float(after.max()), "after_epoch1_range": float(after.max() - after.min()),
                                 "after_epoch1_sd": float(after.std(ddof=1)),
                                 "abs_epoch_to_epoch_change_median": float(np.median(d)),
                                 "abs_epoch_to_epoch_change_max": float(d.max())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val_se.json").write_text(json.dumps(rec, indent=1) + "\n")
    pd.DataFrame({"epoch": m.index, "val_bal_acc": m.to_numpy()}).to_csv(OUT / "summary.csv", index=False)
    lg = rec["logged_val_bal_acc"]
    line = ("Validation balanced accuracy (A4, λ = 0, seed 42; lesion-level ISIC validation split, {} images / {} lesions) "
            "has a lesion-clustered bootstrap SE of {:.3f} at the best-val checkpoint (epoch {}) and {:.3f} at the last "
            "checkpoint (epoch {}) ({} resamples); after epoch 1 it ranges {:.3f}–{:.3f} across epochs 2–39 (median "
            "epoch-to-epoch change {:.3f}).").format(
        len(val), len(groups), rec["best-val"]["boot_se"], rec["best-val"]["epoch"], rec["last"]["boot_se"],
        rec["last"]["epoch"], n_boot, lg["after_epoch1_min"], lg["after_epoch1_max"], lg["abs_epoch_to_epoch_change_median"])
    L = ["# Validation bal acc noise, A4 λ = 0 seed 42", "",
         "| checkpoint | epoch | val bal acc | bootstrap SE | 95 % CI |", "|---|---:|---:|---:|---|"]
    for ck in ("best-val", "last"):
        r = rec[ck]
        L.append("| {} | {} | {:.3f} | {:.3f} | [{:.3f}, {:.3f}] |".format(ck, r["epoch"], r["val_bal_acc"], r["boot_se"], *r["ci95"]))
    L += ["", "Logged val bal acc: epoch 1 = {:.3f}; epochs 2–39 min {:.3f}, max {:.3f}, range {:.3f}, SD {:.3f}; "
          "|epoch-to-epoch change| median {:.3f}, max {:.3f}.".format(
              lg["epoch1"], lg["after_epoch1_min"], lg["after_epoch1_max"], lg["after_epoch1_range"], lg["after_epoch1_sd"],
              lg["abs_epoch_to_epoch_change_median"], lg["abs_epoch_to_epoch_change_max"]), "",
          "Methods line:", "", line, "", "LaTeX row (checkpoint & epoch & val bal acc & SE):", "", "```"]
    L += ["{} & {} & {:.3f} & {:.3f} \\\\".format(ck, rec[ck]["epoch"], rec[ck]["val_bal_acc"], rec[ck]["boot_se"])
          for ck in ("best-val", "last")] + ["```", ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
