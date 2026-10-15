#!/usr/bin/env python3
"""Paper B R5 P8 trajectory (claim (c); decisions/paperB_r5_p8_traj.md). Descriptive, no pass/fail.

Runs: traj_ladv0_s42 and traj_ladv2_s42 (A4 recipe, P8 stage `traj`, per-epoch weights in
checkpoints/r5_p8/traj/<tag>/epochs/). Optional extra: main λ = 2 seeds 52–82 (`--main`), which also save per-epoch
weights; evaluated only if requested.

Subcommands:
  run TAG    per epoch: ID bal acc, held-out leakage (P6 V1), Mahalanobis AUROC on pad_heldout and Fitzpatrick17k
             (detector fit on ISIC train, z_lesion_norm) -> <out>/<tag>.jsonl (one line per epoch, resumable)
  report     summary.csv, TIMING.json, REPORT.md, trajectory.png (adversary train accuracy from lightning_logs)
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import r5_p9v2 as V  # noqa: E402
from eval_fitz_domain_axis import FolderJpegDataset  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402

OUT = R.P8_DIR / "traj" / "per_epoch"
TAGS = ("traj_ladv0_s42", "traj_ladv2_s42")
MAIN_EXTRA = tuple("main_ladv2_s{}".format(s) for s in (52, 62, 72, 82))
CURVES = ("heldout_leakage_V1", "maha_pad_heldout", "maha_fitz", "id_bal_acc", "adv_train_acc")
SMOOTH = 3
BAND = 1


def dirs(tag):
    if tag.startswith("traj_"):
        return R.CKPT / "r5_p8" / "traj" / tag / "epochs", R.P8_DIR / "traj" / "runs" / tag
    return R.CKPT / "r5_p8" / "main" / tag / "epochs", R.P8_DIR / "main" / "runs" / tag


def run(tag):
    ck_dir, _ = dirs(tag)
    out = OUT / "{}.jsonl".format(tag)
    done = {json.loads(l)["epoch"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    ckpts = sorted(ck_dir.glob("ep-*.ckpt"))
    if not ckpts:
        raise SystemExit("no per-epoch checkpoints in {}".format(ck_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = R.load_data()
    tf = build_val_transform_robust()
    sets = {"isic_train": SkinDataset(data["train"], transform=tf), "isic_test": SkinDataset(data["test"], transform=tf),
            "pad_heldout": SkinDataset(data["pad_heldout"], transform=tf),
            "fitzpatrick17k": FolderJpegDataset(sorted(R.p3.FITZ_DIR.glob("*.jpg")), tf)}
    G = V.groups()
    OUT.mkdir(parents=True, exist_ok=True)
    for ck in ckpts:
        ep = int(ck.stem.split("-")[1])
        if ep in done:
            continue
        net = R.load_net(ck, device)
        F = {}
        for name, ds in sets.items():
            pack = R.x13.collect_depths(net, DataLoader(ds, batch_size=64, shuffle=False, num_workers=6), device)
            F[name] = {k: pack[k] for k in ("z_lesion_norm", "logits", "labels")}
        sc = R.Scorer(F["isic_train"])
        te = F["isic_test"]
        assert np.array_equal(te["labels"].astype(int), G["yi"]), "isic_test row order"
        sid = sc.scores(te)["mahalanobis_classcond"]
        rec = {"tag": tag, "epoch": ep,
               "id_bal_acc": float(balanced_accuracy_score(te["labels"], te["logits"].argmax(1))),
               "heldout_leakage_V1": V.v1_probe(te["z_lesion_norm"], F["pad_heldout"]["z_lesion_norm"], G["gi"], G["gp"])["bal_acc_mean"],
               "maha_pad_heldout": R.auroc(sid, sc.scores(F["pad_heldout"])["mahalanobis_classcond"]),
               "maha_fitz": R.auroc(sid, sc.scores(F["fitzpatrick17k"])["mahalanobis_classcond"])}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


def load(tag):
    _ck, run_dir = dirs(tag)
    f = OUT / "{}.jsonl".format(tag)
    if not f.exists():
        return None
    df = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()]).sort_values("epoch")
    m = pd.read_csv(run_dir / "lightning_logs" / "version_0" / "metrics.csv").groupby("epoch").mean(numeric_only=True)
    df["adv_train_acc"] = df.epoch.map(m["train/acc_adv"]) if "train/acc_adv" in m else np.nan
    df["grl_alpha"] = df.epoch.map(m["train/grl_a"]) if "train/grl_a" in m else np.nan
    return df.reset_index(drop=True)


def timing(df, col):
    """Epochs (0-based) at which the SMOOTH-epoch centred running mean first covers 10 % / 50 % / 90 % of its change
    from epoch 0 to the last epoch."""
    y = df[col].rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
    ep = df.epoch.to_numpy()
    tot = y[-1] - y[0]
    out = {"start": float(y[0]), "end": float(y[-1]), "change": float(tot)}
    for q in (0.1, 0.5, 0.9):
        hit = np.flatnonzero((y - y[0]) / tot >= q) if tot != 0 else np.array([], int)
        out["t{}".format(int(q * 100))] = int(ep[hit[0]]) if hit.size else None
    return out


def lagcorr(df, a, b, max_lag=5):
    """Pearson correlation of first differences, corr(Δa[t], Δb[t + lag]) for lag in [-max_lag, max_lag]."""
    da, db = np.diff(df[a].to_numpy()), np.diff(df[b].to_numpy())
    res = {}
    for lag in range(-max_lag, max_lag + 1):
        x, y = (da[:len(da) - lag], db[lag:]) if lag >= 0 else (da[-lag:], db[:len(db) + lag])
        res[lag] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None
    return res


def describe(tl, tm):
    if tl is None or tm is None:
        return "not defined (no change)"
    d = tm - tl
    return "precedes" if d < -BAND else "follows" if d > BAND else "tracks"


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in TAGS + MAIN_EXTRA if (OUT / "{}.jsonl".format(t)).exists()]
    D = {t: load(t) for t in tags}
    pd.concat([d.assign(tag=t) for t, d in D.items()]).to_csv(OUT.parent / "summary.csv", index=False)
    T, L = {}, ["# P8 per-epoch trajectory (claim (c); decisions/paperB_r5_p8_traj.md)", "",
                "Descriptive only. Epochs 0-based (epoch 39 = last.ckpt). Timing on a {}-epoch centred running mean; "
                "t50 = first epoch covering 50 % of the change from epoch 0 to epoch 39. Label: Mahalanobis t50 − leakage "
                "t50 < −{b} → precedes, > +{b} → follows, otherwise tracks.".format(SMOOTH, b=BAND), ""]
    for t, d in D.items():
        tm = {c: timing(d, c) for c in CURVES if d[c].notna().all()}
        lc = {m: lagcorr(d, "heldout_leakage_V1", m) for m in ("maha_pad_heldout", "maha_fitz")}
        lab = {m: describe(tm["heldout_leakage_V1"]["t50"], tm[m]["t50"]) for m in ("maha_pad_heldout", "maha_fitz")}
        T[t] = {"timing": tm, "lagcorr_diff_leak_vs": lc, "label_t50": lab}
        L += ["## {}".format(t), "",
              "| curve | epoch 0 | epoch 39 | change | t10 | t50 | t90 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c, v in tm.items():
            L.append("| {} | {:.3f} | {:.3f} | {:+.3f} | {} | {} | {} |".format(c, v["start"], v["end"], v["change"],
                                                                         v["t10"], v["t50"], v["t90"]))
        for m in ("maha_pad_heldout", "maha_fitz"):
            best = max(((k, v) for k, v in lc[m].items() if v is not None), key=lambda kv: abs(kv[1]), default=(None, None))
            L.append("")
            L.append("{} vs held-out leakage: t50 label **{}**; lagged correlation of first differences (lag > 0 = "
                     "Mahalanobis later), largest |r| at lag {} (r = {}).".format(
                         m, lab[m], best[0], "—" if best[1] is None else "{:.2f}".format(best[1])))
        L.append("")
    l2 = [t for t in ("traj_ladv2_s42",) + MAIN_EXTRA if t in T]
    if l2:
        L += ["## λ = 2 across seeds ({}: {})".format(len(l2), ", ".join(l2)), "",
              "Seed 42 is the trajectory replicate (`traj_ladv2_s42`); seeds 52–82 are the P8 main runs.", "",
              "| run | " + " | ".join("{} t10/t50/t90".format(c) for c in CURVES) + " | label pad_heldout | label Fitzpatrick |",
              "|---|" + "---|" * (len(CURVES) + 2)]
        for t in l2:
            tm = T[t]["timing"]
            L.append("| {} | ".format(t) + " | ".join(
                "{}/{}/{}".format(tm[c]["t10"], tm[c]["t50"], tm[c]["t90"]) if c in tm else "—" for c in CURVES)
                + " | {} | {} |".format(T[t]["label_t50"]["maha_pad_heldout"], T[t]["label_t50"]["maha_fitz"]))
        L += ["", "Spread across seeds (mean ± SD, min–max):", "", "| quantity | mean ± SD | min–max | n |", "|---|---|---|---:|"]
        spread = {}
        for c in CURVES:
            for q in ("t10", "t50", "t90"):
                v = [T[t]["timing"][c][q] for t in l2 if c in T[t]["timing"] and T[t]["timing"][c][q] is not None]
                spread["{} {}".format(c, q)] = v
        for m in ("maha_pad_heldout", "maha_fitz"):
            spread["{} t50 − leakage t50".format(m)] = [
                T[t]["timing"][m]["t50"] - T[t]["timing"]["heldout_leakage_V1"]["t50"] for t in l2
                if T[t]["timing"][m]["t50"] is not None and T[t]["timing"]["heldout_leakage_V1"]["t50"] is not None]
        for k, v in spread.items():
            L.append("| {} | {} | {} | {} |".format(k, R.pm(R.ms(v), 1), "{}–{}".format(min(v), max(v)) if v else "—", len(v)))
        for m in ("maha_pad_heldout", "maha_fitz"):
            labs = [T[t]["label_t50"][m] for t in l2]
            L.append("")
            L.append("{} labels: {}".format(m, ", ".join("{} {}".format(labs.count(x), x) for x in sorted(set(labs)))))
        L.append("")
        T["_lambda2_spread"] = spread
    (OUT.parent / "TIMING.json").write_text(json.dumps(T, indent=1) + "\n")

    fig, ax = plt.subplots(len(D), 2, figsize=(12, 3.6 * len(D)), squeeze=False)
    for i, (t, d) in enumerate(D.items()):
        for c, lab in (("maha_pad_heldout", "Maha AUROC pad_heldout"), ("maha_fitz", "Maha AUROC Fitzpatrick"),
                       ("heldout_leakage_V1", "held-out leakage (V1)")):
            ax[i, 0].plot(d.epoch, d[c], marker=".", label=lab)
        ax[i, 0].axhline(0.5, ls="--", c="grey", lw=0.8)
        ax[i, 0].set_title(t), ax[i, 0].set_xlabel("epoch (0-based)"), ax[i, 0].legend(fontsize=8)
        for c, lab in (("adv_train_acc", "adversary train acc"), ("id_bal_acc", "ID bal acc")):
            ax[i, 1].plot(d.epoch, d[c], marker=".", label=lab)
        ax[i, 1].axhline(0.5, ls="--", c="grey", lw=0.8)
        ax[i, 1].set_title(t), ax[i, 1].set_xlabel("epoch (0-based)"), ax[i, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT.parent / "trajectory.png", dpi=150)
    L += ["![trajectory](trajectory.png)", ""]
    if len(l2) > 1:
        A = pd.concat([D[t].assign(tag=t) for t in l2])
        g = A.groupby("epoch")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for c, lab, k in (("maha_pad_heldout", "Maha AUROC pad_heldout", 0), ("maha_fitz", "Maha AUROC Fitzpatrick", 0),
                          ("heldout_leakage_V1", "held-out leakage (V1)", 0), ("adv_train_acc", "adversary train acc", 1),
                          ("id_bal_acc", "ID bal acc", 1)):
            mu, sd = g[c].mean(), g[c].std()
            ax[k].plot(mu.index, mu, label=lab)
            ax[k].fill_between(mu.index, mu - sd, mu + sd, alpha=0.2)
        for a in ax:
            a.axhline(0.5, ls="--", c="grey", lw=0.8)
            a.set_xlabel("epoch (0-based)")
            a.legend(fontsize=8)
        fig.suptitle("λ = 2, mean ± SD over {} seeds".format(len(l2)))
        fig.tight_layout()
        fig.savefig(OUT.parent / "trajectory_lambda2_seeds.png", dpi=150)
        L += ["![λ = 2 across seeds](trajectory_lambda2_seeds.png)", ""]
    (OUT.parent / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "report": report}[sys.argv[1]]()
