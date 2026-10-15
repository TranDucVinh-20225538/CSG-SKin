#!/usr/bin/env python3
"""Paper B R5 S1/S2 features (decisions/paperB_r5_s.md). Usage: r5_s_extract.py <worker> <n_workers>.

Runs i with i % n_workers == worker, over 35 lesion-level runs then the 28 phase16 BCN↔HAM runs:
  lesion run  -> isic_val.npz next to the run's cached features (same path as R2 item 4 extract_features:
                 fin.load_csg(best_checkpoint), build_val_transform_robust, x13.collect_depths, REQUIRED keys)
  phase16 run -> results/paperB/r5/s_flip/features/<run>/{id_train,id_val,id_test,ood}.npz with z, logits, labels,
                 group (lesion_id), same path as phase16_b1_coarse / bootstrap_phase16_reverse (SingleDannLightning,
                 build_eval_tf(448), collect_all). The run's stored Mahalanobis AUROC is recomputed as a check.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import train_phase3_sweep as p3  # noqa: E402,F401  (installs the transformers stub)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import bootstrap_ood_ci as B  # noqa: E402
import eval_final_gpu as fin  # noqa: E402
import extract_phase13_features as x13  # noqa: E402
import phase16_b1_coarse as b1  # noqa: E402
import phase16_b1_rev_coarse as b1r  # noqa: E402
import r2_common as C  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402
import train_r2_item4_lesion_split as T  # noqa: E402
from phase16_b0 import build_eval_tf  # noqa: E402
from src.datasets.skin_dataset import SkinDataset, build_val_transform_robust  # noqa: E402
from train_phase15_1_single_dann import SingleDannLightning, collect_all  # noqa: E402

ITEM4 = C.R2 / "item4"
P16 = C.PAPERB / "results" / "paperB" / "phase16"
OUT16 = C.PAPERB / "results" / "paperB" / "r5" / "s_flip" / "features"
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PHASE16 = {"b1": ("bcn_ham", {"id_train": "bcn_train", "id_val": "bcn_val", "id_test": "bcn_test", "ood": "ham_heldout"}),
           "b1_rev": ("ham_bcn", {"id_train": "ham_train", "id_val": "ham_val", "id_test": "ham_test",
                                  "ood": "bcn_heldout"})}


def lesion_tags():
    tags = []
    for lam in LAMS:
        tags += [d.name for d in sorted((ITEM4 / "features").glob("runB_orth1_ladv{}_s*".format(C.lam_tag(lam))))]
    assert len(tags) == 35, len(tags)
    return [("lesion", t) for t in tags]


def phase16_runs():
    out = []
    for sub, (prefix, _) in PHASE16.items():
        out += [(sub, d.name) for d in sorted((P16 / sub).glob(prefix + "_ladv*_s*")) if (d / "summary.json").exists()]
    assert len(out) == 28, len(out)
    return out


def all_runs():
    return lesion_tags() + phase16_runs()


def do_lesion(tag, device):
    out = ITEM4 / "features" / tag / "isic_val.npz"
    if out.exists():
        print("skip_done", tag, flush=True)
        return
    summary = json.loads((ITEM4 / "runs" / tag / "summary.json").read_text())
    net = fin.load_csg(Path(summary["best_checkpoint"]), device)
    _tr, val, _te = T.split_record()[1]
    loader = DataLoader(SkinDataset(val, transform=build_val_transform_robust()), batch_size=64, shuffle=False,
                        num_workers=8)
    with torch.no_grad():
        pack = x13.collect_depths(net, loader, device)
    assert len(pack["labels"]) == len(val)
    assert np.array_equal(np.asarray(pack["labels"]).astype(int), val.label_idx.to_numpy().astype(int)), tag
    np.savez_compressed(out, **{k: pack[k] for k in x13.REQUIRED})
    print("features", tag, "isic_val", pack["z_lesion_norm"].shape, flush=True)


def do_phase16(sub, run, device):
    out = OUT16 / run
    keys = PHASE16[sub][1]
    if all((out / "{}.npz".format(k)).exists() for k in keys):
        print("skip_done", run, flush=True)
        return
    summary = json.loads((P16 / sub / run / "summary.json").read_text())
    _df, splits, _meta, _hyg = (b1 if sub == "b1" else b1r).load_or_create_splits()
    lit = SingleDannLightning.load_from_checkpoint(summary["best_checkpoint"], strict=False)
    net = lit.net.to(device).eval()
    tf = build_eval_tf(448)
    packs = {}
    for k, split in keys.items():
        frame = splits[split]
        with torch.no_grad():
            p = collect_all(net, p2.make_loader(SkinDataset(frame, transform=tf), 64, 8, False), device)
        assert len(p["labels"]) == len(frame)
        p["group"] = frame.lesion_id.astype(str).to_numpy()
        packs[k] = p
    sid, sood = B.maha_scores(packs["id_train"]["z"], packs["id_train"]["labels"], packs["id_test"]["z"],
                              packs["ood"]["z"])
    a, stored = B.auroc_from_scores(sid, sood), float(summary["ood_auroc_maha"]["unrestricted"])
    print("check", run, "maha {:.6f} stored {:.6f} diff {:.2e}".format(a, stored, a - stored), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for k, p in packs.items():
        np.savez_compressed(out / "{}.npz".format(k), **p)
    (out / "check.json").write_text(json.dumps({"run": run, "maha_auroc_recomputed": a, "maha_auroc_stored": stored,
                                                "abs_diff": abs(a - stored)}, indent=1) + "\n")


def main():
    worker, n = int(sys.argv[1]), int(sys.argv[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (kind, name) in enumerate(all_runs()):
        if i % n != worker:
            continue
        if kind == "lesion":
            do_lesion(name, device)
        else:
            do_phase16(kind, name, device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
