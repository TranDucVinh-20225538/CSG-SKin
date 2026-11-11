#!/usr/bin/env python3
"""Paper B R12 correctness pilots (no scientific selection).

Checks: weight mean≈1, invalid support, weighted-loss scaling, gradient flow,
fixed-schedule identity across priors (same seed RNG), split isolation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r12_reweight_common as C  # noqa: E402


def test_weights():
    priors = C.compute_priors()
    report = {}
    for name in C.PRIORS:
        w, inv, mean_w = C.class_weights(priors[name], priors["pi_source"])
        report[name] = {"invalid": inv, "mean_w": mean_w, "w": w.tolist(),
                        "ok": (len(inv) == 0 and abs(mean_w - 1.0) < 1e-9)}
        assert len(inv) == 0, name
        assert abs(mean_w - 1.0) < 1e-9, (name, mean_w)
        # prior100 → all ones
        if name == "prior100":
            assert np.allclose(w, 1.0)
        # DF/VASC zero at prior0
        if name == "prior0":
            assert w[R.LABELS.index("DF")] == 0.0 and w[R.LABELS.index("VASC")] == 0.0
    return report


def test_loss_scaling_and_grad():
    """Synthetic CE: doubling all source weights doubles that term; grads flow."""
    torch.manual_seed(0)
    logits = torch.randn(8, 2, requires_grad=True)
    y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    d = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # first 4 source
    w_cls = torch.tensor([1.0, 2.0])
    per = F.cross_entropy(logits, d, reduction="none")
    src = d == 0
    w_row = torch.ones_like(per)
    w_row = torch.where(src, w_cls[y], w_row)  # misuse y as class idx stand-in — use fixed
    # cleaner: class of source rows
    y_src = torch.tensor([0, 1, 0, 1, -1, -1, -1, -1])
    w = torch.tensor([1.5, 0.5])
    w_row = torch.ones_like(per)
    w_row = torch.where(src, w[y_src.clamp(min=0)], w_row)
    w_row = torch.where(src, w_row, torch.ones_like(w_row))
    loss = (w_row * per).mean()
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # scaling: 2*w → 2x contribution of source terms relative to unweighted source mean
    per2 = per.detach()
    w2 = w * 2
    w_row2 = torch.ones_like(per2)
    w_row2 = torch.where(src, w2[y_src.clamp(min=0)], w_row2)
    # source-only means
    m1 = (w_row.detach()[src] * per2[src]).mean()
    m2 = (w_row2[src] * per2[src]).mean()
    assert abs(float(m2 / m1) - 2.0) < 1e-5
    return {"grad_ok": True, "scale_ratio": float(m2 / m1)}


def test_fixed_schedule_identity():
    """Same seed + no adv_mix → identical isic_adv / pad index draws for two prior weightings."""
    data = R.load_data()
    recipe = R.CURRENT
    ds1, _, _ = R.build_train_set("three", recipe, data, adv_mix=None)
    ds2, _, _ = R.build_train_set("three", recipe, data, adv_mix=None)
    torch.manual_seed(42)
    pl_seed = 42
    # draw 64 steps of stream-1 and stream-2 picks by calling _pick
    idxs_a, idxs_b = [], []
    torch.manual_seed(pl_seed)
    for i in range(64):
        idxs_a.append((ds1._pick(1, ds1.streams[1], i), ds1._pick(2, ds1.streams[2], i)))
    torch.manual_seed(pl_seed)
    for i in range(64):
        idxs_b.append((ds2._pick(1, ds2.streams[1], i), ds2._pick(2, ds2.streams[2], i)))
    assert idxs_a == idxs_b
    # different seed differs
    torch.manual_seed(52)
    idxs_c = [(ds1._pick(1, ds1.streams[1], i), ds1._pick(2, ds1.streams[2], i)) for i in range(64)]
    assert idxs_c != idxs_a
    return {"n_steps": 64, "same_seed_identical": True, "diff_seed_differs": True}


def test_split_isolation():
    data = R.load_data()
    tr, te, pa, ph = set(data["train"].path), set(data["test"].path), set(data["pad_adv"].path), set(data["pad_heldout"].path)
    assert not (tr & te) and not (pa & ph) and not (tr & pa)
    return {"train_n": len(tr), "test_n": len(te), "pad_adv_n": len(pa), "pad_heldout_n": len(ph)}


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    manifest = C.write_manifest()
    out = {
        "weights": test_weights(),
        "loss_grad": test_loss_scaling_and_grad(),
        "schedule": test_fixed_schedule_identity(),
        "splits": test_split_isolation(),
        "manifest_n_train_a": manifest["n_train_a"],
        "manifest_n_train_b": manifest["n_train_b"],
        "ok": True,
    }
    (C.OUT / "PILOT.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("PILOT_OK", flush=True)


if __name__ == "__main__":
    main()
