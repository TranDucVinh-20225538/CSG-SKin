#!/usr/bin/env python3
"""30-minute check: does GRL reverse the domain head's own gradient?

GRL must flip only the stream into the encoder. The domain head must see
ordinary CE-minimising gradients. A minus-sign on the head trains it to
get worse → uniform → encoder gradient 0. That would make Phase 12
inconclusive, not outcome (c).

Uses random tensors. No WILDS reload. No retrain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT))
import train_phase12_1b as t12

OUT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase14/camelyon_b1")


def vec(params):
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel()))
        else:
            chunks.append(p.grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.zeros(1)


def cosine(a, b):
    n = float(a.norm() * b.norm())
    if n <= 0:
        return float("nan")
    return float((a * b).sum() / n)


def run_once(seed=0):
    torch.manual_seed(seed)
    model = t12.MatchedDannDenseNet121(pretrained=False)
    model.eval()  # Dropout would make two forwards incomparable
    model.grl.set_lambd(1.0)
    x = torch.randn(12, 3, 224, 224)
    d = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    z = model.feat_bn(model.backbone(x))
    z.retain_grad()

    # A: production graph — domain_head(GRL(z))
    model.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad = None
    dlog_a = model.domain_head(model.grl(z))
    loss_a = F.cross_entropy(dlog_a, d)
    loss_a.backward(retain_graph=True)
    g_head_a = vec(model.domain_head.parameters())
    g_z_a = z.grad.detach().float().reshape(-1).cpu().clone()
    g_enc_a = vec(list(model.backbone.parameters()) + list(model.feat_bn.parameters()))

    # B: same z, GRL bypassed
    model.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad = None
    dlog_b = model.domain_head(z)
    loss_b = F.cross_entropy(dlog_b, d)
    loss_b.backward()
    g_head_b = vec(model.domain_head.parameters())
    g_z_b = z.grad.detach().float().reshape(-1).cpu().clone()
    g_enc_b = vec(list(model.backbone.parameters()) + list(model.feat_bn.parameters()))

    # C: isolate last Linear of the head — its grad must not go through GRL at all
    last = model.domain_head[-1]
    # already captured in g_head_*

    return {
        "ce_with_grl": float(loss_a.detach()),
        "ce_no_grl": float(loss_b.detach()),
        "head_cosine_grl_vs_nogrl": cosine(g_head_a, g_head_b),
        "encoder_cosine_grl_vs_nogrl": cosine(g_enc_a, g_enc_b),
        "z_cosine_grl_vs_nogrl": cosine(g_z_a, g_z_b),
        "head_grad_norm_grl": float(g_head_a.norm()),
        "head_grad_norm_nogrl": float(g_head_b.norm()),
        "encoder_grad_norm_grl": float(g_enc_a.norm()),
        "encoder_grad_norm_nogrl": float(g_enc_b.norm()),
        "adv_acc_grl": float((dlog_a.argmax(1) == d).float().mean()),
    }


def classify(rec):
    c_head = rec["head_cosine_grl_vs_nogrl"]
    c_enc = rec.get("z_cosine_grl_vs_nogrl", rec["encoder_cosine_grl_vs_nogrl"])
    # Head should match (+1). Feature stream into the encoder should flip (−1).
    head_ok = c_head > 0.90
    head_flipped = c_head < -0.90
    enc_ok = c_enc < -0.90
    if head_flipped:
        return {
            "verdict": "sign_error_head_flipped",
            "reopen_wilds": True,
            "drop_from_results": False,
            "reading": (
                "GRL is reversing the domain head's own gradient "
                "(cosine vs no-GRL CE = {:.3f}). One minus sign. "
                "Phase 12 is an implementation bug, not a Camelyon result. WILDS reopens."
            ).format(c_head),
        }
    if head_ok and enc_ok:
        return {
            "verdict": "grl_graph_correct",
            "reopen_wilds": False,
            "drop_from_results": True,
            "reading": (
                "GRL graph is the standard one: head cosine vs no-GRL = {:.3f} (not flipped), "
                "encoder cosine = {:.3f} (flipped). The head's failure to leave ln(3) is not "
                "this sign error. Phase 12 stays inconclusive — implementation never produced "
                "a working adversary — and is not reported as a Camelyon result."
            ).format(c_head, c_enc),
        }
    return {
        "verdict": "unclear",
        "reopen_wilds": False,
        "drop_from_results": True,
        "reading": (
            "Sign check not decisive (head cosine {:.3f}, encoder cosine {:.3f}). "
            "Thirty-minute bound hit. Phase 12 is not reported as a result."
        ).format(c_head, c_enc),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [run_once(s) for s in (0, 1, 2)]
    mean = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
    verdict = classify(mean)
    blob = {"per_seed": recs, "mean": mean, **verdict}
    (OUT / "grl_sign_check.json").write_text(json.dumps(blob, indent=2) + "\n")
    lines = [
        "# Phase 14.B — GRL sign check (30 minutes, then stop)",
        "",
        verdict["reading"],
        "",
        "| seed | head cos (GRL vs no-GRL) | ∂z cos | encoder cos | ||g_head|| GRL | ||g_head|| no-GRL |",
        "|---:|---|---|---|---|---|",
    ]
    for i, r in enumerate(recs):
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4g} | {:.4g} |".format(
                i, r["head_cosine_grl_vs_nogrl"], r["z_cosine_grl_vs_nogrl"],
                r["encoder_cosine_grl_vs_nogrl"],
                r["head_grad_norm_grl"], r["head_grad_norm_nogrl"],
            )
        )
    lines.append("")
    lines.append(
        "Expected if GRL is correct: head cosine ≈ +1, encoder cosine ≈ −1. "
        "Expected if the minus sign hits the head: head cosine ≈ −1."
    )
    lines.append("")
    lines.append("Reopen WILDS: **{}**. Drop Phase 12 from Results: **{}**.".format(
        "yes" if verdict["reopen_wilds"] else "no",
        "yes" if verdict["drop_from_results"] else "no",
    ))
    lines.append("")
    if verdict["drop_from_results"]:
        lines.append(
            "**Phase 12 does not support a conclusion.** The adversary never trained "
            "(CE = ln 3 at every epoch). No Camelyon17 result is reported. "
            "Limitations: tried, implementation unresolved, left for later work. "
            "Not outcome (c) in the sense of 'Camelyon resists invariance'."
        )
    else:
        lines.append(
            "Sign error confirmed. The WILDS extension is an implementation bug and reopens. "
            "No Camelyon number enters the paper until a working adversary is shown."
        )
    (OUT / "PHASE14_B_GRL_SIGN.md").write_text("\n".join(lines) + "\n")
    print(verdict["verdict"], flush=True)
    print(verdict["reading"], flush=True)
    print("wrote", OUT / "PHASE14_B_GRL_SIGN.md")


if __name__ == "__main__":
    main()
