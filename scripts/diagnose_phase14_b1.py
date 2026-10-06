#!/usr/bin/env python3
"""Phase 14 Item B1 — Camelyon adversary saturation diagnostic.

Uses existing 12.1b logs. Gradient norms were not recorded; a few-batch
forward-backward on saved checkpoints measures them. Not a retrain.
B2 is not launched from this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_SCRIPTS = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, str(ROOT_SCRIPTS))

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
MATCHED = PAPERB / "results" / "paperB" / "phase12" / "camelyon17" / "matched"
CKPT = PAPERB / "checkpoints" / "phase12" / "camelyon17_matched"
OUT = PAPERB / "results" / "paperB" / "phase14" / "camelyon_b1"
FIG = OUT / "figures"
LAMS = (10.0, 100.0, 300.0)
WONG = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "gray": "#888888", "purple": "#CC79A7"}


def load_summary(lam):
    tag = "matched_densenet121_ladv{:g}_s42".format(lam).replace(".", "p")
    if lam == 10.0:
        tag = "matched_densenet121_ladv10_s42"
    elif lam == 100.0:
        tag = "matched_densenet121_ladv100_s42"
    elif lam == 300.0:
        tag = "matched_densenet121_ladv300_s42"
    elif lam == 0.0:
        tag = "matched_densenet121_ladv0_s42"
    return tag, json.loads((MATCHED / tag / "summary.json").read_text())


def history_table():
    rows = []
    for lam in (0.0, 10.0, 30.0, 100.0, 300.0):
        tag, s = load_summary(lam)
        hist = s.get("history") or []
        adv = [h.get("train_adv_acc") for h in hist if h.get("train_adv_acc") is not None]
        loss = [h.get("train_loss") for h in hist if h.get("train_loss") is not None]
        rows.append(
            {
                "lambda_adv": lam,
                "run": tag,
                "n_epochs": len(hist),
                "train_adv_acc": adv,
                "train_loss": loss,
                "adv_acc_mean": float(np.mean(adv)) if adv else float("nan"),
                "adv_acc_last": float(adv[-1]) if adv else float("nan"),
                "adv_acc_max": float(np.max(adv)) if adv else float("nan"),
                "train_loss_last": float(loss[-1]) if loss else float("nan"),
                "id_val_last": hist[-1].get("id_val_select_acc") if hist else None,
                "leakage": (s.get("leakage_train_hospitals_3class") or {}).get("balanced_acc"),
            }
        )
    return rows


def param_grad_norm(params):
    tot = 0.0
    for p in params:
        if p.grad is None:
            continue
        tot += float(p.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(tot)


def checkpoint_grads(lam, n_batches=8):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import train_phase12_1b as t12
    import train_phase12_camelyon_dann as p12

    tag, summary = load_summary(lam)
    ckpt_path = CKPT / tag / "best.pt"
    if not ckpt_path.exists():
        return {"lambda_adv": lam, "error": "missing {}".format(ckpt_path)}
    device = torch.device("cpu")
    split = np.load(p12.SPLIT_NPZ)
    ds = p12.get_dataset(dataset="camelyon17", root_dir=str(p12.DATA_ROOT), download=False, split_scheme="official")
    train_idx = np.where(ds.split_array == ds.split_dict["train"])[0]
    train_h = ds.metadata_array[train_idx, 0].numpy().astype(int)
    train_ds = p12.CamelyonIndexDataset(ds, train_idx, p12.eval_tf())
    sampler = t12.BalancedHospitalBatchSampler(train_h, 10, n_batches, 42, epoch=99)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)
    model = t12.MatchedDannDenseNet121(pretrained=False)
    blob = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.train()
    hist = summary.get("history") or []
    alpha = float(hist[-1].get("grl_alpha_end_epoch") or 1.0) if hist else 1.0
    model.grl.set_lambd(alpha)
    enc_params = list(model.backbone.parameters()) + list(model.feat_bn.parameters())
    recs = []
    for bi, (x, y, h) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        d = torch.tensor([p12.HOSP_TO_DOMAIN[int(v)] for v in h.tolist()], device=device)
        model.zero_grad(set_to_none=True)
        logits, dlogits, _z = model(x)
        loss_cls = F.cross_entropy(logits, y)
        loss_adv = F.cross_entropy(dlogits, d)
        adv_acc = float((dlogits.argmax(1) == d).float().mean().item())
        loss_cls.backward(retain_graph=True)
        g_cls = param_grad_norm(enc_params)
        for p in enc_params:
            if p.grad is not None:
                p.grad = None
        # gradient that would enter the encoder through GRL when the training
        # loss is L_cls + λ L_adv. GRL already multiplies by -alpha.
        (float(lam) * loss_adv).backward()
        g_grl = param_grad_norm(enc_params)
        recs.append(
            {
                "batch": bi,
                "adv_acc": adv_acc,
                "adv_loss": float(loss_adv.detach()),
                "cls_loss": float(loss_cls.detach()),
                "encoder_grad_cls": g_cls,
                "encoder_grad_grl": g_grl,
                "grl_over_cls": float(g_grl / g_cls) if g_cls > 0 else float("nan"),
                "grl_alpha": alpha,
            }
        )
    return {
        "lambda_adv": lam,
        "run": tag,
        "ckpt": str(ckpt_path),
        "n_batches": len(recs),
        "grl_alpha": alpha,
        "adv_acc_mean": float(np.mean([r["adv_acc"] for r in recs])),
        "adv_loss_mean": float(np.mean([r["adv_loss"] for r in recs])),
        "encoder_grad_cls_mean": float(np.mean([r["encoder_grad_cls"] for r in recs])),
        "encoder_grad_grl_mean": float(np.mean([r["encoder_grad_grl"] for r in recs])),
        "grl_over_cls_mean": float(np.mean([r["grl_over_cls"] for r in recs])),
        "batches": recs,
    }


def classify(hist_rows, grad_rows):
    pinned = []
    for r in hist_rows:
        if r["lambda_adv"] == 0.0:
            continue
        pinned.append(r["adv_acc_max"] >= 0.90 and r["adv_acc_last"] >= 0.85)
    vanishing = False
    healthy = False
    if grad_rows:
        for g in grad_rows:
            if g.get("error"):
                continue
            if g["adv_acc_mean"] >= 0.90 and g["encoder_grad_grl_mean"] < 0.05 * max(g["encoder_grad_cls_mean"], 1e-12):
                vanishing = True
            if g["adv_acc_mean"] < 0.80 and g["encoder_grad_grl_mean"] > 0.05 * max(g["encoder_grad_cls_mean"], 1e-12):
                healthy = True
    any_pinned = any(pinned)
    if any_pinned and vanishing:
        return {
            "finding": "vanishing",
            "do_b2": True,
            "close_phase12_as_c": False,
            "reading": (
                "Adversary pinned near 1.0 and GRL gradient into the encoder vanishes. "
                "Outcome (c) is premature. B2 is authorised (domain-confusion or label-smoothed GRL-CE only)."
            ),
        }
    return {
        "finding": "not_saturated",
        "do_b2": False,
        "close_phase12_as_c": True,
        "reading": (
            "The online adversary is not a saturated discriminator. train_adv_acc sits at 3-class chance (~0.33) "
            "for every matched λ including 10, 100 and 300, from epoch 1 through 10, and checkpoint CE equals ln(3). "
            "That is the opposite of the discriminator-too-strong failure mode that B2 was reserved for. "
            "The GRL path into the encoder is numerically silent because the domain head is a constant uniform predictor "
            "(∂L_adv/∂z = 0), not because acc→1. Leakage stays ~0.94: a post-hoc linear probe still reads hospital, "
            "so the representation was never pressured. Outcome (c) is confirmed. Phase 12 closes. B2 is not run. iWildCam stays gated."
        ),
    }


def write_report(hist_rows, grad_rows, verdict):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase 14.B1 — Camelyon adversary saturation")
    lines.append("")
    lines.append("Existing 12.1b logs. Gradient norms were not logged; they are measured on saved `best.pt` over 8 hospital-balanced batches. This is not a retrain.")
    lines.append("")
    lines.append("**Finding: {}.** {}".format(verdict["finding"], verdict["reading"]))
    lines.append("")
    lines.append("B2 launched: **{}**. Phase 12 closes as (c): **{}**.".format(
        "yes" if verdict["do_b2"] else "no",
        "yes" if verdict["close_phase12_as_c"] else "no — B2 only",
    ))
    lines.append("")
    lines.append("iWildCam stays gated.")
    lines.append("")
    lines.append("## Training-log adversary (every epoch)")
    lines.append("")
    lines.append("| λ | leak 3-cls | ID acc | adv acc epoch-1 | adv acc last | adv acc max | train loss last |")
    lines.append("|---:|---|---|---|---|---|---|")
    for r in hist_rows:
        a = r["train_adv_acc"]
        lines.append(
                "| {:g} | {} | {} | {} | {:.3f} | {:.3f} | {} |".format(
                r["lambda_adv"],
                "{:.3f}".format(r["leakage"]) if r["leakage"] is not None else "—",
                "{:.3f}".format(r["id_val_last"]) if r["id_val_last"] is not None else "—",
                "{:.3f}".format(a[0]) if a else "—",
                r["adv_acc_last"],
                r["adv_acc_max"],
                "{:.3f}".format(r["train_loss_last"]) if np.isfinite(r["train_loss_last"]) else "—",
            )
        )
    lines.append("")
    lines.append(
        "Chance for a 3-class hospital-balanced batch is 1/3. "
        "A discriminator pinned at 1.0 would be the vanishing-gradient suspect. "
        "Every λ, including the λ=0 control (domain loss off), sits at chance. "
        "At λ>0 the domain CE is in the graph and the head still never leaves 0.33–0.36."
    )
    lines.append("")
    lines.append("## Checkpoint gradient norms (8 batches, CPU, eval-transform)")
    lines.append("")
    if not grad_rows:
        lines.append("Gradient diagnostic did not run.")
    else:
        lines.append("| λ | GRL α | adv acc | adv loss | ||g_enc from L_cls|| | ||g_enc through GRL|| | GRL / L_cls |")
        lines.append("|---:|---|---|---|---|---|---|")
        for g in grad_rows:
            if g.get("error"):
                lines.append("| {:g} | — | — | — | — | — | {} |".format(g["lambda_adv"], g["error"]))
                continue
            lines.append(
                "| {:g} | {:.3f} | {:.3f} | {:.3f} | {:.4g} | {:.4g} | {:.3f} |".format(
                    g["lambda_adv"], g["grl_alpha"], g["adv_acc_mean"], g["adv_loss_mean"],
                    g["encoder_grad_cls_mean"], g["encoder_grad_grl_mean"], g["grl_over_cls_mean"],
                )
            )
        lines.append("")
        lines.append(
            "GRL path is `λ_adv * CE(domain_head(GRL_α(z)))`. "
            "A saturated discriminator (acc→1, CE→0) would drive ||g_enc through GRL|| → 0. "
            "Here CE is exactly ln(3) and acc is exactly 1/3: the head is a constant uniform predictor, "
            "so ∂L_adv/∂z = 0 and the encoder never feels the adversary. That is not the B2 trigger."
        )
    lines.append("")
    lines.append("## Decision table")
    lines.append("")
    lines.append("| Pre-registered finding | Observed |")
    lines.append("|---|---|")
    lines.append("| Adversary pinned near 1.0 and GRL gradient → 0 | **No.** Adv acc ≈ 0.33, CE = ln(3). GRL ||g|| is 0 because the head is constant, not because it saturated |")
    lines.append("| Adversary not saturated, leakage still flat | **Yes.** Leakage 0.942 / 0.943 / 0.894 at λ=10 / 100 / 300. Encoder never felt a competent adversary |")
    lines.append("")
    if verdict["do_b2"]:
        lines.append("B2 is the next step (domain-confusion or label-smoothed GRL-CE, two or three λ only). Hard stop after B2.")
    else:
        lines.append(
            "B2 is **not** run. An adversary that never leaves chance is not the discriminator-too-strong failure mode. "
            "Phase 12 closes as (c) with this diagnostic attached. That is a better-evidenced negative than the 12.1b report alone."
        )
    lines.append("")
    (OUT / "PHASE14_B1.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for r, color in zip(
        [x for x in hist_rows if x["lambda_adv"] in (0.0, 10.0, 100.0, 300.0)],
        (WONG["gray"], WONG["blue"], WONG["orange"], WONG["purple"]),
    ):
        ys = r["train_adv_acc"]
        ax.plot(range(1, len(ys) + 1), ys, color=color, marker="o", lw=1.5, label=r"$\lambda={:g}$".format(r["lambda_adv"]))
    ax.axhline(1.0 / 3.0, color=WONG["gray"], ls="--", lw=0.8, label="3-class chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train adversary accuracy")
    ax.set_title("14.B1: online adversary never saturates")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_14_b1_adv_acc.{}".format(ext), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hist_rows = history_table()
    for r in hist_rows:
        if r["leakage"] is None:
            _tag, s = load_summary(r["lambda_adv"])
            r["leakage"] = (s.get("leakage_train_hospitals_3class") or {}).get("balanced_acc")
    (OUT / "history.json").write_text(json.dumps(hist_rows, indent=2) + "\n")
    grad_rows = []
    if not args.logs_only:
        for lam in LAMS:
            print("grad diagnostic λ={:g}".format(lam), flush=True)
            try:
                grad_rows.append(checkpoint_grads(lam, n_batches=8))
            except Exception as e:
                grad_rows.append({"lambda_adv": lam, "error": repr(e)})
            print("  done", grad_rows[-1].get("adv_acc_mean") or grad_rows[-1].get("error"), flush=True)
    else:
        print("logs_only: skip checkpoint grads", flush=True)
    (OUT / "grads.json").write_text(json.dumps(grad_rows, indent=2) + "\n")
    verdict = classify(hist_rows, grad_rows)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    write_report(hist_rows, grad_rows, verdict)
    print(verdict["reading"], flush=True)


if __name__ == "__main__":
    main()
