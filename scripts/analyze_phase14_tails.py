#!/usr/bin/env python3
"""Phase 14 Item A — is the inversion a tail effect?

Cached Phase 13 features only. No new CSG training.
Outputs under results/paperB/phase14/tails/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
FEAT = PAPERB / "results" / "paperB" / "phase13" / "features"
OUT = PAPERB / "results" / "paperB" / "phase14" / "tails"
FIG = OUT / "figures"
CLASS_NAMES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
QUANTILES = (1, 5, 25, 50, 75, 90, 95, 99, 99.9)
TAIL_PCTS = (90.0, 95.0, 99.0)
SPACES = ("z_lesion_norm", "backbone_raw_lesion")
OOD_SPLITS = ("pad_heldout", "fitzpatrick17k")
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
JOBS = []
for _lam in LAMS:
    _seeds = (42, 52, 62, 72, 82) if _lam in (0.0, 2.0, 8.0) else (42, 52, 62)
    for _s in _seeds:
        JOBS.append((_lam, _s))
HEADLINE_EPS = 1e-3
WONG = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "gray": "#888888",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
}


def lam_tag(x):
    return "{:g}".format(float(x)).replace(".", "p")


def mean_std(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def fmt(m, s):
    if not np.isfinite(m):
        return "—"
    return "{:.3f} ± {:.3f}".format(m, s)


def has_features(tag):
    d = FEAT / tag
    return all((d / "{}.npz".format(s)).exists() for s in ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k"))


def load_split(tag, split):
    return dict(np.load(FEAT / tag / "{}.npz".format(split)))


def pooled_centered(z, y, n_classes=8):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    means = np.zeros((n_classes, z.shape[1]), dtype=np.float64)
    present = []
    for c in range(n_classes):
        mask = y == c
        if not np.any(mask):
            continue
        means[c] = z[mask].mean(axis=0)
        present.append(c)
    centered = z - means[y]
    return means, centered, present


def maha_min_sq(z, means, precision):
    z = np.asarray(z, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    p = np.asarray(precision, dtype=np.float64)
    zp = z @ p
    zpz = np.einsum("nd,nd->n", zp, z)
    mp = means @ p
    mpm = np.einsum("cd,cd->c", mp, means)
    d2 = zpz[:, None] + mpm[None, :] - 2.0 * (zp @ means.T)
    return d2.min(axis=1)


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float("nan"), "std": float("nan"), "skewness": float("nan"), "kurtosis": float("nan")}
    m = float(x.mean())
    s = float(x.std(ddof=1))
    if s <= 0:
        return {"n": n, "mean": m, "std": 0.0, "skewness": float("nan"), "kurtosis": float("nan")}
    z = (x - m) / s
    return {
        "n": int(n),
        "mean": m,
        "std": s,
        "skewness": float((z ** 3).mean()),
        "kurtosis": float((z ** 4).mean() - 3.0),
    }


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    return {str(q): float(np.quantile(x, q / 100.0)) for q in QUANTILES}


def score_pack(ztr, ytr, z):
    means, centered, present = pooled_centered(ztr, ytr, n_classes=8)
    denom = max(len(ztr) - len(present), 1)
    cov = (centered.T @ centered) / denom
    cov = cov + float(HEADLINE_EPS) * np.eye(cov.shape[0])
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    return maha_min_sq(z, means, prec)


def auroc_pair(sid, sood):
    if len(sid) == 0 or len(sood) == 0:
        return float("nan")
    y = np.concatenate([np.zeros(len(sid)), np.ones(len(sood))])
    return float(roc_auc_score(y, np.concatenate([sid, sood])))


def tail_drop(sid, sood, pct):
    pooled = np.concatenate([sid, sood])
    thr = float(np.quantile(pooled, pct / 100.0))
    mid = sid <= thr
    mood = sood <= thr
    return {
        "percentile": pct,
        "threshold": thr,
        "n_id_kept": int(mid.sum()),
        "n_ood_kept": int(mood.sum()),
        "n_id_dropped": int((~mid).sum()),
        "n_ood_dropped": int((~mood).sum()),
        "auroc_full": auroc_pair(sid, sood),
        "auroc_bulk": auroc_pair(sid[mid], sood[mood]),
    }


def dist_block(scores):
    return {"quantiles": quantiles(scores), **moments(scores)}


def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def analyze_run(tag, lam, seed):
    tr = load_split(tag, "isic_train")
    idp = load_split(tag, "isic_test")
    hold = load_split(tag, "pad_heldout")
    fitz = load_split(tag, "fitzpatrick17k")
    packs = {"pad_heldout": hold, "fitzpatrick17k": fitz}
    yid = np.asarray(idp["labels"], dtype=int)
    logits = np.asarray(idp["logits"], dtype=np.float64)
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == yid

    out = {"run_name": tag, "lambda_adv": float(lam), "seed": int(seed), "spaces": {}}
    scores = {}
    for space in SPACES:
        ztr, ytr = tr[space], tr["labels"]
        sid = score_pack(ztr, ytr, idp[space])
        block = {"id": dist_block(sid), "ood": {}, "a2": {}, "a3": None}
        sood_cache = {}
        for oname, opack in packs.items():
            sood = score_pack(ztr, ytr, opack[space])
            sood_cache[oname] = sood.astype(np.float32)
            block["ood"][oname] = dist_block(sood)
            a2 = {"full_auroc": auroc_pair(sid, sood)}
            for pct in TAIL_PCTS:
                a2["drop_p{:g}".format(pct)] = tail_drop(sid, sood, pct)
            block["a2"][oname] = a2
        scores[space] = {"sid": sid.astype(np.float32), "sood": sood_cache}
        if space == "z_lesion_norm":
            k = max(int(round(0.01 * len(sid))), 1)
            order = np.argsort(sid)[::-1]
            top = order[:k]
            cls_all = {name: int((yid == i).sum()) for i, name in enumerate(CLASS_NAMES)}
            cls_tail = {name: int((yid[top] == i).sum()) for i, name in enumerate(CLASS_NAMES)}
            rare_mask = np.isin(yid, [5, 6])
            block["a3"] = {
                "k": int(k),
                "rule": "top 1% of ID Mahalanobis scores on isic_test",
                "class_counts_all": cls_all,
                "class_counts_tail": cls_tail,
                "rare_df_vasc_frac_all": float(rare_mask.mean()),
                "rare_df_vasc_frac_tail": float(rare_mask[top].mean()),
                "acc_all": float(correct.mean()),
                "acc_tail": float(correct[top].mean()),
                "mean_conf_all": float(conf.mean()),
                "mean_conf_tail": float(conf[top].mean()),
                "median_conf_tail": float(np.median(conf[top])),
                "frac_incorrect_tail": float((~correct[top]).mean()),
                "frac_incorrect_and_common_tail": float(((~correct[top]) & (~rare_mask[top])).mean()),
                "frac_correct_and_rare_tail": float((correct[top] & rare_mask[top]).mean()),
                "frac_correct_and_common_tail": float((correct[top] & (~rare_mask[top])).mean()),
                "by_class": [],
            }
            for i, name in enumerate(CLASS_NAMES):
                m_all = yid == i
                m_top = yid[top] == i
                block["a3"]["by_class"].append(
                    {
                        "class": name,
                        "n_all": int(m_all.sum()),
                        "n_tail": int(m_top.sum()),
                        "share_all": float(m_all.mean()),
                        "share_tail": float(m_top.mean()) if k else 0.0,
                        "acc_all": float(correct[m_all].mean()) if m_all.any() else float("nan"),
                        "acc_tail": float(correct[top][m_top].mean()) if m_top.any() else float("nan"),
                        "mean_conf_all": float(conf[m_all].mean()) if m_all.any() else float("nan"),
                        "mean_conf_tail": float(conf[top][m_top].mean()) if m_top.any() else float("nan"),
                    }
                )
        out["spaces"][space] = block
    out["_scores"] = scores
    return out


def gather(per_run, space, path_fn):
    by = defaultdict(list)
    for r in per_run:
        by[r["lambda_adv"]].append(path_fn(r["spaces"][space]))
    return {lam: mean_std(by[lam]) for lam in LAMS}


def plot_densities(per_run):
    FIG.mkdir(parents=True, exist_ok=True)
    show_lams = (0.0, 0.25, 2.0)
    for space, title in (
        ("z_lesion_norm", r"$z_\mathrm{lesion\_norm}$ Maha"),
        ("backbone_raw_lesion", "backbone Maha"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
        for ax, lam in zip(axes, show_lams):
            recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12 and r.get("seed") == 42]
            if not recs:
                recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12][:1]
            for r in recs:
                sc = (r.get("_scores") or {}).get(space) or {}
                sid = np.asarray(sc.get("sid", []))
                sood = sc.get("sood") or {}
                sh = np.asarray(sood.get("pad_heldout", []))
                sf = np.asarray(sood.get("fitzpatrick17k", []))
                if sid.size == 0:
                    continue
                for sc, color, lab, ls in (
                    (sid, WONG["blue"], "ISIC test", "-"),
                    (sh, WONG["orange"], "pad_heldout", "-"),
                    (sf, WONG["purple"], "Fitzpatrick17k", "--"),
                ):
                    if sc.size == 0:
                        continue
                    ax.hist(
                        np.log10(np.clip(sc, 1e-12, None)),
                        bins=40,
                        density=True,
                        histtype="step",
                        color=color,
                        lw=1.4,
                        ls=ls,
                        label=lab,
                    )
            ax.set_title(r"$\lambda={:g}$".format(lam), fontsize=10)
            ax.set_xlabel(r"$\log_{10}$ Maha")
            ax.axvline(0, color=WONG["gray"], lw=0.4)
        axes[0].set_ylabel("density")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("14.A1: {} score densities (log axis)".format(title), fontsize=11, y=1.03)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / "fig_14_a1_density_{}.{}".format(space, ext), dpi=300)
        plt.close(fig)

    # quantile tracks
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharex=True)
    for ax, space, title in zip(
        axes,
        SPACES,
        (r"$z_\mathrm{lesion\_norm}$", "backbone"),
    ):
        for q, color, ls in ((50, WONG["gray"], ":"), (95, WONG["blue"], "-"), (99, WONG["orange"], "--")):
            key = str(q)
            for oname, mk in (("id", "o"), ("pad_heldout", "s"), ("fitzpatrick17k", "^")):
                ys, es = [], []
                for lam in LAMS:
                    recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12]
                    if oname == "id":
                        vals = [r["spaces"][space]["id"]["quantiles"][key] for r in recs]
                    else:
                        vals = [r["spaces"][space]["ood"][oname]["quantiles"][key] for r in recs]
                    m, s = mean_std(vals)
                    ys.append(m)
                    es.append(s)
                label = None
                if q == 95 and oname == "id":
                    label = "ID p{:g}".format(q)
                elif q == 95 and oname == "pad_heldout":
                    label = "hold p{:g}".format(q)
                elif q == 50 and oname == "id":
                    label = "ID median"
                elif q == 50 and oname == "pad_heldout":
                    label = "hold median"
                ax.errorbar(
                    list(LAMS), ys, yerr=es, color=color if oname != "fitzpatrick17k" else WONG["purple"],
                    marker=mk, lw=1.2, ls=ls, alpha=0.9 if oname != "fitzpatrick17k" else 0.55, label=label,
                )
        ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
        ax.set_ylabel("Maha quantile")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("14.A1: medians vs upper tails", fontsize=11, y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_14_a1_quantiles.{}".format(ext), dpi=300)
    plt.close(fig)

    # A2 AUROC vs tail cut
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, ood in zip(axes, OOD_SPLITS):
        for space, color, lab in (
            ("z_lesion_norm", WONG["orange"], r"$z_\mathrm{norm}$"),
            ("backbone_raw_lesion", WONG["blue"], "backbone"),
        ):
            for cut, ls, mk in (("full", "-", "o"), ("drop_p90", "--", "s"), ("drop_p95", ":", "^"), ("drop_p99", "-.", "D")):
                ys, es = [], []
                for lam in LAMS:
                    recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12]
                    if cut == "full":
                        vals = [r["spaces"][space]["a2"][ood]["full_auroc"] for r in recs]
                    else:
                        vals = [r["spaces"][space]["a2"][ood][cut]["auroc_bulk"] for r in recs]
                    m, s = mean_std(vals)
                    ys.append(m)
                    es.append(s)
                label = "{} {}".format(lab, "full" if cut == "full" else cut.replace("drop_", "")) if space == "z_lesion_norm" else None
                ax.errorbar(list(LAMS), ys, yerr=es, color=color, marker=mk, lw=1.3, ls=ls, label=label)
        ax.axhline(0.5, color=WONG["gray"], ls=":", lw=0.8)
        ax.set_xlabel(r"$\lambda_\mathrm{adv}$")
        ax.set_title(ood, fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Mahalanobis AUROC")
    fig.suptitle("14.A2: AUROC after dropping the pooled upper tail", fontsize=11, y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / "fig_14_a2_tail_drop.{}".format(ext), dpi=300)
    plt.close(fig)


def classify_a2(per_run):
    """Inversion disappears in the bulk → tail-driven; persists → not a tail effect."""
    recs2 = [r for r in per_run if abs(r["lambda_adv"] - 2.0) < 1e-12]
    full = mean_std([r["spaces"]["z_lesion_norm"]["a2"]["pad_heldout"]["full_auroc"] for r in recs2])[0]
    b90 = mean_std([r["spaces"]["z_lesion_norm"]["a2"]["pad_heldout"]["drop_p90"]["auroc_bulk"] for r in recs2])[0]
    b95 = mean_std([r["spaces"]["z_lesion_norm"]["a2"]["pad_heldout"]["drop_p95"]["auroc_bulk"] for r in recs2])[0]
    b99 = mean_std([r["spaces"]["z_lesion_norm"]["a2"]["pad_heldout"]["drop_p99"]["auroc_bulk"] for r in recs2])[0]
    inverted = full < 0.48
    # disappears: bulk back at or above chance while full is inverted
    gone_90 = inverted and b90 >= 0.48
    persist_90 = inverted and b90 < 0.48
    return {
        "lambda": 2.0,
        "ood": "pad_heldout",
        "space": "z_lesion_norm",
        "full": full,
        "bulk_p90": b90,
        "bulk_p95": b95,
        "bulk_p99": b99,
        "inverted_at_full": bool(inverted),
        "tail_is_driver": bool(gone_90),
        "bulk_still_inverted": bool(persist_90),
        "reading": (
            "Inversion at λ=2 ({:.3f}) disappears after dropping the pooled upper 10% ({:.3f}). "
            "The tail is the driver."
        ).format(full, b90)
        if gone_90
        else (
            "Inversion at λ=2 ({:.3f}) persists in the bulk after dropping p90 ({:.3f}), p95 ({:.3f}), p99 ({:.3f}). "
            "The tail hypothesis is wrong. This line closes."
        ).format(full, b90, b95, b99)
        if persist_90
        else (
            "Full AUROC at λ=2 is {:.3f}; bulk p90 is {:.3f}. Neither a clean tail-driven disappearance "
            "nor a clean bulk persistence. Report the numbers; do not force a story."
        ).format(full, b90),
    }


def write_report(per_run, verdict):
    lines = []
    lines.append("# Phase 14.A — is the inversion a tail effect?")
    lines.append("")
    lines.append("Cached Phase 13 features. No new training. Mahalanobis: class-conditional shared covariance, `reg_eps=1e-3`, fit on ISIC train only.")
    lines.append("")
    lines.append("**Headline (13.3, unchanged).** OOD ECE on pad_heldout rises 0.247 → 0.746 while ID ECE stays flat at ~0.10. Mean OOD confidence overtakes ID by λ=2. That is the paper's lead result.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict["reading"])
    lines.append("")
    lines.append(
        "Pre-registered prediction: medians converge as λ rises while ID p95/p99 stay above OOD. "
        "A2: inversion disappears when the pooled upper tail is dropped → tail-driven; persists in the bulk → not a tail effect."
    )
    lines.append("")

    def qtable(space, split_key, which):
        lines.append("| λ | n | p1 | p5 | p25 | p50 | p75 | p90 | p95 | p99 | p99.9 | mean | std | skew | kurt |")
        lines.append("|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lam in LAMS:
            recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12]
            def gq(q):
                if which == "id":
                    vals = [r["spaces"][space]["id"]["quantiles"][q] for r in recs]
                else:
                    vals = [r["spaces"][space]["ood"][split_key]["quantiles"][q] for r in recs]
                return fmt(*mean_std(vals))
            def gm(k):
                if which == "id":
                    vals = [r["spaces"][space]["id"][k] for r in recs]
                else:
                    vals = [r["spaces"][space]["ood"][split_key][k] for r in recs]
                return fmt(*mean_std(vals))
            lines.append(
                "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    lam, len(recs),
                    gq("1"), gq("5"), gq("25"), gq("50"), gq("75"), gq("90"), gq("95"), gq("99"), gq("99.9"),
                    gm("mean"), gm("std"), gm("skewness"), gm("kurtosis"),
                )
            )
        lines.append("")

    for space, title in (
        ("z_lesion_norm", r"z_lesion_norm (primary)"),
        ("backbone_raw_lesion", "backbone_raw_lesion"),
    ):
        lines.append("## A1 — score distributions, {}".format(title))
        lines.append("")
        lines.append("### ISIC test")
        lines.append("")
        qtable(space, None, "id")
        lines.append("### pad_heldout")
        lines.append("")
        qtable(space, "pad_heldout", "ood")
        lines.append("### Fitzpatrick17k")
        lines.append("")
        qtable(space, "fitzpatrick17k", "ood")

        # median / p95 comparison
        lines.append("Median and upper-tail comparison (ID vs pad_heldout):")
        lines.append("")
        lines.append("| λ | ID p50 | hold p50 | ID−hold p50 | ID p95 | hold p95 | ID−hold p95 | ID p99 | hold p99 | ID−hold p99 |")
        lines.append("|---:|---|---|---|---|---|---|---|---|---|")
        for lam in LAMS:
            recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12]
            def d(a, b):
                return fmt(*mean_std([a[i] - b[i] for i in range(len(a))]))
            def col(fn):
                return [fn(r) for r in recs]
            id50 = col(lambda r: r["spaces"][space]["id"]["quantiles"]["50"])
            h50 = col(lambda r: r["spaces"][space]["ood"]["pad_heldout"]["quantiles"]["50"])
            id95 = col(lambda r: r["spaces"][space]["id"]["quantiles"]["95"])
            h95 = col(lambda r: r["spaces"][space]["ood"]["pad_heldout"]["quantiles"]["95"])
            id99 = col(lambda r: r["spaces"][space]["id"]["quantiles"]["99"])
            h99 = col(lambda r: r["spaces"][space]["ood"]["pad_heldout"]["quantiles"]["99"])
            lines.append(
                "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    lam, fmt(*mean_std(id50)), fmt(*mean_std(h50)), d(id50, h50),
                    fmt(*mean_std(id95)), fmt(*mean_std(h95)), d(id95, h95),
                    fmt(*mean_std(id99)), fmt(*mean_std(h99)), d(id99, h99),
                )
            )
        lines.append("")

        lines.append("## A2 — AUROC after dropping the pooled upper tail, {}".format(title))
        lines.append("")
        for ood in OOD_SPLITS:
            lines.append("### {}".format(ood))
            lines.append("")
            lines.append("| λ | n | full | drop p90 | drop p95 | drop p99 | n_id kept p90 | n_ood kept p90 |")
            lines.append("|---:|---:|---|---|---|---|---|---|")
            for lam in LAMS:
                recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12]
                def g(path):
                    return fmt(*mean_std([path(r) for r in recs]))
                a2 = lambda r: r["spaces"][space]["a2"][ood]
                lines.append(
                    "| {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                        lam, len(recs),
                        g(lambda r: a2(r)["full_auroc"]),
                        g(lambda r: a2(r)["drop_p90"]["auroc_bulk"]),
                        g(lambda r: a2(r)["drop_p95"]["auroc_bulk"]),
                        g(lambda r: a2(r)["drop_p99"]["auroc_bulk"]),
                        g(lambda r: a2(r)["drop_p90"]["n_id_kept"]),
                        g(lambda r: a2(r)["drop_p90"]["n_ood_kept"]),
                    )
                )
            lines.append("")

    lines.append("## A3 — what is in the ID tail? (z_lesion_norm, top 1%)")
    lines.append("")
    lines.append(
        "Phase 2.5b already showed class restriction does not fix the inversion (0.414 vs 0.409) "
        "and reweighting strengthens it (0.240). The tail cannot be *purely* rare-class-driven. "
        "If DF/VASC are over-represented, that is a note, not the mechanism."
    )
    lines.append("")
    lines.append("| λ | n | acc all | acc tail | conf all | conf tail | rare frac all | rare frac tail | incorrect∧common | correct∧rare | correct∧common |")
    lines.append("|---:|---:|---|---|---|---|---|---|---|---|---|")
    for lam in LAMS:
        recs = [r for r in per_run if abs(r["lambda_adv"] - lam) < 1e-12]
        def g(k):
            return fmt(*mean_std([r["spaces"]["z_lesion_norm"]["a3"][k] for r in recs]))
        lines.append(
            "| {:g} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                lam, len(recs),
                g("acc_all"), g("acc_tail"), g("mean_conf_all"), g("mean_conf_tail"),
                g("rare_df_vasc_frac_all"), g("rare_df_vasc_frac_tail"),
                g("frac_incorrect_and_common_tail"), g("frac_correct_and_rare_tail"),
                g("frac_correct_and_common_tail"),
            )
        )
    lines.append("")
    lines.append("Class share of the top-1% ID tail vs base rate, λ=0 and λ=2:")
    lines.append("")
    lines.append("| class | base share | tail share λ=0 | tail share λ=2 | acc tail λ=2 | conf tail λ=2 |")
    lines.append("|---|---|---|---|---|---|")
    recs0 = [r for r in per_run if abs(r["lambda_adv"] - 0.0) < 1e-12]
    recs2 = [r for r in per_run if abs(r["lambda_adv"] - 2.0) < 1e-12]
    for i, name in enumerate(CLASS_NAMES):
        def g0(k):
            return fmt(*mean_std([r["spaces"]["z_lesion_norm"]["a3"]["by_class"][i][k] for r in recs0]))
        def g2(k):
            return fmt(*mean_std([r["spaces"]["z_lesion_norm"]["a3"]["by_class"][i][k] for r in recs2]))
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                name, g0("share_all"), g0("share_tail"), g2("share_tail"), g2("acc_tail"), g2("mean_conf_tail"),
            )
        )
    lines.append("")

    recs2 = [r for r in per_run if abs(r["lambda_adv"] - 2.0) < 1e-12]
    inc = mean_std([r["spaces"]["z_lesion_norm"]["a3"]["frac_incorrect_and_common_tail"] for r in recs2])[0]
    rare = mean_std([r["spaces"]["z_lesion_norm"]["a3"]["rare_df_vasc_frac_tail"] for r in recs2])[0]
    corr_c = mean_std([r["spaces"]["z_lesion_norm"]["a3"]["frac_correct_and_common_tail"] for r in recs2])[0]
    acc_t = mean_std([r["spaces"]["z_lesion_norm"]["a3"]["acc_tail"] for r in recs2])[0]
    conf_t = mean_std([r["spaces"]["z_lesion_norm"]["a3"]["mean_conf_tail"] for r in recs2])[0]
    conf_a = mean_std([r["spaces"]["z_lesion_norm"]["a3"]["mean_conf_all"] for r in recs2])[0]
    lines.append("### A3 reading at λ=2")
    lines.append("")
    lines.append(
        "Top-1% ID tail: accuracy {:.3f}, mean softmax confidence {:.3f} (all-ID {:.3f}). "
        "DF+VASC share of the tail {:.3f}. Incorrect common-class share {:.3f}. "
        "Correct common-class share {:.3f}.".format(acc_t, conf_t, conf_a, rare, inc, corr_c)
    )
    lines.append("")
    if rare > 0.50:
        lines.append(
            "Rare classes dominate the tail. That cannot be the whole mechanism: Phase 2.5b class restriction left the inversion intact (0.414 vs 0.409)."
        )
    elif inc > 0.40:
        lines.append(
            "The tail is largely misclassified common-class cases (within-class outliers or mislabels), not rare-class mass. That is compatible with Phase 2.5b."
        )
    elif corr_c > 0.40:
        lines.append(
            "The tail is largely *correct* common-class cases sitting far from the class-conditional cloud — within-class outliers / artifacts, not errors and not rare classes."
        )
    else:
        lines.append("The tail is mixed. No single identity (rare class / error / correct outlier) dominates. Report the breakdown; do not force a mechanism.")
    lines.append("")
    lines.append("Figures: `phase14/tails/figures/fig_14_a1_*.{png,pdf}`, `fig_14_a2_tail_drop.{png,pdf}`.")
    (OUT / "PHASE14_A_TAILS.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    per_run = []
    (OUT / "per_run").mkdir(exist_ok=True)
    for lam, seed in JOBS:
        tag = "runB_orth1_ladv{}_s{}".format(lam_tag(lam), seed)
        dest = OUT / "per_run" / "{}.json".format(tag)
        if dest.exists():
            rec = json.loads(dest.read_text())
            per_run.append(rec)
            print("resume", tag, flush=True)
            continue
        if not has_features(tag):
            raise RuntimeError("missing features for {}".format(tag))
        print("analyze", tag, flush=True)
        rec = analyze_run(tag, lam, seed)
        serial = {k: v for k, v in rec.items() if k != "_scores"}
        dest.write_text(json.dumps(serial, indent=2) + "\n")
        per_run.append(rec)
    # plots need raw scores at λ=0, 0.25, 2 seed 42
    for rec in per_run:
        if rec.get("seed") == 42 and rec["lambda_adv"] in (0.0, 0.25, 2.0) and "_scores" not in rec:
            print("scores for plot", rec["run_name"], flush=True)
            rec.update(analyze_run(rec["run_name"], rec["lambda_adv"], rec["seed"]))
    verdict = classify_a2(per_run)
    serial_all = [{k: v for k, v in r.items() if k != "_scores"} for r in per_run]
    (OUT / "all_runs.json").write_text(json.dumps(serial_all, indent=2) + "\n")
    (OUT / "a2_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    plot_densities(per_run)
    write_report(per_run, verdict)
    print("verdict", verdict["reading"], flush=True)
    print("wrote", OUT / "PHASE14_A_TAILS.md", flush=True)


if __name__ == "__main__":
    main()
