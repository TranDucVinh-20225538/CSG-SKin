#!/usr/bin/env python3
"""Interim R2 send: oracle λ=2 (5 seeds) + matched IWDAN λ=2 + R1 DANN shift100 last.ckpt.

Gap recovered locked (decisions/paperB_r6_r2.md): (method−DANN)/(λ0−DANN) per unseen set;
λ0 = R1 shift100 λ=0 last.ckpt (same recipe), computed per set. Always co-report absolute
Δ = method−DANN; flag small |λ0−DANN|. Does not close the pre-committed fix reading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r6_common as R  # noqa: E402
import r6_report as RR  # noqa: E402

OUT = R.RES / "r6" / "r2" / "cur"
UNSEEN = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198")
SEEDS = R.SEEDS


def main():
    rt = "cur"
    r1 = RR.item_dir("r1", rt)
    r2 = RR.item_dir("r2", rt)

    def pack(root, prefix_fmt, lam_tag):
        fd, missing = {}, []
        for s in SEEDS:
            tag = prefix_fmt.format(lam=lam_tag, s=s)
            f = root / "features" / tag
            if (f / "isic_test.npz").exists():
                fd[s] = f
            else:
                missing.append(tag)
        return fd, missing

    dann_f, miss_d = pack(r1, "r1_cur_shift100_ladv{lam}_s{s}", "2")
    l0_f, miss_0 = pack(r1, "r1_cur_shift100_ladv{lam}_s{s}", "0")
    iw_f, miss_iw = pack(r2, "r2_cur_iwdan_ladv{lam}_s{s}", "2")
    ora_f, miss_o = pack(r2, "r2_cur_iwdan_oracle_ladv{lam}_s{s}", "2")

    if miss_o:
        raise SystemExit("oracle λ=2 incomplete: " + ", ".join(miss_o))
    if miss_iw:
        raise SystemExit("IWDAN λ=2 incomplete: " + ", ".join(miss_iw))
    if miss_d or miss_0:
        raise SystemExit("R1 DANN/λ0 missing: " + ", ".join(miss_d + miss_0))

    data = R.load_data()
    lesion_g, _ = R.T.lesion_groups(data["test"])
    idg = R.idx_groups(lesion_g)

    lines = [
        "# Paper B R6 R2 — interim: oracle λ = 2 vs IWDAN λ = 2 vs R1 DANN",
        "",
        f"Commit `{R.C.git_head()}`. Order deviation 2026-10-17 (oracle λ=2 advanced; criteria unchanged).",
        "",
        "**Baselines (locked):** DANN = R1 shift 100% λ=2 `last.ckpt` (pad ≈ 0.368, Fitz ≈ 0.340, leakage ≈ 0.634); "
        "λ0 = R1 shift 100% λ=0 `last.ckpt`, **AUROC per unseen set**. Same recipe `current`, seeds "
        "`(42,52,62,72,82)`, lesion-level / Phase 2 split.",
        "",
        "**Gap recovered (locked):** `(method−DANN)/(λ0−DANN)` Mahalanobis, paired by seed; always co-report "
        "absolute Δ. If mean `|λ0−DANN| < 0.05` → flag **small denom — prefer absolute Δ**.",
        "",
        "Pre-committed fix reading **not** closed here — wait for CDAN and λ = 0.5.",
        "",
        "## Per-seed Mahalanobis (point)",
        "",
        "| seed | DANN pad | IWDAN pad | oracle pad | DANN Fitz | IWDAN Fitz | oracle Fitz |",
        "|---:|---|---|---|---|---|---|",
    ]

    def pt(feat, ood):
        sc = RR._score_packs(feat, ood)
        return {s: R.auroc(*sc[s]) for s in SEEDS if s in sc}

    rows_json = {"seeds": {}, "gap_recovered": {}, "paired_delta": {}}
    for s in SEEDS:
        dpad = R.auroc(*RR._score_packs({s: dann_f[s]}, "pad_heldout")[s])
        ipad = R.auroc(*RR._score_packs({s: iw_f[s]}, "pad_heldout")[s])
        opad = R.auroc(*RR._score_packs({s: ora_f[s]}, "pad_heldout")[s])
        dfitz = R.auroc(*RR._score_packs({s: dann_f[s]}, "fitzpatrick17k")[s])
        ifitz = R.auroc(*RR._score_packs({s: iw_f[s]}, "fitzpatrick17k")[s])
        ofitz = R.auroc(*RR._score_packs({s: ora_f[s]}, "fitzpatrick17k")[s])
        lines.append(
            f"| {s} | {dpad:.3f} | {ipad:.3f} | {opad:.3f} | {dfitz:.3f} | {ifitz:.3f} | {ofitz:.3f} |"
        )
        rows_json["seeds"][str(s)] = {
            "dann_pad": dpad, "iwdan_pad": ipad, "oracle_pad": opad,
            "dann_fitz": dfitz, "iwdan_fitz": ifitz, "oracle_fitz": ofitz,
        }

    lines += ["", "## Gap recovered and paired Δ vs DANN (per unseen set)", "",
              "Absolute Δ is primary when the denom is small; ratio is still printed for every set.",
              "",
              "| method | set | Δ vs DANN [cluster CI] | gap recovered (method−DANN)/(λ0−DANN) [seed boot CI · Δabs · denom] |",
              "|---|---|---|---|"]

    for name, feat in (("IWDAN", iw_f), ("IWDAN_ORACLE", ora_f)):
        for ood in UNSEEN:
            try:
                og, _ = R.ood_groups(ood, data)
            except FileNotFoundError:
                continue
            dlt = RR.paired_delta_cluster_ci(feat, dann_f, ood, idg, og)
            gap = RR.gap_recovered_seed_boot(feat, dann_f, l0_f, ood)
            key = f"{name}|2|{ood}"
            if dlt:
                rows_json["paired_delta"][key] = dlt
            if gap:
                rows_json["gap_recovered"][key] = gap
            lines.append("| {} | {} | {} | {} |".format(
                name, ood, RR._fmt_delta(dlt), RR._fmt_gap(gap)))

    lines += [
        "",
        "## Cell means (Mahalanobis)",
        "",
    ]
    for ood in UNSEEN:
        def mean_pts(feat):
            sc = RR._score_packs(feat, ood)
            v = [R.auroc(*sc[s]) for s in SEEDS if s in sc]
            return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        dm, ds = mean_pts(dann_f)
        im, i_s = mean_pts(iw_f)
        om, os_ = mean_pts(ora_f)
        lines.append(f"- **{ood}:** DANN {dm:.3f}±{ds:.3f}; IWDAN {im:.3f}±{i_s:.3f}; oracle {om:.3f}±{os_:.3f}")

    lines += ["", f"**On origin/paperb/reviewer-r1:** _(fill after push)._", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    rep = OUT / "REPORT_ORACLE_L2_INTERIM.md"
    rep.write_text("\n".join(lines) + "\n")
    (OUT / "oracle_l2_interim.json").write_text(json.dumps(rows_json, indent=2, default=float) + "\n")
    print("wrote", rep, flush=True)


if __name__ == "__main__":
    main()
