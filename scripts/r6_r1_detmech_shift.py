#!/usr/bin/env python3
"""Paper B R6 R1 post hoc: detmech at shift 0% vs 100% (decisions/paperB_r6_r1_detmech_shift.md). CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
sys.path.insert(0, "/data2/hpcshared/Vinh/CSG-Skin")
import r5cpu_common as R5  # noqa: E402
import r6_common as R  # noqa: E402

OUT = R.RES / "r6" / "r1" / "cur" / "detmech_shift"
R1_FEAT = R.RES / "r6" / "r1" / "cur" / "features"
A4_FEAT = R.RES / "r5" / "p8" / "main" / "features"
OODS = ("pad_heldout", "fitzpatrick17k", "mednode", "sd198")
MIN_N = 20
RNG = 20261005
FITZ_MAP = {
    "melanoma": "MEL", "basal cell carcinoma": "BCC", "squamous cell carcinoma": "SCC",
    "actinic keratosis": "AK", "dermatofibroma": "DF", "congenital nevus": "NV",
}


def healthy(summary_path):
    if not summary_path.exists():
        return None
    s = json.loads(summary_path.read_text())
    m = s.get("metrics") or {}
    bal = (m.get("id") or {}).get("bal_acc")
    deg = m.get("degraded")
    if deg is None and bal is not None:
        deg = bool(bal < 0.25)
    return {"seed": s.get("seed"), "id_bal": bal, "degraded": bool(deg), "tag": s.get("tag")}


def load_feats(feat_dir):
    need = ("isic_train", "isic_test") + OODS
    return R.load_feats(Path(feat_dir), need)


def ood_labels(name, pack, n):
    """Return int labels length n; -1 = unmapped / unavailable."""
    y = np.asarray(pack["labels"], int)
    if name == "pad_heldout":
        return y
    if name in ("mednode", "sd198"):
        meta = pd.read_csv(R.P7A_SETS / "{}.csv".format(name), keep_default_na=False)
        assert len(meta) == n, (name, len(meta), n)
        out = np.full(n, -1, dtype=int)
        for i, m in enumerate(meta["mapped"].tolist()):
            if m in R.LABELS:
                out[i] = R.LABELS.index(m)
        return out
    if name == "fitzpatrick17k":
        stems = R5.fitz_stems()
        assert len(stems) == n
        fz = pd.read_csv(R.RES / "phase11_third_domain" / "fitzpatrick17k.csv").drop_duplicates("md5hash").set_index("md5hash")
        out = np.full(n, -1, dtype=int)
        for i, st in enumerate(stems):
            lab = fz.label.get(st) if st in fz.index else None
            m = FITZ_MAP.get(lab) if isinstance(lab, str) else None
            if m in R.LABELS:
                out[i] = R.LABELS.index(m)
        return out
    return np.full(n, -1, dtype=int)


def groups_for(name):
    if name == "pad_heldout":
        return R5.pad_heldout_patients()
    if name == "fitzpatrick17k":
        return R5.fitz_groups("cluster_union")
    if name in ("mednode", "sd198"):
        meta = pd.read_csv(R.P7A_SETS / "{}.csv".format(name), keep_default_na=False)
        return meta["group"].astype(str).to_numpy()
    raise KeyError(name)


def class_mix(y, mask=None):
    y = np.asarray(y, int)
    if mask is not None:
        y = y[mask]
    y = y[y >= 0]
    c = np.bincount(y, minlength=R.NCLS).astype(float)
    s = c.sum()
    return c / s if s > 0 else c


def ood_w_to_id(y_ood, pi_id):
    y = np.asarray(y_ood, int)
    pi_o = class_mix(y)
    w = np.zeros(len(y), dtype=np.float64)
    for c in range(R.NCLS):
        m = y == c
        if m.any() and pi_o[c] > 0:
            w[m] = pi_id[c] / pi_o[c]
    # only labelled OOD enters; zero-weight rows dropped via mask in auroc call
    return w


def auroc_w(sid, sood, w_ood=None, ood_mask=None):
    sid, sood = np.asarray(sid, float), np.asarray(sood, float)
    if ood_mask is not None:
        sood = sood[ood_mask]
        if w_ood is not None:
            w_ood = np.asarray(w_ood, float)[ood_mask]
    if w_ood is not None:
        w_ood = np.asarray(w_ood, float)
        keep = w_ood > 0
        sood, w_ood = sood[keep], w_ood[keep]
        if len(sood) == 0:
            return float("nan")
        return R5.auroc(sid, sood, w_ood=w_ood)
    return R5.auroc(sid, sood)


def pooled_boot(score_pairs, gi, go, n_boot, seed=RNG, w_ood=None, ood_mask=None):
    """score_pairs: {seed: (sid, sood)}; optional w_ood {seed: weights} and shared ood_mask."""
    rng = np.random.default_rng(seed)
    id_draws = R5.draws(gi, n_boot, rng)
    ood_draws = R5.draws(go, n_boot, np.random.default_rng(seed + 7))
    seeds = sorted(score_pairs)
    pts = {}
    for s in seeds:
        sid, so = score_pairs[s]
        ww = None if w_ood is None else w_ood[s]
        pts[s] = auroc_w(sid, so, w_ood=ww, ood_mask=ood_mask)
    mat = np.full((n_boot, len(seeds)), np.nan)
    for b, (ii, oo) in enumerate(zip(id_draws, ood_draws)):
        for j, s in enumerate(seeds):
            sid, so = score_pairs[s]
            ww = None if w_ood is None else w_ood[s]
            if ood_mask is not None:
                use = np.asarray(ood_mask, bool).copy()
                take = np.zeros(len(so), bool)
                take[oo] = True
                use &= take
                if not use.any():
                    continue
                mat[b, j] = auroc_w(sid[ii], so, w_ood=ww, ood_mask=use)
            else:
                if ww is None:
                    mat[b, j] = R5.auroc(sid[ii], so[oo])
                else:
                    mat[b, j] = R5.auroc(sid[ii], so[oo], w_ood=ww[oo])
    ok = np.isfinite(mat).any(1)
    pooled = np.nanmean(mat[ok], 1) if ok.any() else np.array([])
    lo, hi = (np.nanpercentile(pooled, [2.5, 97.5]) if len(pooled) else (np.nan, np.nan))
    v = np.array([pts[s] for s in seeds], float)
    return {
        "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
        "ci_lo": float(lo), "ci_hi": float(hi), "n_seeds": len(seeds),
        "per_seed": {int(s): (None if pts[s] != pts[s] else float(pts[s])) for s in seeds},
    }


def discover_cells():
    cells = []
    # A4 λ=0 baseline
    a4 = []
    for s in R.SEEDS:
        d = A4_FEAT / "main_ladv0_s{}".format(s)
        summ = R.RES / "r5" / "p8" / "main" / "runs" / "main_ladv0_s{}".format(s) / "summary.json"
        # summary may live elsewhere
        if not summ.exists():
            # try glob
            cands = list((R.RES / "r5" / "p8").glob("**/main_ladv0_s{}/summary.json".format(s)))
            summ = cands[0] if cands else summ
        info = healthy(summ) if summ.exists() else {"seed": s, "id_bal": None, "degraded": False}
        if d.exists() and not info.get("degraded"):
            a4.append({"seed": s, "feat_dir": d, "id_bal": info.get("id_bal")})
    cells.append({"cell": "A4_lam0", "label": "A4 λ=0 (baseline)", "shift_pct": None, "lambda_adv": 0.0,
                  "runs": a4})
    # R1 λ=0 if any
    r1_0 = []
    for d in sorted(R1_FEAT.glob("r1_cur_shift*_ladv0_s*")):
        # exclude ladv0p5
        if "ladv0p" in d.name:
            continue
        seed = int(d.name.rsplit("_s", 1)[1])
        summ = R.RES / "r6" / "r1" / "cur" / "runs" / d.name / "summary.json"
        info = healthy(summ) if summ.exists() else {"seed": seed, "degraded": False}
        if not info.get("degraded"):
            r1_0.append({"seed": seed, "feat_dir": d, "id_bal": info.get("id_bal")})
    if r1_0:
        cells.append({"cell": "R1_lam0", "label": "R1 λ=0 (baseline)", "shift_pct": None, "lambda_adv": 0.0,
                      "runs": r1_0})
    for sh in (0, 100):
        runs, skipped = [], []
        for s in R.SEEDS:
            d = R1_FEAT / "r1_cur_shift{}_ladv2_s{}".format(sh, s)
            summ = R.RES / "r6" / "r1" / "cur" / "runs" / d.name / "summary.json"
            info = healthy(summ) if summ.exists() else {"seed": s, "degraded": True, "id_bal": None}
            if not d.exists():
                skipped.append({"seed": s, "reason": "missing_features"})
                continue
            if info.get("degraded"):
                skipped.append({"seed": s, "reason": "degraded", "id_bal": info.get("id_bal")})
                continue
            runs.append({"seed": s, "feat_dir": d, "id_bal": info.get("id_bal")})
        cells.append({"cell": "R1_lam2_shift{}".format(sh), "label": "R1 λ=2 shift {}%".format(sh),
                      "shift_pct": sh, "lambda_adv": 2.0, "runs": runs, "skipped": skipped})
    return cells


def analyse_cell(cell, n_boot):
    runs = cell["runs"]
    if not runs:
        return {"cell": cell["cell"], "empty": True}
    gi = R5.id_test_groups()
    # score each seed
    scored = {}
    for r in runs:
        Fs = load_feats(r["feat_dir"])
        sc = R.Scorer(Fs["isic_train"])
        sid = sc.scores(Fs["isic_test"])["mahalanobis_classcond"]
        yid = Fs["isic_test"]["labels"].astype(int)
        pi_id = class_mix(yid)
        id_nv = float(np.mean(sc.nearest_class(Fs["isic_test"]["z_lesion_norm"]) == R.NV))
        pack = {"sid": sid, "yid": yid, "pi_id": pi_id, "id_nv_nearest": id_nv, "sc": sc, "Fs": Fs}
        scored[r["seed"]] = pack

    out = {"cell": cell["cell"], "label": cell["label"], "shift_pct": cell.get("shift_pct"),
           "lambda_adv": cell["lambda_adv"], "seeds": [r["seed"] for r in runs],
           "skipped": cell.get("skipped", []), "ood": {}}

    for ood in OODS:
        go = groups_for(ood)
        # per-seed sood + nearest + labels
        pairs, nv_pairs, non_pairs = {}, {}, {}
        w_ood, nv_fracs = {}, []
        y_ref = None
        for s, pack in scored.items():
            Fs, sc = pack["Fs"], pack["sc"]
            so = sc.scores(Fs[ood])["mahalanobis_classcond"]
            z = Fs[ood]["z_lesion_norm"]
            nc = sc.nearest_class(z)
            y = ood_labels(ood, Fs[ood], len(so))
            if y_ref is None:
                y_ref = y
            pairs[s] = (pack["sid"], so)
            nv_m, non_m = nc == R.NV, nc != R.NV
            nv_pairs[s] = (pack["sid"], so[nv_m]) if nv_m.any() else (pack["sid"], so[:0])
            non_pairs[s] = (pack["sid"], so[non_m]) if non_m.any() else (pack["sid"], so[:0])
            nv_fracs.append(float(nv_m.mean()))
            w_ood[s] = ood_w_to_id(y, pack["pi_id"])

        # For NV/non-NV subset AUROC, bootstrap on subset groups
        def boot_partition(which):
            rng = np.random.default_rng(RNG + (0 if which == "nv" else 1) + (abs(hash(ood)) % 1000))
            id_draws = R5.draws(gi, n_boot, rng)
            ood_draws = R5.draws(go, n_boot, np.random.default_rng(RNG + 99))
            seeds = sorted(scored)
            pts = []
            mat = np.full((n_boot, len(seeds)), np.nan)
            for j, s in enumerate(seeds):
                pack = scored[s]
                so = pack["sc"].scores(pack["Fs"][ood])["mahalanobis_classcond"]
                nc = pack["sc"].nearest_class(pack["Fs"][ood]["z_lesion_norm"])
                m = nc == R.NV if which == "nv" else nc != R.NV
                pts.append(R5.auroc(pack["sid"], so[m]) if m.any() else float("nan"))
                for b, (ii, oo) in enumerate(zip(id_draws, ood_draws)):
                    sel = np.asarray(oo, int)[m[np.asarray(oo, int)]]
                    if len(sel) == 0:
                        continue
                    mat[b, j] = R5.auroc(pack["sid"][ii], so[sel])
            v = np.asarray(pts, float)
            pooled = np.nanmean(mat, 1)
            ok = pooled[np.isfinite(pooled)]
            lo, hi = (np.nanpercentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
            return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                    "ci_lo": float(lo), "ci_hi": float(hi), "n_seeds": int(np.isfinite(v).sum())}

        unw = pooled_boot(pairs, gi, go, n_boot, seed=RNG)
        # reweight only where labels exist
        lab_mask = y_ref >= 0
        if lab_mask.any():
            rw = pooled_boot(pairs, gi, go, n_boot, seed=RNG + 3, w_ood=w_ood, ood_mask=lab_mask)
        else:
            rw = None
        nv_b = boot_partition("nv")
        non_b = boot_partition("non")

        # class-matched
        cm_rows = []
        if lab_mask.any():
            for c in range(R.NCLS):
                name = R.LABELS[c]
                # ID class c / OOD class c — groups sliced
                mid = scored[runs[0]["seed"]]["yid"] == c
                mood = y_ref == c
                n_id, n_o = int(mid.sum()), int(mood.sum())
                elig = n_id >= MIN_N and n_o >= MIN_N
                if n_id == 0 or n_o == 0:
                    cm_rows.append({"class": name, "n_id": n_id, "n_ood": n_o, "eligible": False})
                    continue
                gi_c, go_c = gi[mid], go[mood]
                cm_pairs = {}
                for s, pack in scored.items():
                    so = pack["sc"].scores(pack["Fs"][ood])["mahalanobis_classcond"]
                    cm_pairs[s] = (pack["sid"][pack["yid"] == c], so[y_ref == c])
                bb = pooled_boot(cm_pairs, gi_c, go_c, n_boot, seed=RNG + 20 + c)
                cm_rows.append({"class": name, "n_id": n_id, "n_ood": n_o, "eligible": elig, **{
                    k: bb[k] for k in ("mean", "sd", "ci_lo", "ci_hi")}})

        id_nv = float(np.mean([scored[s]["id_nv_nearest"] for s in scored]))
        out["ood"][ood] = {
            "nv_frac_ood_mean": float(np.mean(nv_fracs)),
            "nv_frac_ood_sd": float(np.std(nv_fracs, ddof=1)) if len(nv_fracs) > 1 else 0.0,
            "id_nv_nearest_base": id_nv,
            "maha_unweighted": unw,
            "maha_ood_reweighted_to_id_mix": rw,
            "maha_nv_nearest_ood": nv_b,
            "maha_non_nv_nearest_ood": non_b,
            "class_matched": cm_rows,
            "n_labelled_ood": int(lab_mask.sum()),
        }
    return out


def fmt(b):
    if b is None:
        return "—"
    return "{:.3f} ± {:.3f} [{:.3f}, {:.3f}]".format(b["mean"], b["sd"], b["ci_lo"], b["ci_hi"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre = R5.prereg()
    n_boot = int(pre["n_resamples"])
    cells = discover_cells()
    results = []
    for cell in cells:
        print("cell", cell["label"], "n_runs", len(cell["runs"]), "skipped", cell.get("skipped"), flush=True)
        rec = analyse_cell(cell, n_boot)
        results.append(rec)
        (OUT / "{}.json".format(cell["cell"])).write_text(json.dumps(rec, indent=1, default=float) + "\n")

    # summary rows
    rows = []
    for rec in results:
        if rec.get("empty"):
            continue
        for ood, blk in rec["ood"].items():
            rw = blk["maha_ood_reweighted_to_id_mix"]
            rows.append({
                "cell": rec["cell"], "label": rec["label"], "shift_pct": rec["shift_pct"],
                "ood": ood, "n_seeds": len(rec["seeds"]),
                "nv_frac_ood": blk["nv_frac_ood_mean"], "id_nv_base": blk["id_nv_nearest_base"],
                "maha": blk["maha_unweighted"]["mean"], "maha_ci_lo": blk["maha_unweighted"]["ci_lo"],
                "maha_ci_hi": blk["maha_unweighted"]["ci_hi"],
                "maha_rw": None if rw is None else rw["mean"],
                "maha_rw_ci_lo": None if rw is None else rw["ci_lo"],
                "maha_rw_ci_hi": None if rw is None else rw["ci_hi"],
                "maha_nv_nearest": blk["maha_nv_nearest_ood"]["mean"],
                "maha_non_nv_nearest": blk["maha_non_nv_nearest_ood"]["mean"],
            })
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)

    # Reading on pad_heldout primarily (has full labels); also note others
    by = {r["cell"]: r for r in results if not r.get("empty")}
    L = [
        "# Paper B R6 R1 — post hoc detmech: shift 0% vs 100% (λ = 2)",
        "",
        "Design: `decisions/paperB_r6_r1_detmech_shift.md`. Commit `{}`. "
        "Mahalanobis on z_lesion_norm; fit ISIC train; healthy seeds only (ID bal ≥ 0.25). "
        "Cluster CIs: pad=patient, Fitz=P13 cluster_union, MED/SD=p7a group; ID=lesion. "
        "PREREGISTER {} draws. R1 λ=0 features absent → baseline is A4 λ=0 only.".format(
            R5.git_head(), n_boot),
        "",
        "## Cells",
        "",
    ]
    for rec in results:
        if rec.get("empty"):
            L.append("- {}: empty".format(rec.get("cell")))
            continue
        L.append("- **{}**: seeds {}; skipped {}".format(
            rec["label"], rec["seeds"], rec.get("skipped")))
    L += ["", "## Per set (pooled)", ""]
    for ood in OODS:
        L += ["### {}".format(ood), "",
              "| cell | nv_frac OOD (ID base) | Maha | Maha OOD→ID mix | NV-nearest OOD | non-NV-nearest OOD |",
              "|---|---|---|---|---|---|"]
        for rec in results:
            if rec.get("empty") or ood not in rec.get("ood", {}):
                continue
            b = rec["ood"][ood]
            rw = b["maha_ood_reweighted_to_id_mix"]
            L.append("| {} | {:.3f} (ID {:.3f}) | {} | {} | {} | {} |".format(
                rec["label"], b["nv_frac_ood_mean"], b["id_nv_nearest_base"],
                fmt(b["maha_unweighted"]), fmt(rw),
                fmt(b["maha_nv_nearest_ood"]), fmt(b["maha_non_nv_nearest_ood"])))
        L.append("")

    # class-matched tables for each cell × pad (and note others)
    L += ["## Class-matched Mahalanobis (eligible n≥20 both sides)", ""]
    for rec in results:
        if rec.get("empty"):
            continue
        L.append("### {}".format(rec["label"]))
        for ood in OODS:
            cm = rec["ood"][ood].get("class_matched") or []
            if not cm:
                L.append("- {}: no labels for class-matched.".format(ood))
                continue
            L += ["", "**{}**".format(ood), "",
                  "| class | n_ID | n_OOD | mean ± sd [CI] | in reading |",
                  "|---|---:|---:|---|:---:|"]
            for r in cm:
                if "mean" not in r:
                    L.append("| {} | {} | {} | — | no |".format(r["class"], r["n_id"], r["n_ood"]))
                    continue
                L.append("| {} | {} | {} | {:.3f} ± {:.3f} [{:.3f}, {:.3f}] | {} |".format(
                    r["class"], r["n_id"], r["n_ood"], r["mean"], r["sd"], r["ci_lo"], r["ci_hi"],
                    "yes" if r["eligible"] else "no"))
        L.append("")

    # Reading
    def rw_pad(cell_id):
        r = by.get(cell_id)
        if not r:
            return None
        return r["ood"]["pad_heldout"]["maha_ood_reweighted_to_id_mix"]

    def non_nv(cell_id, ood="pad_heldout"):
        r = by.get(cell_id)
        return None if not r else r["ood"][ood]["maha_non_nv_nearest_ood"]["mean"]

    c0, c100 = "R1_lam2_shift0", "R1_lam2_shift100"
    rw0, rw100 = rw_pad(c0), rw_pad(c100)
    reading_lines = ["## Reading (post hoc)", ""]
    if rw0 and rw100:
        a0, a100 = rw0["mean"], rw100["mean"]
        if a0 >= 0.5 and a100 < 0.5:
            tag = "label_shift_drives_3a_and_3b"
            txt = ("Reweighted Mahalanobis on pad_heldout ≥ 0.5 at shift 0% ({:.3f}) but < 0.5 at shift 100% "
                   "({:.3f}) → label shift drives both 3a and 3b.".format(a0, a100))
        elif a0 >= 0.5 and a100 >= 0.5:
            tag = "3b_needs_label_shift_only_through_3a"
            txt = ("Reweighted Mahalanobis ≥ 0.5 at both shifts (0%: {:.3f}, 100%: {:.3f}) → 3b needs label "
                   "shift only through 3a.".format(a0, a100))
        else:
            tag = "mixed_or_other"
            txt = ("Reweighted pad_heldout: shift0={:.3f}, shift100={:.3f} — neither pre-committed branch; "
                   "see non-NV-nearest / class-matched.".format(a0, a100))
        reading_lines += ["**{}.** {}".format(tag, txt), ""]
        # within-class pull at shift 0
        nn0 = non_nv(c0)
        nn100 = non_nv(c100)
        elig_below = []
        if c0 in by:
            for r in by[c0]["ood"]["pad_heldout"].get("class_matched") or []:
                if r.get("eligible") and r.get("mean") is not None and r["mean"] < 0.5:
                    elig_below.append(r["class"])
        reading_lines.append(
            "Within-class / non-nevus at shift 0% (pad_heldout): non-NV-nearest Maha = {:.3f}; "
            "eligible class-matched < 0.5: {}.".format(
                nn0 if nn0 is not None else float("nan"),
                ", ".join(elig_below) if elig_below else "none"))
        if nn0 is not None and nn0 >= 0.5 and not elig_below:
            reading_lines.append(
                "At shift 0%, within-class pull on pad_heldout **recovers** (non-NV-nearest and class-matched ≥ 0.5).")
        elif nn0 is not None and nn0 < 0.5:
            reading_lines.append(
                "At shift 0%, non-NV-nearest still < 0.5 → within-class pull does **not** fully disappear "
                "(not only the nevus shift).")
        reading_lines.append("")
    else:
        reading_lines += ["Insufficient cells for the pre-committed reading.", ""]

    L += reading_lines
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    R5.write_json(OUT / "results.json", {"cells": results, "reading": reading_lines})
    print("\n".join(L[-40:]))
    print("wrote", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
