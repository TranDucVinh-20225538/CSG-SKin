#!/usr/bin/env python3
"""Paper B R6 split-stability (decisions/paperB_r6_split_stability.md).

Modes:
  prepare   write split records + path hashes for seeds {42,7,123,2027}; verify seed 42 vs REF
  verify42  exit 0 only if seed-42 path hashes match the frozen reference (must pass before train)
  train     --task_id 0..17  (or --isic_split_seed/--lambda_adv/--seed)
  report    write REPORT.md / summary.csv / BOOTSTRAP.json / VERDICT.json

Grid: isic_split_seed ∈ {7,123,2027} × λ ∈ {0,2} × train seed ∈ {42,52,62} = 18 runs.
PAD split fixed (Phase 2 split_seed=42). Primary = last.ckpt. Recipe = current (Table 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_phase3_sweep as p3  # noqa: E402  (transformers stub)
import r5_common as R  # noqa: E402
import train_r2_item4_lesion_split as T4  # noqa: E402
import train_phase2_pad_holdout as p2  # noqa: E402

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = ROOT / "results" / "paperB" / "r6" / "split_stability"
SPLITS = OUT / "splits"
RUNS = OUT / "runs"
CKPT = ROOT / "checkpoints" / "r6_split_stability"
PY = str(ROOT / ".venv" / "bin" / "python")

SPLIT_SEEDS_NEW = (7, 123, 2027)
SPLIT_SEEDS_ALL = (42,) + SPLIT_SEEDS_NEW
LAMBDAS = (0.0, 2.0)
TRAIN_SEEDS = (42, 52, 62)
REF42 = SPLITS / "seed_42" / "REF_HASHES.json"
DECISION = ROOT / "decisions" / "paperB_r6_split_stability.md"

# Frozen sha256 of sorted image path lists for isic_split_seed=42 (computed 2026-10-21 before any new train).
# Must match results of isic_splits_by_lesion(..., 42) on the current master CSV.
SEED42_REF = {
    "isic_split_seed": 42,
    "train_sha256": "fa6fca6b9a37b87adb0d2aa8c4f37b4d2ff63948d9bf80704cf0905548502ff8",
    "val_sha256": "5adc1d6372bae072bde4738d6fdc86e7579377330b3650ed3eb10971239cf947",
    "test_sha256": "0156db35350f97a016784dcabe6d3e55a537c19f74e221e4e7f38eccd2048e5d",
    "n_train": 16211,
    "n_val": 4053,
    "n_test": 5067,
    "hash_unit": "sorted full image paths, sha256 of newline-joined UTF-8",
}


def grid():
    """Task id 0..17 → (isic_split_seed, lambda_adv, train_seed)."""
    out = []
    for ss in SPLIT_SEEDS_NEW:
        for lam in LAMBDAS:
            for seed in TRAIN_SEEDS:
                out.append((ss, lam, seed))
    return out


def run_tag(ss, lam, seed):
    return "ss_s{}_ladv{}_s{}".format(ss, R.lam_tag(lam), seed)


def path_sha256(frame):
    blob = "\n".join(sorted(frame.path.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), int(len(frame))


def compute_split(ss):
    df = p2.load_master()
    isic = df[df.domain == "isic"].copy().reset_index(drop=True)
    train, val, test = T4.isic_splits_by_lesion(isic, isic_split_seed=ss)
    rec, _ = T4.split_record(isic_split_seed=ss)
    th, tn = path_sha256(train)
    vh, vn = path_sha256(val)
    eh, en = path_sha256(test)
    hashes = {
        "isic_split_seed": int(ss),
        "train_sha256": th, "val_sha256": vh, "test_sha256": eh,
        "n_train": tn, "n_val": vn, "n_test": en,
        "hash_unit": SEED42_REF["hash_unit"],
        "pad_split": "Phase 2 JSON pad_adv/pad_heldout (split_seed=42, fixed)",
    }
    return rec, hashes, (train, val, test)


def write_split(ss):
    d = SPLITS / "seed_{}".format(ss)
    d.mkdir(parents=True, exist_ok=True)
    rec, hashes, _ = compute_split(ss)
    (d / "split_record.json").write_text(json.dumps(rec, indent=2) + "\n")
    (d / "path_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


def verify_seed42(strict=True):
    """Return (ok, message). If strict and fail → SystemExit."""
    SPLITS.mkdir(parents=True, exist_ok=True)
    if not REF42.exists():
        REF42.parent.mkdir(parents=True, exist_ok=True)
        REF42.write_text(json.dumps(SEED42_REF, indent=2) + "\n")
    ref = json.loads(REF42.read_text())
    got = write_split(42)
    mism = []
    for k in ("train_sha256", "val_sha256", "test_sha256", "n_train", "n_val", "n_test"):
        if got[k] != ref[k]:
            mism.append("{}: got {} expected {}".format(k, got[k], ref[k]))
    if mism:
        msg = "SEED-42 SPLIT HASH MISMATCH — STOP. " + "; ".join(mism)
        if strict:
            raise SystemExit(msg)
        return False, msg
    return True, "seed 42 path hashes match REF_HASHES.json"


def cmd_prepare(_a):
    OUT.mkdir(parents=True, exist_ok=True)
    ok, msg = verify_seed42(strict=True)
    print(msg)
    for ss in SPLIT_SEEDS_NEW:
        h = write_split(ss)
        print("wrote split seed", ss, "n", h["n_train"], h["n_val"], h["n_test"],
              "test_sha", h["test_sha256"][:12])
    print("prepare ok; decision", DECISION.name)


def cmd_verify42(_a):
    ok, msg = verify_seed42(strict=True)
    print(msg)


def cmd_train(a):
    ok, msg = verify_seed42(strict=True)
    print(msg)
    if a.task_id is not None:
        ss, lam, seed = grid()[int(a.task_id)]
    else:
        ss, lam, seed = int(a.isic_split_seed), float(a.lambda_adv), int(a.seed)
    tag = run_tag(ss, lam, seed)
    write_split(ss)
    cmd = [PY, str(ROOT / "scripts" / "train_r5_dual_isic.py"),
           "--item", "split_stability",
           "--backbone", "effb3",
           "--lambda_adv", str(lam),
           "--seed", str(seed),
           "--isic_split_seed", str(ss),
           "--run_tag", tag,
           "--out_root", str(RUNS),
           "--ckpt_root", str(CKPT),
           "--recipe", "current",
           "--num_workers", str(a.num_workers)]
    if a.skip_done:
        cmd.append("--skip_done")
    if a.resume:
        cmd.append("--resume")
    if a.dry_run:
        cmd.append("--dry_run")
    print("exec", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _heldout_v1(run_dir, isic_split_seed):
    """P6 V1 on z_lesion_norm for this run's ISIC test fold."""
    import numpy as np
    import r5_p9v2 as V
    import r2_common as C

    cache = Path(run_dir) / "heldout_probe_V1.json"
    if cache.exists():
        return float(json.loads(cache.read_text())["V1"])
    feat_te = Path(run_dir) / "features" / "isic_test.npz"
    feat_ho = Path(run_dir) / "features" / "pad_heldout.npz"
    if not feat_te.exists() or not feat_ho.exists():
        return None
    _rec, (_tr, _va, test) = T4.split_record(isic_split_seed=isic_split_seed)
    gi, _ = T4.lesion_groups(test)
    meta = C.pad_heldout_meta()
    gp = np.asarray(meta.patient_id.astype(str).tolist(), dtype=str)
    te = np.load(feat_te)
    ho = np.load(feat_ho)
    zkey = "z_lesion_norm" if "z_lesion_norm" in te else "z"
    assert len(te["labels"]) == len(test), "isic_test length vs split"
    assert np.array_equal(te["labels"].astype(int), test.label_idx.astype(int).to_numpy()), "isic_test label order"
    val = float(V.v1_probe(te[zkey], ho[zkey], np.asarray(gi).astype(str), gp)["bal_acc_mean"])
    cache.write_text(json.dumps({
        "V1": val, "protocol": "P6 V1 / A3", "isic_split_seed": int(isic_split_seed),
    }, indent=1) + "\n")
    return val


def load_run(ss, lam, seed):
    tag = run_tag(ss, lam, seed)
    d = RUNS / tag
    sp = d / "summary.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text())
    s["_dir"] = str(d)
    s["_tag"] = tag
    s["isic_split_seed"] = ss
    # R1 rule (decisions/paperB_r6_r1.md): degraded if ID bal < 2× chance = 0.25
    s["degraded"] = bool(float(s["id_balanced_acc"]) < 0.25)
    v1 = _heldout_v1(d, ss)
    if v1 is None:
        v1 = float(s["leakage"]["test_vs_pad_heldout"]["bal_acc_mean"])
        s["leakage_heldout_source"] = "test_vs_pad_heldout"
    else:
        s["leakage_heldout_source"] = "P6_V1"
    s["leakage_heldout"] = v1
    return s


def cell_mean_sd(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def verdict_from_cells(cells):
    """cells[(ss, lam)] = dict with pad/fitz/med/sd means (non-degraded)."""
    # NO-GO: any split with pad λ=2 mean >= 0.5
    for ss in SPLIT_SEEDS_NEW:
        c2 = cells.get((ss, 2.0))
        if c2 is None or c2.get("pad") is None:
            # undefined λ=2 pad → cannot be GO; treat as not < 0.5 for NO-GO check only if we have a mean >= 0.5
            continue
        if c2["pad"] >= 0.5:
            return "NO-GO", "split {} pad_heldout λ=2 mean={:.3f} ≥ 0.5".format(ss, c2["pad"])

    # GO needs all 3 splits: pad λ=2 mean < 0.5 AND < λ=0; and ≥2/3 splits with ≥2/3 clinical < 0.5
    clinical_ok_splits = 0
    for ss in SPLIT_SEEDS_NEW:
        c0 = cells.get((ss, 0.0))
        c2 = cells.get((ss, 2.0))
        if not c0 or not c2 or c2.get("pad") is None or c0.get("pad") is None:
            return "INCONCLUSIVE", "split {} missing non-degraded pad means".format(ss)
        if not (c2["pad"] < 0.5 and c2["pad"] < c0["pad"]):
            return "INCONCLUSIVE", (
                "split {} pad λ=2 mean={:.3f} not (<0.5 and < λ=0={:.3f})".format(ss, c2["pad"], c0["pad"]))
        clin = [c2.get(k) for k in ("fitz", "med", "sd")]
        if any(x is None for x in clin):
            return "INCONCLUSIVE", "split {} missing clinical means".format(ss)
        n_below = sum(1 for x in clin if x < 0.5)
        if n_below >= 2:
            clinical_ok_splits += 1
    if clinical_ok_splits >= 2:
        return "GO", "all 3 splits invert pad; {}/3 splits have ≥2/3 clinical means < 0.5".format(clinical_ok_splits)
    return "INCONCLUSIVE", "pad GO conditions met but only {}/3 splits have ≥2/3 clinical < 0.5".format(
        clinical_ok_splits)


def _ood_groups(ood, n_o):
    import pandas as pd
    if ood == "pad_heldout":
        return R.pad_heldout_groups()
    if ood == "fitzpatrick17k":
        p13 = R.R5 / "p13" / "fitz_clusters.csv"
        if p13.exists():
            paths = sorted(p3.FITZ_DIR.glob("*.jpg"))
            cl = pd.read_csv(p13).set_index("md5hash")["cluster_union"]
            return R.idx_groups([str(cl[p.stem]) for p in paths])
        return [np.array([i]) for i in range(n_o)]
    if ood in R.EXTRA_SETS:
        return R.idx_groups(
            pd.read_csv(R.P7A_SETS / "{}.csv".format(ood), keep_default_na=False).group.astype(str))
    return [np.array([i]) for i in range(n_o)]


def _bal_acc(y, pred, n_classes=8):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    recalls = []
    for c in range(n_classes):
        m = y == c
        if not np.any(m):
            continue
        recalls.append(float((pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def bootstrap_id_bal(per_seed_logits_labels, id_groups, rng, n_boot):
    """per_seed: {seed: (logits, labels)}. Lesion-cluster bootstrap of balanced accuracy."""
    gi = [id_groups[j] for j in range(len(id_groups))]
    di = [np.concatenate([gi[j] for j in rng.integers(0, len(gi), len(gi))]) for _ in range(n_boot)]
    draws, per = {}, {}
    for s, (logits, labels) in per_seed_logits_labels.items():
        pred = np.asarray(logits).argmax(1)
        y = np.asarray(labels, int)
        pts = np.array([_bal_acc(y[di[b]], pred[di[b]]) for b in range(n_boot)])
        draws[s] = pts
        lo, hi = np.quantile(pts, [0.025, 0.975])
        point = _bal_acc(y, pred)
        per[str(s)] = {"point": point, "ci_lo": float(lo), "ci_hi": float(hi),
                       "ci_entirely_below_0.5": bool(hi < 0.5)}
    pooled = np.mean(np.stack(list(draws.values())), 0)
    lo, hi = np.quantile(pooled, [0.025, 0.975])
    k = int(sum(p["ci_entirely_below_0.5"] for p in per.values()))
    return {"mean": float(np.mean([p["point"] for p in per.values()])),
            "sd": float(np.std([p["point"] for p in per.values()], ddof=1)) if len(per) > 1 else 0.0,
            "pooled_ci": [float(lo), float(hi)], "k_below": k, "n_seeds": len(per), "per_seed": per}


def bootstrap_cell_split(ss, runs):
    """Per-split lesion groups for ID side (fixes n=5066 vs 5067). runs: {seed: run_dir}."""
    import numpy as np
    n_boot, rng_seed = R.prereg()
    _rec, (_tr, _va, test) = T4.split_record(isic_split_seed=ss)
    g_les, _ = T4.lesion_groups(test)
    gid = R.idx_groups(g_les)
    rng = np.random.default_rng(rng_seed)
    S = {s: dict(np.load(Path(d) / "scores.npz")) for s, d in runs.items()
         if (Path(d) / "scores.npz").exists()}
    if not S:
        return {}
    # sanity: ID score length matches this split's test
    n_id = len(next(iter(S.values()))["id__mahalanobis_classcond"])
    assert n_id == len(test), "id scores n={} vs test n={} for split {}".format(n_id, len(test), ss)
    out = {}
    for ood in ("pad_heldout", "fitzpatrick17k", "mednode", "sd198"):
        if not all("{}__mahalanobis_classcond".format(ood) in v for v in S.values()):
            continue
        n_o = len(next(iter(S.values()))["{}__mahalanobis_classcond".format(ood)])
        go = _ood_groups(ood, n_o)
        out[ood] = {
            "mahalanobis_classcond": R.bootstrap_cell(
                {s: (v["id__mahalanobis_classcond"], v["{}__mahalanobis_classcond".format(ood)])
                 for s, v in S.items()},
                gid, go, rng, n_boot)}
    # ID bal acc from features logits
    LL = {}
    for s, d in runs.items():
        te = Path(d) / "features" / "isic_test.npz"
        if te.exists():
            pack = np.load(te)
            LL[s] = (pack["logits"], pack["labels"])
    if LL:
        out["id_bal_acc"] = bootstrap_id_bal(LL, gid, rng, n_boot)
    return out


def cmd_report(_a):
    import csv
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    by_cell = defaultdict(list)
    for ss, lam, seed in grid():
        s = load_run(ss, lam, seed)
        if s is None:
            rows.append({"isic_split_seed": ss, "lambda": lam, "seed": seed, "status": "MISSING"})
            continue
        ood = s["ood"]
        row = {
            "isic_split_seed": ss, "lambda": lam, "seed": seed, "tag": s["_tag"],
            "status": "degraded" if s["degraded"] else "ok",
            "id_bal_acc": s["id_balanced_acc"],
            "leakage_heldout": s["leakage_heldout"],
            "leakage_heldout_source": s["leakage_heldout_source"],
            "maha_pad_heldout": ood["pad_heldout"]["mahalanobis_classcond"],
            "maha_fitz": ood["fitzpatrick17k"]["mahalanobis_classcond"],
            "maha_mednode": ood.get("mednode", {}).get("mahalanobis_classcond"),
            "maha_sd198": ood.get("sd198", {}).get("mahalanobis_classcond"),
            "degraded": s["degraded"],
            "gpu_hours": s.get("gpu_hours"),
        }
        rows.append(row)
        by_cell[(ss, lam)].append(row)

    keys = ["isic_split_seed", "lambda", "seed", "tag", "status", "id_bal_acc", "leakage_heldout",
            "maha_pad_heldout", "maha_fitz", "maha_mednode", "maha_sd198", "degraded", "gpu_hours",
            "leakage_heldout_source"]
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cells_agg = {}
    boot_out = {}
    for ss in SPLIT_SEEDS_NEW:
        for lam in LAMBDAS:
            rs = [r for r in by_cell[(ss, lam)] if not r.get("degraded") and r.get("status") != "MISSING"]
            all_rs = [r for r in by_cell[(ss, lam)] if r.get("status") != "MISSING"]

            def col(name, _rs=rs):
                return [r[name] for r in _rs if r.get(name) is not None]

            agg = {
                "n_ok": len(rs), "n_all": len(all_rs),
                "degraded_seeds": [r["seed"] for r in all_rs if r.get("degraded")],
            }
            for name, key in [("id", "id_bal_acc"), ("leak", "leakage_heldout"), ("pad", "maha_pad_heldout"),
                              ("fitz", "maha_fitz"), ("med", "maha_mednode"), ("sd", "maha_sd198")]:
                m, sd, n = cell_mean_sd(col(key))
                agg[name] = m
                agg[name + "_sd"] = sd
                agg[name + "_n"] = n
            cells_agg[(ss, lam)] = agg
            dirs = {int(r["seed"]): str(RUNS / r["tag"]) for r in rs}
            key = "ss{}_l{}".format(ss, R.lam_tag(lam))
            if dirs:
                blk = bootstrap_cell_split(ss, dirs)
                # serialise
                ser = {}
                for ood, dets in blk.items():
                    if ood == "id_bal_acc":
                        m = dets
                        ser[ood] = {k: m[k] for k in ("mean", "sd", "pooled_ci", "k_below", "n_seeds")}
                    else:
                        ser[ood] = {}
                        for d, m in dets.items():
                            ser[ood][d] = {k: m[k] for k in ("mean", "sd", "pooled_ci", "k_below", "n_seeds", "rule")}
                boot_out[key] = ser
                # attach CI into cells_agg for LaTeX
                if "pad_heldout" in ser:
                    m = ser["pad_heldout"]["mahalanobis_classcond"]
                    agg["pad_ci"] = m["pooled_ci"]
                    agg["pad_k"] = m["k_below"]
                if "fitzpatrick17k" in ser:
                    m = ser["fitzpatrick17k"]["mahalanobis_classcond"]
                    agg["fitz_ci"] = m["pooled_ci"]
                    agg["fitz_k"] = m["k_below"]
                if "mednode" in ser:
                    m = ser["mednode"]["mahalanobis_classcond"]
                    agg["med_ci"] = m["pooled_ci"]
                    agg["med_k"] = m["k_below"]
                if "sd198" in ser:
                    m = ser["sd198"]["mahalanobis_classcond"]
                    agg["sd_ci"] = m["pooled_ci"]
                    agg["sd_k"] = m["k_below"]
                if "id_bal_acc" in ser:
                    m = ser["id_bal_acc"]
                    agg["id_ci"] = m["pooled_ci"]
                    agg["id_k"] = m["k_below"]

    (OUT / "BOOTSTRAP.json").write_text(json.dumps(boot_out, indent=2) + "\n")

    cells_for_verdict = {}
    for k, agg in cells_agg.items():
        cells_for_verdict[k] = {x: agg[x] for x in ("pad", "fitz", "med", "sd")}
    verdict, reason = verdict_from_cells(cells_for_verdict)
    (OUT / "VERDICT.json").write_text(json.dumps({
        "verdict": verdict, "reason": reason,
        "criteria": "decisions/paperB_r6_split_stability.md (verbatim GO/NO-GO/INCONCLUSIVE)",
        "commit": R.git_head(),
    }, indent=2) + "\n")

    # REPORT.md
    L = [
        "# Paper B R6 — split stability",
        "",
        "Protocol: `decisions/paperB_r6_split_stability.md`. Primary checkpoint: `last.ckpt`. "
        "Recipe: `current` (Table 1 dual EffNet-B3). PAD split fixed (`split_seed=42`). "
        "Degraded: ID bal acc < 0.25 (excluded from means).",
        "",
        "## Verdict: **{}**".format(verdict),
        "",
        reason,
        "",
        "## Per-run",
        "",
        "| split | λ | seed | status | ID bal | leak (held-out) | Maha pad | Fitz | MED | SD-198 |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("status") == "MISSING":
            L.append("| {} | {:g} | {} | MISSING | — | — | — | — | — | — |".format(
                r["isic_split_seed"], r["lambda"], r["seed"]))
            continue
        L.append(
            "| {ss} | {lam:g} | {seed} | {st} | {id:.3f} | {lk:.3f} | {pad:.3f} | {fz:.3f} | {md} | {sd} |".format(
                ss=r["isic_split_seed"], lam=r["lambda"], seed=r["seed"], st=r["status"],
                id=r["id_bal_acc"], lk=r["leakage_heldout"], pad=r["maha_pad_heldout"], fz=r["maha_fitz"],
                md=("{:.3f}".format(r["maha_mednode"]) if r["maha_mednode"] is not None else "—"),
                sd=("{:.3f}".format(r["maha_sd198"]) if r["maha_sd198"] is not None else "—"),
            ))

    L += ["", "## Cell means (non-degraded)", "",
          "| split | λ | n | ID bal | leak | Maha pad | Fitz | MED | SD-198 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ss in SPLIT_SEEDS_NEW:
        for lam in LAMBDAS:
            a = cells_agg[(ss, lam)]
            def fmt(name):
                if a[name] is None:
                    return "—"
                return "{:.3f} ± {:.3f}".format(a[name], a[name + "_sd"] or 0.0)
            L.append("| {} | {:g} | {} | {} | {} | {} | {} | {} | {} |".format(
                ss, lam, a["n_ok"], fmt("id"), fmt("leak"), fmt("pad"), fmt("fitz"), fmt("med"), fmt("sd")))

    L += ["", "## Bootstrap (per-split ID lesion groups; PREREGISTER_BOOTSTRAP.json)", "",
          "ID side rebuilt from each `isic_split_seed` test fold (not the seed-42 table). "
          "pad_heldout by patient; Fitzpatrick17k by P13 `cluster_union`; MED-NODE/SD-198 by `group` column; "
          "ID bal acc by lesion.", "",
          "| cell | metric | mean ± sd | pooled CI | k_below |",
          "|---|---|---|---|---:|"]
    for k in sorted(boot_out):
        blk = boot_out[k]
        if "id_bal_acc" in blk:
            m = blk["id_bal_acc"]
            L.append("| {} | ID bal | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] | {}/{} |".format(
                k, m["mean"], m["sd"], m["pooled_ci"][0], m["pooled_ci"][1], m["k_below"], m["n_seeds"]))
        for ood, lab in (("pad_heldout", "Maha pad"), ("fitzpatrick17k", "Maha Fitz"),
                         ("mednode", "Maha MED"), ("sd198", "Maha SD-198")):
            if ood not in blk:
                continue
            m = blk[ood]["mahalanobis_classcond"]
            L.append("| {} | {} | {:.3f} ± {:.3f} | [{:.3f}, {:.3f}] | {}/{} |".format(
                k, lab, m["mean"], m["sd"], m["pooled_ci"][0], m["pooled_ci"][1],
                m["k_below"], m["n_seeds"]))

    tex_rows = [
        "% tab:splits rows with pooled 95\\% CI (PREREGISTER_BOOTSTRAP; per-split lesion groups).",
        "% Columns: split, lam, ID, ID[lo,hi], leak, pad, pad[lo,hi], Fitz, Fitz[lo,hi], MED, MED[lo,hi], SD, SD[lo,hi]",
    ]
    for ss in SPLIT_SEEDS_NEW:
        for lam in LAMBDAS:
            a = cells_agg[(ss, lam)]

            def msd(name):
                return "{:.3f}\\,$\pm$\\,{:.3f}".format(a[name], a[name + "_sd"] or 0.0)

            def cij(prefix):
                c = a.get(prefix + "_ci")
                return "[{:.3f}, {:.3f}]".format(c[0], c[1]) if c else "--"

            tex_rows.append(
                "{ss} & {lam:g} & ${id}$ & ${idci}$ & ${lk}$ & ${pad}$ & ${padci}$ & ${fz}$ & ${fzci}$ "
                "& ${md}$ & ${mdci}$ & ${sd}$ & ${sdci}$ \\\\".format(
                    ss=ss, lam=lam, id=msd("id"), idci=cij("id"), lk=msd("leak"),
                    pad=msd("pad"), padci=cij("pad"), fz=msd("fitz"), fzci=cij("fitz"),
                    md=msd("med"), mdci=cij("med"), sd=msd("sd"), sdci=cij("sd")))
    (OUT / "LATEX_SPLITS_WITH_CI.tex").write_text("\n".join(tex_rows) + "\n")

    L += ["", "## LaTeX (`LATEX_SPLITS_WITH_CI.tex`)", "", "```latex"] + tex_rows + ["```", "",
          "## Criteria (pre-registered, unchanged)", "",
          "- **GO:** ở cả 3 split, AUROC Mahalanobis trên pad_heldout tại λ=2 có mean < 0.5 VÀ thấp hơn λ=0 cùng split; "
          "và ở ≥ 2/3 split, ít nhất 2/3 tập lâm sàng (Fitz, MED-NODE, SD-198) có mean < 0.5.",
          "- **NO-GO:** có ≥ 1 split mà pad_heldout λ=2 có mean ≥ 0.5.",
          "- **Còn lại: INCONCLUSIVE.**",
          ""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "cells.json").write_text(json.dumps(
        {"{}_{}".format(ss, R.lam_tag(lam)): cells_agg[(ss, lam)] for ss in SPLIT_SEEDS_NEW for lam in LAMBDAS},
        indent=2) + "\n")
    print("verdict", verdict, reason)
    print("wrote", OUT / "REPORT.md")
    print("wrote", OUT / "LATEX_SPLITS_WITH_CI.tex")
    print("wrote", OUT / "BOOTSTRAP.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "verify42", "train", "report", "list"))
    ap.add_argument("--task_id", type=int, default=None)
    ap.add_argument("--isic_split_seed", type=int, default=None)
    ap.add_argument("--lambda_adv", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args()
    if a.mode == "list":
        for i, (ss, lam, seed) in enumerate(grid()):
            print(i, run_tag(ss, lam, seed), ss, lam, seed)
        return
    {"prepare": cmd_prepare, "verify42": cmd_verify42, "train": cmd_train, "report": cmd_report}[a.mode](a)


if __name__ == "__main__":
    main()
