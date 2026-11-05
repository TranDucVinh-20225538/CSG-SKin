#!/usr/bin/env python3
"""R11 level b: redraw bootstrap summaries from cached features; compare to stored files.

Writes results/paperB/r11/b/<same name> and results/paperB/r11/b/COMPARE.md.
If any |diff| > 1e-9 on mean/sd/ci/per-seed CI, or k/Rule A differ: report and stop (exit 2).
Does not modify stored repo bootstrap files.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r5_ckpt_audit as A  # noqa: E402
import r5cpu_common as R5  # noqa: E402
import r6_common as R  # noqa: E402
import r6_report as RR  # noqa: E402
import r5_t1_last as T1  # noqa: E402

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = R.RES / "r11" / "b"
TOL = 1e-9
DET = "mahalanobis_classcond"


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")


def near(a, b, tol=TOL):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


# ---- redraws ----

def redraw_bootstrap_t1():
    """Same logic as r5_t1_last.report() boots block; write only BOOTSTRAP.json to OUT."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(T1.OUT / "summary.csv")
    data = R.load_data()
    id_g, _ = R.T.lesion_groups(data["test"])
    id_groups = R.idx_groups(id_g)
    og = {"pad_heldout": R.ood_groups("pad_heldout", data), "fitzpatrick17k": T1.fitz_groups()}
    boots = {}
    for lam, d in df[df.checkpoint == "last"].groupby("lambda"):
        seeds = sorted(d.seed.unique())
        for ck, root in (("best", T1.BEST_FEAT), ("last", T1.OUT / "features_last")):
            runs = {s: root / d[d.seed == s].run.iloc[0] for s in seeds}
            for o in T1.OODS:
                b = R.bootstrap_cell(runs, o, id_groups, og[o][0])
                boots["{:g}|{}|{}".format(lam, ck, o)] = {
                    "unit_ood": og[o][1], "unit_id": "lesion",
                    "mahalanobis_classcond": b["mahalanobis_classcond"],
                    "rule_A": b["rule_A"],
                    **{k: {x: b[k][x] for x in ("mean", "ci_lo", "ci_hi", "k_below", "n_seeds")}
                       for k in R.BOOT_DETS if k != DET and k in b},
                }
                print("BOOTSTRAP", f"{lam:g}|{ck}|{o}", boots[f"{lam:g}|{ck}|{o}"][DET]["mean"], flush=True)
    wjson(OUT / "BOOTSTRAP.json", boots)
    return OUT / "BOOTSTRAP.json"


def redraw_unseen_ece():
    """Call restored r5_t1last_unseen_report.py --report_only into a scratch OUT, copy bootstrap file."""
    # Patch: run module by importing main after redirecting its OUT/FEAT
    import r5_t1last_unseen_report as U
    scratch = OUT / "_unseen_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    # Point FEAT at real features; write JSON into scratch then copy bootstrap-bearing file
    real_out = U.OUT
    real_feat = U.FEAT
    U.OUT = scratch
    U.FEAT = real_feat  # features stay
    sys.argv = ["r5_t1last_unseen_report.py", "--report_only"]
    U.main()
    src = scratch / "UNSEEN_ECE.json"
    dst = OUT / "UNSEEN_ECE.json"
    shutil.copy2(src, dst)
    U.OUT = real_out
    U.FEAT = real_feat
    print("wrote", dst, flush=True)
    return dst


def redraw_b1():
    import r10b_b1_shift50_boots as B1
    real = B1.OUT
    B1.OUT = OUT
    B1.main()
    B1.OUT = real
    # rename to stored name
    src = OUT / "B1_shift50_boots.json"
    assert src.exists()
    return src


def redraw_b3():
    import r10b_b3_a4_ci as B3
    real = B3.OUT
    B3.OUT = OUT
    B3.main()
    B3.OUT = real
    return OUT / "B3_a4_ci.json"


def redraw_b4():
    import r10b_b4_splits_scin as B4
    real = B4.OUT
    B4.OUT = OUT
    B4.main()
    B4.OUT = real
    return OUT / "B4_splits_scin.json"


def redraw_scin_report():
    """Re-run boots_for via r6_r2_scin_report.build_src; write only to r11/b/."""
    import r6_r2_scin_report as S
    data = R.load_data()
    idg = R.idx_groups(R.T.lesion_groups(data["test"])[0])
    rt = "cur"
    root = R.RES / "r6" / "r2" / rt
    src = S.build_src(root, rt)
    boots = RR.boots_for(src, [S.OOD], data, idg, dets=R.BOOT_DETS)
    boot_out = {}
    for lab, lam, _pairs in src:
        key = "{}|{:g}".format(lab, lam)
        b = boots.get((lab, lam, S.OOD))
        if b is None:
            continue
        boot_out[key] = b
    payload = {
        "commit": R.C.git_head(),
        "design": "decisions/paperB_r6_r2_scin.md",
        "ood": S.OOD,
        "unit_ood": "P7a group column",
        "mapping": S._mapping_note(),
        "bootstrap": boot_out,
        "note": "R11 level b redraw",
    }
    wjson(OUT / "scin_report.json", payload)
    return OUT / "scin_report.json"


# ---- compare ----

def collect_diffs_boot_t1(stored, new):
    diffs = []
    n = 0
    for key in sorted(set(stored) | set(new)):
        if key not in stored or key not in new:
            diffs.append((key, "missing_key", stored.get(key), new.get(key)))
            n += 1
            continue
        for det in (DET,):
            if det not in stored[key] or det not in new[key]:
                continue
            s, t = stored[key][det], new[key][det]
            for f in ("mean", "sd", "ci_lo", "ci_hi"):
                n += 1
                if not near(s.get(f), t.get(f)):
                    diffs.append((f"{key}.{det}.{f}", f, s.get(f), t.get(f)))
            n += 1
            if int(s.get("k_below", -1)) != int(t.get("k_below", -2)):
                diffs.append((f"{key}.{det}.k_below", "k", s.get("k_below"), t.get("k_below")))
            n += 1
            ra_s = stored[key].get("rule_A")
            ra_t = new[key].get("rule_A")
            if bool(ra_s) != bool(ra_t):
                diffs.append((f"{key}.rule_A", "rule_A", ra_s, ra_t))
            # per-seed CI
            for sid, sp in (s.get("per_seed") or {}).items():
                tp = (t.get("per_seed") or {}).get(sid)
                if tp is None:
                    diffs.append((f"{key}.{det}.per_seed.{sid}", "missing", sp, None))
                    n += 1
                    continue
                for f in ("ci_lo", "ci_hi"):
                    n += 1
                    if not near(sp.get(f), tp.get(f)):
                        diffs.append((f"{key}.{det}.per_seed.{sid}.{f}", f, sp.get(f), tp.get(f)))
    return n, diffs


def collect_diffs_unseen(stored, new):
    diffs = []
    n = 0
    sb, nb = stored["bootstrap"], new["bootstrap"]
    for key in sorted(set(sb) | set(nb)):
        if key not in sb or key not in nb:
            diffs.append((key, "missing_key", sb.get(key), nb.get(key)))
            n += 1
            continue
        s, t = sb[key], nb[key]
        for f in ("mean", "ci_lo", "ci_hi"):
            n += 1
            if not near(s.get(f), t.get(f)):
                diffs.append((f"{key}.{f}", f, s.get(f), t.get(f)))
        # sd optional
        if "sd" in s or "sd" in t:
            n += 1
            if not near(s.get("sd"), t.get("sd")):
                diffs.append((f"{key}.sd", "sd", s.get("sd"), t.get("sd")))
        n += 1
        if int(s.get("k_below", -1)) != int(t.get("k_below", -2)):
            diffs.append((f"{key}.k_below", "k", s.get("k_below"), t.get("k_below")))
        n += 1
        if bool(s.get("rule_A")) != bool(t.get("rule_A")):
            diffs.append((f"{key}.rule_A", "rule_A", s.get("rule_A"), t.get("rule_A")))
        # no per-seed in stored UNSEEN bootstrap
    return n, diffs


def collect_diffs_r10b_rows(stored, new, name):
    diffs = []
    n = 0
    sr, nr = {json.dumps(r, sort_keys=True, default=str): r for r in stored["rows"]}, None
    # match by lambda/set/split
    def key(r):
        return (r.get("shift"), r.get("split"), r.get("lambda"), r.get("set"), r.get("backbone"))
    smap = {key(r): r for r in stored["rows"]}
    nmap = {key(r): r for r in new["rows"]}
    for k in sorted(set(smap) | set(nmap), key=str):
        if k not in smap or k not in nmap:
            diffs.append((f"{name}:{k}", "missing_row", smap.get(k), nmap.get(k)))
            n += 1
            continue
        s, t = smap[k], nmap[k]
        for f in ("mean", "sd"):
            n += 1
            if not near(s.get(f), t.get(f)):
                diffs.append((f"{name}:{k}.{f}", f, s.get(f), t.get(f)))
        # ci list
        for i, f in enumerate(("ci_lo", "ci_hi")):
            n += 1
            sv = (s.get("ci") or [None, None])[i]
            tv = (t.get("ci") or [None, None])[i]
            if not near(sv, tv):
                diffs.append((f"{name}:{k}.{f}", f, sv, tv))
        n += 1
        if s.get("k") != t.get("k"):
            diffs.append((f"{name}:{k}.k", "k", s.get("k"), t.get("k")))
        n += 1
        if bool(s.get("rule_A")) != bool(t.get("rule_A")):
            diffs.append((f"{name}:{k}.rule_A", "rule_A", s.get("rule_A"), t.get("rule_A")))
        # per_seed if present
        for sid, sp in (s.get("per_seed") or {}).items():
            tp = (t.get("per_seed") or {}).get(sid)
            n += 1
            if tp is None:
                diffs.append((f"{name}:{k}.per_seed.{sid}", "missing", sp, None))
                continue
            for f in ("ci_lo", "ci_hi"):
                n += 1
                if not near(sp.get(f), tp.get(f)):
                    diffs.append((f"{name}:{k}.per_seed.{sid}.{f}", f, sp.get(f), tp.get(f)))
    return n, diffs


def collect_diffs_scin(stored, new):
    diffs = []
    n = 0
    sb, nb = stored["bootstrap"], new["bootstrap"]
    for key in sorted(set(sb) | set(nb)):
        if key not in sb or key not in nb:
            diffs.append((key, "missing_key", None, None))
            n += 1
            continue
        s, t = sb[key], nb[key]
        for det in (DET,):
            if det not in s or det not in t:
                continue
            ss, tt = s[det], t[det]
            for f in ("mean", "sd", "ci_lo", "ci_hi"):
                n += 1
                if not near(ss.get(f), tt.get(f)):
                    diffs.append((f"{key}.{det}.{f}", f, ss.get(f), tt.get(f)))
            n += 1
            if int(ss.get("k_below", -1)) != int(tt.get("k_below", -2)):
                diffs.append((f"{key}.{det}.k_below", "k", ss.get("k_below"), tt.get("k_below")))
            n += 1
            if bool(s.get("rule_A")) != bool(t.get("rule_A")):
                diffs.append((f"{key}.rule_A", "rule_A", s.get("rule_A"), t.get("rule_A")))
            for sid, sp in (ss.get("per_seed") or {}).items():
                tp = (tt.get("per_seed") or {}).get(sid)
                for f in ("ci_lo", "ci_hi"):
                    n += 1
                    if tp is None or not near(sp.get(f), tp.get(f)):
                        diffs.append((f"{key}.{det}.per_seed.{sid}.{f}", f,
                                      sp.get(f), None if tp is None else tp.get(f)))
    return n, diffs


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    stored_root = R.RES
    pairs = []

    print("=== redraw BOOTSTRAP.json ===", flush=True)
    redraw_bootstrap_t1()
    pairs.append(("r5/t1last/BOOTSTRAP.json", OUT / "BOOTSTRAP.json", "boot_t1"))

    print("=== redraw UNSEEN_ECE.json ===", flush=True)
    redraw_unseen_ece()
    pairs.append(("r5/t1last/UNSEEN_ECE.json", OUT / "UNSEEN_ECE.json", "unseen"))

    print("=== redraw B1 ===", flush=True)
    redraw_b1()
    pairs.append(("r10b/B1_shift50_boots.json", OUT / "B1_shift50_boots.json", "b1"))

    print("=== redraw B3 ===", flush=True)
    redraw_b3()
    pairs.append(("r10b/B3_a4_ci.json", OUT / "B3_a4_ci.json", "b3"))

    print("=== redraw B4 ===", flush=True)
    redraw_b4()
    pairs.append(("r10b/B4_splits_scin.json", OUT / "B4_splits_scin.json", "b4"))

    print("=== redraw scin_report.json ===", flush=True)
    redraw_scin_report()
    pairs.append(("r6/r2/cur/scin_report.json", OUT / "scin_report.json", "scin"))

    table = []
    all_diffs = []
    for rel, newp, kind in pairs:
        stored = json.loads((stored_root / rel).read_text())
        new = json.loads(newp.read_text())
        if kind == "boot_t1":
            n, diffs = collect_diffs_boot_t1(stored, new)
        elif kind == "unseen":
            n, diffs = collect_diffs_unseen(stored, new)
        elif kind in ("b1", "b3", "b4"):
            n, diffs = collect_diffs_r10b_rows(stored, new, kind)
        else:
            n, diffs = collect_diffs_scin(stored, new)
        table.append({"file": rel, "n_compared": n, "n_differing": len(diffs)})
        for d in diffs:
            all_diffs.append({"file": rel, "field": d[0], "kind": d[1], "stored": d[2], "redrawn": d[3]})
        print(f"COMPARE {rel}: n={n} differing={len(diffs)}", flush=True)

    compare = {
        "tol": TOL,
        "cpu_hours": (time.time() - t0) / 3600.0,
        "table": table,
        "diffs": all_diffs,
        "ok": len(all_diffs) == 0,
    }
    wjson(OUT / "COMPARE.json", compare)
    md = ["# R11 level b — stored vs redrawn", "",
          f"Tolerance |diff| ≤ {TOL}. CPU hours {compare['cpu_hours']:.3f}.", "",
          "| file | n compared | n differing |", "|---|---:|---:|"]
    for r in table:
        md.append(f"| `{r['file']}` | {r['n_compared']} | {r['n_differing']} |")
    md += ["", "## Differing values (stored → redrawn)", ""]
    if not all_diffs:
        md.append("None.")
    else:
        for d in all_diffs:
            md.append(f"- `{d['file']}` `{d['field']}`: {d['stored']!r} → {d['redrawn']!r}")
    (OUT / "COMPARE.md").write_text("\n".join(md) + "\n")
    print("wrote", OUT / "COMPARE.md", "ok=", compare["ok"], flush=True)
    if not compare["ok"]:
        print("STOP: differences found; not fixing.", flush=True)
        sys.exit(2)
    # reproduce_rule_a against redrawn files in a scratch checkout copy of JSON paths
    scratch = OUT / "_rule_a_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    # Build a mini tree under scratch/results/paperB matching reproduce_rule_a paths
    for rel, newp, _ in pairs:
        dest = scratch / "results" / "paperB" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(newp, dest)
    # also need paper tex
    tex_src = ROOT / "paper" / "midl" / "paper_b.tex"
    tex_dst = scratch / "paper" / "midl" / "paper_b.tex"
    tex_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tex_src, tex_dst)
    script = scratch / "scripts" / "reproduce_rule_a.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "reproduce_rule_a.py", script)
    # patch ROOT inside by running from scratch — reproduce_rule_a uses Path(__file__).parents[1]
    r = subprocess.run([sys.executable, str(script)], cwd=str(scratch), capture_output=True, text=True)
    (OUT / "rule_a_on_redrawn.txt").write_text(r.stdout + r.stderr)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print("reproduce_rule_a on redrawn failed", r.returncode, flush=True)
        sys.exit(r.returncode)
    print("level b OK", flush=True)


if __name__ == "__main__":
    main()
