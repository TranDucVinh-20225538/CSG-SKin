#!/usr/bin/env python3
"""Harvest cite-able numbers from R5/R6 reports into a JSON store + staged manifest keys.

Sources: R1, D1, S1/S2 (s_flip), clinical, P6, P8.
Writes `results/paperB/r5/staged_cite_numbers.json` (verifier-readable) and appends
`staged_*` entries (flag `"staged": true`) to `paper/number_manifest.json`.
Does not edit .tex. Idempotent for staged_* / staged:true entries.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
MAN = ROOT / "paper" / "number_manifest.json"
RES = ROOT / "results" / "paperB"
STORE_REL = "r5/staged_cite_numbers.json"
STORE = RES / STORE_REL


def rnd3(x):
    return f"{float(x):.3f}"


def load_json(p: Path):
    return json.loads(p.read_text())


def put(store, path, value, meta=None):
    """path: list of str keys into nested dict store['values']."""
    cur = store.setdefault("values", {})
    for k in path[:-1]:
        cur = cur.setdefault(str(k), {})
    leaf = str(path[-1])
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    cur[leaf] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    if meta:
        store.setdefault("meta", {})["/".join(str(x) for x in path)] = meta


def entry(eid, written, key, label=None, note=""):
    return {
        "id": eid,
        "written": written if isinstance(written, str) else rnd3(written),
        "label": label,
        "file": STORE_REL,
        "key": [str(k) for k in key],
        "staged": True,
        "optional": True,
        "note": note or "staged from R5/R6 reports for verifier prep",
    }


def num_ok(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x) and abs(x) <= 1e6


def walk_json(obj, path, out_leaves, max_list=64):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(v, path + [k], out_leaves, max_list)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:max_list]):
            walk_json(v, path + [i], out_leaves, max_list)
    elif isinstance(obj, (int, float, np.floating, np.integer)) and not isinstance(obj, bool):
        if num_ok(obj):
            out_leaves.append((path, float(obj)))


def interesting(path, patterns):
    joined = "/".join(str(x) for x in path)
    return bool(re.search(patterns, joined, re.I))


def harvest_csv(store, entries, rel, family, col_pat, max_rows=500):
    p = RES / rel
    if not p.is_file():
        return
    rows = list(csv.DictReader(p.open()))
    for i, r in enumerate(rows[:max_rows]):
        for col, val in r.items():
            if not re.search(col_pat, col, re.I):
                continue
            if not num_ok(val):
                continue
            key = [family, "csv", str(i), col]
            put(store, key, float(val), {"source": rel})
            entries.append(entry(f"staged_{family}_csv_{i}_{col}", float(val), ["values"] + key,
                                 note=f"{family} {rel}"))


def harvest_r1(store, entries):
    cells = RES / "r6/r1/cur/cells.json"
    if cells.is_file():
        data = load_json(cells)
        items = data if isinstance(data, list) else data.get("cells", data.get("rows", []))
        for i, c in enumerate(items):
            if not isinstance(c, dict):
                continue
            leaves = []
            walk_json(c, ["r1", "cells", i], leaves)
            for path, v in leaves:
                if interesting(path, r"(mean|sd|auroc|leak|bal|acc|maha|ece|ci_)"):
                    put(store, path, v, {"source": "r6/r1/cur/cells.json"})
                    entries.append(entry("staged_" + "_".join(str(x) for x in path)[:140], v,
                                         ["values"] + [str(x) for x in path], note="R1 cells.json"))
    harvest_csv(store, entries, "r6/r1/cur/summary.csv", "r1",
                r"(mean|auroc|leak|bal|acc|maha|ece|sd)")
    lm = RES / "r6/r1/cur/leakage_matched/summary.csv"
    if lm.is_file():
        harvest_csv(store, entries, "r6/r1/cur/leakage_matched/summary.csv", "r1_lm",
                    r"(mean|auroc|leak|bal|acc|maha|sd|ci)")


def harvest_d1(store, entries):
    p = RES / "r5/d1_domain/D1_RESULTS.json"
    if not p.is_file():
        return
    leaves = []
    walk_json(load_json(p), ["d1"], leaves)
    for path, v in leaves:
        if interesting(path, r"(rho|spearman|auroc|mean|sd|acc|leak|cosine|proj|corr|ci_)"):
            put(store, path, v, {"source": "r5/d1_domain/D1_RESULTS.json"})
            entries.append(entry("staged_" + "_".join(str(x) for x in path)[:140], v,
                                 ["values"] + [str(x) for x in path], note="D1"))


def harvest_sflip(store, entries):
    p = RES / "r5/s_flip/results.json"
    if p.is_file():
        leaves = []
        walk_json(load_json(p), ["s_flip"], leaves)
        for path, v in leaves:
            if interesting(path, r"(auroc|mean|sd|leak|bal|acc|rho|frac|rate)"):
                put(store, path, v, {"source": "r5/s_flip/results.json"})
                entries.append(entry("staged_" + "_".join(str(x) for x in path)[:140], v,
                                     ["values"] + [str(x) for x in path], note="S1/S2 s_flip"))
    harvest_csv(store, entries, "r5/s_flip/summary.csv", "s_flip",
                r"(mean|auroc|leak|bal|acc|sd)")


def harvest_clinical(store, entries):
    p = RES / "r5/posthoc/clinical/results.json"
    if p.is_file():
        leaves = []
        walk_json(load_json(p), ["clinical"], leaves, max_list=128)
        for path, v in leaves:
            if interesting(path, r"(accept|nv|rate|mean|sd|ci_|frac|share|n_)"):
                joined = "/".join(str(x) for x in path).lower()
                lab = None
                if "fitz" in joined:
                    lab = "fitzpatrick"
                elif "sd198" in joined:
                    lab = "sd198"
                elif "pad" in joined:
                    lab = "pad_heldout"
                put(store, path, v, {"source": "r5/posthoc/clinical/results.json"})
                entries.append(entry("staged_" + "_".join(str(x) for x in path)[:140], v,
                                     ["values"] + [str(x) for x in path], label=lab, note="clinical"))
    harvest_csv(store, entries, "r5/posthoc/clinical/summary.csv", "clinical",
                r".*", max_rows=400)


def harvest_p6(store, entries):
    harvest_csv(store, entries, "r5/p6/summary.csv", "p6",
                r"(bal|auroc|mean|sd|V[0-3])")
    # aggregate a few per_run headline means into store for λ=0 and λ=2 if present
    pr = RES / "r5/p6/per_run"
    if pr.is_dir():
        for p in sorted(pr.glob("*.json")):
            rec = load_json(p)
            lam = rec.get("lambda")
            seed = rec.get("seed")
            z = rec.get("z_lesion_norm") or {}
            for var in ("V0", "V1", "V2", "V3"):
                block = z.get(var) or {}
                if "bal_acc_mean" in block:
                    path = ["p6", "per_run", f"l{lam}_s{seed}", "z_lesion_norm", var, "bal_acc_mean"]
                    put(store, path, block["bal_acc_mean"], {"source": str(p.relative_to(RES))})
                    entries.append(entry(
                        f"staged_p6_l{lam}_s{seed}_{var}_bal",
                        block["bal_acc_mean"], ["values"] + path, note="P6 per_run"))


def harvest_p8(store, entries):
    for name in ("SELECTED_RECIPE.json", "RECIPE_DECISION.json", "MAIN_RECIPE_RULE.json",
                 "selection_stageA.json", "selection_stageB.json"):
        p = RES / "r5/p8" / name
        if not p.is_file():
            continue
        leaves = []
        walk_json(load_json(p), ["p8", name.replace(".json", "")], leaves)
        for path, v in leaves:
            if interesting(path, r"(score|mean|sd|auroc|leak|bal|acc|lambda|weight)"):
                put(store, path, v, {"source": f"r5/p8/{name}"})
                entries.append(entry("staged_" + "_".join(str(x) for x in path)[:140], v,
                                     ["values"] + [str(x) for x in path], note="P8"))
    harvest_csv(store, entries, "r5/p8/summary.csv", "p8",
                r"(mean|auroc|leak|bal|acc|sd|ece)")


def main():
    store = {"commit_note": "staged cite numbers for R1/D1/S/clinical/P6/P8", "values": {}, "meta": {}}
    entries = []
    for fn in (harvest_r1, harvest_d1, harvest_sflip, harvest_clinical, harvest_p6, harvest_p8):
        before = len(entries)
        fn(store, entries)
        print(f"{fn.__name__}: +{len(entries) - before}", flush=True)

    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(store, indent=2) + "\n")

    # dedupe by id
    by_id = {e["id"]: e for e in entries}
    staged = list(by_id.values())

    man = load_json(MAN)
    kept = [e for e in man if not (isinstance(e, dict) and (
        e.get("staged") is True or str(e.get("id", "")).startswith("staged_")))]
    new_man = kept + staged
    MAN.write_text(json.dumps(new_man, indent=2) + "\n")
    print(f"wrote {STORE} ({len(store['meta'])} leaves)", flush=True)
    print(f"manifest: kept {len(kept)} + staged {len(staged)} = {len(new_man)}", flush=True)


if __name__ == "__main__":
    main()
