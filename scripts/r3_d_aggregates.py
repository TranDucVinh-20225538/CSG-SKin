#!/usr/bin/env python3
"""R3 D: seed aggregates (mean, s.d., n) for results whose means were printed only in markdown.

Reads the per-run values already stored in item2_results.json and item4_results.json; no recomputation.
Writes r2/item2/item2_aggregate.json and r2/item4/item4_aggregate.json."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402


def ms(xs):
    m, s, n = C.mean_sd(xs)
    return {"mean": m, "sd": s, "n": n}


def item2():
    d = json.loads((C.R2 / "item2" / "item2_results.json").read_text())
    vals = defaultdict(list)
    for r in d["per_run"]:
        for block, rec in r.items():
            if not isinstance(rec, dict):
                continue
            for f, v in rec.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals[(block, "{:g}".format(r["lambda"]), f)].append(v)
    out = defaultdict(lambda: defaultdict(dict))
    for (block, lam, f), xs in vals.items():
        out[block][lam][f] = ms(xs)
    (C.R2 / "item2" / "item2_aggregate.json").write_text(json.dumps(
        {"source": "r2/item2/item2_results.json per_run (commit {})".format(d["commit"]), "by_block": out}, indent=1) + "\n")


def item4():
    d = json.loads((C.R2 / "item4" / "item4_results.json").read_text())
    out = {}
    for split, by in d["tables"].items():
        out[split] = {}
        for lam, rows in by.items():
            out[split]["{:g}".format(float(lam))] = {k: ms([r[k] for r in rows]) for k in rows[0] if k != "seed"}
    (C.R2 / "item4" / "item4_aggregate.json").write_text(json.dumps(
        {"source": "r2/item4/item4_results.json tables (commit {}); seeds 42, 52, 62".format(d["commit"]), "by_split": out},
        indent=1) + "\n")


def flatten(obj, pre=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, "{}.{}".format(pre, k) if pre else str(k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield pre, float(obj)


def run_groups(rel_dir, out_name):
    """Seed aggregates of every numeric summary.json leaf, grouped by run name without its _s<seed> suffix."""
    base = C.PAPERB / "results" / "paperB" / rel_dir
    vals = defaultdict(lambda: defaultdict(list))
    for p in sorted(base.glob("*_s*/summary.json")):
        m = re.fullmatch(r"(.+)_s(\d+)", p.parent.name)
        if not m:
            continue
        for k, v in flatten(json.loads(p.read_text())):
            vals[m.group(1)][k].append(v)
    out = {g: {k: ms(xs) for k, xs in d.items()} for g, d in vals.items()}
    (base / out_name).write_text(json.dumps({"source": "{}/*_s*/summary.json".format(rel_dir), "by_group": out}, indent=1) + "\n")


if __name__ == "__main__":
    item2()
    item4()
    run_groups("phase15/single_dann", "single_dann_aggregate.json")
    run_groups("phase15b/mmd", "mmd_aggregate.json")
    run_groups("phase3_sweep", "sweep_aggregate.json")
