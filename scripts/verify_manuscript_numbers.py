#!/usr/bin/env python3
"""Cross-check every decimal in the manuscript against the result JSON files.

Each number in the .tex is matched at the precision it is written: 0.53 matches
any source value that rounds to 0.53, while 0.534 must round to 0.534. For every
number the report names the JSON file and key it came from, or says no source
exists. Numbers that are not results -- years, design constants, citation and
page numbers -- are allowlisted by value with a stated reason.

The check that matters most is the label check. A number can match a real source
and still be wrong because it is attached to the wrong dataset or the wrong
lambda in the prose; that is how 0.833 (pad_heldout) once appeared as a
Fitzpatrick value. For every number we read the dataset, lambda and
representation from the surrounding text and compare them against the tags in
the matching file path and key.

Usage:
    verify_manuscript_numbers.py --tex paper/midl/paper_b.tex --results results/paperB
    verify_manuscript_numbers.py ... --out /tmp/report.md --fail-on problem
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# Allowlist: value -> reason. These are not results and are never looked up.
# --------------------------------------------------------------------------
ALLOWLIST = {
    "0.5": "chance floor / balanced-accuracy floor for a two-domain problem",
    "0.05": "significance level",
    "0.9": "pre-registered monitor bar",
    "0.90": "pre-registered monitor bar",
    "0.95": "confidence level",
    "0.99": "confidence level",
    "0.45": "pre-registered detector-headroom bound",
    "0.55": "pre-registered detector-headroom bound",
    "0.60": "pre-registered adversary-CE gate",
    "0.10": "pre-registered leakage-movement gate",
    "0.02": "pre-registered ID-accuracy tolerance",
    "0.03": "pre-registered ID-ECE tolerance",
    "0.25": "swept lambda value (design constant)",
    "0.2": "classifier learning-rate multiplier",
    "1": "design constant",
    "2": "design constant / swept lambda",
    "3": "design constant",
    "4": "design constant / swept lambda",
    "5": "design constant",
    "6": "design constant",
    "8": "design constant / swept lambda",
    "10": "design constant",
    "15": "ECE bins",
    "16": "latent dimension",
    "27": "number of training runs",
    "30": "adversary learning-rate multiplier",
    "40": "training epochs",
    "50": "kNN k",
    "64": "context latent dimension",
    "100": "adversary LR multiplier in the sensitivity grid",
    "200": "kNN k",
    "1536": "pre-projection backbone dimension",
    "2000": "bootstrap resamples",
    "2048": "ResNet-50 feature dimension",
    "90": "percentile",
    "95": "percentile",
    "99": "percentile",
    # published dataset sizes, which are facts about the data rather than
    # results of this study; each is cited in the Data section
    "25331": "ISIC 2019 total images (cited)",
    "2298": "PAD-UFES-20 total images (cited)",
    "12413": "BCN20000 contribution to ISIC 2019 (cited)",
    "10015": "HAM10000 contribution to ISIC 2019 (cited)",
    "819": "MSK4 contribution to ISIC 2019 (cited)",
    "3887": "Fitzpatrick17k images obtained",
    "16577": "Fitzpatrick17k images listed (cited)",
}
ALLOWLIST_YEAR_RANGE = (1900, 2100)

# --------------------------------------------------------------------------
# Numbers whose provenance is a markdown report or a hand derivation, because
# the phase that produced them emitted no JSON. Each was checked by hand once
# and is reprinted on every run so it stays visible rather than silently
# passing. Fix the upstream phase to emit JSON and the entry can be deleted.
# --------------------------------------------------------------------------
DOC_SOURCED = {
    "4.879": ("results/paperB/phase13/PHASE13_REPORT.md",
              "PR(z_lesion_norm) table, lambda=0: 4.879 +/- 0.148; phase 13 "
              "wrote no JSON"),
    "84.5": ("results/paperB/PHASE1_6_REPORT.md",
             "k=1 variance share of z_context; phase 1.6 wrote no JSON"),
    "11.6": ("results/paperB/PHASE1_6_REPORT.md",
             "discordant-pair count in millions, same table"),
    "3041": ("results/paperB/r2/item4/REPORT.md",
             "image-level lesion overlap; recomputable from "
             "r2/item4/isic_image_split_with_lesion.csv"),
    "60.0": ("results/paperB/r2/item4/REPORT.md",
             "same overlap as a percentage (3041/5067)"),
    "50.8": ("results/paperB/r2/item4/split_record.json",
             "derived: test NV 2575 / 5067 = 0.5082"),
    "16211": ("results/paperB/r2/item4/split_record.json", "train.n"),
    "4053": ("results/paperB/r2/item4/split_record.json", "val.n"),
    "5067": ("results/paperB/r2/item4/split_record.json", "test.n"),
    "2084": ("results/paperB/r2/item4/split_record.json",
             "n_null_lesion_all_isic"),
    "2026": ("results/paperB/r2/item1/REPORT.md",
             "lesion-disjoint ISIC test subset size"),
}

# --------------------------------------------------------------------------
# Tag vocabularies. Context patterns -> canonical tag; source patterns likewise.
# --------------------------------------------------------------------------
DATASET_PATTERNS = {
    "pad_heldout": [r"pad\\?_heldout", r"pad-ufes", r"held-out pad"],
    "pad_adv": [r"pad\\?_adv"],
    "pad_full": [r"pad\\?_full"],
    "fitzpatrick": [r"fitzpatrick", r"fitz17k", r"\bfitz\b"],
    "isic": [r"\bisic\b", r"ham10000", r"\bbcn\b", r"\bham\b"],
    "camelyon": [r"camelyon", r"hospital"],
    "iwildcam": [r"iwildcam"],
}
SOURCE_DATASET_PATTERNS = {
    "pad_heldout": [r"pad_heldout"],
    "pad_adv": [r"pad_adv"],
    "pad_full": [r"pad_full"],
    "fitzpatrick": [r"fitzpatrick", r"fitz"],
    "isic": [r"isic", r"\bid\b", r"ham", r"bcn"],
    "camelyon": [r"camelyon", r"hospital", r"site_probe"],
    "iwildcam": [r"iwildcam"],
}
# pad_full/pad_adv must not silently satisfy a pad_heldout claim
DATASET_EXCLUSIVE = {"pad_heldout", "pad_adv", "pad_full", "fitzpatrick", "isic",
                     "camelyon", "iwildcam"}

REPR_PATTERNS = {
    "imagenet": [r"frozen imagenet", r"imagenet resnet"],
    "context": [r"context branch", r"z_\{?c\}?", r"\bz_c\b"],
    "backbone": [r"pre-projection", r"backbone"],
    "zlesion": [r"z_\{?\\ell\}?", r"lesion latent", r"16-dimensional"],
}
SOURCE_REPR_PATTERNS = {
    "imagenet": [r"imagenet"],
    "context": [r"z_context", r"z_c\b", r"context"],
    "backbone": [r"backbone"],
    "zlesion": [r"z_lesion"],
}

LAMBDA_CONTEXT = re.compile(
    r"(?:\\lam|\\lambda_\{?\\?mathrm\{adv\}\}?|\\lambda)\s*(?:\{?=\}?|=)\s*"
    r"(\d+(?:\.\d+)?)")
LAMBDA_SOURCE = re.compile(r"(?:ladv|lambda[_=]?)(\d+(?:p\d+)?)")

# --------------------------------------------------------------------------
# TeX handling
# --------------------------------------------------------------------------
BLANK_COMMANDS = [
    r"\\cite[tp]?\*?\{[^{}]*\}",
    r"\\label\{[^{}]*\}",
    r"\\(?:eq)?ref\{[^{}]*\}",
    r"\\url\{[^{}]*\}",
    r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\jmlr(?:volume|year|workshop)\{[^{}]*\}",
    r"\\bibliography\{[^{}]*\}",
    r"\\documentclass(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\usepackage(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\newcommand\{[^{}]*\}",
    r"10\^\{-?\d+\}",
    r"\\\\",
    r"p\{[\d.]+cm\}",
    r"\d+(?:\.\d+)?\s*(?:pt|cm|em|ex|in|dpi)\b",
]


def strip_tex(text: str) -> str:
    """Blank out non-content regions, preserving every byte offset."""
    out = list(text)

    def blank(a: int, b: int) -> None:
        for i in range(a, b):
            if out[i] != "\n":
                out[i] = " "

    # line comments
    pos = 0
    for line in text.splitlines(keepends=True):
        m = re.search(r"(?<!\\)%", line)
        if m:
            blank(pos + m.start(), pos + len(line.rstrip("\n")))
        pos += len(line)

    joined = "".join(out)
    for pat in BLANK_COMMANDS:
        for m in re.finditer(pat, joined):
            blank(m.start(), m.end())
        joined = "".join(out)
    return joined


NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+|\d+)(?:\.(\d+))?(?![\w])")


def extract_numbers(stripped: str):
    """Yield (raw, value, decimals, start, end)."""
    for m in NUM_RE.finditer(stripped):
        int_part = m.group(1).replace("{,}", "")
        dec_part = m.group(2)
        raw = int_part + ("." + dec_part if dec_part else "")
        try:
            value = float(raw)
        except ValueError:
            continue
        yield raw, value, len(dec_part) if dec_part else 0, m.start(), m.end()


def context_of(text: str, start: int, end: int, width: int = 260) -> str:
    return text[max(0, start - width):min(len(text), end + width)]


def label_positions(text: str):
    """Every dataset / representation mention with its offset.

    Attribution has to be by proximity, not by presence in a window. In
    "AUROC 0.988 on Fitzpatrick17k and 0.833 on pad_heldout" both labels sit
    inside any sensible window, so a set-based check calls either number
    consistent and the one failure mode this script exists to catch -- a value
    attached to the wrong dataset -- passes silently.
    """
    ds, rp = [], []
    low = text.lower()
    for name, pats in DATASET_PATTERNS.items():
        for pat in pats:
            for m in re.finditer(pat, low):
                ds.append((m.start(), m.end(), name))
    for name, pats in REPR_PATTERNS.items():
        for pat in pats:
            for m in re.finditer(pat, low):
                rp.append((m.start(), m.end(), name))
    lam = [(m.start(), m.end(), float(m.group(1)))
           for m in LAMBDA_CONTEXT.finditer(text)]
    return sorted(ds), sorted(rp), sorted(lam)


def nearest(mentions, start, end, limit):
    """The label this number is attached to, and the others within limit.

    A label that *follows* the number wins over a nearer one that precedes it.
    English puts the referent after the value -- "0.988 on Fitzpatrick17k and
    0.833 on pad_heldout" -- so plain proximity attributes 0.833 to
    Fitzpatrick, which is the error this check exists to detect.
    """
    after, before, others = (None, None), (None, None), set()
    for a, b, tag in mentions:
        if a <= start and end <= b:
            d, side = 0, "after"
        elif a >= end:
            d, side = a - end, "after"
        else:
            d, side = start - b, "before"
        if d > limit:
            continue
        others.add(tag)
        slot = after if side == "after" else before
        if slot[1] is None or d < slot[1]:
            if side == "after":
                after = (tag, d)
            else:
                before = (tag, d)
    best = after[0] if after[0] is not None else before[0]
    if best is not None:
        others.discard(best)
    return best, others


def tags_from_context(ctx: str):
    low = ctx.lower()
    datasets = {name for name, pats in DATASET_PATTERNS.items()
                if any(re.search(p, low) for p in pats)}
    # pad_heldout also matches a bare "pad"; keep the specific one if present
    if "pad_heldout" in datasets:
        datasets.discard("pad_full")
    reprs = {name for name, pats in REPR_PATTERNS.items()
             if any(re.search(p, low) for p in pats)}
    lambdas = {float(x) for x in LAMBDA_CONTEXT.findall(ctx)}
    return datasets, reprs, lambdas


def tags_from_source(path: str, keypath: str):
    low = (path + "|" + keypath).lower()
    datasets = {name for name, pats in SOURCE_DATASET_PATTERNS.items()
                if any(re.search(p, low) for p in pats)}
    reprs = {name for name, pats in SOURCE_REPR_PATTERNS.items()
             if any(re.search(p, low) for p in pats)}
    lambdas = set()
    for tok in LAMBDA_SOURCE.findall(low):
        lambdas.add(float(tok.replace("p", ".")))
    m = re.search(r'"?lambda"?[=:\s]+(\d+(?:\.\d+)?)', low)
    if m:
        lambdas.add(float(m.group(1)))
    return datasets, reprs, lambdas


# --------------------------------------------------------------------------
# Source index
# --------------------------------------------------------------------------
class Source:
    __slots__ = ("path", "keypath", "value", "datasets", "reprs", "lambdas")

    def __init__(self, path, keypath, value):
        self.path = path
        self.keypath = keypath
        self.value = value
        self.datasets, self.reprs, self.lambdas = tags_from_source(path, keypath)


def walk_json(node, path, keypath, out, lam_hint=None):
    if isinstance(node, dict):
        hint = lam_hint
        for k in ("lambda", "lambda_adv", "ladv"):
            if isinstance(node.get(k), (int, float)):
                hint = float(node[k])
        for k, v in node.items():
            walk_json(v, path, f"{keypath}|{k}" if keypath else str(k), out, hint)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk_json(v, path, f"{keypath}[{i}]", out, lam_hint)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        if isinstance(node, float) and (math.isnan(node) or math.isinf(node)):
            return
        kp = keypath
        if lam_hint is not None and "lambda" not in kp.lower():
            kp = f"{kp} (lambda={lam_hint:g})"
        out.append(Source(path, kp, float(node)))


def derive_aggregates(node, path, keypath, out):
    """Means and s.d. over lists of records, grouped by lambda.

    Most manuscript numbers are seed means, which appear in no file as a leaf.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            derive_aggregates(v, path, f"{keypath}|{k}" if keypath else str(k), out)
        return
    if not (isinstance(node, list) and node and
            all(isinstance(r, dict) for r in node)):
        return

    groups = defaultdict(list)
    for rec in node:
        lam = None
        for k in ("lambda", "lambda_adv", "ladv"):
            if isinstance(rec.get(k), (int, float)):
                lam = float(rec[k])
        groups[lam].append(rec)

    for lam, recs in groups.items():
        flat = defaultdict(list)
        for rec in recs:
            leaves = []
            walk_json(rec, path, "", leaves)
            for s in leaves:
                base = re.sub(r"\s*\(lambda=[^)]*\)$", "", s.keypath)
                if re.search(r"(^|\|)(seed|lambda|n|run)(\||$)", base):
                    continue
                flat[base].append(s.value)
        tag = f"lambda={lam:g}" if lam is not None else "all"
        for base, vals in flat.items():
            if len(vals) < 2:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            kp = f"{keypath}[{tag}]|{base}"
            out.append(Source(path, kp + "|mean", mean))
            out.append(Source(path, kp + "|sd", math.sqrt(var)))


def build_index(results_root: str):
    sources = []
    n_files = 0
    for dirpath, _dirs, files in os.walk(results_root):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, results_root)
            try:
                with open(full, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (ValueError, OSError):
                continue
            n_files += 1
            walk_json(data, rel, "", sources)
            derive_aggregates(data, rel, "", sources)
    return sources, n_files


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def allowlisted(raw: str, value: float):
    if raw in ALLOWLIST:
        return ALLOWLIST[raw]
    if "." not in raw and ALLOWLIST_YEAR_RANGE[0] <= value <= ALLOWLIST_YEAR_RANGE[1]:
        return "year or four-digit non-result integer"
    return None


def match(num_value, decimals, sources):
    """Sources whose value rounds to the written number at written precision."""
    hits = []
    for s in sources:
        if round(s.value, decimals) == round(num_value, decimals):
            hits.append(s)
    return hits


def classify(near_ds, near_rp, near_lam, hits):
    """Return (status, note, chosen_hits).

    The nearest dataset mention is binding: a value that exists only under a
    different dataset is a mismatch even if the other dataset is also named
    nearby. Lambda and representation are checked the same way.
    """
    if not hits:
        return "NO-SOURCE", "no file contains this value at this precision", []

    def ok(s):
        if near_ds and s.datasets and near_ds not in s.datasets:
            if (s.datasets | {near_ds}) & DATASET_EXCLUSIVE:
                return False
        if near_lam is not None and s.lambdas and near_lam not in s.lambdas:
            return False
        if near_rp and s.reprs and near_rp not in s.reprs:
            return False
        return True

    consistent = [s for s in hits if ok(s)]
    if consistent:
        return "OK", "", consistent

    wrong_ds = sorted({d for s in hits for d in s.datasets})
    wrong_lam = sorted({l for s in hits for l in s.lambdas})
    bits = []
    if near_ds:
        bits.append("nearest label is %s but the value exists only under %s"
                    % (near_ds, ", ".join(wrong_ds) or "another dataset"))
    if near_lam is not None and wrong_lam:
        bits.append("nearest lambda is %g but the value occurs at %s"
                    % (near_lam, ", ".join("%g" % x for x in wrong_lam)))
    if near_rp:
        bits.append("nearest representation is %s" % near_rp)
    return "LABEL-MISMATCH", "; ".join(bits) or "label inconsistent", hits



# --------------------------------------------------------------------------
# Manifest mode
# --------------------------------------------------------------------------
# The free scan above cannot catch a value attached to the wrong dataset. With
# ~140k indexed values a three-decimal number collides with roughly 140 of them,
# so a source carrying any given label almost always exists: the 0.833 /0.988
# swap passes a set-based *and* a nearest-label check for that reason.
#
# A manifest removes the ambiguity by naming the one source each number is
# supposed to come from. For every entry we check three things: the key still
# exists and still holds that value at the written precision; the number appears
# in the .tex; and the dataset mention nearest to it there is the declared one.
# The third check is what catches a swap.


def resolve_key(data, keypath):
    """keypath is a list of components.

    It cannot be a '|'-joined string: the result files use '|' inside single
    keys, e.g. "Mahalanobis|pad_heldout|full|pooled".
    """
    if isinstance(keypath, str):
        keypath = [keypath]
    cur = data
    for part in keypath:
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None, "key not found at %r" % part
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        return None, "key is not numeric (%r)" % type(cur).__name__
    return float(cur), None


def resolve_list(data, keypath):
    if isinstance(keypath, str):
        keypath = [keypath]
    cur = data
    for part in keypath:
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None, "list key not found at %r" % part
    return cur, None


def check_manifest(manifest_path, results_root, stripped, ds_pos, line_of):
    with open(manifest_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    results = []
    for e in entries:
        eid = e.get("id", "?")
        written = str(e["written"])
        dec = len(written.split(".")[1]) if "." in written else 0
        want = float(written)
        full = os.path.join(results_root, e["file"])
        problems, warnings = [], []

        if not os.path.exists(full):
            problems.append("missing file %s" % e["file"])
            got = None
        else:
            with open(full, encoding="utf-8") as fh:
                data = json.load(fh)
            if "list" in e:
                # most headline numbers are seed means and exist in no file as
                # a leaf, so the manifest may name a list plus a leaf and a stat
                cur, err = resolve_list(data, e["list"])
                if err:
                    problems.append(err); got = None
                elif not isinstance(cur, list):
                    problems.append("%r is not a list" % e["list"]); got = None
                else:
                    recs = cur
                    for wk, wv in (e.get("where") or {}).items():
                        recs = [r for r in recs if r.get(wk) == wv]
                    leaves = e.get("leaves") or [e["leaf"]]
                    per_leaf = []
                    for leaf in leaves:
                        lv = [resolve_key(r, leaf)[0] for r in recs]
                        lv = [x for x in lv if x is not None]
                        if lv:
                            per_leaf.append(sum(lv) / len(lv))
                    vals = [resolve_key(r, e["leaf"])[0] for r in recs] \
                        if "leaves" not in e else per_leaf
                    vals = [x for x in vals if x is not None]
                    if not vals:
                        problems.append("no values at leaf %r" % e["leaf"])
                        got = None
                    else:
                        n = e.get("n")
                        if n is not None and len(vals) != n:
                            problems.append("expected n=%d seeds, found %d"
                                            % (n, len(vals)))
                        mean = sum(vals) / len(vals)
                        stat = e.get("stat", "mean")
                        if stat == "min":
                            got = min(vals)
                        elif stat == "max":
                            got = max(vals)
                        elif stat == "sd":
                            if len(vals) < 2:
                                problems.append("sd needs >= 2 values")
                                got = None
                            else:
                                got = math.sqrt(sum((v - mean) ** 2 for v in vals)
                                                / (len(vals) - 1))
                        else:
                            got = mean
                if got is not None and round(got, dec) != round(want, dec):
                    problems.append("source gives %.6g, which is %.*f at the "
                                    "written precision, not %s"
                                    % (got, dec, round(got, dec), written))
            else:
                got, err = resolve_key(data, e["key"])
                if err:
                    problems.append(err)
                elif round(got, dec) != round(want, dec):
                    problems.append("source holds %.6g, which is %.*f at the "
                                    "written precision, not %s"
                                    % (got, dec, round(got, dec), written))

        # the number must appear in the tex, with the declared label nearest it
        occurrences = []
        for raw, value, d, a, b in extract_numbers(stripped):
            if raw == written:
                occurrences.append((a, b))
        if not occurrences:
            problems.append("not present in the manuscript")
        elif e.get("label"):
            oks = []
            for a, b in occurrences:
                near, _ = nearest(ds_pos, a, b, 200)
                oks.append((near, line_of[a]))
            good = [x for x in oks if x[0] == e["label"]]
            labelled = [x for x in oks if x[0] is not None]
            if good:
                pass
            elif not labelled:
                warnings.append("appears only in contexts with no nearby "
                                "dataset label (line(s) %s); a table cell takes "
                                "its label from the column header, which this "
                                "check cannot read"
                                % ", ".join(str(l) for _, l in oks))
            else:
                problems.append("appears at line(s) %s but the nearest label is "
                                "%s, not %s"
                                % (", ".join(str(l) for _, l in labelled),
                                   ", ".join(sorted({str(n) for n, _ in labelled})),
                                   e["label"]))
        results.append((eid, written, e.get("label", ""), problems, got,
                        warnings))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/midl/paper_b.tex")
    ap.add_argument("--results", default="results/paperB")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-decimals", type=int, default=2,
                    help="numbers with fewer decimals are reported as weak")
    ap.add_argument("--manifest", default=None,
                    help="JSON list of {id, written, label, file, key} entries; "
                         "the only mode that reliably catches a value attached "
                         "to the wrong dataset")
    ap.add_argument("--fail-on", choices=["never", "problem"], default="never")
    args = ap.parse_args()

    with open(args.tex, encoding="utf-8") as fh:
        raw_tex = fh.read()
    stripped = strip_tex(raw_tex)
    line_of = [0] * (len(raw_tex) + 1)
    ln = 1
    for i, ch in enumerate(raw_tex):
        line_of[i] = ln
        if ch == "\n":
            ln += 1
    line_of[len(raw_tex)] = ln

    sources, n_files = build_index(args.results)
    ds_pos, rp_pos, lam_pos = label_positions(stripped)

    rows = []
    for raw, value, dec, start, end in extract_numbers(stripped):
        reason = allowlisted(raw, value)
        if reason:
            rows.append(dict(line=line_of[start], raw=raw, status="ALLOWLISTED",
                             note=reason, hits=[], weak=False))
            continue
        hits = match(value, dec, sources)
        near_ds, other_ds = nearest(ds_pos, start, end, 200)
        near_rp, _ = nearest(rp_pos, start, end, 200)
        near_lam, _ = nearest(lam_pos, start, end, 120)
        status, note, chosen = classify(near_ds, near_rp, near_lam, hits)
        if status != "OK" and raw in DOC_SOURCED:
            src, why = DOC_SOURCED[raw]
            status, note, chosen = "OK-DOC", "%s -- %s" % (src, why), []
        rows.append(dict(line=line_of[start], raw=raw, status=status, note=note,
                         hits=chosen, weak=dec < args.min_decimals,
                         ctx_ds=([near_ds] if near_ds else []) + sorted(other_ds),
                         ctx_lam=([near_lam] if near_lam is not None else [])))

    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1

    L = []
    L.append("# Manuscript number verification\n")
    L.append("tex: `%s`  \nresults: `%s` (%d JSON files, %d indexed values "
             "including derived seed means)\n" % (args.tex, args.results, n_files,
                                                  len(sources)))
    L.append("| status | count |")
    L.append("|---|---:|")
    for k in ("OK", "OK-DOC", "ALLOWLISTED", "LABEL-MISMATCH", "NO-SOURCE"):
        L.append("| %s | %d |" % (k, counts[k]))
    L.append("")

    doc = [r for r in rows if r["status"] == "OK-DOC"]
    L.append("## Verified against a markdown report, not against JSON (%d)\n"
             % len(doc))
    L.append("These pass, but by a hand check recorded in the script rather than "
             "by matching a result file. Making the upstream phase emit JSON "
             "would retire each entry.\n")
    if doc:
        L.append("| line | number | provenance |")
        L.append("|---:|---|---|")
        for r in doc:
            L.append("| %d | `%s` | %s |" % (r["line"], r["raw"], r["note"]))
    else:
        L.append("None.")
    L.append("")

    for status, title in (("LABEL-MISMATCH", "Label mismatches"),
                          ("NO-SOURCE", "No source found")):
        bad = [r for r in rows if r["status"] == status]
        L.append("## %s (%d)\n" % (title, len(bad)))
        if not bad:
            L.append("None.\n")
            continue
        L.append("| line | number | context labels | note |")
        L.append("|---:|---|---|---|")
        for r in bad:
            L.append("| %d | `%s`%s | %s | %s |" % (
                r["line"], r["raw"], " (weak)" if r.get("weak") else "",
                ", ".join(r.get("ctx_ds") or []) +
                (" lam=" + ",".join("%g" % x for x in r.get("ctx_lam") or [])
                 if r.get("ctx_lam") else ""),
                r["note"]))
        L.append("")

    L.append("## Matched numbers and their sources\n")
    L.append("| line | number | source file | key |")
    L.append("|---:|---|---|---|")
    for r in rows:
        if r["status"] != "OK":
            continue
        h = r["hits"][0]
        extra = "" if len(r["hits"]) == 1 else " (+%d more)" % (len(r["hits"]) - 1)
        L.append("| %d | `%s`%s | `%s` | `%s`%s |" % (
            r["line"], r["raw"], " (weak)" if r["weak"] else "",
            h.path, h.keypath, extra))
    man_bad = 0
    if args.manifest:
        res = check_manifest(args.manifest, args.results, stripped, ds_pos, line_of)
        man_bad = sum(1 for r in res if r[3])
        L.append("## Manifest check (%d entries, %d failing)\n"
                 % (len(res), man_bad))
        L.append("| id | written | label | verdict |")
        L.append("|---|---|---|---|")
        for eid, written, label, probs, got, warns in res:
            verdict = ("**" + "; ".join(probs) + "**") if probs else (
                "ok (" + "; ".join(warns) + ")" if warns else "ok")
            L.append("| `%s` | `%s` | %s | %s |" % (
                eid, written, label or "--", verdict))
        L.append("")

    report = "\n".join(L) + "\n"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print("wrote %s" % args.out)
    print("OK %d | allowlisted %d | LABEL-MISMATCH %d | NO-SOURCE %d"
          % (counts["OK"], counts["ALLOWLISTED"], counts["LABEL-MISMATCH"],
             counts["NO-SOURCE"]))
    if counts["OK-DOC"]:
        print("verified against markdown only: %d" % counts["OK-DOC"])
    if args.manifest:
        print("manifest: %d failing" % man_bad)
    if args.fail_on == "problem" and (counts["LABEL-MISMATCH"] or
                                      counts["NO-SOURCE"] or man_bad):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
