#!/usr/bin/env python3
"""Cross-check every number in the manuscript against results/paperB/**/*.json (+ CSV and a few feature-derived counts).

Two modes, and the second is the one that catches a value attached to the wrong dataset.

The free scan below indexes every value it can find and asks whether any source matches
the printed number. That is the right tool for a wrong number or a missing source, but it
cannot establish that a number belongs to the dataset the sentence names: with ~140k
indexed values a three-decimal number collides with about 140 of them, so a source
carrying any given label almost always exists. 0.833 matches 43 values, one of them a
Fitzpatrick AUROC, which is why an earlier check "verified" 84.5% against a cosine AUROC
of 84.4486 and why swapping 0.833 and 0.988 between pad_heldout and Fitzpatrick passes
every heuristic.

Manifest mode removes the ambiguity. paper/number_manifest.json names, for each
load-bearing number, the one file and key it comes from and the dataset it belongs to;
each entry is checked against that key alone, and against the dataset mention nearest to
the number in the .tex. Attribution prefers a label that FOLLOWS the number, because
"0.988 on Fitzpatrick17k and 0.833 on pad_heldout" puts the referent after the value and
plain proximity assigns 0.833 to Fitzpatrick -- the exact error being guarded against.

Report only; never edits the manuscript. Categories per number:
  match      some source rounds (nearest, half up) to the printed value at the printed precision
  last-digit nearest source differs by at most one unit in the last printed digit (rounding / truncation)
  mismatch   no source within one last-digit unit (wrong number, wrong source, or source not in results/)
Flags on matches: lambda-source (every matching source carries a λ not mentioned in the sentence),
pad_full-only (every matching source is pad_full), pair (mean ± s.d. both found in one source group).

Usage: python scripts/verify_manuscript_numbers.py [--paper-root DIR] [--out results/paperB/manuscript_number_check.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

# Resolved from this file's location so the script runs on the cluster and locally
# alike; override with --root / --results.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "paperB"
TEX = ("paper/paper_b.tex", "paper/midl/paper_b.tex")
MAX_LIST = 64
SEED_RE = re.compile(r"_s\d{2,3}(?=\b|_|\.)")
LAM_SRC = re.compile(r"ladv(\d+(?:p\d+)?)")
LAM_TEX = re.compile(r"(?:\\lam|λ|\\lambda(?:_\{\\mathrm\{adv\}\})?)\s*\$?\s*(?:=|\\to|\\geq|>|<)\s*\$?\s*(\d+(?:\.\d+)?)")
CONCEPTS = {
    "mahalanobis": ("maha",), "knn": ("knn",), "cosine": ("cos",), "msp": ("msp",), "energy": ("energy",),
    "ece": ("ece",), "leakage": ("leak", "probe"), "fitzpatrick": ("fitz",), "pad\\_heldout": ("pad_heldout", "heldout"),
    "pad\\_full": ("pad_full",), "balanced": ("bal",), "confidence": ("conf", "msp"), "camelyon": ("camelyon",),
    "iwildcam": ("iwild",), "context": ("context", "z_c"), "imagenet": ("imagenet", "r50", "resnet"),
    "effnet": ("effb", "efficientnet"), "erm": ("erm",), "recall": ("recall",), "accuracy": ("acc",),
    "nevus": ("nv",), "actinic": ("ak",), "squamous": ("scc",), "melanoma": ("mel",), "auroc": ("auroc", "auc"),
    "isic": ("isic",), "ham": ("ham",), "bcn": ("bcn",),
}


def rnd(v, d):
    return float(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rnd_vec(v, d):
    f = 10.0 ** d
    return np.round(np.floor(np.asarray(v) * f + 0.5 + 1e-9) / f, d)


# ---------------------------------------------------------------- index
class Index:
    def __init__(self):
        self.vals, self.src = [], []

    def add(self, v, s):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if np.isfinite(v) and not isinstance(v, bool):
            self.vals.append(v)
            self.src.append(s)

    def walk(self, obj, s, leaves):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.walk(v, "{}.{}".format(s, k), leaves)
        elif isinstance(obj, list):
            if len(obj) > MAX_LIST and all(isinstance(x, (int, float)) for x in obj[:5]):
                return
            for i, v in enumerate(obj):
                self.walk(v, "{}[{}]".format(s, i), leaves)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            leaves[s] = float(obj)

    def finish(self):
        o = np.argsort(self.vals)
        self.v = np.asarray(self.vals)[o]
        self.s = [self.src[i] for i in o]
        keys = list(CONCEPTS)
        cache = {}
        mask = np.zeros(len(self.s), np.int64)
        lam = np.full(len(self.s), np.nan)
        for i, src in enumerate(self.s):
            head = src.rsplit("::", 1)
            k = src
            if k not in cache:
                toks = set(re.split(r"[^a-z0-9]+", src.lower()))
                mm = 0
                for j, c in enumerate(keys):
                    if any(_hit(pt, toks) for pt in CONCEPTS[c]):
                        mm |= 1 << j
                lt = lam_of_src(src)
                cache[k] = (mm, np.nan if lt is None else lt)
            mask[i], lam[i] = cache[k]
        self.mask, self.lam, self.keys = mask, lam, keys
        self.mean = np.array(["[mean" in x for x in self.s])

    def rng(self, lo, hi):
        return np.searchsorted(self.v, lo), np.searchsorted(self.v, hi, side="right")

    def near(self, x, tol):
        a, b = np.searchsorted(self.v, x - tol), np.searchsorted(self.v, x + tol, side="right")
        return [(self.v[i], self.s[i]) for i in range(a, b)]


def build_index():
    idx = Index()
    groups = defaultdict(dict)  # seed-stripped file -> seed -> leaves
    for p in sorted(RESULTS.rglob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        rel = str(p.relative_to(RESULTS))
        leaves = {}
        idx.walk(obj, "", leaves)
        for k, v in leaves.items():
            idx.add(v, rel + "::" + k)
        m = SEED_RE.search(rel)
        if m:
            groups[SEED_RE.sub("_s*", rel)][m.group(0)] = leaves
        # per-run lists inside one file (e.g. per_run arrays) are covered by their own leaves
    for p in sorted(RESULTS.rglob("*.csv")):
        if p.stat().st_size > 5e6:
            continue
        rel = str(p.relative_to(RESULTS))
        with open(p, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                key = "|".join("{}={}".format(k, v) for k, v in row.items() if v and not _isnum(v))[:160]
                for k, v in row.items():
                    if v and _isnum(v):
                        idx.add(v, "{}::row{}[{}]::{}".format(rel, i, k, key))
    for g, seeds in derived_groups().items():
        groups[g].update(seeds)
        for sd, leaves in seeds.items():
            for k, v in leaves.items():
                idx.add(v, g.replace("_s*", sd) + "::" + k)
    for g, seeds in groups.items():
        if len(seeds) < 2:
            continue
        keys = set.intersection(*[set(l) for l in seeds.values()])
        for k in keys:
            a = np.array([seeds[s][k] for s in seeds])
            idx.add(a.mean(), "{} [mean n={}]::{}".format(g, len(a), k))
            idx.add(a.std(ddof=1), "{} [sd n={}]::{}".format(g, len(a), k))
    idx.finish()
    return idx


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def derived_groups():
    """Counts not stored in any JSON: 6-class (PAD-restricted) predictions on pad_heldout from Phase 13 features."""
    feat = RESULTS / "phase13" / "features"
    pad = np.array([0, 1, 2, 3, 4, 7])
    names = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
    out = defaultdict(dict)
    for d in sorted(feat.glob("runB_orth1_ladv*_s*")):
        f = d / "pad_heldout.npz"
        if not f.exists():
            continue
        z = np.load(f)
        lg, y = z["logits"], z["labels"].astype(int)
        pr = pad[lg[:, pad].argmax(1)]
        lv = {"accuracy": float((pr == y).mean())}
        for c in pad:
            n = names[c]
            lv["pred_count_" + n] = float((pr == c).sum())
            lv["correct_" + n] = float(((pr == c) & (y == c)).sum())
            lv["recall_" + n] = float(((pr == c) & (y == c)).sum() / max((y == c).sum(), 1))
        m = SEED_RE.search(d.name)
        out["derived/phase13_pad_heldout_6class/" + SEED_RE.sub("_s*", d.name)][m.group(0)] = lv
    return out


# ---------------------------------------------------------------- tex
NUM = re.compile(r"(?<![\w.\\{])(\d{1,3}(?:\{,\}\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)(?![\w.]*\d)")


def strip_tex(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    end = re.search(r"\\begin\{thebibliography\}|\\bibliography\{", s)
    return s[: end.start()] if end else s


def macros(s):
    out = {}
    for m in re.finditer(r"\\newcommand\{(\\[A-Za-z]+)\}\{([^{}]*)\}", s):
        if re.search(r"\d", m.group(2)):
            out[m.group(1)] = m.group(2)
    return out


def body(s):
    b = s.find("\\begin{document}")
    return b if b >= 0 else 0


SKIP_CMD = re.compile(r"\\(?:cite|ref|label|url|includegraphics|eqref|href|bibitem|input|usepackage|documentclass|"
                      r"setlength|vspace|hspace|resizebox|scalebox|textwidth|linewidth|columnwidth)\*?(?:\[[^\]]*\])?\{[^{}]*\}")


_PROC = {}  # path -> macro-expanded, comment-stripped body, for the manifest label check


def tex_numbers(path):
    raw = strip_tex(Path(path).read_text())
    mac = macros(raw)
    s = raw[body(raw):]
    for k in sorted(mac, key=len, reverse=True):
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda _m, v=mac[k]: v, s)
    s = SKIP_CMD.sub(lambda m: " " * len(m.group(0)), s)
    s = re.sub(r"\\begin\{(?:tabular|figure|table)\}\{?[^}\n]*\}?", lambda m: " " * len(m.group(0)), s)
    in_math = _math_mask(s)
    _PROC[str(path)] = s
    for m in NUM.finditer(s):
        t = m.group(1)
        is_dec = "." in t
        is_grp = "{,}" in t
        if not (is_dec or is_grp or in_math[m.start()]):
            continue
        if not is_dec and re.fullmatch(r"(19|20)\d\d", t):
            continue
        val = float(t.replace("{,}", ""))
        d = len(t.split(".")[1]) if is_dec else 0
        a, b = max(0, m.start() - 80), m.end() + 80
        ctx = re.sub(r"\s+", " ", s[a:b])
        pa = s.rfind("\n\n", 0, m.start())
        pb = s.find("\n\n", m.end())
        pa, pb = max(pa, m.start() - 700), (pb if pb >= 0 else len(s))
        sent = s[max(pa, 0): min(pb, m.end() + 700)]
        para = s.count("\n\n", 0, m.start())
        pct = s[m.end(): m.end() + 3].startswith("\\%")
        pm = re.match(r"\s*(?:\\,)?\s*\$?\s*\\pm\s*\$?\s*(?:\\,)?\s*(\d+\.\d+)", s[m.end(): m.end() + 30])
        line = s.count("\n", 0, m.start()) + 1 + raw[: body(raw)].count("\n")
        yield {"file": path, "line": line, "text": t, "val": val, "d": d, "ctx": ctx, "sent": sent, "pct": pct, "para": para,
               "pos": m.start(), "end": m.end(), "sd": float(pm.group(1)) if pm else None}


def _math_mask(s):
    mask = np.zeros(len(s) + 1, bool)
    on, i = False, 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            i += 2
            continue
        if s[i] == "$":
            on = not on
        mask[i] = on
        i += 1
    return mask


# ---------------------------------------------------------------- check
def lam_of_src(src):
    m = LAM_SRC.search(src)
    return float(m.group(1).replace("p", ".")) if m else None


_TOK = {}


def _tokens(src):
    t = _TOK.get(src)
    if t is None:
        t = _TOK[src] = set(re.split(r"[^a-z0-9]+", src.lower()))
    return t


def _hit(pat, toks):
    return any(t.startswith(pat) for t in toks)


def relevance(sent, src, lams):
    sl, toks = sent.lower(), _tokens(src)
    r = sum(1 for k, pats in CONCEPTS.items() if k in sl and any(_hit(p, toks) for p in pats))
    lt = lam_of_src(src)
    if lams and lt is not None:
        r += 2 if lt in lams else -2
    return r


def check(n, idx):
    x, d = n["val"], n["d"]
    unit = 10.0 ** (-d)
    win = max(unit * 30, abs(x) * 0.08)
    parts = []
    lo, hi = idx.rng(x - win, x + win)
    parts.append((np.arange(lo, hi), 1.0))
    if n["pct"]:
        lo, hi = idx.rng((x - win) / 100, (x + win) / 100)
        parts.append((np.arange(lo, hi), 100.0))
    ii = np.concatenate([p for p, _ in parts])
    if ii.size == 0:
        n["lamset"] = set()
        return "mismatch", None, 0, ["no source within window"]
    vals = np.concatenate([idx.v[p] * f for p, f in parts])
    sl = n["sent"].lower()
    smask = sum(1 << j for j, c in enumerate(idx.keys) if c in sl)
    lams = {float(v) for v in LAM_TEX.findall(n["sent"])}
    m = idx.mask[ii] & smask
    rel = sum((m >> j) & 1 for j in range(len(idx.keys))).astype(int)
    rel = rel + idx.mean[ii].astype(int)
    lt = idx.lam[ii]
    if lams:
        has = ~np.isnan(lt)
        inl = np.isin(lt, list(lams))
        rel = rel + np.where(has & inl, 2, 0) - np.where(has & ~inl, 2, 0)
    top = rel.max()
    sel = rel >= (top - 1 if top >= 2 else top)
    rounded = rnd_vec(vals[sel], d)
    pv, pi = vals[sel], ii[sel]
    ex = rounded == x
    cl = (~ex) & (np.abs(pv - x) <= unit * 1.0001)
    status = "match" if ex.any() else ("last-digit" if cl.any() else "mismatch")
    flags = ["context score {}".format(int(top))]
    if top <= 0:
        flags.append("weak (no context overlap)")
    if ex.any():
        allx = np.abs(vals - x) <= unit
        allx &= rnd_vec(vals, d) == x
        tags = idx.lam[ii[allx]]
        if lams and tags.size and not np.isnan(tags).any() and not np.isin(tags, list(lams)).any():
            flags.append("lambda-source (sentence λ {}; sources λ {})".format(sorted(lams), sorted(set(tags.tolist()))))
        if all("pad_full" in idx.s[j] for j in pi[ex]):
            flags.append("pad_full-only")
    use = ex if ex.any() else (cl if cl.any() else np.ones(len(pv), bool))
    cand = [(pv[k], idx.s[pi[k]], idx.mean[pi[k]]) for k in np.flatnonzero(use)]
    cand.sort(key=lambda t: (not t[2], abs(t[0] - x), len(t[1])))
    if n["sd"] is not None and cand:
        key = lambda src: (src.split("::")[0].split(" [")[0], src.split("::")[-1])
        sdu = 10.0 ** (-len(str(n["sd"]).split(".")[1])) * 0.51
        lo, hi = idx.rng(n["sd"] - sdu, n["sd"] + sdu)
        sds = {key(idx.s[j]) for j in range(lo, hi) if "[sd" in idx.s[j]}
        if any(key(c[1]) in sds for c in cand if c[2]):
            flags.append("pair")
    n["lamset"] = set(idx.lam[pi[ex]][~np.isnan(idx.lam[pi[ex]])].tolist()) if ex.any() else set()
    best = (cand[0][0], cand[0][1]) if cand else None
    return status, best, int(ex.sum()), flags


def paragraph_lambda_outliers(rows):
    """Flag a number whose matched sources carry a single λ that no other number in its paragraph matches,
    when the rest of the paragraph matches at least two other λ values."""
    by_para = defaultdict(list)
    for r in rows:
        by_para[(r[0], r[1]["para"])].append(r)
    for rs in by_para.values():
        single = [(r, next(iter(r[1]["lamset"]))) for r in rs if len(r[1]["lamset"]) == 1]
        counts = defaultdict(int)
        for _, l in single:
            counts[l] += 1
        for r, l in single:
            others = {k for k, c in counts.items() if k != l}
            if counts[l] == 1 and len(others) >= 2:
                r[5].append("lambda-outlier in paragraph (this λ {}; others {})".format(l, sorted(others)))


# ------------------------------------------------------------- manifest mode
# Canonical dataset names as the manifest declares them, with the patterns that
# name them in the .tex. Kept separate from CONCEPTS: these must be exclusive,
# because pad_full satisfying a pad_heldout claim is the circularity this paper
# already corrected once.
MANIFEST_DS = {
    "pad_heldout": (r"pad\\?_heldout", r"held-out pad"),
    "pad_adv": (r"pad\\?_adv",),
    "pad_full": (r"pad\\?_full",),
    "fitzpatrick": (r"fitzpatrick", r"fitz17k"),
    "isic": (r"\bisic\b", r"ham10000", r"\bbcn\b"),
    "camelyon": (r"camelyon", r"hospital"),
    "iwildcam": (r"iwildcam",),
}


def ds_positions(text):
    out = []
    low = text.lower()
    for name, pats in MANIFEST_DS.items():
        for pat in pats:
            for m in re.finditer(pat, low):
                out.append((m.start(), m.end(), name))
    return sorted(out)


def nearest_label(mentions, start, end, limit=200):
    """The label this number is attached to, preferring one that follows it."""
    after = before = (None, None)
    for a, b, tag in mentions:
        if a <= start and end <= b:
            d, side = 0, "after"
        elif a >= end:
            d, side = a - end, "after"
        else:
            d, side = start - b, "before"
        if d > limit:
            continue
        if side == "after":
            if after[1] is None or d < after[1]:
                after = (tag, d)
        elif before[1] is None or d < before[1]:
            before = (tag, d)
    return after[0] if after[0] is not None else before[0]


def resolve_path(data, keypath):
    """keypath is a LIST of components: result keys contain '|' themselves,
    e.g. "Mahalanobis|pad_heldout|full|pooled"."""
    if isinstance(keypath, str):
        keypath = [keypath]
    cur = data
    for part in keypath:
        try:
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return None, "key not found at {!r}".format(part)
    return cur, None


def _leaf_value(rec, leaf):
    v, err = resolve_path(rec, leaf)
    if err or isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def check_manifest(path, results, numbers):
    """numbers: list of dicts from tex_numbers, so the label check sees real offsets."""
    entries = json.loads(Path(path).read_text())
    by_text = defaultdict(list)
    for n in numbers:
        by_text[n["text"]].append(n)

    out = []
    for e in entries:
        eid = e.get("id", "?")
        written = str(e["written"])
        d = len(written.split(".")[1]) if "." in written else 0
        want = float(written.replace("{,}", ""))
        probs, warns, got = [], [], None

        f = Path(results) / e["file"]
        if not f.exists():
            probs.append("missing file {}".format(e["file"]))
        else:
            data = json.loads(f.read_text())
            if "list" in e:
                node, err = resolve_path(data, e["list"])
                if err:
                    probs.append(err)
                elif not isinstance(node, list):
                    probs.append("{!r} is not a list".format(e["list"]))
                else:
                    recs = node
                    for wk, wv in (e.get("where") or {}).items():
                        recs = [r for r in recs if r.get(wk) == wv]
                    if "leaves" in e:
                        vals = []
                        for leaf in e["leaves"]:
                            lv = [_leaf_value(r, leaf) for r in recs]
                            lv = [x for x in lv if x is not None]
                            if lv:
                                vals.append(sum(lv) / len(lv))
                    else:
                        vals = [_leaf_value(r, e["leaf"]) for r in recs]
                        vals = [x for x in vals if x is not None]
                    if not vals:
                        probs.append("no values at leaf {!r}".format(e.get("leaf")))
                    else:
                        if e.get("n") is not None and len(vals) != e["n"]:
                            probs.append("expected n={} seeds, found {}".format(e["n"], len(vals)))
                        stat = e.get("stat", "mean")
                        mean = sum(vals) / len(vals)
                        if stat == "min":
                            got = min(vals)
                        elif stat == "max":
                            got = max(vals)
                        elif stat == "sd":
                            if len(vals) < 2:
                                probs.append("sd needs >= 2 values")
                            else:
                                got = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                        else:
                            got = mean
            else:
                v, err = resolve_path(data, e["key"])
                if err:
                    probs.append(err)
                elif isinstance(v, bool) or not isinstance(v, (int, float)):
                    probs.append("key is not numeric")
                else:
                    got = float(v)
            if got is not None and rnd(got, d) != rnd(want, d):
                probs.append("source gives {:.6g}, which is {:.{p}f} at the written "
                             "precision, not {}".format(got, rnd(got, d), written, p=d))

        occ = by_text.get(written, [])
        if not occ:
            probs.append("not present in the manuscript")
        elif e.get("label"):
            seen = [(nearest_label(ds_positions(_PROC[str(n["file"])]), n["pos"], n["end"]), n["line"])
                    for n in occ]
            labelled = [x for x in seen if x[0] is not None]
            if any(t == e["label"] for t, _ in seen):
                pass
            elif not labelled:
                warns.append("appears only where no dataset label is nearby (line(s) {}); "
                             "a table cell takes its label from the column header, which "
                             "this check cannot read"
                             .format(", ".join(str(l) for _, l in seen)))
            else:
                probs.append("appears at line(s) {} but the nearest label is {}, not {}"
                             .format(", ".join(str(l) for _, l in labelled),
                                     ", ".join(sorted({t for t, _ in labelled})), e["label"]))
        out.append((eid, written, e.get("label", ""), got, probs, warns))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="repo root (default: this script's repo)")
    ap.add_argument("--results", default=None, help="results dir (default: <root>/results/paperB)")
    ap.add_argument("--paper-root", default=None, help="where paper/ lives (default: <root>)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ints", action="store_true", help="also report bare integers inside math (counts)")
    ap.add_argument("--manifest", default=None,
                    help="JSON list of {id, written, label, file, key} entries. The only "
                         "mode that reliably catches a value attached to the wrong dataset")
    ap.add_argument("--allowlist", default=None,
                    help="JSON list of {written, context, reason}: constants and derived values with no "
                         "source file. A free-scan row whose printed text equals `written` and whose "
                         "context matches the regex `context` is reported as allowlisted")
    ap.add_argument("--fail-on", choices=["never", "manifest", "problem"], default="never",
                    help="manifest: exit 1 if any manifest entry fails. "
                         "problem: also exit 1 on any free-scan mismatch")
    a = ap.parse_args()
    global ROOT, RESULTS
    if a.root:
        ROOT = Path(a.root).resolve()
        RESULTS = ROOT / "results" / "paperB"
    if a.results:
        RESULTS = Path(a.results).resolve()
    if a.paper_root is None:
        a.paper_root = str(ROOT)
    if a.out is None:
        a.out = str(RESULTS / "manuscript_number_check.md")
    idx = build_index()
    print("index: {} values".format(len(idx.v)), file=sys.stderr)
    rows = []
    for t in TEX:
        p = Path(a.paper_root) / t
        if not p.exists():
            continue
        for n in tex_numbers(p):
            if n["d"] == 0 and not a.ints and "{,}" not in n["text"] and n["val"] < 1000:
                continue
            st, best, nex, flags = check(n, idx)
            rows.append((t, n, st, best, nex, flags))
    paragraph_lambda_outliers(rows)
    allow = json.loads(Path(a.allowlist).read_text()) if a.allowlist else []
    for i, (t, n, st, best, nex, flags) in enumerate(rows):
        if st == "match":
            continue
        txt = n["text"] + ("%" if n["pct"] else "")
        hit = next((e for e in allow if e["written"] == txt and re.search(e["context"], n["ctx"])), None)
        if hit:
            rows[i] = (t, n, "allowlisted", best, nex, flags + ["allowlist: " + hit["reason"]])
    order = {"mismatch": 0, "last-digit": 1, "allowlisted": 2, "match": 3}
    rows.sort(key=lambda r: (order[r[2]], not any(f.startswith(("lambda", "pad_full")) for f in r[5]), r[0], r[1]["line"]))
    cnt = defaultdict(int)
    for r in rows:
        cnt[(r[0], r[2])] += 1
    L = ["# Manuscript number check", "",
         "Generated by `scripts/verify_manuscript_numbers.py`. Report only; nothing in the manuscript was changed.", "",
         "Rounding convention: nearest, half up, at the printed precision.", "",
         "| File | match | allowlisted | last-digit | mismatch |", "|---|---:|---:|---:|---:|"]
    for t in TEX:
        L.append("| {} | {} | {} | {} | {} |".format(t, cnt[(t, "match")], cnt[(t, "allowlisted")], cnt[(t, "last-digit")],
                                                     cnt[(t, "mismatch")]))
    L += ["", "Flags: `lambda-source` = every exactly matching source carries a λ not in the sentence; `pad_full-only`; "
          "`pair` = mean and s.d. found in one source group; `weak` = matched only by value, no context overlap.", "",
          "| File | Line | Printed | Status | Nearest source value | Source | #exact | Flags | Context |",
          "|---|---:|---|---|---|---|---:|---|---|"]
    for t, n, st, best, nex, flags in rows:
        bv, bs = (("{:.6g}".format(best[0]), best[1][:150]) if best else ("—", "—"))
        L.append("| {} | {} | {} | {} | {} | `{}` | {} | {} | {} |".format(
            t.split("/")[-2] if "midl" in t else "main", n["line"], n["text"] + ("%" if n["pct"] else ""), st, bv,
            bs.replace("|", "/"), nex, "; ".join(flags), n["ctx"].replace("|", "/")))
    man_bad = 0
    if a.manifest:
        res = check_manifest(a.manifest, RESULTS, [r[1] for r in rows])
        man_bad = sum(1 for r in res if r[4])
        M = ["", "## Manifest check ({} entries, {} failing)".format(len(res), man_bad), "",
             "Each entry is checked against one declared source key and against the dataset "
             "mention nearest to the number in the .tex. This is the only check here that "
             "can catch a correct value attached to the wrong dataset.", "",
             "| id | written | label | source value | verdict |", "|---|---|---|---|---|"]
        for eid, written, label, got, probs, warns in res:
            verdict = ("**" + "; ".join(probs) + "**") if probs else (
                "ok (" + "; ".join(warns) + ")" if warns else "ok")
            M.append("| `{}` | `{}` | {} | {} | {} |".format(
                eid, written, label or "--",
                "{:.6g}".format(got) if got is not None else "--",
                verdict.replace("|", "/")))
        L += M
    Path(a.out).write_text("\n".join(L) + "\n")
    for t in TEX:
        print(t, {k: cnt[(t, k)] for k in order}, file=sys.stderr)
    if a.manifest:
        print("manifest: {} failing".format(man_bad), file=sys.stderr)
    print("wrote", a.out, file=sys.stderr)
    bad_scan = sum(cnt[(t, "mismatch")] for t in TEX)
    if a.fail_on in ("manifest", "problem") and man_bad:
        return 1
    if a.fail_on == "problem" and bad_scan:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
