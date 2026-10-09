#!/usr/bin/env python3
"""Cross-check every number in the manuscript against results/paperB/**/*.json (+ CSV and a few feature-derived counts).

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

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
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


def tex_numbers(path):
    raw = strip_tex(Path(path).read_text())
    mac = macros(raw)
    s = raw[body(raw):]
    for k in sorted(mac, key=len, reverse=True):
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda _m, v=mac[k]: v, s)
    s = SKIP_CMD.sub(lambda m: " " * len(m.group(0)), s)
    s = re.sub(r"\\begin\{(?:tabular|figure|table)\}\{?[^}\n]*\}?", lambda m: " " * len(m.group(0)), s)
    in_math = _math_mask(s)
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
               "sd": float(pm.group(1)) if pm else None}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-root", default=str(ROOT.parent / "CSG-Skin-paperB-main"))
    ap.add_argument("--out", default=str(RESULTS / "manuscript_number_check.md"))
    ap.add_argument("--ints", action="store_true", help="also report bare integers inside math (counts)")
    a = ap.parse_args()
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
    order = {"mismatch": 0, "last-digit": 1, "match": 2}
    rows.sort(key=lambda r: (order[r[2]], not any(f.startswith(("lambda", "pad_full")) for f in r[5]), r[0], r[1]["line"]))
    cnt = defaultdict(int)
    for r in rows:
        cnt[(r[0], r[2])] += 1
    L = ["# Manuscript number check", "",
         "Generated by `scripts/verify_manuscript_numbers.py`. Report only; nothing in the manuscript was changed.", "",
         "Rounding convention: nearest, half up, at the printed precision.", "",
         "| File | match | last-digit | mismatch |", "|---|---:|---:|---:|"]
    for t in TEX:
        L.append("| {} | {} | {} | {} |".format(t, cnt[(t, "match")], cnt[(t, "last-digit")], cnt[(t, "mismatch")]))
    L += ["", "Flags: `lambda-source` = every exactly matching source carries a λ not in the sentence; `pad_full-only`; "
          "`pair` = mean and s.d. found in one source group; `weak` = matched only by value, no context overlap.", "",
          "| File | Line | Printed | Status | Nearest source value | Source | #exact | Flags | Context |",
          "|---|---:|---|---|---|---|---:|---|---|"]
    for t, n, st, best, nex, flags in rows:
        bv, bs = (("{:.6g}".format(best[0]), best[1][:150]) if best else ("—", "—"))
        L.append("| {} | {} | {} | {} | {} | `{}` | {} | {} | {} |".format(
            t.split("/")[-2] if "midl" in t else "main", n["line"], n["text"] + ("%" if n["pct"] else ""), st, bv,
            bs.replace("|", "/"), nex, "; ".join(flags), n["ctx"].replace("|", "/")))
    Path(a.out).write_text("\n".join(L) + "\n")
    for t in TEX:
        print(t, {k: cnt[(t, k)] for k in order}, file=sys.stderr)
    print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
