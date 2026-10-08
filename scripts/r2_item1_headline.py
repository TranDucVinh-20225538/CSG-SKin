"""R2 Item 1 headline block: matched-set comparison of frozen ImageNet vs the model's own z_lesion^norm detector,
pre-registered rule stated beside the post-hoc kNN, and per-class AUROC (ID class c vs OOD).
Writes item1/headline.json and replaces the block between markers in item1/REPORT.md."""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

OUT = C.R2 / "item1"
R50 = C.R2 / "features" / "imagenet_resnet50"
OODS = ("pad_heldout", "fitzpatrick17k")
IDS = ("full", "disjoint")
CLS = ("MEL", "NV", "BCC", "AK", "BKL", "SCC")
BEGIN, END = "<!-- HEADLINE BEGIN -->", "<!-- HEADLINE END -->"


def block(ztr, ytr, zte, yte, zoo, disjoint):
    dets = {"Mahalanobis": C.Maha(ztr, ytr), "kNN k=50": C.KNN(ztr)}
    out = {}
    for dn, det in dets.items():
        sid = det.score(zte)
        for o in OODS:
            so = det.score(zoo[o])
            for idv in IDS:
                m = np.ones(len(yte), bool) if idv == "full" else disjoint
                out[(dn, o, idv, "pooled")] = C.auroc(sid[m], so)
                for c in CLS:
                    mm = m & (yte == C.LABELS.index(c))
                    out[(dn, o, idv, c)] = C.auroc(sid[mm], so) if mm.sum() >= 20 else None
    return out


def main():
    disjoint, _g, _te = C.isic_test_groups()
    r50 = {s: dict(np.load(R50 / "{}.npz".format(s))) for s in ("isic_train", "isic_test", "pad_heldout", "fitzpatrick17k")}
    yte = r50["isic_test"]["labels"].astype(int)
    base = block(r50["isic_train"]["z"], r50["isic_train"]["labels"].astype(int), r50["isic_test"]["z"], yte,
                 {o: r50[o]["z"] for o in OODS}, disjoint)
    model = {0.0: [], 2.0: []}
    for lam, seed, tag in C.runs_with_features():
        if lam not in model:
            continue
        tr, te = C.load13(tag, "isic_train"), C.load13(tag, "isic_test")
        assert np.array_equal(te["labels"].astype(int), yte), "ISIC test order differs between feature sets"
        model[lam].append(block(tr["z_lesion_norm"], tr["labels"].astype(int), te["z_lesion_norm"], yte,
                                {o: C.load13(tag, o)["z_lesion_norm"] for o in OODS}, disjoint))
        print(tag, flush=True)

    def m(lam, key):
        return C.mean_sd([r[key] for r in model[lam]])

    L = [BEGIN, "## Headline (matched sets)", "",
         "All entries on the same ID set within a row group (full ISIC test, n = 5067, unless marked lesion-disjoint, n = 2026) "
         "and the same OOD set. The model columns are z_lesion^norm, mean ± s.d. over 5 seeds; frozen ImageNet is one deterministic encoder.", "",
         "### Pre-registered rule vs post-hoc detector", "",
         "**Pre-registered:** Mahalanobis on frozen ImageNet ResNet-50 ≥ 0.90 in all four cells (2 OOD sets × full / lesion-disjoint ID). "
         "**Not met:** pad_heldout gives 0.833 (full) and 0.848 (disjoint). kNN k=50 on the same features gives 0.913 / 0.915 on pad_heldout; "
         "kNN was not the pre-registered score and is reported for information only. The bar is not cleared.", "",
         "| OOD | ID set | Detector | Frozen ImageNet | Model λ=0 | Model λ=2 | Pre-registered |", "|---|---|---|---|---|---|---|"]
    for o in OODS:
        for idv in IDS:
            for dn in ("Mahalanobis", "kNN k=50"):
                k = (dn, o, idv, "pooled")
                L.append("| {} | {} | {} | {:.3f} | {} | {} | {} |".format(
                    o, "full" if idv == "full" else "lesion-disjoint", dn, base[k], C.fmt(*m(0.0, k)[:2]), C.fmt(*m(2.0, k)[:2]),
                    "yes (≥ 0.90)" if dn == "Mahalanobis" else "no"))
    pf, ff = base[("Mahalanobis", "pad_heldout", "full", "pooled")], base[("Mahalanobis", "fitzpatrick17k", "full", "pooled")]
    p0, f0 = m(0.0, ("Mahalanobis", "pad_heldout", "full", "pooled"))[0], m(0.0, ("Mahalanobis", "fitzpatrick17k", "full", "pooled"))[0]
    p2, f2 = m(2.0, ("Mahalanobis", "pad_heldout", "full", "pooled"))[0], m(2.0, ("Mahalanobis", "fitzpatrick17k", "full", "pooled"))[0]
    L += ["", "**Reading — partial mitigation, below the pre-registered bar.** The frozen encoder never inverts (≥ 0.83 everywhere) "
          "and beats the model at λ=2 on both sets (pad_heldout {:.3f} vs {:.3f}; Fitzpatrick {:.3f} vs {:.3f}). Against the model "
          "before adversarial training (λ=0) it is higher on Fitzpatrick ({:.3f} vs {:.3f}) but not on pad_heldout ({:.3f} vs {:.3f}; "
          "within one s.d.). It is a partial mitigation, not a fix.".format(pf, p2, ff, f2, ff, f0, pf, p0), "",
          "### Per-class AUROC (ID class c vs OOD), Mahalanobis", "",
          "Within one ID class the score does not depend on the class mix of the ID set, unlike the pooled AUROC.", ""]
    for o in OODS:
        L += ["**{}**".format(o), "",
              "| ID class | Model λ=2, full | Model λ=2, lesion-disjoint | Model λ=0, full | Frozen ImageNet, full |", "|---|---|---|---|---|"]
        for c in CLS + ("pooled",):
            k = lambda idv: ("Mahalanobis", o, idv, c)
            a, b, z0 = m(2.0, k("full")), m(2.0, k("disjoint")), m(0.0, k("full"))
            L.append("| {} | {} | {} | {} | {} |".format(
                "pooled (all 8)" if c == "pooled" else c, C.fmt(*a[:2]), C.fmt(*b[:2]) if b[2] else "— (n<20)",
                C.fmt(*z0[:2]), "{:.3f}".format(base[k("full")])))
        nonnv = [m(2.0, ("Mahalanobis", o, idv, c))[0] for c in ("MEL", "BCC", "AK", "BKL", "SCC") for idv in IDS]
        L += ["", "Non-NV classes (MEL, BCC, AK, BKL, SCC) at λ=2, both ID sets: {:.2f}–{:.2f}.".format(min(nonnv), max(nonnv)), ""]
    L.append(END)

    rep = OUT / "REPORT.md"
    s = rep.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + "\n".join(L) + s[s.index(END) + len(END):]
    else:
        anchor = s.index("Statistics fit on ISIC train only")
        s = s[:anchor] + "\n".join(L) + "\n\n" + s[anchor:]
    s = s.replace("**Verdict:** Frozen ImageNet degrades -> there is no cheap fix.",
                  "**Verdict (pre-registered decision table):** Frozen ImageNet degrades -> there is no cheap fix "
                  "(Mahalanobis 0.833 on pad_heldout < 0.90). Reading: partial mitigation, see Headline.")
    rep.write_text(s)
    (OUT / "headline.json").write_text(json.dumps({
        "imagenet": {"|".join(k): v for k, v in base.items()},
        "model": {str(l): [{"|".join(k): v for k, v in r.items()} for r in rs] for l, rs in model.items()},
        "commit": C.git_head()}, indent=1))


if __name__ == "__main__":
    main()
