#!/usr/bin/env python3
"""R2 Item 5 exploratory: Camelyon17 two hospitals, λ ∈ {0, 0.3, 1}, seeds 42–44. The Item 5 verdict stays negative."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/data2/cmdir/home/toandq/CSG-Skin-paperB/scripts")
import r2_common as C  # noqa: E402

OUT = C.R2 / "item5"
RUNS = OUT / "runs"
LAMS, SEEDS = (0.0, 0.3, 1.0), (42, 43, 44)
DETS = (("mahalanobis_classcond_sharedcov", "Maha"), ("cosine_max", "Cosine"), ("knn_k50", "kNN"), ("MSP", "MSP"), ("Energy_T1", "Energy"))


def load(lam, seed):
    p = RUNS / "camelyon2_ladv{}_s{}".format(C.lam_tag(lam), seed) / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    d = s["ood"]["b_hold"]["detectors"]
    r = {lab: d[k] for k, lab in DETS}
    r.update({"probe": s["leakage_site_probe_a_vs_bhold"]["balanced_acc_mean"],
              "adv_min": s["adversary"]["min_window_adv_ce"], "adv_last": s["adversary"]["last_window"]["adv_ce"],
              "id_acc": s["id_test"]["acc"], "id_acc_slide": s["id_test_slide_disjoint"]["acc"], "seed": seed})
    return r


COLS = [lab for _, lab in DETS] + ["probe", "adv_min", "adv_last", "id_acc", "id_acc_slide"]
NAMES = {"probe": "Hospital probe bal acc", "adv_min": "Adv CE min", "adv_last": "Adv CE last",
         "id_acc": "ID acc", "id_acc_slide": "ID acc slide-disjoint"}


def main():
    tab = {lam: [r for r in (load(lam, s) for s in SEEDS) if r is not None] for lam in LAMS}
    missing = ["λ={:g} s{}".format(l, s) for l in LAMS for s in SEEDS if load(l, s) is None]
    r1 = tab[1.0]
    below = [r["Maha"] < 0.5 for r in r1]
    probes = [r["probe"] for r in r1]
    if len(r1) == len(SEEDS):
        answer = ("Mahalanobis < 0.5 at λ=1 on {} of {} seeds ({}); hospital probe at λ=1: {} (λ=0: {}).".format(
            sum(below), len(below), ", ".join("s{} {:.3f}".format(r["seed"], r["Maha"]) for r in r1),
            ", ".join("{:.3f}".format(p) for p in probes),
            ", ".join("{:.3f}".format(r["probe"]) for r in tab[0.0])))
    else:
        answer = "Incomplete: missing {}.".format(", ".join(missing))
    hdr = ["λ", "n"] + [NAMES.get(c, c + " (B held-out)") for c in COLS]
    L = ["# R2 Item 5 — exploratory seeds (Camelyon17, two hospitals)\n",
         "Commit: `{}`. **Exploratory.** Not pre-registered; the Item 5 verdict (negative, leakage gate failed) is unchanged. "
         "Same code, recipe and primary assignment (A = hospital 3, B = hospital 0, held-out slides of B for detection).\n".format(C.git_head()),
         "**Question:** does Mahalanobis drop below 0.5 at λ = 1 on every seed while the probe stays ≈ 0.95?\n",
         "**Answer:** {}\n".format(answer),
         "## Mean ± s.d. over seeds 42–44\n",
         "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for lam, rs in tab.items():
        L.append("| {:g} | {} | ".format(lam, len(rs)) + " | ".join(C.fmt(*C.mean_sd([r[c] for r in rs])[:2]) for c in COLS) + " |")
    L += ["", "## Per seed\n", "| λ | seed | " + " | ".join(hdr[2:]) + " |", "|---:|---:|" + "---|" * len(COLS)]
    for lam, rs in tab.items():
        for r in rs:
            L.append("| {:g} | {} | ".format(lam, r["seed"]) + " | ".join("{:.3f}".format(r[c]) for c in COLS) + " |")
    L += ["", "## Caveats\n",
          "- Exploratory: added after the pre-registered coarse scan returned a negative; it does not reopen the gates.",
          "- Detectors: ID = A patch-level test, OOD = held-out slides of hospital B. Probe: logistic ID-vs-B_heldout on encoder features, 70/30, 3 probe seeds.",
          "- Adversary CE at chance is ln 2 = 0.693."]
    (OUT / "EXPLORATORY_REPORT.md").write_text("\n".join(L) + "\n")
    (OUT / "exploratory_results.json").write_text(json.dumps({"table": {str(k): v for k, v in tab.items()}, "answer": answer,
                                                              "missing": missing, "commit": C.git_head()}, indent=1) + "\n")
    print(answer)


if __name__ == "__main__":
    main()
