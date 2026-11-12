"""Mechanism figure for paper_b.tex (Figure 2).

(a) Share of OOD images whose nearest ISIC-train class centroid (class-conditional Mahalanobis,
    z_lesion^norm) is nevus. Source: results/paperB/r6/r1/cur/runs/r1_cur_*/summary.json,
    field metrics.ood.<set>.nv_nearest_frac and metrics.id_nv_nearest_frac. Runs with ID balanced
    accuracy < 0.25 are excluded (degraded rule). Mean +- s.d. over seeds.
(b) Share of melanomas predicted nevus with confidence > 0.9 (Table tab:clinical, main recipe,
    last checkpoint, five seeds, bootstrap 95% intervals).
Run from paper/midl:  python figures/fig_mechanism.py
"""
import glob, json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[3] / "results/paperB/r6/r1/cur/runs"
SETS = [("pad_heldout", "pad_heldout"), ("fitzpatrick17k", "Fitzpatrick17k"),
        ("mednode", "MED-NODE"), ("sd198", "SD-198")]
CONDS = [("shift100_ladv0", r"$\lambda_{\mathrm{adv}}=0$", "#9a9a9a"),
         ("shift100_ladv2", r"$\lambda_{\mathrm{adv}}=2$, label shift", "#D55E00"),
         ("shift0_ladv2", r"$\lambda_{\mathrm{adv}}=2$, no label shift", "#0072B2")]

def runs(cond):
    out = []
    for f in sorted(glob.glob(str(ROOT / f"r1_cur_{cond}_s*/summary.json"))):
        m = json.load(open(f))["metrics"]
        if m["id"]["bal_acc"] >= 0.25:
            out.append(m)
    return out

data = {c: runs(c) for c, _, _ in CONDS}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200,
                               gridspec_kw={"width_ratios": [1.7, 1]})
w = 0.26
for j, (c, lab, col) in enumerate(CONDS):
    ms = [st.mean(r["ood"][s]["nv_nearest_frac"] for r in data[c]) for s, _ in SETS]
    sd = [st.stdev(r["ood"][s]["nv_nearest_frac"] for r in data[c]) for s, _ in SETS]
    xs = [i + (j - 1) * w for i in range(len(SETS))]
    axa.bar(xs, ms, w, yerr=sd, color=col, label=lab, capsize=2, error_kw={"lw": 0.7})
    print(c, len(data[c]), [round(v, 3) for v in ms])
idnv = st.mean(r["id_nv_nearest_frac"] for r in data["shift100_ladv2"])
axa.axhline(idnv, color="k", ls=(0, (3, 2)), lw=0.8)
axa.text(3.45, idnv + 0.02, "ISIC test", fontsize=6.5, ha="right")
axa.set_xticks(range(len(SETS)), [n for _, n in SETS], fontsize=7)
axa.set_ylabel("share nearest to nevus centroid", fontsize=7)
axa.set_ylim(0, 1)
axa.tick_params(labelsize=7)
axa.set_title("(a) OOD images by nearest class centroid", fontsize=7.5)

# (b) MEL rows from clinical_t1last (share_pred_nv_conf09 with bootstrap CI), in %
# R10 C6: was hard-coded; now reads results/paperB/r5/posthoc/clinical_t1last/summary.csv
import csv
clin = Path(__file__).resolve().parents[3] / "results/paperB/r5/posthoc/clinical_t1last/summary.csv"
mel = {}  # name -> [(mean, lo, hi) for λ=0, λ=2] in percent
ood_lab = {"fitzpatrick17k": "Fitzpatrick17k", "sd198": "SD-198"}
with clin.open() as f:
    for row in csv.DictReader(f):
        if row["class"] != "MEL" or not row["cell"].startswith("main_last"):
            continue
        ood = row["ood"]
        if ood not in ood_lab:
            continue
        cell = row["cell"]
        if "λ=0" in cell:
            lam = 0
        elif "λ=2" in cell:
            lam = 2
        else:
            continue
        vals = (100.0 * float(row["share_pred_nv_conf09"]),
                100.0 * float(row["nv09_ci_lo"]),
                100.0 * float(row["nv09_ci_hi"]))
        mel.setdefault(ood_lab[ood], [None, None])
        mel[ood_lab[ood]][0 if lam == 0 else 1] = vals
order = ["Fitzpatrick17k", "SD-198"]
for name in order:
    if name not in mel or any(v is None for v in mel[name]):
        raise SystemExit("missing clinical MEL rows for {}".format(name))
for i, name in enumerate(order):
    for j, ((m, lo, hi), col) in enumerate(zip(mel[name], ["#9a9a9a", "#D55E00"])):
        x = i + (j - 0.5) * 0.35
        axb.bar(x, m, 0.35, color=col, yerr=[[m - lo], [hi - m]], capsize=2, error_kw={"lw": 0.7})
        print("panel_b", name, "lam", (0, 2)[j], [round(m, 1), round(lo, 1), round(hi, 1)])
axb.set_xticks([0, 1], order, fontsize=7)
axb.set_ylabel("melanomas called nevus, conf. $>0.9$ (%)", fontsize=7)
axb.set_ylim(0, 100)
axb.tick_params(labelsize=7)
axb.set_title(r"(b) Melanomas called nevus ($\lambda_{\mathrm{adv}}$ 0 vs 2)", fontsize=7.5)
for a in (axa, axb):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
h, l = axa.get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = Path(__file__).resolve().parent
fig.savefig(out / "fig_mechanism.pdf", bbox_inches="tight")
fig.savefig(out / "fig_mechanism.png", bbox_inches="tight")
