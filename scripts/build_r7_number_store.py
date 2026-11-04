"""Build results/paperB/r7/r7_number_store.json from the R7 / R7b / P9 v2 result files and add the
matching entries to paper/number_manifest.json (ids prefixed "r7_"; re-running replaces them).

Sources: r7/t1, r7/t1b, r7/t2 summary.csv (mean and sd over seeds, sd with ddof=1), r7/t1/CASE.json,
r7/c3/coverage.json, r7b/c1/summary.csv, r7b/c3b/summary.csv, r7b/t1c/SELECTION.json,
r5/p9/v2/BOOTSTRAP.json and NO_GO5.json.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paperB"
STORE = RES / "r7" / "r7_number_store.json"
STORE_REL = "r7/r7_number_store.json"
MANIFEST = ROOT / "paper" / "number_manifest.json"


def rows(rel):
    with open(RES / rel) as f:
        return list(csv.DictReader(f))


def agg(recs, cols):
    out = {}
    for c in cols:
        v = [float(r[c]) for r in recs]
        out[c] = {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
    return out


def build():
    s = {}
    cols = ["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc", "maha_mednode", "maha_sd198", "maha_scin"]
    g = defaultdict(list)
    for r in rows("r7/t1/summary.csv"):
        g["{}|lam{}".format(r["arm"].split()[0].lower(), int(float(r["lambda_adv"])))].append(r)
    s["t1"] = {k: agg(v, cols) for k, v in g.items()}
    s["t1"]["gap"] = json.loads((RES / "r7/t1/CASE.json").read_text())["gap"]

    s["t1b"] = agg(rows("r7/t1b/summary.csv"), ["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc"])

    g = defaultdict(list)
    for r in rows("r7/t2/summary.csv"):
        g["frac{}|lam{}".format(r["frac"], int(float(r["lambda_adv"])))].append(r)
    s["t2"] = {k: agg(v, ["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc"]) for k, v in g.items()}

    s["c1"] = {}
    for r in rows("r7b/c1/summary.csv"):
        if r["ood"] != "pad_heldout":
            continue
        lo, hi = json.loads(r["fpr95tpr_pooled_ci"])
        m = float(r["fpr95tpr_mean"])
        if float(r["lambda"]) not in (0.0, 2.0):
            continue
        s["c1"]["lam{}".format(int(float(r["lambda"])))] = {
            "fpr95tpr_mean": m, "fpr95tpr_pct": 100 * m, "fpr95tpr_sd": float(r["fpr95tpr_sd"]),
            "ci_lo": lo, "ci_hi": hi, "flagged_mean": float(r["ood_flagged_at_95id_mean"])}

    cov = json.loads((RES / "r7/c3/coverage.json").read_text())
    tv = {t["variable"]: t["tv"] for t in cov["tv"]}
    s["c3"] = {"tv_type": tv["fitzpatrick (fitzpatrick_scale)"], "tv_isic": tv["isic_mapped_label"]}

    s["c3b"] = {}
    for r in rows("r7b/c3b/summary.csv"):
        if r["detector"] != "mahalanobis_classcond":
            continue
        s["c3b"]["lam{}|{}".format(r["lambda"], r["setting"])] = {
            "auroc": float(r["auroc_mean"]), "n_images": int(r["n_images"])}

    sel = json.loads((RES / "r7b/t1c/SELECTION.json").read_text())
    s["t1c"] = {arm: {str(c["mult"]): c["val_balanced_acc"] for c in sel[arm + "_candidates"]}
                for arm in ("dual", "erm")}

    b = json.loads((RES / "r5/p9/v2/BOOTSTRAP.json").read_text())
    s["p9"] = {
        "best_pad": b["best"]["pad_heldout"]["mahalanobis_classcond"]["mean"],
        "best_pad_ci_lo": b["best"]["pad_heldout"]["mahalanobis_classcond"]["pooled_ci"][0],
        "best_pad_ci_hi": b["best"]["pad_heldout"]["mahalanobis_classcond"]["pooled_ci"][1],
        "best_fitz": b["best"]["fitzpatrick17k"]["mahalanobis_classcond"]["mean"],
        "control_pad": b["control_lambda0"]["pad_heldout"]["mahalanobis_classcond"]["mean"],
        "leakage": json.loads((RES / "r5/p9/v2/NO_GO5.json").read_text())["mean_v1"],
        "control_pad_ci_lo": b["control_lambda0"]["pad_heldout"]["mahalanobis_classcond"]["pooled_ci"][0],
        "control_pad_ci_hi": b["control_lambda0"]["pad_heldout"]["mahalanobis_classcond"]["pooled_ci"][1],
        "target": json.loads((RES / "r5/p9/v2/TARGET.json").read_text())["target"],
    }
    d4 = json.loads((RES / "r8/d4/D4.json").read_text())
    s["d4"] = {bb: {k: {"mean": x["mean"], "ci_lo": x["ci_lo"], "ci_hi": x["ci_hi"]} for k, x in d4[bb].items()}
               for bb in ("effb3_dual", "dinov2_dual")}

    # R8 D1 screen (seed 42): V1 range and pass count per backbone x objective; D2 five-seed means
    d1 = list(csv.DictReader(open(RES / "r8/d1/summary.csv")))
    s["r8_d1"] = {"n_runs": len(d1), "gpu_hours": sum(float(r["gpu_hours"]) for r in d1)}
    for r in d1:
        g = s["r8_d1"].setdefault("{}|{}".format(r["backbone"], r["penalty"] if r["family"] == "align" else "grl"),
                                  {"v1": [], "n_pass": 0})
        g["v1"].append(float(r["v1"]))
        g["n_pass"] += r["pass_flag"] == "True"
        s["r8_d1"]["run|" + r["tag"]] = {"v1": float(r["v1"]), "id_bal_acc": float(r["id_balanced_acc"])}
    for k, g in s["r8_d1"].items():
        if isinstance(g, dict) and "v1" in g and isinstance(g["v1"], list):
            g.update(n=len(g["v1"]), v1_min=min(g["v1"]), v1_max=max(g["v1"]))
    d2 = defaultdict(list)
    for r in csv.DictReader(open(RES / "r8/d2/summary.csv")):
        d2[r["arm"]].append(r)
    s["r8_d2"] = {}
    for arm, rs in d2.items():
        for c in ("v1", "leakage", "id_bal_acc", "ood_auroc", "fitz_auroc"):
            v = [float(r[c]) for r in rs if r[c] != ""]
            if len(v) > 1:
                s["r8_d2"].setdefault(arm, {})[c] = {"mean": statistics.mean(v), "sd": statistics.stdev(v)}
    # R1 retrained runs: share of melanomas predicted nevus with confidence > 0.9 (r5/posthoc/clinical)
    s["clin_r1"] = {}
    for r in csv.DictReader(open(RES / "r5/posthoc/clinical/summary.csv")):
        v = list(r.values())
        if v[0].startswith("R1 shift") and v[2] == "MEL":
            s["clin_r1"]["{}|{}".format(v[0], v[1])] = 100.0 * float(v[7])
    # R10b bootstrap CIs: B1 shift 50% (tab:shiftfull), B3 A4 rows (tab:unseen), B4 SCIN splits (tab:splits)
    s["r10b"] = {}
    for f, keyf in (("B1_shift50_boots", lambda r: "b1|{}|{}".format(r["lambda"], r["set"])),
                    ("B3_a4_ci", lambda r: "b3|{}|{}".format(r["lambda"], r["set"])),
                    ("B4_splits_scin", lambda r: "b4|{}|{}".format(r["split"], r["lambda"]))):
        for r in json.loads((RES / "r10b" / (f + ".json")).read_text())["rows"]:
            s["r10b"][keyf(r)] = {"mean": r["mean"], "sd": r["sd"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1],
                                  "k_below": int(r["k"].split("/")[0]), "rule_a": bool(r["rule_A"])}
    s["r8_d2"]["outcome"] = json.loads((RES / "r8/d2/OUTCOME.json").read_text())
    s["r8_d2"]["gpu_hours_align"] = sum(float(r["gpu_hours"]) for r in d2["align MK-MMD w30"])
    return s


def entries():
    e = []

    def add(eid, written, key, optional=False, note=None, cell=None):
        d = {"id": "r7_" + eid, "written": written, "file": STORE_REL, "key": key}
        if cell:
            d["cell"] = cell
        if optional:
            d["optional"] = True
        if note:
            d["note"] = note
        e.append(d)

    t1 = {
        "erm|lam0": ["0.983", "0.002", "0.565", "0.015", "0.591", "0.060", "0.704", "0.059", "0.452", "0.033", "0.702", "0.053", "0.781", "0.074"],
        "dual|lam0": ["0.947", "0.007", "0.470", "0.015", "0.831", "0.029", "0.778", "0.016", "0.580", "0.030", "0.790", "0.005", "0.716", "0.041"],
        "dual|lam2": ["0.576", "0.034", "0.462", "0.022", "0.448", "0.045", "0.472", "0.067", "0.363", "0.045", "0.461", "0.066", "0.551", "0.064"],
    }
    cols = ["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc", "maha_mednode", "maha_sd198", "maha_scin"]
    # tab:dino keys rows by the first column, so the two "dual DINOv2" rows collide; the lambda = 2 row
    # gets value-only entries (sd values sit after \pm and are not tokenized, hence optional).
    t1_rows = {"erm|lam0": "ERMDINOv2", "dual|lam0": "dualDINOv2", "dual|lam2": None}
    for arm, w in t1.items():
        for i, c in enumerate(cols):
            for j, stat in enumerate(("mean", "sd")):
                row = t1_rows[arm]
                cell = {"table": "tab:dino", "row": row, "col": 2 + i, "stat": stat} if row else None
                add("t1_{}_{}_{}".format(arm.replace("|", "_"), c, stat), w[2 * i + j], ["t1", arm, c, stat],
                    optional=(row is None and stat == "sd"), cell=cell)
    add("t1_gap", "0.095", ["t1", "gap"])

    for c, (m, sd) in zip(["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc"],
                          [("0.884", "0.019"), ("0.555", "0.023"), ("0.546", "0.033"), ("0.476", "0.069")]):
        i = ["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc"].index(c)
        for stat, w in (("mean", m), ("sd", sd)):
            add("t1b_{}_{}".format(c, stat), w, ["t1b", c, stat],
                cell={"table": "tab:single-dino", "row": "0.884pm0.019", "col": i, "stat": stat})

    t2 = {
        "frac0.25|lam0": ["0.898", "0.032", "0.472", "0.019", "0.819", "0.022", "0.724", "0.043"],
        "frac0.25|lam2": ["0.632", "0.007", "0.477", "0.002", "0.479", "0.013", "0.407", "0.011"],
        "frac0.5|lam2": ["0.575", "0.030", "0.459", "0.022", "0.415", "0.043", "0.397", "0.069"],
    }
    t2_rows = {"frac0.25|lam0": "25%(407images)", "frac0.25|lam2": "25%", "frac0.5|lam2": "50%"}
    for arm, w in t2.items():
        for i, c in enumerate(["leakage", "id_bal_acc", "ood_auroc", "fitz_auroc"]):
            for j, stat in enumerate(("mean", "sd")):
                add("t2_{}_{}_{}".format(arm.replace("|", "_"), c, stat), w[2 * i + j], ["t2", arm, c, stat],
                    cell={"table": "tab:padadv", "row": t2_rows[arm], "col": 2 + i, "stat": stat})

    for lam, vals in (("lam0", ["0.911", "0.060", "0.882", "0.940", "91.1"]),
                      ("lam2", ["0.961", "0.007", "0.951", "0.970", "96.1"])):
        for k, w in zip(["fpr95tpr_mean", "fpr95tpr_sd", "ci_lo", "ci_hi", "fpr95tpr_pct"], vals):
            add("c1_{}_{}".format(lam, k), w, ["c1", lam, k])
    add("c1_lam2_flagged", "0.039", ["c1", "lam2", "flagged_mean"])

    add("c3_tv_type", "0.257", ["c3", "tv_type"])
    add("c3_tv_isic", "0.063", ["c3", "tv_isic"])

    for lam, settings in (("0.0", [("types_1_2", "0.733"), ("types_3_4", "0.711"), ("types_5_6", "0.695"),
                                   ("reweight_type", "0.719"), ("reweight_type_x_isic", "0.714")]),
                          ("2.0", [("types_1_2", "0.423"), ("types_3_4", "0.390"), ("types_5_6", "0.411"),
                                   ("reweight_type", "0.415"), ("reweight_type_x_isic", "0.424"),
                                   ("unknown", "0.479")])):
        for st, w in settings:
            add("c3b_lam{}_{}".format(lam, st), w, ["c3b", "lam{}|{}".format(lam, st), "auroc"])
    for st, w in (("types_1_2", "819"), ("types_3_4", "2{,}078"), ("types_5_6", "837"), ("unknown", "153")):
        # bare integers are not tokenized without --ints; the value is still checked
        add("c3b_n_{}".format(st), w, ["c3b", "lam2.0|{}".format(st), "n_images"], optional="{," not in w)

    for arm, ws in (("dual", ["0.444", "0.402", "0.236"]), ("erm", ["0.516", "0.504", "0.418"])):
        for mult, w in zip(["0.1", "0.3", "1.0"], ws):
            add("t1c_{}_{}".format(arm, mult), w, ["t1c", arm, mult])

    add("p9_leak_2dp", "0.70", ["p9", "leakage"])
    add("p9_pad_2dp", "0.52", ["p9", "best_pad"])
    add("p9_fitz_2dp", "0.62", ["p9", "best_fitz"])
    for k, w in (("best_pad", "0.524"), ("best_pad_ci_lo", "0.498"), ("best_pad_ci_hi", "0.550"),
                 ("leakage", "0.699"), ("control_pad", "0.686")):
        add("p9_" + k, w, ["p9", k])
    for k, w in (("control_pad_ci_lo", "0.665"), ("control_pad_ci_hi", "0.706"), ("target", "0.552")):
        add("p9_" + k, w, ["p9", k])

    # D4 (paired lambda = 2 minus lambda = 0); signed values are matched without the sign
    d4 = {"effb3_dual": {"id_bal_acc": ("-0.009", "-0.026", "+0.006"), "pad_heldout": ("-0.444", "-0.465", "-0.417"),
                         "fitzpatrick17k": ("-0.307", "-0.334", "-0.275")},
          "dinov2_dual": {"id_bal_acc": ("-0.008", "-0.014", "-0.002"), "pad_heldout": ("-0.383", "-0.428", "-0.329"),
                          "fitzpatrick17k": ("-0.306", "-0.361", "-0.244"), "mednode": ("-0.218", "-0.253", "-0.183"),
                          "sd198": ("-0.329", "-0.372", "-0.270")}}
    for bb, qs in d4.items():
        for q, ws in qs.items():
            for stat, w in zip(("mean", "ci_lo", "ci_hi"), ws):
                add("d4_{}_{}_{}".format(bb, q, stat), w, ["d4", bb, q, stat], optional=False,
                    note="Appendix Competence (D4)")

    # R8 D1 screen (tab:r8 and §4.1) and D2 follow-up (App. r8)
    add("r8_d1_gpu", "18.95", ["r8_d1", "gpu_hours"])
    for g, (lo, hi) in {"effb3|grl": ("0.685", "0.727"), "effb3|mkmmd": ("0.810", "0.894"),
                        "effb3|coral": ("0.981", "0.993"), "dinov2_vits14|grl": ("0.772", "0.930"),
                        "dinov2_vits14|mkmmd": ("0.600", "0.665"), "dinov2_vits14|coral": ("0.616", "0.990")}.items():
        add("r8_d1_{}_v1_min".format(g), lo, ["r8_d1", g, "v1_min"])
        add("r8_d1_{}_v1_max".format(g), hi, ["r8_d1", g, "v1_max"])
    pre = "r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_"
    add("r8_d1_coral10_v1", "0.650", ["r8_d1", "run|" + pre + "coral10", "v1"])
    add("r8_d1_coral_degraded_id", "0.125", ["r8_d1", "run|" + pre + "coral30", "id_bal_acc"])
    add("r8_d1_mkmmd30_id", "0.555", ["r8_d1", "run|" + pre + "mkmmd30", "id_bal_acc"])
    for arm, cols in {"align MK-MMD w30": {"v1": ("0.604", "0.018"), "leakage": ("0.691", "0.030"),
                                           "id_bal_acc": ("0.547", None), "ood_auroc": ("0.540", "0.017"),
                                           "fitz_auroc": ("0.519", "0.028")}}.items():
        for c, (m, sd) in cols.items():
            add("r8_d2_align_{}_mean".format(c), m, ["r8_d2", arm, c, "mean"])
            if sd:
                add("r8_d2_align_{}_sd".format(c), sd, ["r8_d2", arm, c, "sd"], optional=True,
                    note="sd follows \\pm and is not tokenized")
    for cell, ds, w in (("R1 shift 100% λ=2", "fitzpatrick17k", "53.0"), ("R1 shift 0% λ=2", "fitzpatrick17k", "21.4"),
                        ("R1 shift 100% λ=2", "sd198", "83.2"), ("R1 shift 0% λ=2", "sd198", "33.8")):
        add("clin_r1_{}_{}".format(cell.split()[2].rstrip("%"), ds), w, ["clin_r1", "{}|{}".format(cell, ds)],
            note="§4.4 and Fig. 1 caption; MEL predicted NV conf > 0.9, %")
    r10b = json.loads(STORE.read_text())["r10b"] if STORE.exists() else build()["r10b"]
    for k, v in r10b.items():
        tag = k.replace("|", "_").replace(".", "p")
        # tab:unseen prints A4 at λ=0 as a mean only (no interval column)
        for stat in (("mean",) if k.startswith("b3|0.0|") else ("mean", "ci_lo", "ci_hi")):
            add("r10b_{}_{}".format(tag, stat), "{:.3f}".format(v[stat]), ["r10b", k, stat],
                note="R10b bootstrap (2,000 draws, rng 20261005)")
    add("r8_d2_gap", "0.018", ["r8_d2", "outcome", "gap"])
    add("r8_d2_gpu", "3.64", ["r8_d2", "gpu_hours_align"])
    return e


def main():
    STORE.write_text(json.dumps(build(), indent=1))
    m = [x for x in json.loads(MANIFEST.read_text()) if not x["id"].startswith("r7_")]
    new = entries()
    MANIFEST.write_text(json.dumps(m + new, indent=1))
    print("store: {}; manifest: {} entries ({} r7_)".format(STORE.relative_to(ROOT), len(m) + len(new), len(new)))


if __name__ == "__main__":
    main()
