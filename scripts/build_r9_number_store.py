"""Build results/paperB/r9/r9_number_store.json from the R9 per-run summaries and add the matching
entries to paper/number_manifest.json (ids prefixed "r9_"; re-running replaces them).

E1: six-class runs (r9/e1/runs) against the R1 references (r6/r1/cur/runs), Mahalanobis AUROC.
Degraded runs (ID test balanced accuracy < 0.25) are excluded from means, as in Table tab:shift
(this drops shift 0% lambda=2 seed 52, ID 0.139). E2: r9/e2/S/runs, r9/e2/erm_ref, GATE_S.json.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paperB"
STORE = RES / "r9" / "r9_number_store.json"
STORE_REL = "r9/r9_number_store.json"
MANIFEST = ROOT / "paper" / "number_manifest.json"
SEEDS = [42, 52, 62, 72, 82]
SETS = ["pad_heldout", "fitzpatrick17k", "mednode", "sd198"]


def summ(p):
    return json.loads((RES / p / "summary.json").read_text())["metrics"]


def runs(kind, lam):
    out = []
    for s in SEEDS:
        p = ("r9/e1/runs/e1_6cls_ladv{}_s{}" if kind == "six" else "r6/r1/cur/runs/r1_cur_" + kind + "_ladv{}_s{}").format(lam, s)
        m = summ(p)
        if m["id"]["bal_acc"] >= 0.25:
            out.append(m)
    return out


def ms(v):
    return {"mean": statistics.mean(v), "sd": statistics.stdev(v)}


def build():
    s = {"e1": {}, "e2": {}}
    for lam, key in (("2", "lam2"), ("0p5", "lam0.5")):
        r = {k: runs(k, lam) for k in ("six", "shift100", "shift0")}
        d = {"n": {k: len(v) for k, v in r.items()},
             "id_bal_acc": ms([m["id"]["bal_acc"] for m in r["six"]])}
        for st in SETS:
            v = {k: ms([m["ood"][st]["mahalanobis_classcond"] for m in r[k]]) for k in r}
            v["recovery"] = (v["six"]["mean"] - v["shift100"]["mean"]) / (v["shift0"]["mean"] - v["shift100"]["mean"])
            v["delta_six_minus_100"] = v["six"]["mean"] - v["shift100"]["mean"]
            d[st] = v
        d["max_delta_labelled"] = max(d[x]["delta_six_minus_100"] for x in SETS[1:])
        s["e1"][key] = d
    cfg = json.loads((RES / "r9/e1/runs/e1_6cls_ladv2_s42/summary.json").read_text())
    s["e1"]["pi_6cls"] = dict(zip(["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"], cfg["pi_6cls"]))
    s["e1"]["tv_isic"] = cfg["tv_to_pi_isic"]
    s["e1"]["tv_pad"] = cfg["tv_to_pi_pad"]
    g = json.loads((RES / "r9/e2/GATE_S.json").read_text())
    s["e2"]["a7_test_mean"] = statistics.mean(summ("r9/e2/S/runs/A7_e80_ladv0_s{}".format(x))["id"]["bal_acc"] for x in (42, 52))
    s["e2"]["gate_mean"] = g["id_test_mean"]
    s["e2"]["threshold"] = g["threshold"]
    s["e2"]["erm80_test_mean"] = statistics.mean(
        json.loads((RES / "r9/e2/erm_ref/single_dann_ladv0_s{}/summary.json".format(x)).read_text())["metrics"]["id"]["bal_acc"] for x in (42, 52))
    return s


def entries():
    e = []

    def add(eid, written, key):
        e.append({"id": "r9_" + eid, "written": written, "file": STORE_REL, "key": key})

    w = {"lam2": {"pad_heldout": ("0.372", "0.034", "0.368", "0.017", "0.458", "0.003", "0.050"),
                  "fitzpatrick17k": ("0.362", "0.032", "0.340", "0.065", "0.691", "0.031", "0.063"),
                  "mednode": ("0.312", "0.050", "0.298", "0.032", "0.541", "0.027", "0.058"),
                  "sd198": ("0.339", "0.039", "0.325", "0.069", "0.694", "0.031", "0.037")},
         "lam0.5": {"pad_heldout": ("0.374", "0.037", "0.361", "0.029", "0.498", "0.025", "0.101"),
                    "fitzpatrick17k": ("0.394", "0.043", "0.385", "0.023", "0.672", "0.046", "0.031"),
                    "mednode": ("0.332", "0.036", "0.310", "0.055", "0.525", "0.050", "0.103"),
                    "sd198": ("0.384", "0.051", "0.365", "0.029", "0.664", "0.058", "0.063")}}
    for lam, sets in w.items():
        for st, vals in sets.items():
            i = 0
            for k in ("six", "shift100", "shift0"):
                for stat in ("mean", "sd"):
                    add("e1_{}_{}_{}_{}".format(lam, st, k, stat), vals[i], ["e1", lam, st, k, stat])
                    i += 1
            add("e1_{}_{}_recovery".format(lam, st), vals[6], ["e1", lam, st, "recovery"])
    add("e1_lam2_id_mean", "0.467", ["e1", "lam2", "id_bal_acc", "mean"])
    add("e1_lam2_id_sd", "0.010", ["e1", "lam2", "id_bal_acc", "sd"])
    add("e1_lam05_id_mean", "0.470", ["e1", "lam0.5", "id_bal_acc", "mean"])
    add("e1_lam05_id_sd", "0.010", ["e1", "lam0.5", "id_bal_acc", "sd"])
    add("e1_max_delta", "0.022", ["e1", "lam2", "max_delta_labelled"])
    for c, v in (("MEL", "0.182"), ("NV", "0.518"), ("BCC", "0.134"), ("AK", "0.035"), ("BKL", "0.106"), ("SCC", "0.025")):
        add("e1_pi_" + c, v, ["e1", "pi_6cls", c])
    add("e1_tv_isic", "0.019", ["e1", "tv_isic"])
    add("e1_tv_pad", "0.579", ["e1", "tv_pad"])
    add("e2_a7_test", "0.522", ["e2", "a7_test_mean"])
    add("e2_gate_mean", "0.522", ["e2", "gate_mean"])
    add("e2_threshold", "0.528", ["e2", "threshold"])
    add("e2_erm80", "0.557", ["e2", "erm80_test_mean"])
    return e


def main():
    STORE.write_text(json.dumps(build(), indent=1))
    m = [x for x in json.loads(MANIFEST.read_text()) if not x["id"].startswith("r9_")]
    new = entries()
    MANIFEST.write_text(json.dumps(m + new, indent=1))
    print("store: {}; manifest: {} entries ({} r9_)".format(STORE.relative_to(ROOT), len(m) + len(new), len(new)))


if __name__ == "__main__":
    main()
