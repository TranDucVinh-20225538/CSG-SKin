#!/usr/bin/env python3
"""R4 manifest entries from results/paperB/r4/r4_results.json, merged into paper/number_manifest.json.

Table 1 entries are anchored to their cell (`cell`: table label, row λ as printed, 0-based column, mean|sd), so a value
printed elsewhere in the paper cannot satisfy them. MMD/GRL entries are text numbers marked `optional`: checked
(value and dataset label) wherever the text prints them, a warning if it does not.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "r4/r4_results.json"
R = json.loads((ROOT / "results" / "paperB" / SRC).read_text())

IMAGE_TABLE = "tab:dial"           # image-level Table 1 as it stands in the draft
LESION_TABLE = "tab:dial_lesion"   # lesion-level Table 1; set to the draft's label once the table is in the .tex
COLS = (("leakage", 2, "Leakage"), ("id_bal", 3, "ID bal acc"), ("id_ece", 4, "ID ECE"), ("ood_ece", 5, "OOD ECE"),
        ("maha", 6, "OOD AUROC"), ("fitz", 7, "Fitz AUROC"), ("xdom", 8, "X-dom bal acc"))
# image-level table prints ± only in these cells; the lesion-level table prints ± everywhere
IMAGE_SD = {"0": {"leakage", "maha", "fitz", "xdom"}, "2": {"leakage", "maha", "fitz", "xdom"},
            "0.25": {"xdom"}, "1": {"xdom"}}


def cell_entry(eid, value, key, table, row, col, col_name, stat, note):
    w = "{:.3f}".format(value) if stat != "n" else str(int(value))
    return {"id": eid, "written": w, "label": None, "file": SRC, "key": key,
            "cell": {"table": table, "row": row, "col": col, "col_name": col_name, "stat": "sd" if stat == "sd" else "mean"},
            "note": note}


def table_entries(block, table, prefix, sd_cols, note):
    out = []
    for lam, r in R[block].items():
        if not r or (sd_cols is not None and lam not in sd_cols):
            continue
        some = next(iter(c for c in r.values() if isinstance(c, dict)))
        out.append(cell_entry("{}_n_l{}".format(prefix, lam), some["n"], [block, lam, "leakage", "n"],
                              table, lam, 1, "n", "n", note))
        for col, idx, name in COLS:
            if col not in r:
                continue
            out.append(cell_entry("{}_{}_l{}".format(prefix, col, lam), r[col]["mean"], [block, lam, col, "mean"],
                                  table, lam, idx, name, "mean", note))
            if (sd_cols is None or col in sd_cols[lam]) and r[col]["sd"] is not None:
                out.append(cell_entry("{}_{}_sd_l{}".format(prefix, col, lam), r[col]["sd"], [block, lam, col, "sd"],
                                      table, lam, idx, name, "sd", note))
    return out


def mmd_entries():
    out = []
    for arm_key in ("mmd_arm", "grl_arm"):
        for name, r in R[arm_key].items():
            for det, d in r["pad_heldout"].items():
                out.append({"id": "{}_{}_{}".format(arm_key, name, det), "written": "{:.3f}".format(d["mean"]),
                            "label": "pad_heldout", "file": SRC, "key": [arm_key, name, "pad_heldout", det, "mean"],
                            "optional": True, "note": "R4: pad_heldout, like-for-like MMD vs GRL"})
            for col in ("leakage", "id_acc", "id_bal"):
                out.append({"id": "{}_{}_{}".format(arm_key, name, col), "written": "{:.3f}".format(r[col]["mean"]),
                            "label": None, "file": SRC, "key": [arm_key, name, col, "mean"],
                            "optional": True, "note": "R4: MMD vs GRL"})
    return out


def main():
    man = ROOT / "paper" / "number_manifest.json"
    entries = json.loads(man.read_text())
    new = (table_entries("image_table1", IMAGE_TABLE, "t1img", IMAGE_SD, "R4: image-level Table 1 at n = 5, cell-anchored")
           + table_entries("lesion_table1", LESION_TABLE, "t1les", None, "R4: lesion-level Table 1, n = 5, cell-anchored")
           + mmd_entries())
    drop = ("t1img_", "t1les_", "mmd_arm_", "grl_arm_")
    entries = [e for e in entries if not e["id"].startswith(drop)] + new
    table_only = {  # pre-R4 entries whose number is printed only in the image-level Table 1
        "resnet50_fitz": {"row": "ResNet-50", "col": 6, "col_name": "Fitz AUROC", "stat": "mean"},
        "maha_pad_l2_sd": {"row": "2", "col": 6, "col_name": "OOD AUROC", "stat": "sd"},
        "img_fitz_maha_sd_l2": {"row": "2", "col": 7, "col_name": "Fitz AUROC", "stat": "sd"},
    }
    for e in entries:
        if e["id"] in table_only:
            e["cell"] = {"table": IMAGE_TABLE, **table_only[e["id"]]}
    man.write_text(json.dumps(entries, indent=2) + "\n")
    for f in ("number_manifest_lesion_table1.json", "number_manifest_r4_mmd.json"):
        (ROOT / "paper" / f).unlink(missing_ok=True)
    print(len(entries), "entries;", sum(1 for e in new if e.get("cell")), "cell-anchored R4,",
          sum(1 for e in new if e.get("optional")), "optional MMD/GRL")


if __name__ == "__main__":
    main()
