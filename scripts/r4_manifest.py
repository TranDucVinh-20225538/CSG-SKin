#!/usr/bin/env python3
"""R4 manifest entries from results/paperB/r4/r4_results.json.

- paper/number_manifest.json: image-level Table 1 rows λ ∈ {0.25, 1} at n = 5 (`written` = the n = 5 value, so the
  entry fails until the draft prints it); replaces earlier entries with the same id.
- paper/number_manifest_lesion_table1.json: lesion-level Table 1, all seven λ (staged until the table is in the .tex).
- paper/number_manifest_r4_mmd.json: MMD and GRL arms on pad_heldout (staged until the text cites them).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "r4/r4_results.json"
R = json.loads((ROOT / "results" / "paperB" / SRC).read_text())
LABEL = {"maha": "pad_heldout", "fitz": "fitzpatrick"}
SD_COLS = {"leakage", "maha", "fitz", "xdom"}


def entry(eid, v, key, label=None, note=""):
    return {"id": eid, "written": "{:.3f}".format(v), "label": label, "file": SRC, "key": key, "note": note}


def cells(block, root_key, prefix, cols, note, sd_cols=SD_COLS):
    out = []
    for lam, r in R[block].items():
        if not r:
            continue
        for col in cols:
            if col not in r:
                continue
            out.append(entry("{}_{}_l{}".format(prefix, col, lam), r[col]["mean"], [root_key, lam, col, "mean"],
                             LABEL.get(col), note))
            if col in sd_cols and r[col]["sd"] is not None:
                out.append(entry("{}_{}_sd_l{}".format(prefix, col, lam), r[col]["sd"], [root_key, lam, col, "sd"],
                                 None, note))
    return out


def main():
    man = ROOT / "paper" / "number_manifest.json"
    entries = json.loads(man.read_text())
    img = [e for e in cells("image_table1", "image_table1", "t1img",
                            ("leakage", "id_bal", "id_ece", "ood_ece", "maha", "fitz", "xdom"),
                            "R4: image-level Table 1 at n = 5")
           if e["id"].endswith(("_l0.25", "_l1"))]
    ids = {e["id"] for e in img}
    entries = [e for e in entries if e["id"] not in ids] + img
    man.write_text(json.dumps(entries, indent=2) + "\n")

    lcols = ("leakage", "id_bal", "id_ece", "ood_ece", "maha", "fitz", "xdom")
    les = cells("lesion_table1", "lesion_table1", "t1les", lcols, "R4: lesion-level Table 1, n = 5 at every λ", set(lcols))
    (ROOT / "paper" / "number_manifest_lesion_table1.json").write_text(json.dumps(les, indent=2) + "\n")

    mmd = []
    for arm_key, arms in (("mmd_arm", R["mmd_arm"]), ("grl_arm", R["grl_arm"])):
        for name, r in arms.items():
            for det, d in r["pad_heldout"].items():
                mmd.append(entry("{}_{}_{}".format(arm_key, name, det), d["mean"], [arm_key, name, "pad_heldout", det, "mean"],
                                 "pad_heldout", "R4: pad_heldout, like-for-like MMD vs GRL"))
            for col in ("leakage", "id_acc", "id_bal"):
                mmd.append(entry("{}_{}_{}".format(arm_key, name, col), r[col]["mean"], [arm_key, name, col, "mean"],
                                 None, "R4: MMD vs GRL"))
    (ROOT / "paper" / "number_manifest_r4_mmd.json").write_text(json.dumps(mmd, indent=2) + "\n")
    print(len(entries), "main entries ({} image-level R4);".format(len(img)), len(les), "lesion Table 1;", len(mmd), "MMD/GRL")


if __name__ == "__main__":
    main()
