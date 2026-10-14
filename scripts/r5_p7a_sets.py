#!/usr/bin/env python3
"""Paper B R5 P7(a): image manifests of the unseen clinical sets (decisions/paperB_r5_p7a.md).

One CSV per set in results/paperB/r5/p7a/sets/: path, source_label, mapped (ISIC class or ""), group (resample unit).
Reads file names and label files only; no image is decoded.
"""
import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "unseen_clinical"
OUT = ROOT / "results" / "paperB" / "r5" / "p7a" / "sets"

SD198_MAP = {
    "NV": ["Blue_Nevus", "Compound_Nevus", "Congenital_Nevus", "Dysplastic_Nevus", "Halo_Nevus", "Junction_Nevus",
           "Nail_Nevus", "Nevus_Incipiens", "Nevus_Spilus"],
    "MEL": ["Malignant_Melanoma", "Lentigo_Maligna_Melanoma"],
    "BCC": ["Basal_Cell_Carcinoma"],
    "AK": ["Actinic_solar_Damage(Actinic_Keratosis)"],
    "BKL": ["Seborrheic_Keratosis", "Solar_Lentigo"],
    "DF": ["Dermatofibroma"],
    "VASC": ["Angioma", "Pyogenic_Granuloma", "Strawberry_Hemangioma"],
    "SCC": ["Bowen's_Disease", "Keratoacanthoma"],
}
SCIN_MAP = {"Melanocytic Nevus": "NV", "Melanoma": "MEL", "Basal Cell Carcinoma": "BCC", "Actinic Keratosis": "AK",
            "Seborrheic Keratosis": "BKL", "SCC/SCCIS": "SCC", "Dermatofibroma": "DF", "Hemangioma": "VASC",
            "Pyogenic granuloma": "VASC"}
IMG = (".jpg", ".jpeg", ".png", ".bmp")


def mednode():
    rows = []
    for cls, mapped in (("melanoma", "MEL"), ("naevus", "NV")):
        for p in sorted((DATA / "mednode").rglob("*")):
            if p.suffix.lower() in IMG and p.parent.name == cls:
                rows.append((str(p), cls, mapped, p.stem))
    return rows


def sd198():
    inv = {c: k for k, v in SD198_MAP.items() for c in v}
    rows = []
    for d in sorted((DATA / "sd198" / "sd-198" / "images").iterdir()):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG:
                rows.append((str(p), d.name, inv.get(d.name, ""), "{}/{}".format(d.name, p.stem)))
    return rows


def scin():
    base = DATA / "scin"
    cases = pd.read_csv(base / "scin_cases.csv", dtype={"case_id": str})
    labels = pd.read_csv(base / "scin_labels.csv", dtype={"case_id": str}).set_index("case_id")
    rows = []
    for _, c in cases.iterrows():
        top, mapped = "", ""
        w = labels.weighted_skin_condition_label.get(c.case_id)
        if isinstance(w, str) and w.strip() not in ("", "{}"):
            k, v = max(ast.literal_eval(w).items(), key=lambda x: x[1])
            top = k
            if v >= 0.5:
                mapped = SCIN_MAP.get(k, "")
        for col in ("image_1_path", "image_2_path", "image_3_path"):
            if isinstance(c[col], str) and c[col]:
                p = base / "images" / Path(c[col]).name
                if p.exists():
                    rows.append((str(p), top, mapped, c.case_id))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("mednode", mednode), ("sd198", sd198), ("scin", scin)):
        df = pd.DataFrame(fn(), columns=["path", "source_label", "mapped", "group"])
        df.to_csv(OUT / "{}.csv".format(name), index=False)
        print(name, len(df), "images,", df.group.nunique(), "groups; mapped:",
              df[df.mapped != ""].mapped.value_counts().to_dict())


if __name__ == "__main__":
    main()
