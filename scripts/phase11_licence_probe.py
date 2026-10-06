#!/usr/bin/env python3
"""Phase 11 (non-GPU): licence record + local availability probe for a third dermatology domain.

Does not download. Does not train. Writes phase11_availability.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = PAPERB / "results" / "paperB" / "phase11_third_domain"
SEARCH_ROOTS = [
    Path("/data2/hpcshared/Vinh/CSG-Skin/data"),
    Path("/data2/cmdir/home/toandq/CSG-Skin-paperB"),
]
NAME_HINTS = {
    "ddi": ["ddi", "diverse_dermatology", "diverse-dermatology"],
    "derm7pt": ["derm7pt", "derm_7pt", "7point", "7-point"],
    "fitzpatrick17k": ["fitzpatrick17k", "fitzpatrick_17k", "fitz17k"],
}


def walk_hits(hints, max_hits=12):
    hits = []
    skip = {".git", ".venv", "node_modules", "__pycache__", "checkpoints", "wandb", "site-packages", "lib64", "lib"}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
            depth = dirpath.count(os.sep) - str(root).count(os.sep)
            if depth > 4:
                dirnames[:] = []
                continue
            base = os.path.basename(dirpath).lower()
            if any(h in base for h in hints):
                hits.append(dirpath)
                if len(hits) >= max_hits:
                    return hits
    return hits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "chosen_dataset": "Fitzpatrick17k",
        "choice_reason": (
            "DDI requires a Stanford research-use agreement that cannot be completed from this job. "
            "Derm7pt is password-gated (CC BY-NC-ND). Fitzpatrick17k annotations are publicly posted "
            "(CC BY-NC-SA 3.0) and are the only candidate that can be acquired without an interactive licence grant. "
            "HAM10000 is rejected: it overlaps ISIC 2018/2019 and is not an unseen domain."
        ),
        "evaluation_status": "not_run",
        "gpu": False,
        "licences": {
            "ISIC_2019": {
                "used_in": "ID training / ID test / Phase 4 semantic OOD",
                "licence": "ISIC Archive / challenge release: CC BY-NC 4.0 (non-commercial). Individual contributing centres may impose additional terms.",
                "citation": "Tschandl et al., The HAM10000 dataset, Sci. Data 2018; Combalia et al., BCN20000, arXiv:1908.02288; ISIC 2019 Challenge.",
                "url": "https://challenge.isic-archive.com/data/#2019",
                "status_in_manuscript": "Limitation #5 previously unrecorded; recorded here.",
            },
            "PAD_UFES_20": {
                "used_in": "domain-adversary stream + covariate-shift OOD (Phases 0–3, 6, 9)",
                "licence": "CC BY 4.0",
                "citation": "Pacheco et al., PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones, Data in Brief 32:106221, 2020.",
                "url": "https://data.mendeley.com/datasets/zr7vgbcyr2",
                "status_in_manuscript": "Limitation #5 previously unrecorded; recorded here.",
            },
            "DDI": {
                "used_in": "candidate Phase 11; NOT acquired",
                "licence": "Stanford DDI research-use agreement. Not a public Commons licence. Redistribution prohibited without Stanford approval.",
                "citation": "Daneshjou et al., Disparities in dermatology AI: Diverse Dermatology Images, Sci. Adv. 2022.",
                "url": "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50fba9823c79",
                "label_space_note": "DDI diagnoses do not cover the ISIC 8-class set 1:1. Mapping would be many-to-one / drop-unmapped; must be frozen before any eval.",
                "availability": "blocked_on_agreement",
            },
            "Derm7pt": {
                "used_in": "candidate Phase 11; NOT acquired",
                "licence": "CC BY-NC-ND 4.0 (no derivatives) + password from the authors.",
                "citation": "Kawahara et al., 7-Point Checklist, IEEE JBHI 2019.",
                "url": "https://derm.cs.sfu.ca/Welcome.html",
                "label_space_note": "7-point checklist + diagnosis; not native ISIC-8. Mapping required.",
                "availability": "blocked_on_password",
            },
            "Fitzpatrick17k": {
                "used_in": "chosen Phase 11 third domain (pending image acquisition)",
                "licence_annotations": "CC BY-NC-SA 3.0 (Groh et al. annotations / metadata).",
                "licence_images": (
                    "Images were collected from the public web (DermNet, Atlas Dermatologico, etc.). "
                    "Underlying photograph copyright is mixed and is NOT transferred by the annotation licence. "
                    "Use is non-commercial research only; do not redistribute the image files."
                ),
                "citation": "Groh et al., Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset, CVPRW 2021.",
                "url": "https://github.com/mattgroh/fitzpatrick17k",
                "label_space_note": (
                    "Primary labels are disease names (~114) plus Fitzpatrick skin type I–VI. "
                    "This is not the ISIC 8-class taxonomy. Proposed mapping (to be locked before eval): "
                    "melanoma→MEL, nevus/melanocytic nevus→NV, basal cell carcinoma→BCC, "
                    "actinic keratosis→AK, seborrheic keratosis / lichenoid keratosis / solar lentigo→BKL, "
                    "dermatofibroma→DF, vascular / angioma / pyogenic granuloma→VASC, squamous cell carcinoma→SCC. "
                    "Unmapped diseases are excluded from 8-class diagnosis metrics but remain valid as OOD images "
                    "for z_context domain detection (the actual Phase 11 test does not need 8-class labels)."
                ),
                "availability": "not_on_disk_yet",
            },
        },
        "local_hits": {},
    }
    for key, hints in NAME_HINTS.items():
        try:
            report["local_hits"][key] = walk_hits(hints)
        except Exception as e:
            report["local_hits"][key] = ["ERROR: {}".format(e)]
    (OUT / "phase11_availability.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "LICENCES.md").write_text(_md(report))
    print("wrote", OUT / "LICENCES.md")
    print("hits", json.dumps(report["local_hits"], indent=2))


def _md(report):
    lines = [
        "# Phase 11 — third-domain licences and availability",
        "",
        "Chosen dataset: **{}**.".format(report["chosen_dataset"]),
        "",
        report["choice_reason"],
        "",
        "Evaluation itself is inference-only and is **not** started here.",
        "",
        "## Licences (ISIC, PAD, and the three candidates)",
        "",
    ]
    for name, rec in report["licences"].items():
        lines.append("### {}".format(name.replace("_", " ")))
        for k, v in rec.items():
            lines.append("- **{}**: {}".format(k, v))
        lines.append("")
    lines.append("## Local filesystem probe")
    lines.append("")
    for k, v in report["local_hits"].items():
        lines.append("- `{}`: {}".format(k, v if v else "no hits"))
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
