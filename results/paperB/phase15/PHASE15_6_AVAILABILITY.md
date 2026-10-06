# Phase 15.6 — chest X-ray availability (before compute)

MIMIC-CXR needs PhysioNet credentialing. CheXpert needs Stanford registration.
Do not start training until a readable pair exists.

| Dataset | Found | Paths tried |
|---|---|---|
| CheXpert | no | /data2/cmdir/home/toandq/data/CheXpert, /data2/cmdir/home/toandq/data/chexpert, /data2/hpcshared/Vinh/data/CheXpert, /data2/cmdir/home/toandq/CSG-Skin-paperB/data/CheXpert |
| NIH ChestX-ray14 | no | /data2/cmdir/home/toandq/data/NIH, /data2/cmdir/home/toandq/data/chestxray14, /data2/hpcshared/Vinh/data/NIH |
| MIMIC-CXR | no | /data2/cmdir/home/toandq/data/MIMIC-CXR, /data2/cmdir/home/toandq/data/mimic-cxr, /data2/hpcshared/Vinh/data/MIMIC-CXR |

Not enough local data (found: none). 15.6 is future work until CheXpert↔NIH or MIMIC↔CheXpert is on disk and licensed.

