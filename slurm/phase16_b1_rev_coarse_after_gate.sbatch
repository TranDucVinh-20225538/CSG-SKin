#!/bin/bash
# Submit manually after λ=0 gate pass:
#   MHA=$(python3 -c "import json;print(json.load(open('.../ham_bcn_ladv0_s42/summary.json'))['ood_detectors_bcn_heldout']['mahalanobis_classcond'])")
#   awk -v m=$MHA 'BEGIN{exit !(m>=0.65)}' && sbatch slurm/phase16_b1_rev_coarse.sbatch

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
SUM="${ROOT}/results/paperB/phase16/b1_rev/ham_bcn_ladv0_s42/summary.json"
if [ ! -f "$SUM" ]; then echo "missing $SUM"; exit 1; fi
MHA=$(python3 -c "import json; print(json.load(open('$SUM'))['ood_detectors_bcn_heldout']['mahalanobis_classcond'])")
python3 -c "import sys; sys.exit(0 if float('$MHA') >= 0.65 else 1)" || { echo "GATE FAIL maha=$MHA < 0.65 — not submitting coarse"; exit 0; }
echo "GATE PASS maha=$MHA — submitting rev coarse array"
sbatch "${ROOT}/slurm/phase16_b1_rev_coarse.sbatch"
