#!/bin/bash
#SBATCH --job-name=r14-prep
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_prepare_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_prepare_%j.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
mkdir -p logs results/paperB/r14_pair3 data/r14_pair3
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
echo START_PREPARE $(date)
.venv/bin/python scripts/r14_pair3_prepare_data.py
echo START_PHASH $(date)
.venv/bin/python scripts/r14_pair3_overlap_phash.py
# refresh PRIORS after any phash drops
.venv/bin/python - <<'PY'
import sys, json
sys.path.insert(0,'scripts')
import r14_pair3_common as C
import r14_pair3_prepare_data as P
adv=C.load_frame('milk_adv'); hold=C.load_frame('milk_heldout')
P.write_priors(adv, hold)
print('PRIORS refreshed')
PY
echo DONE_PREPARE $(date)
