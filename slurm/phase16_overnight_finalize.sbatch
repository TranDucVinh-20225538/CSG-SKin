#!/bin/bash
#SBATCH --job-name=paperB-p16-morning
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=14:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_overnight_finalize_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_overnight_finalize_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
cd "${ROOT}"

need_rev_coarse() {
  local missing=0
  for L in 0 0.1 0.25 0.5 1 2 4 8; do
    T=$(python3 -c "print('{:g}'.format(float('$L')).replace('.','p'))")
    F="${ROOT}/results/paperB/phase16/b1_rev/ham_bcn_ladv${T}_s42/summary.json"
    [ -f "$F" ] || missing=$((missing+1))
  done
  echo "$missing"
}

need_rev_dense() {
  local missing=0
  for L in 0 0.25 1; do
    for S in 52 62; do
      T=$(python3 -c "print('{:g}'.format(float('$L')).replace('.','p'))")
      F="${ROOT}/results/paperB/phase16/b1_rev/ham_bcn_ladv${T}_s${S}/summary.json"
      [ -f "$F" ] || missing=$((missing+1))
    done
  done
  echo "$missing"
}

echo "=== overnight finalize start $(date) ==="
while true; do
  MC=$(need_rev_coarse)
  MD=$(need_rev_dense)
  echo "$(date) rev_coarse_missing=$MC rev_dense_missing=$MD"
  "${PY}" "${ROOT}/scripts/phase16_morning_build.py" || true
  if [ "$MC" -eq 0 ] && [ "$MD" -eq 0 ]; then
    echo "ALL COMPLETE $(date)"
    "${PY}" "${ROOT}/scripts/phase16_b1_rev_aggregate.py"
    "${PY}" "${ROOT}/scripts/phase16_morning_build.py"
    touch "${ROOT}/results/paperB/phase16/OVERNIGHT_DONE.flag"
    exit 0
  fi
  sleep 300
done
