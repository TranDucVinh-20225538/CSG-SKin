#!/bin/bash
# Paper B R6 split-stability launcher (decisions/paperB_r6_split_stability.md).
# Usage: GO=1 bash scripts/launch_r6_split_stability.sh [AFTER_JOBID]
# Chain: prepare+verify42 (local) -> GPU array 0-17%3 -> CPU report
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
cd "${ROOT}"
mkdir -p logs results/paperB/r6/split_stability
if [[ "${GO:-0}" != "1" ]]; then echo "compute HOLD: set GO=1 to submit"; exit 1; fi

echo "=== prepare + seed-42 hash gate (must pass) $(date) ==="
"${ROOT}/.venv/bin/python" scripts/r6_split_stability.py prepare
"${ROOT}/.venv/bin/python" scripts/r6_split_stability.py verify42

DEP=()
if [[ -n "${1:-}" ]]; then DEP=(--dependency=afterany:$1); fi
GPU=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} slurm/r6_split_stability.sbatch)
# report after any completion (partial tables ok; re-run report when all done)
REP=$(sbatch --parsable --dependency=afterany:${GPU} --cpus-per-task=8 --mem=32G --time=04:00:00 \
  --job-name=paperB-ss-rep \
  --output="${ROOT}/logs/r6ss_report_%j.out" --error="${ROOT}/logs/r6ss_report_%j.err" \
  --wrap="cd ${ROOT} && export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts && ${ROOT}/.venv/bin/python scripts/r6_split_stability.py report")
echo "split_stability GPU=${GPU} report=${REP}"
echo "last job: ${REP}"
