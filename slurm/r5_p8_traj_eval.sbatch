#!/bin/bash
#SBATCH --job-name=paperB-r5-p8traj-eval
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8trajeval_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8trajeval_%A_%a.err
# Per-epoch evaluation of the P8 main λ = 2 runs, seeds 52/62/72/82 (decisions/paperB_r5_p8_traj.md, 2026-10-15 23:20).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
TAGS=(main_ladv2_s52 main_ladv2_s62 main_ladv2_s72 main_ladv2_s82)
echo "=== P8 traj eval ${TAGS[${SLURM_ARRAY_TASK_ID}]} start $(date) host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
"${ROOT}/.venv/bin/python" scripts/r5_p8_traj.py run "${TAGS[${SLURM_ARRAY_TASK_ID}]}"
echo "=== done $(date) ==="
