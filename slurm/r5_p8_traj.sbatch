#!/bin/bash
#SBATCH --job-name=paperB-r5-p8traj
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8traj_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p8traj_%A_%a.err
# P8 trajectory runs (decisions/paperB_r5_p8_traj.md): task 0 = λ 0 s42, task 1 = λ 2 s42 (A4, per-epoch weights),
# then per-epoch evaluation on the same GPU.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
TAGS=(traj_ladv0_s42 traj_ladv2_s42)
echo "=== P8 traj task=${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) commit=$(git rev-parse --short HEAD) ==="
"${ROOT}/.venv/bin/python" scripts/train_r5_p8.py --stage traj --task_id "${SLURM_ARRAY_TASK_ID}" --resume --skip_done --num_workers 6
"${ROOT}/.venv/bin/python" scripts/r5_p8_traj.py run "${TAGS[${SLURM_ARRAY_TASK_ID}]}"
echo "=== done $(date) ==="
