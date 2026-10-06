#!/usr/bin/env bash
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" \
  "${ROOT}/results/paperB/phase15/objectives" \
  "${ROOT}/checkpoints/phase15/objectives"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
export PYTHONUNBUFFERED=1

echo "=== 15.2 dry_run erm w=0 seed 42 ==="
"${PY}" "${ROOT}/scripts/train_phase15_2_objectives.py" --objective erm --weight 0 --seed 42 --dry_run

SBATCH="${ROOT}/slurm/phase15_2.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p152
#SBATCH --array=0-38
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_2_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_2_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
# 39 jobs: erm×3 + 4 objectives × 3 weights × 3 seeds
OBJ=(erm erm erm coral coral coral coral coral coral coral coral coral mmd mmd mmd mmd mmd mmd mmd mmd mmd irm irm irm irm irm irm irm irm irm groupdro groupdro groupdro groupdro groupdro groupdro groupdro groupdro groupdro)
W=(0 0 0 1 1 1 10 10 10 100 100 100 1 1 1 10 10 10 100 100 100 1 1 1 10 10 10 100 100 100 0.1 0.1 0.1 1 1 1 10 10 10)
SEEDS=(42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== 15.2 ${OBJ[$TID]} w=${W[$TID]} seed=${SEEDS[$TID]} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/train_phase15_2_objectives.py --objective "${OBJ[$TID]}" --weight "${W[$TID]}" --seed "${SEEDS[$TID]}" --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== 15.2 done $(date) ==="
EOF

echo "=== sbatch 15.2 array 0-38 ==="
sbatch "${SBATCH}"
squeue -u "$(whoami)" | head -30
