#!/usr/bin/env bash
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" \
  "${ROOT}/results/paperB/phase15/pacs" \
  "${ROOT}/checkpoints/phase15/pacs"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
export PYTHONUNBUFFERED=1

echo "=== 15.3 dry_run hold=photo λ=0 seed 42 ==="
"${PY}" "${ROOT}/scripts/train_phase15_3_pacs.py" --heldout_domain photo --lambda_adv 0 --seed 42 --dry_run

if [[ -f "${ROOT}/results/paperB/phase15/pacs/DATA_MISSING.md" ]]; then
  echo "PACS missing — not submitting 15.3. See results/paperB/phase15/pacs/DATA_MISSING.md"
  exit 0
fi

SBATCH="${ROOT}/slurm/phase15_3.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p153
#SBATCH --array=0-59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_3_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase15_3_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
# 4 domains × 5 λ × 3 seeds
HOLD=(photo photo photo photo photo photo photo photo photo photo photo photo photo photo photo art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting art_painting cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon cartoon sketch sketch sketch sketch sketch sketch sketch sketch sketch sketch sketch sketch sketch sketch sketch)
LAM=(0 0 0 0.25 0.25 0.25 1 1 1 2 2 2 8 8 8 0 0 0 0.25 0.25 0.25 1 1 1 2 2 2 8 8 8 0 0 0 0.25 0.25 0.25 1 1 1 2 2 2 8 8 8 0 0 0 0.25 0.25 0.25 1 1 1 2 2 2 8 8 8)
SEEDS=(42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62 42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== 15.3 hold=${HOLD[$TID]} λ=${LAM[$TID]} seed=${SEEDS[$TID]} start $(date) ==="
${PY} ${ROOT}/scripts/train_phase15_3_pacs.py --heldout_domain "${HOLD[$TID]}" --lambda_adv "${LAM[$TID]}" --seed "${SEEDS[$TID]}" --skip_done --num_workers 4 --max_epochs 30
echo "=== 15.3 done $(date) ==="
EOF

echo "=== sbatch 15.3 array 0-59 ==="
sbatch "${SBATCH}"
squeue -u "$(whoami)" | head -30
