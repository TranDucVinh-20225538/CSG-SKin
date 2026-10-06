#!/usr/bin/env bash
# Phase 12.1 Camelyon17 coarse λ scan. 4 GPU tasks, 1 seed. Dense sweep is NOT launched.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" \
  "${ROOT}/results/paperB/phase12/camelyon17/coarse" \
  "${ROOT}/checkpoints/phase12/camelyon17"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== dry_run λ=0 seed 42 ==="
"${PY}" "${ROOT}/scripts/train_phase12_camelyon_dann.py" --lambda_adv 0 --seed 42 --dry_run

SBATCH="${ROOT}/slurm/phase12_coarse.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p12c
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p12coarse_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p12coarse_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMS=(0 0.1 1 10)
TID=${SLURM_ARRAY_TASK_ID}
LAM="${LAMS[$TID]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== P12 coarse lambda_adv=${LAM} seed=42 start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase12_camelyon_dann.py --lambda_adv "${LAM}" --seed 42 --skip_done --num_workers 4 --batch_size 32 --epochs 10
echo "=== P12 coarse lambda_adv=${LAM} seed=42 done $(date) ==="
EOF

AGG="${ROOT}/slurm/phase12_coarse_agg.sbatch"
cat > "${AGG}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p12cagg
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p12coarseagg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p12coarseagg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
${PY} ${ROOT}/scripts/aggregate_phase12_coarse.py
echo "=== P12 coarse aggregate done $(date) ==="
EOF

echo "=== sbatch coarse array 0-3 ==="
JID=$(sbatch --parsable "${SBATCH}")
echo "array job ${JID}"
sbatch --dependency=afterok:${JID} "${AGG}"
squeue -u "$(whoami)" | head -20
echo "Dense sweep NOT launched."
