#!/usr/bin/env bash
# Phase 12.1b: derm-matched adversary.
# Array 60636 already covers λ ∈ {10, 30, 100, 300}.
# This script's array definition is the historical 0-3 launch.
# Matched λ=0 is a separate job (blocking control). Do NOT submit the coarse dense grid.
# Do NOT start iWildCam.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" \
  "${ROOT}/results/paperB/phase12/camelyon17/matched" \
  "${ROOT}/results/paperB/phase12/figures" \
  "${ROOT}/checkpoints/phase12/camelyon17_matched"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== dry_run λ=0 seed 42 (matched control) ==="
"${PY}" "${ROOT}/scripts/train_phase12_1b.py" --lambda_adv 0 --seed 42 --dry_run

SBATCH="${ROOT}/slurm/phase12_1b.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p121b
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p121b_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p121b_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
LAMS=(10 30 100 300)
TID=${SLURM_ARRAY_TASK_ID}
LAM="${LAMS[$TID]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== P12.1b matched lambda_adv=${LAM} seed=42 start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase12_1b.py --lambda_adv "${LAM}" --seed 42 --skip_done --num_workers 4 --epochs 10
echo "=== P12.1b matched lambda_adv=${LAM} seed=42 done $(date) ==="
EOF

L0="${ROOT}/slurm/phase12_1b_l0.sbatch"
cat > "${L0}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p121b0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p121b0_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p121b0_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== P12.1b matched lambda_adv=0 seed=42 start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase12_1b.py --lambda_adv 0 --seed 42 --skip_done --num_workers 4 --epochs 10
echo "=== P12.1b matched lambda_adv=0 seed=42 done $(date) ==="
EOF

AGG="${ROOT}/slurm/phase12_1b_agg.sbatch"
cat > "${AGG}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p121bagg
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p121bagg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p121bagg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
${PY} ${ROOT}/scripts/aggregate_phase12_1b.py
echo "=== P12.1b aggregate done $(date) ==="
EOF

echo "This launcher does not resubmit 60636. Use submit_phase12_1b_lambda0.sh for the control."
echo "Dense cliff grid NOT submitted. iWildCam NOT started."
