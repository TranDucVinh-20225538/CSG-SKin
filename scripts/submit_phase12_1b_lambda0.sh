#!/usr/bin/env bash
# Matched-recipe λ=0 control. Rebind aggregate to wait on 60636 AND this job.
# Do not cancel 60636. Do not start iWildCam or a dense grid.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== dry_run λ=0 seed 42 ==="
"${PY}" "${ROOT}/scripts/train_phase12_1b.py" --lambda_adv 0 --seed 42 --dry_run

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

echo "=== cancel old aggregate 60637 (it would run without matched λ=0) ==="
scancel 60637 || true

echo "=== sbatch matched λ=0 ==="
J0=$(sbatch --parsable "${L0}")
echo "lambda0 job ${J0}"

echo "=== sbatch aggregate afterok:60636,afterok:${J0} ==="
sbatch --dependency=afterok:60636,afterok:${J0} "${AGG}"
squeue -u "$(whoami)" | head -20
echo "60636 left running. Dense cliff grid NOT submitted. iWildCam NOT started."
