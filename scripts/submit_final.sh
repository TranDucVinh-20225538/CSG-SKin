#!/usr/bin/env bash
# Paper B final package: Phase 6 (pad_heldout xfer) + Phase 3.5 (mechanism). Inference only.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" \
  "${ROOT}/results/paperB/phase6_xfer" \
  "${ROOT}/results/paperB/phase35_mech" \
  "${ROOT}/results/paperB/figures" \
  "${ROOT}/results/paperB/phase10"

export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== dry_run task 0 (CSG λ=0 s42) ==="
"${PY}" "${ROOT}/scripts/eval_final_gpu.py" --task_id 0 --dry_run
echo "=== dry_run task 27 (baseline s42) ==="
"${PY}" "${ROOT}/scripts/eval_final_gpu.py" --task_id 27 --dry_run

SBATCH="${ROOT}/slurm/final_gpu.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-final
#SBATCH --array=0-36
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalgpu_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalgpu_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== final gpu task ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/eval_final_gpu.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done --num_workers 4 --batch_size 64
echo "=== final gpu task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
EOF

AGG="${ROOT}/slurm/final_agg.sbatch"
cat > "${AGG}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-finalagg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalagg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/finalagg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== final aggregate start $(date) ==="
${PY} ${ROOT}/scripts/aggregate_final_package.py
echo "=== final aggregate done $(date) ==="
EOF

echo "=== sbatch GPU array 0-36 ==="
JID=$(sbatch --parsable "${SBATCH}")
echo "array job ${JID}"
sbatch --dependency=afterok:${JID} "${AGG}"
squeue -u "$(whoami)" | head -50
