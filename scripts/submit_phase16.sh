#!/usr/bin/env bash
# Phase 1.6a+d — CPU-only so we do not compete with Phase 2/4 GPU arrays.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" "${ROOT}/results/paperB/phase1_6"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== dry_run 1.6a ==="
"${PY}" "${ROOT}/scripts/eval_supervised_domain_head.py" --cpu --dry_run --num_workers 0 --batch_size 8

SBATCH="${ROOT}/slurm/phase16.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p16
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase16_%j.err
# No --gres: do not compete with Phase 2/4 for GPUs.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== Phase 1.6 start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/eval_supervised_domain_head.py --cpu --num_workers 8 --batch_size 32
if ls ${ROOT}/data/fitzpatrick17k/images/*.jpg >/dev/null 2>&1; then
  ${PY} ${ROOT}/scripts/eval_fitz_domain_axis.py --cpu --num_workers 8
else
  echo "Fitzpatrick images not ready; skip 1.6d extract"
fi
echo "=== Phase 1.6 done $(date) ==="
EOF

echo "=== sbatch Phase 1.6 CPU ==="
sbatch "${SBATCH}"
squeue -u "$(whoami)"
echo "Phase 2/4 not modified. Phase 3 not submitted."
