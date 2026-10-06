#!/usr/bin/env bash
# Phase 4 — semantic OOD (ISIC only). 2 configs × 3 methods × 3 seeds = 18-task array.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/results/paperB/phase4_semantic_ood" "${ROOT}/slurm"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== dry_run 4a/4b × 3 methods ==="
for cfg in 4a 4b; do
  for method in baseline effb3 runB_orth1; do
    "${PY}" "${ROOT}/scripts/train_phase4_semantic_ood.py" --config "${cfg}" --method "${method}" --seed 42 --dry_run
  done
done

SBATCH="${ROOT}/slurm/phase4.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p4
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase4_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase4_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
CONFIGS=(4a 4b)
METHODS=(baseline effb3 runB_orth1)
SEEDS=(42 52 62)
TID=${SLURM_ARRAY_TASK_ID}
CFG="${CONFIGS[$((TID / 9))]}"
METHOD="${METHODS[$(( (TID % 9) / 3 ))]}"
SEED="${SEEDS[$((TID % 3))]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
echo "=== Phase 4 ${CFG} ${METHOD} seed=${SEED} start $(date) host=$(hostname) ==="
${PY} -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
${PY} ${ROOT}/scripts/train_phase4_semantic_ood.py --config "${CFG}" --method "${METHOD}" --seed "${SEED}" --resume --skip_done --num_workers 4 --max_epochs 40
echo "=== Phase 4 ${CFG} ${METHOD} seed=${SEED} done $(date) ==="
EOF

echo "=== sbatch Phase 4 array ==="
sbatch "${SBATCH}"
squeue -u "$(whoami)"
echo "Phase 3 is gated. Do not submit λ_adv sweep."
