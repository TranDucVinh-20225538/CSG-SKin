#!/usr/bin/env bash
# Phase 2.5 — CPU inference array. Does not take GPUs from Phase 4.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" "${ROOT}/results/paperB/phase2_5"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"

echo "=== 2.5c axis distributions ==="
"${PY}" "${ROOT}/scripts/eval_phase25_axis.py"

echo "=== dry_run 2.5a/b one cell ==="
"${PY}" "${ROOT}/scripts/eval_phase25.py" --cpu --dry_run --num_workers 0 --batch_size 8 --method runB_orth1 --seed 42

SBATCH="${ROOT}/slurm/phase25.sbatch"
cat > "${SBATCH}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p25
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase25_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase25_%A_%a.err
# No --gres: do not compete with Phase 4 for GPUs.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
METHODS=(runB_orth1 baseline_soft effb3_control)
SEEDS=(42 52 62 72 82)
TID=${SLURM_ARRAY_TASK_ID}
METHOD="${METHODS[$((TID / 5))]}"
SEED="${SEEDS[$((TID % 5))]}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== Phase 2.5 ${METHOD} seed=${SEED} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/eval_phase25.py --cpu --num_workers 4 --batch_size 32 --method "${METHOD}" --seed "${SEED}" --skip_done
echo "=== Phase 2.5 ${METHOD} seed=${SEED} done $(date) ==="
EOF

AGG="${ROOT}/slurm/phase25_agg.sbatch"
cat > "${AGG}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p25agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase25agg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/phase25agg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
${PY} ${ROOT}/scripts/aggregate_phase25.py
echo "=== Phase 2.5 aggregate done $(date) ==="
EOF

echo "=== sbatch Phase 2.5 CPU array ==="
JID=$(sbatch --parsable "${SBATCH}")
echo "array job ${JID}"
sbatch --dependency=afterok:${JID} "${AGG}"
squeue -u "$(whoami)" | head -40
echo "Phase 4 not modified. Phase 2.5 is CPU."
