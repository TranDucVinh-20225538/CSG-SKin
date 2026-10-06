#!/usr/bin/env bash
# Phase 13: CPU extraction of Phase 3 features + analyze + toy.
# Does NOT request GPUs. Does NOT cancel Phase 12.
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/logs" "${ROOT}/slurm" \
  "${ROOT}/results/paperB/phase13/features" \
  "${ROOT}/results/paperB/phase13/figures" \
  "${ROOT}/results/paperB/phase13/toy"
export PYTHONPATH="/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts"
export CUDA_VISIBLE_DEVICES=""

echo "=== dry_run extract task 0 + analyze + toy ==="
"${PY}" "${ROOT}/scripts/extract_phase13_features.py" --task_id 0 --dry_run
"${PY}" "${ROOT}/scripts/analyze_phase13.py"
"${PY}" "${ROOT}/scripts/train_phase13_toy.py" --dry_run

EXT="${ROOT}/slurm/phase13_extract.sbatch"
cat > "${EXT}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p13ext
#SBATCH --array=0-26
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13ext_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13ext_%A_%a.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P13 extract task ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
${PY} ${ROOT}/scripts/extract_phase13_features.py --task_id "${SLURM_ARRAY_TASK_ID}" --skip_done --cpu --num_workers 4 --batch_size 32
echo "=== P13 extract task ${SLURM_ARRAY_TASK_ID} done $(date) ==="
EOF

TOY="${ROOT}/slurm/phase13_toy.sbatch"
cat > "${TOY}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p13toy
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13toy_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13toy_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P13 toy start $(date) ==="
${PY} ${ROOT}/scripts/train_phase13_toy.py
echo "=== P13 toy done $(date) ==="
EOF

AGG="${ROOT}/slurm/phase13_agg.sbatch"
cat > "${AGG}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=paperB-p13agg
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13agg_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/p13agg_%j.err

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
PY="${ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
echo "=== P13 analyze start $(date) ==="
${PY} ${ROOT}/scripts/analyze_phase13.py
echo "=== P13 analyze done $(date) ==="
EOF

echo "=== sbatch Phase 13 CPU extract + toy; aggregate after extract ==="
JEXT=$(sbatch --parsable "${EXT}")
echo "extract array ${JEXT}"
JTOY=$(sbatch --parsable "${TOY}")
echo "toy ${JTOY}"
sbatch --dependency=afterok:${JEXT} "${AGG}"
squeue -u "$(whoami)" | head -30
echo "Phase 12 left running. No GPU requested."
