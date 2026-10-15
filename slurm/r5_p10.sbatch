#!/bin/bash
#SBATCH --job-name=paperB-r5-p10
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5p10_%A_%a.err
# STAGE (env): ladder (5 tasks, sequential %1: rungs L1_bank, L2_ganin, L3_strong, L4_lam3, L5_lam10; seed 42)
#              stage2 (2 tasks: the first gate-meeting rung at seeds 43, 44; no-op if none met the gate)
# Submit through scripts/launch_r5_p10.sh.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
T=${SLURM_ARRAY_TASK_ID}
case "${STAGE}" in
  ladder) RUNGS=(L1_bank L2_ganin L3_strong L4_lam3 L5_lam10); ARGS=(--rung "${RUNGS[$T]}" --seed 42) ;;
  stage2) S2=(43 44); ARGS=(--stage2 --seed "${S2[$T]}") ;;
  *) echo "unknown STAGE ${STAGE}"; exit 2 ;;
esac
echo "=== R5 P10 stage=${STAGE} task=${T} args=${ARGS[*]} start $(date) host=$(hostname) ==="
"${ROOT}/.venv/bin/python" -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
"${ROOT}/.venv/bin/python" scripts/train_r5_p10_camelyon_toolbox.py "${ARGS[@]}" --recipe "${RECIPE:-current}" \
  --skip_done --num_workers 6
echo "=== done $(date) ==="
