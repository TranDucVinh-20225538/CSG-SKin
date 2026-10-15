#!/bin/bash
#SBATCH --job-name=paperB-pack-test
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/pack_test_%x_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/pack_test_%x_%j.err
# Packing test (decisions/paperB_r5_pack_test.md). ARM (env): solo (submit with --cpus-per-task=8) or packed (16).
# Each arm: deterministic 2-epoch run(s), then normal-mode 3-epoch run(s). packed = seeds 42 and 52 at the same time.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python scripts/r5_pack_test.py --num_workers 6"
echo "=== pack test ${ARM} start $(date) host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "${ARM}" == "solo" ]]; then
  ${PY} --arm det_solo --seed 42 --epochs 2 --deterministic
  ${PY} --arm speed_solo --seed 42 --epochs 3
else
  ${PY} --arm det_packed --seed 42 --epochs 2 --deterministic &
  ${PY} --arm det_packed --seed 52 --epochs 2 --deterministic &
  wait
  ${PY} --arm speed_packed --seed 42 --epochs 3 &
  ${PY} --arm speed_packed --seed 52 --epochs 3 &
  wait
fi
echo "=== done $(date) ==="
