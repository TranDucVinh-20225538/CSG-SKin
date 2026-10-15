#!/bin/bash
#SBATCH --job-name=paperB-r5-frozen-extract
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5frozen_%A_%a.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r5frozen_%A_%a.err
# GPU (light). Paper B R5 P3 (full frames) / P12 / Q5: frozen-encoder features on results/paperB/r5/frozen/frames.
# Needs STEP=frames of slurm/paperB_r5r6_cpu.sbatch first. Encoders by array index; MONET (index 3) is included (user
# decision 2026-10-15, licence noted in decisions/paperB_r5_p12.md); weights prefetched to the Hugging Face cache.
# Q1 CXR frozen monitor (Q5):  sbatch --array=0 --export=ALL,FRAMES_DIR=<Q1 frames dir>,OUT_ROOT=results/paperB/r5/frozen/features_cxr ...

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=${ROOT}/scripts
cd "${ROOT}"
ENCODERS=(imagenet_r50 dinov2_b14 dinov2_l14 monet)
ENC="${ENCODERS[${SLURM_ARRAY_TASK_ID}]}"
FRAMES_DIR="${FRAMES_DIR:-${ROOT}/results/paperB/r5/frozen/frames}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/paperB/r5/frozen/features}"
echo "=== frozen extract ${ENC} start $(date) host=$(hostname) frames=${FRAMES_DIR} ==="
PY="${ROOT}/.venv/bin/python"
# .venv's transformers needs torch >= 2.5 (it has 2.1.1); MONET runs in the benhda env (torch 2.4.1, transformers 4.46)
if [[ "${ENC}" == "monet" ]]; then PY=/data2/cmdir/home/toandq/.conda/envs/benhda/bin/python; fi
"${PY}" scripts/r5_frozen_extract.py --encoder "${ENC}" --frames_dir "${FRAMES_DIR}" --out_root "${OUT_ROOT}"
echo "=== done $(date) ==="
