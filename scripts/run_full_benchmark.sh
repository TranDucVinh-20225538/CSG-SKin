#!/usr/bin/env bash
set -euo pipefail

# Full benchmark runner:
# - Train 3 seeds per method
# - Run leakage check per checkpoint
# - Aggregate per method
# - Append to results_final_v1.csv
#
# Methods:
#   1) baseline_soft
#   2) runA_grl
#   3) runB
#
# Notes:
# - Edit *_EXTRA_FLAGS below if you want to tweak configs.
# - Default metadata is soft lesion-only csv.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(42 52 62)
METADATA="${METADATA:-data/master_metadata_lesion_only_soft.csv}"
FINAL_CSV="${FINAL_CSV:-results_final_v1.csv}"

# Optional: set DRY_RUN=1 to print commands only.
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    echo "[RUN] $*"
    eval "$@"
  fi
}

join_by_comma() {
  local IFS=","
  echo "$*"
}

find_best_ckpt() {
  local dir="$1"
  local ckpt
  ckpt="$(ls -1t "$dir"/best-*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "ERROR: no best-*.ckpt in ${dir}" >&2
    exit 1
  fi
  echo "$ckpt"
}

train_and_probe_csg() {
  local method="$1"
  local csg_flags="$2"
  local -a summary_paths=()
  local -a leakage_paths=()

  for seed in "${SEEDS[@]}"; do
    local run_name="${method}_s${seed}"
    local run_dir="results/csg_lite/${run_name}"
    local ckpt_dir="checkpoints/csg_lite/${run_name}"
    mkdir -p "$run_dir"

    run_cmd "python scripts/train_csg.py \
      --metadata ${METADATA} \
      --backbone_variant b3 \
      --max_epochs 40 \
      --lr 1e-4 \
      --seed ${seed} \
      --run_name ${run_name} \
      ${csg_flags}"

    local best_ckpt
    best_ckpt="$(find_best_ckpt "$ckpt_dir")"

    run_cmd "python scripts/check_leakage.py \
      --metadata ${METADATA} \
      --ckpt ${best_ckpt} \
      --model_type csg \
      --probe_seeds 42,52,62 \
      --shuffle_test_seed 123 \
      --output_json ${run_dir}/leakage.json"

    summary_paths+=("${run_dir}/summary.json")
    leakage_paths+=("${run_dir}/leakage.json")
  done

  local summary_csv leakage_csv
  summary_csv="$(join_by_comma "${summary_paths[@]}")"
  leakage_csv="$(join_by_comma "${leakage_paths[@]}")"
  run_cmd "python scripts/aggregate_results.py \
    --label ${method} \
    --summary_inputs ${summary_csv} \
    --leakage_inputs ${leakage_csv} \
    --output_md results/table1_${method}_aggregate.md \
    --output_json results/table1_${method}_aggregate.json \
    --append_csv ${FINAL_CSV}"
}

train_and_probe_baseline() {
  local method="$1"
  local baseline_flags="$2"
  local -a summary_paths=()
  local -a leakage_paths=()

  for seed in "${SEEDS[@]}"; do
    local run_name="${method}_s${seed}"
    local run_dir="results/baseline/${run_name}"
    local ckpt_dir="checkpoints/baseline/${run_name}"
    mkdir -p "$run_dir"

    run_cmd "python scripts/train_baseline.py \
      --metadata ${METADATA} \
      --max_epochs 40 \
      --lr 2e-4 \
      --seed ${seed} \
      --run_name ${run_name} \
      ${baseline_flags}"

    local best_ckpt
    best_ckpt="$(find_best_ckpt "$ckpt_dir")"

    run_cmd "python scripts/check_leakage.py \
      --metadata ${METADATA} \
      --ckpt ${best_ckpt} \
      --model_type baseline \
      --probe_seeds 42,52,62 \
      --shuffle_test_seed 123 \
      --output_json ${run_dir}/leakage.json"

    summary_paths+=("${run_dir}/summary.json")
    leakage_paths+=("${run_dir}/leakage.json")
  done

  local summary_csv leakage_csv
  summary_csv="$(join_by_comma "${summary_paths[@]}")"
  leakage_csv="$(join_by_comma "${leakage_paths[@]}")"
  run_cmd "python scripts/aggregate_results.py \
    --label ${method} \
    --summary_inputs ${summary_csv} \
    --leakage_inputs ${leakage_csv} \
    --output_md results/table1_${method}_aggregate.md \
    --output_json results/table1_${method}_aggregate.json \
    --append_csv ${FINAL_CSV}"
}

echo "Preparing fresh ${FINAL_CSV}"
if [[ "$DRY_RUN" != "1" ]]; then
  printf "Method,Acc,Bal Acc,ECE,Leakage\n" > "${FINAL_CSV}"
fi

# ========== Config presets ==========
# Baseline soft: lighter transforms.
BASELINE_SOFT_FLAGS="--no_robust_transforms"

# Run A / GRL preset (adjust if your canonical Run A differs).
RUNA_GRL_FLAGS="--lambda_adv 2.0 --lambda_orth 0.0 --lambda_supcon 0.0"

# Run B preset (current best balance in your experiments).
RUNB_FLAGS="--lambda_adv 2.0 --lambda_orth 5.0 --lambda_supcon 0.0"

echo "=== 1) baseline_soft ==="
train_and_probe_baseline "baseline_soft" "${BASELINE_SOFT_FLAGS}"

echo "=== 2) runA_grl ==="
train_and_probe_csg "runA_grl" "${RUNA_GRL_FLAGS}"

echo "=== 3) runB ==="
train_and_probe_csg "runB" "${RUNB_FLAGS}"

echo "Done. Final frozen table: ${FINAL_CSV}"
