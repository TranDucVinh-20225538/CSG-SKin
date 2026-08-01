#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
METADATA="${METADATA:-data/master_metadata_lesion_only_soft.csv}"
SEEDS=(42 52 62 72 82)
PROBE_SEEDS="42,52,62,72,82"

mkdir -p "results/cbm_revision" "results/cbm_revision/figures" "results/cbm_revision/per_seed"

find_best_ckpt() {
  local d="$1"
  local ckpt=""
  ckpt="$(ls -1t "$d"/best-*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$ckpt" ]]; then
    ckpt="$(ls -1t "$d"/last*.ckpt 2>/dev/null | head -n 1 || true)"
  fi
  echo "$ckpt"
}

run_baseline_seed() {
  local seed="$1"
  local run="baseline_soft_s${seed}"
  local summary="results/baseline/${run}/summary.json"
  local leakage="results/baseline/${run}/leakage.json"
  local ckpt_dir="checkpoints/baseline/${run}"

  if [[ ! -f "$summary" ]]; then
    echo "[baseline][seed=${seed}] training..."
    "$PYTHON_BIN" "scripts/train_baseline.py" \
      --metadata "$METADATA" \
      --max_epochs 40 \
      --lr 2e-4 \
      --seed "$seed" \
      --run_name "$run" \
      --no_robust_transforms
  else
    echo "[baseline][seed=${seed}] summary exists, skip train."
  fi

  if [[ ! -f "$leakage" ]]; then
    local ckpt
    ckpt="$(find_best_ckpt "$ckpt_dir")"
    if [[ -z "$ckpt" ]]; then
      echo "ERROR: no checkpoint found in $ckpt_dir" >&2
      exit 1
    fi
    echo "[baseline][seed=${seed}] leakage..."
    "$PYTHON_BIN" "scripts/check_leakage.py" \
      --metadata "$METADATA" \
      --ckpt "$ckpt" \
      --model_type baseline \
      --probe_seeds "$PROBE_SEEDS" \
      --shuffle_test_seed 123 \
      --output_json "$leakage"
  else
    echo "[baseline][seed=${seed}] leakage exists, skip."
  fi
}

run_csg_seed() {
  local seed="$1"
  local method="$2"
  local flags="$3"
  local run="${method}_s${seed}"
  local summary="results/csg_lite/${run}/summary.json"
  local leakage="results/csg_lite/${run}/leakage.json"
  local ckpt_dir="checkpoints/csg_lite/${run}"

  if [[ ! -f "$summary" ]]; then
    echo "[${method}][seed=${seed}] training..."
    "$PYTHON_BIN" "scripts/train_csg.py" \
      --metadata "$METADATA" \
      --backbone_variant b3 \
      --max_epochs 40 \
      --lr 1e-4 \
      --seed "$seed" \
      --run_name "$run" \
      $flags
  else
    echo "[${method}][seed=${seed}] summary exists, skip train."
  fi

  if [[ ! -f "$leakage" ]]; then
    local ckpt
    ckpt="$(find_best_ckpt "$ckpt_dir")"
    if [[ -z "$ckpt" ]]; then
      echo "ERROR: no checkpoint found in $ckpt_dir" >&2
      exit 1
    fi
    echo "[${method}][seed=${seed}] leakage..."
    "$PYTHON_BIN" "scripts/check_leakage.py" \
      --metadata "$METADATA" \
      --ckpt "$ckpt" \
      --model_type csg \
      --probe_seeds "$PROBE_SEEDS" \
      --shuffle_test_seed 123 \
      --output_json "$leakage"
  else
    echo "[${method}][seed=${seed}] leakage exists, skip."
  fi
}

aggregate_method() {
  local label="$1"
  local run_prefix="$2"
  local mode="$3" # baseline|csg
  local -a summaries=()
  local -a leakages=()
  local base_dir="results/${mode}"
  if [[ "$mode" == "csg" ]]; then
    base_dir="results/csg_lite"
  fi
  for s in "${SEEDS[@]}"; do
    summaries+=("${base_dir}/${run_prefix}_s${s}/summary.json")
    leakages+=("${base_dir}/${run_prefix}_s${s}/leakage.json")
  done
  local summary_csv leakage_csv
  summary_csv="$(IFS=,; echo "${summaries[*]}")"
  leakage_csv="$(IFS=,; echo "${leakages[*]}")"

  "$PYTHON_BIN" "scripts/aggregate_results.py" \
    --label "$label" \
    --summary_inputs "$summary_csv" \
    --leakage_inputs "$leakage_csv" \
    --output_md "results/cbm_revision/${run_prefix}_n5.md" \
    --output_json "results/cbm_revision/${run_prefix}_n5.json"
}

echo "=== CBM revision one-shot start ==="
echo "Python: $PYTHON_BIN"
echo "Metadata: $METADATA"

for seed in "${SEEDS[@]}"; do
  run_baseline_seed "$seed"
done
for seed in "${SEEDS[@]}"; do
  run_csg_seed "$seed" "runA_grl" "--lambda_adv 2.0 --lambda_orth 0.0 --lambda_supcon 0.0"
done
for seed in "${SEEDS[@]}"; do
  run_csg_seed "$seed" "runB_orth1" "--lambda_adv 2.0 --lambda_orth 1.0 --lambda_supcon 0.0"
done

echo "=== Aggregate n=5 ==="
aggregate_method "Baseline Soft" "baseline_soft" "baseline"
aggregate_method "Run A / GRL" "runA_grl" "csg"
aggregate_method "Run B (orth=1.0)" "runB_orth1" "csg"

echo "=== Build table_main.csv ==="
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import csv

root = Path("results/cbm_revision")
inputs = [
    ("Baseline Soft", root / "baseline_soft_n5.json"),
    ("Run A / GRL", root / "runA_grl_n5.json"),
    ("Run B (orth=1.0)", root / "runB_orth1_n5.json"),
]
out = root / "table_main.csv"
rows = []
for method, fp in inputs:
    obj = json.loads(fp.read_text(encoding="utf-8"))
    agg = obj["aggregate"]
    rows.append({
        "Method": method,
        "n": obj.get("n_runs", 5),
        "Acc_mean": agg["id_acc"]["mean"],
        "Acc_std": agg["id_acc"]["std"],
        "BAcc_mean": agg["id_balanced_acc"]["mean"],
        "BAcc_std": agg["id_balanced_acc"]["std"],
        "ECE_mean": agg["id_ece"]["mean"],
        "ECE_std": agg["id_ece"]["std"],
        "Leak_mean": agg["z_lesion_acc_mean"]["mean"],
        "Leak_std": agg["z_lesion_acc_mean"]["std"],
    })

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["Method","n","Acc_mean","Acc_std","BAcc_mean","BAcc_std","ECE_mean","ECE_std","Leak_mean","Leak_std"],
    )
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Saved: {out}")
PY

if [[ -f "cbm_revision/scripts/eval_ood_benchmarks.py" ]]; then
  echo "=== Running extended OOD benchmarks ==="
  "$PYTHON_BIN" "cbm_revision/scripts/eval_ood_benchmarks.py"
else
  echo "[INFO] skip Task A extended OOD: cbm_revision/scripts/eval_ood_benchmarks.py not found yet."
fi

if [[ -f "cbm_revision/scripts/stat_tests_cbm.py" ]]; then
  "$PYTHON_BIN" "cbm_revision/scripts/stat_tests_cbm.py"
else
  echo "[INFO] skip Task D stats: cbm_revision/scripts/stat_tests_cbm.py not found yet."
fi

if [[ -f "cbm_revision/scripts/plot_cbm_figures.py" ]]; then
  "$PYTHON_BIN" "cbm_revision/scripts/plot_cbm_figures.py"
else
  echo "[INFO] skip Task E cbm figures: cbm_revision/scripts/plot_cbm_figures.py not found yet."
fi

if [[ -f "cbm_revision/scripts/write_final_summary.py" ]]; then
  "$PYTHON_BIN" "cbm_revision/scripts/write_final_summary.py"
else
  echo "[INFO] skip Task F summary: cbm_revision/scripts/write_final_summary.py not found yet."
fi

if [[ -f "cbm_revision/scripts/run_effb3_control.py" ]]; then
  echo "=== Running EffNet-B3 single-encoder control ==="
  "$PYTHON_BIN" "cbm_revision/scripts/run_effb3_control.py" \
    --metadata "$METADATA" \
    --seeds "$PROBE_SEEDS" \
    --run_train
else
  echo "[INFO] skip EffNet-B3 control: cbm_revision/scripts/run_effb3_control.py not found."
fi

echo "=== Done ==="
echo "Main output: results/cbm_revision/table_main.csv"
