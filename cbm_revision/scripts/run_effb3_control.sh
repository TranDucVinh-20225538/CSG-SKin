#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
METADATA="${METADATA:-data/master_metadata_lesion_only_soft.csv}"

"$PYTHON_BIN" "cbm_revision/scripts/run_effb3_control.py" \
  --metadata "$METADATA" \
  --seeds "42,52,62,72,82" \
  --run_train

echo "Done: results/effb3_control/"

