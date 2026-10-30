#!/bin/bash
#SBATCH --job-name=paperB-r7-finalize
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7finalize_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r7finalize_%j.err
set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
cd "${ROOT}"
"${ROOT}/.venv/bin/python" scripts/r7_finalize.py
# commit if anything new (author vinhtranduc)
export GIT_AUTHOR_NAME='vinhtranduc' GIT_AUTHOR_EMAIL='vinhtran9969@gmail.com'
export GIT_COMMITTER_NAME='vinhtranduc' GIT_COMMITTER_EMAIL='vinhtran9969@gmail.com'
git add results/paperB/r7/ || true
if ! git diff --cached --quiet; then
  TREE=$(git write-tree)
  PARENT=$(git rev-parse HEAD)
  COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "$(cat <<'EOF'
Paper B R7: per-item reports and consolidated REPORT.md after GPU runs.
EOF
)")
  git reset --soft "$COMMIT"
  git push origin paperb/writing-r6 || true
  echo "committed $(git rev-parse --short HEAD)"
fi
