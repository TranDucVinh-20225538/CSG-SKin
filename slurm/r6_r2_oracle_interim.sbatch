#!/bin/bash
#SBATCH --job-name=pB-r2-oraL2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r2_oracle_l2_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6r2_oracle_l2_%j.err
# Interim oracle λ=2 send: report + commit + push (vinhtranduc).

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
GIT=/usr/bin/git
AUTHOR='vinhtranduc <vinhtran9969@gmail.com>'
echo "=== R2 oracle λ=2 interim $(date) commit=$(${GIT} rev-parse --short HEAD) ==="
${PY} scripts/r6_r2_oracle_l2_interim.py

${GIT} add results/paperB/r6/r2/cur/REPORT_ORACLE_L2_INTERIM.md \
  results/paperB/r6/r2/cur/oracle_l2_interim.json
if ! ${GIT} diff --cached --quiet; then
  ${GIT} commit --author="${AUTHOR}" -m "$(cat <<'EOF'
Paper B R2 interim: oracle λ=2 vs IWDAN vs R1 DANN (gap recovered).

EOF
)"
  ${GIT} pull --rebase origin paperb/reviewer-r1 || true
  ${GIT} push origin HEAD || true
fi

TIP=$(${GIT} rev-parse --short HEAD)
export TIP
${PY} - <<'PY'
import os, re
from pathlib import Path
p = Path("results/paperB/r6/r2/cur/REPORT_ORACLE_L2_INTERIM.md")
t = p.read_text()
stamp = "**On origin/paperb/reviewer-r1:** `{}`.\n".format(os.environ["TIP"])
if "On origin/paperb/reviewer-r1" in t:
    t = re.sub(r"\*\*On origin/paperb/reviewer-r1:\*\*.*\n", stamp, t)
else:
    t = t.rstrip() + "\n\n" + stamp
p.write_text(t)
PY
${GIT} add results/paperB/r6/r2/cur/REPORT_ORACLE_L2_INTERIM.md
if ! ${GIT} diff --cached --quiet; then
  ${GIT} commit --author="${AUTHOR}" -m "$(cat <<'EOF'
Paper B R2 interim REPORT: stamp origin hash.

EOF
)"
  ${GIT} push origin HEAD || true
fi
echo "=== interim done tip=$(${GIT} rev-parse --short HEAD) $(date) ==="
