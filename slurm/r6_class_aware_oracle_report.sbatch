#!/bin/bash
#SBATCH --job-name=pB-clsAwareOra
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_class_aware_ora_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r6_class_aware_ora_%j.err
# After oracle s82 (68562_14): probe s82, write oracle-complete REPORT, commit+push.

set -euo pipefail
ROOT="/data2/cmdir/home/toandq/CSG-Skin-paperB"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
export CUDA_VISIBLE_DEVICES=""
cd "${ROOT}"
PY="${ROOT}/.venv/bin/python"
GIT=/usr/bin/git
AUTHOR='vinhtranduc <vinhtran9969@gmail.com>'
echo "=== class-aware oracle-complete $(date) commit=$(${GIT} rev-parse --short HEAD) ==="
# wait briefly for feature extract of s82 if train just finished
for i in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -f results/paperB/r6/r2/cur/features/r2_cur_iwdan_oracle_ladv2_s82/isic_test.npz ]]; then
    break
  fi
  echo "waiting for oracle s82 features… (${i})"
  sleep 60
done
${PY} scripts/r6_class_aware_heldout_probes.py --require_oracle_complete

${GIT} add decisions/paperB_r6_class_aware_heldout.md \
  scripts/r6_class_aware_heldout_probes.py \
  slurm/r6_class_aware_heldout.sbatch \
  slurm/r6_class_aware_oracle_report.sbatch \
  results/paperB/r6/class_aware_heldout/
if ! ${GIT} diff --cached --quiet; then
  ${GIT} commit --author="${AUTHOR}" -m "$(cat <<'EOF'
Paper B: class-aware held-out probes REPORT (oracle λ=2 complete).

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
p = Path("results/paperB/r6/class_aware_heldout/REPORT.md")
if p.exists():
    t = p.read_text()
    stamp = "**On origin/paperb/reviewer-r1:** `{}`.\n".format(os.environ["TIP"])
    t = re.sub(r"\*\*On origin/paperb/reviewer-r1:\*\*.*\n", stamp, t)
    p.write_text(t)
PY
${GIT} add results/paperB/r6/class_aware_heldout/REPORT.md
if ! ${GIT} diff --cached --quiet; then
  ${GIT} commit --author="${AUTHOR}" -m "$(cat <<'EOF'
Paper B class-aware heldout REPORT: stamp origin hash.

EOF
)"
  ${GIT} push origin HEAD || true
fi
echo "=== oracle-complete report tip=$(${GIT} rev-parse --short HEAD) $(date) ==="
