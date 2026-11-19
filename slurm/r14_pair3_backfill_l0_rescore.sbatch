#!/bin/bash
#SBATCH --job-name=r14-bf-score
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --qos=vip2
#SBATCH --output=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_backfill_rescore_%j.out
#SBATCH --error=/data2/cmdir/home/toandq/CSG-Skin-paperB/logs/r14_pair3_backfill_rescore_%j.err
set -euo pipefail
ROOT=/data2/cmdir/home/toandq/CSG-Skin-paperB
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${ROOT}/scripts
.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available())"
# 1) backfill missing OOD features for λ0 s42 redo
.venv/bin/python <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import r6_common as R
import r14_pair3_common as C
import r14_pair3_train as T
import torch

tag = "r14_ladv0_prior100_s42"
feat_dir = C.OUT / "features" / tag
ckpt = C.CKPT / tag / "last.ckpt"
need = ("pad_ufes20", "scin")
missing = [n for n in need if not (feat_dir / f"{n}.npz").exists()]
print("missing", missing, flush=True)
if missing:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = R.load_net(ckpt, device)
    data = C.load_data()
    T.extract_all(net, data, feat_dir, num_workers=4, device=device, light=False)
    print("backfill done", sorted(p.name for p in feat_dir.glob("*.npz")), flush=True)
else:
    print("already complete", flush=True)
PY
# 2) full rescore all 20 tags → CELLS/RESULTS/REPORT base; then we patch REPORT
.venv/bin/python scripts/r14_pair3_score.py
echo DONE_BACKFILL_RESCORE $(date)
