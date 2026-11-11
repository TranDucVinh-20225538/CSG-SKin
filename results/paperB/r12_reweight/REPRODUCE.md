# R12 reproduce commands

Repo: `/data2/cmdir/home/toandq/CSG-Skin-paperB`  
Outputs: `results/paperB/r12_reweight/` (non-overwriting)  
Checkpoints: `checkpoints/r12_reweight/`

```bash
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
export PYTHONPATH=/data2/hpcshared/Vinh/CSG-Skin:${PWD}/scripts

# Pilot (CPU) — correctness only
.venv/bin/python scripts/r12_reweight_pilot.py

# Job table (45 train jobs: 35 Block A + 10 Block B)
.venv/bin/python scripts/r12_reweight_train.py --list

# Train array (GPU, qos vip2)
mkdir -p logs
sbatch slurm/r12_reweight_train.sbatch

# After all array tasks finish:
sbatch --dependency=afterok:<TRAIN_JOBID> slurm/r12_reweight_score.sbatch
# or:
.venv/bin/python scripts/r12_reweight_score.py --write_report
```

Single task resume:  
`.venv/bin/python scripts/r12_reweight_train.py --task_id N --skip_done --resume`

Code version / dirty tree: `CODE_VERSION.txt`, `DIRTY_STATUS.txt`, `DIRTY_DIFFSTAT.txt`.  
Frozen design: `decisions/paperB_r12_reweight.md`, `MANIFEST.json`, `ENDPOINTS.md`.
