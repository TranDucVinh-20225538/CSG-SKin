# Paper B — environment record (Phase 0.5)

Recorded 2026-09-20 on the HPC login node. This closes manuscript Limitation #6 (unreported compute environment). Live `nvidia-smi` is unavailable on the login node (no NVIDIA driver). GPU identity below is from the 2026-08-01 repository audit of the checkpoint-producing `.venv`, and will be re-confirmed on the first SLURM GPU allocation.

## Scheduler

| Item | Value |
|---|---|
| Scheduler | **SLURM** (`sbatch`/`squeue`/`sinfo` at `/cm/shared/apps/slurm/current/bin`, module `slurm/slurm/21.08.8`) |
| Login node | `bright92` (this session; `SLURM_JOB_ID` unset) |
| Partition | `defq` (default), nodes `node[001,002,004]` |
| Node 001 | 256 CPU, 1.0 TB, `gpu:8`, **drained** |
| Node 002 | 240 CPU (2×120), 850 GB, `gpu:8`, MIXED |
| Node 004 | 240 CPU (2×120), 850 GB, `gpu:8`, MIXED |
| Job defaults | `DefCpuPerGPU=12`, `DefMemPerGPU=96000` |
| Default time | 07:00:00 |

Long Paper B jobs will be submitted as SLURM array/single-GPU jobs with checkpointing. Nothing in Phases 2–4 will be run in a foreground login-node shell.

## Hardware

| Item | Value |
|---|---|
| Login CPU | 2× Intel Xeon Gold 6226R @ 2.90 GHz, 64 logical CPUs, 502 GB RAM |
| OS | Ubuntu 20.04.6 LTS, kernel 5.15.0-75-generic |
| GPU (training / audit) | **1× NVIDIA A100-SXM4-80GB** (from `docs/repository_audit.md`, captured from the original `.venv` on a compute node) |
| GPU (this login session) | `nvidia-smi` failed: no driver on `bright92` |
| Scratch / data disk | `/data2` — 175 T, 137 T used, 39 T free |

## Software — original training environment (checkpoint-producing)

From `CSG-Skin/.venv` as frozen in `requirements-server.txt` and `docs/repository_audit.md` (2026-08-01). **This `.venv` is not runnable by the current user:** its `python` symlink points at `/root/anaconda3/bin/python` (permission denied).

| Package | Version |
|---|---|
| Python | 3.13 (venv layout `lib/python3.13`) |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| pytorch-lightning | 2.6.1 |
| torchmetrics | 1.9.0 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| opencv-python | 4.13.0.92 |
| CUDA (compiled) | 13.0 |

## Software — this agent session (Phase 0 CPU checks)

| Item | Value |
|---|---|
| User | `toandq` (uid 1007; groups `toandq`, `ioit`, `hpcioit`) |
| System python | `/usr/bin/python3` 3.8, sklearn 0.24.1, pandas 2.0.3, numpy 1.24.4, torch 2.4.1+cu121, **no torchvision, no pytorch-lightning** |
| Intended Phase 1 runtime | `/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python` (3.11.5; torch + torchvision present) + a dedicated Paper B venv that adds `pytorch-lightning` / `torchmetrics` so the shared `torch-env` is not mutated |
| Repo `.venv` | unusable (root-owned interpreter) |

Exact `pip freeze` of the Phase 1 job will be written next to the Phase 1 outputs once the SLURM GPU job starts.

## Per-epoch wall time (from existing run artifacts)

Estimated as `mtime(last*.ckpt) − mtime(results/.../config.yaml)` for each 40-epoch CSG run. This is wall time of the whole `Trainer.fit`, including validation, not a profiler number.

| Method | n runs used | Fit wall time | Per-epoch |
|---|---:|---|---|
| `runA_grl` | 5 | 0.89–2.04 h | **1.3–3.1 min** |
| `runB_orth1` | 5 | 1.22–1.42 h | **1.8–2.1 min** |
| `runB` (orth=5) | 3 | 1.07–1.50 h | **1.6–2.2 min** |

**Working figure for budget:** CSG-Lite dual-encoder, EffNet-B3 × 2, batch 32 paired (64 images/step), 40 epochs, 1× A100-80GB ≈ **~2.0 min/epoch**, **~1.3–1.5 h/run**.

Baseline ResNet-50 (`baseline_soft`) consecutive-seed last-ckpt gaps are ~0.2–0.4 h when seeds were run back-to-back (~0.3–0.6 min/epoch), consistent with a single encoder and no PAD pairing.

## Write-location deviation (operational)

`CSG-Skin/` is owned by `anhnv:toandq` with mode `755` / files `644`. User `toandq` cannot create `results/paperB/` or `scripts/` inside the repo. All Paper B artifacts for this session live at:

```
/data2/cmdir/home/toandq/CSG-Skin-paperB/
  results/paperB/     # required outputs
  scripts/            # new scripts (eval_ood_dual_branch.py, …)
```

Checkpoints, data, and existing `results/` are read-only from `/data2/hpcshared/Vinh/CSG-Skin/`. Nothing in `results/`, `results/cbm_revision/`, or `results/effb3_control/` is modified.
