#!/usr/bin/env python3
"""Phase 13 features and Phase 6 cross-domain eval for the Phase 3 runs added after p3.JOBS was fixed
(λ ∈ {0.25, 1}, seeds 72/82), via the unchanged extract_phase13_features / eval_final_gpu code.

Usage: r4_extra_phase3_runs.py {p13,xfer} IDX   (IDX 0..3 indexes EXTRA)
"""
import sys

import train_phase3_sweep as p3

EXTRA = [(0.25, 72), (0.25, 82), (1.0, 72), (1.0, 82)]


def main():
    which, idx = sys.argv[1], int(sys.argv[2])
    lam, seed = EXTRA[idx]
    if which == "p13":
        import extract_phase13_features as m
        p3.JOBS.append((lam, seed))
        sys.argv = [sys.argv[0], "--task_id", str(len(p3.JOBS) - 1), "--skip_done", "--cpu",
                    "--num_workers", "4", "--batch_size", "32"]
    elif which == "xfer":
        import eval_final_gpu as m
        m.JOBS.append({"kind": "csg", "lambda_adv": lam, "seed": seed})
        sys.argv = [sys.argv[0], "--task_id", str(len(m.JOBS) - 1), "--skip_done", "--cpu", "--num_workers", "4"]
    else:
        raise SystemExit("which must be p13 or xfer")
    m.main()


if __name__ == "__main__":
    main()
