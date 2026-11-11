# GPU estimate (before / during submit)

Basis: R11 dual CURRENT ~1.0–1.4 h/run; R12 live measure ~3.3 it/s × 506 steps → ~2.0–2.3 h/run + feature extract.
Jobs: 45 train (35 Block A + 10 Block B). Array `%5` but `QOSMaxGRESPerUser` often allows only 1–2 concurrent with other vip2 jobs.
Calendar: ~45×2.2 h ≈ **~4 days** at 1 GPU; shorter when a second GRES frees.
No paid resources. qos=vip2, partition=defq, gres=gpu:1, mem=32G, **time=04:00:00** per train task.
Actual wall times recorded in each `runs/*/summary.json` (`wall_seconds`) — no guessed GPU-hour totals in REPORT.
