# R12 live status

Updated: 2026-11-09T09:35:41

| item | value |
|---|---|
| Train array | `73220` (0–44%5) |
| Score job | `73221` (afterok:73220) |
| Timelimit | train 4 h / score 6 h |
| summaries | 0/45 |
| Monitor | 2h AGENT_LOOP_TICK_r12 |

```
             JOBID ST       TIME NAME
    73220_[1-44%5] PD       0:00 paperB-r12
             73221 PD       0:00 paperB-r12sc
             73190 PD       0:00 seg2_pmet
           73220_0  R       3:57 paperB-r12
             73189  R   12:44:58 seg2_proxy

```

Pilot OK. Manifest frozen (35 A + 10 B). Far-OOD unavailable.
Do not inspect AUROC to retune. REPORT.md after score job.
