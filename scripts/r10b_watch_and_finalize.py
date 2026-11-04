#!/usr/bin/env python3
"""Poll R10b B1–B4 outputs; commit+push each as it appears; write REPORT.md when all done."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
OUT = ROOT / "results" / "paperB" / "r10b"
ITEMS = {
    "B1": {
        "marker": OUT / "B1_shift50_boots.json",
        "paths": [
            "results/paperB/r10b/B1_shift50_boots.json",
            "results/paperB/r10b/B1_shift50_boots.md",
        ],
        "msg": "Paper B R10b B1: R1 shift 50% bootstrap CIs (λ ∈ {0.5, 2}, all OOD incl. SCIN).",
    },
    "B2": {
        "marker": OUT / "B2_scin_boots.json",
        "paths": [
            "results/paperB/r10b/B2_scin_boots.json",
            "results/paperB/r10b/B2_scin_boots.md",
            "results/paperB/r6/r2/cur/REPORT_SCIN.md",
            "results/paperB/r6/r2/cur/scin_report.json",
        ],
        "msg": "Paper B R10b B2: R2 SCIN REPORT with PREREGISTER bootstrap CIs.",
    },
    "B3": {
        "marker": OUT / "B3_a4_ci.json",
        "paths": [
            "results/paperB/r10b/B3_a4_ci.json",
            "results/paperB/r10b/B3_a4_ci.md",
            "results/paperB/r10b/C3_unseen_with_ci.json",
        ],
        "msg": "Paper B R10b B3: A4 tab:unseen pooled CI / k / Rule A.",
    },
    "B4": {
        "marker": OUT / "B4_splits_scin.json",
        "paths": [
            "results/paperB/r10b/B4_splits_scin.json",
            "results/paperB/r10b/B4_splits_scin.md",
        ],
        "msg": "Paper B R10b B4: SCIN Mahalanobis on tab:splits (7/123/2027 × λ ∈ {0,2}).",
    },
}


def commit_push(msg: str, paths: list[str]) -> str:
    existing = [p for p in paths if (ROOT / p).exists()]
    if not existing:
        return ""
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "r10b_commit_push.sh"), msg, *existing],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, flush=True)
        raise RuntimeError(f"commit failed: {r.stderr}")
    # first line of git log -1 is hash
    for line in r.stdout.splitlines():
        if len(line) >= 40 and all(c in "0123456789abcdef" for c in line[:40]):
            return line.split()[0]
    return "unknown"


def latex_block(path: Path) -> str:
    if not path.exists():
        return "_(missing)_"
    t = path.read_text()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            return parts[1].strip()
    return t.strip()


def write_report(hashes: dict[str, str]) -> None:
    b1 = json.loads((OUT / "B1_shift50_boots.json").read_text()) if (OUT / "B1_shift50_boots.json").exists() else {}
    b2 = json.loads((OUT / "B2_scin_boots.json").read_text()) if (OUT / "B2_scin_boots.json").exists() else {}
    b3 = json.loads((OUT / "B3_a4_ci.json").read_text()) if (OUT / "B3_a4_ci.json").exists() else {}
    b4 = json.loads((OUT / "B4_splits_scin.json").read_text()) if (OUT / "B4_splits_scin.json").exists() else {}
    lines = [
        "# Paper B R10b — bootstrap CIs + table fills (CPU)",
        "",
        "Deadline 10 Nov 2026. Branch `paperb/writing-r6`. No `.tex` edits. No training.",
        "Commits authored `vinhtranduc` only. PREREGISTER: 2,000 draws, rng 20261005.",
        "Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.",
        "",
        "## Commit hashes",
        "",
        "| item | hash | CPU hours |",
        "|---|---|---:|",
        f"| B1 | `{hashes.get('B1','')}` | {b1.get('cpu_hours', float('nan')):.3f} |",
        f"| B2 | `{hashes.get('B2','')}` | {b2.get('cpu_hours', float('nan')):.3f} |",
        f"| B3 | `{hashes.get('B3','')}` | {b3.get('cpu_hours', float('nan')):.3f} |",
        f"| B4 | `{hashes.get('B4','')}` | {b4.get('cpu_hours', float('nan')):.3f} |",
        f"| REPORT | `{hashes.get('REPORT','')}` | — |",
        "",
        f"**Total CPU hours (sum of item timers):** "
        f"{sum(x.get('cpu_hours', 0) for x in (b1, b2, b3, b4)):.3f}",
        "",
        "## B1 — R1 shift 50% bootstrap",
        "",
        b1.get("e1", ""),
        "",
        "Seed lists checked against `{42,52,62,72,82}`:",
        "```",
        json.dumps(b1.get("seed_lists", {}), indent=1),
        "```",
        "",
        "LaTeX rows (shift 50% cells):",
        "```",
        "\n".join(b1.get("latex", [])),
        "```",
        "",
        "Old → new: cells previously had means only / `—` CIs under R10 `--no_boot`; "
        "Rule A column newly filled. See `B1_shift50_boots.md` for the table.",
        "",
        "## B2 — R2 SCIN with bootstrap",
        "",
        f"Regenerated `results/paperB/r6/r2/cur/REPORT_SCIN.md` (commit at run: `{b2.get('commit','')}`).",
        "Old → new: Maha k<0.5 and rule A columns were `—` under `--no_boot`; now filled.",
        "",
        "## B3 — A4 rows of tab:unseen",
        "",
        "LaTeX:",
        "```",
        "\n".join(b3.get("latex", [])),
        "```",
        "",
        "Old → new: means-only A4 rows from R10 C3 gain pooled CI, k, Rule A.",
        "",
        "## B4 — SCIN in tab:splits",
        "",
        f"OOD unit: `{b4.get('unit_ood', 'P7a group column')}` (same as SCIN elsewhere / REPORT_SCIN).",
        "",
        "LaTeX:",
        "```",
        "\n".join(b4.get("latex", [])),
        "```",
        "",
        "Old → new: SCIN absent from split-stability table cells → filled for splits "
        "{7,123,2027} × λ ∈ {0,2}.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("wrote", OUT / "REPORT.md", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = {k: False for k in ITEMS}
    hashes: dict[str, str] = {}
    state = OUT / "_watch_state.json"
    if state.exists():
        prev = json.loads(state.read_text())
        done.update(prev.get("done", {}))
        hashes.update(prev.get("hashes", {}))
    print("watching", list(ITEMS), flush=True)
    deadline = time.time() + 14 * 3600
    while time.time() < deadline:
        for name, spec in ITEMS.items():
            if done[name]:
                continue
            if not spec["marker"].exists():
                continue
            # wait a moment for companion md to flush
            time.sleep(2)
            print(f"committing {name}…", flush=True)
            h = commit_push(spec["msg"], spec["paths"])
            hashes[name] = h
            done[name] = True
            state.write_text(json.dumps({"done": done, "hashes": hashes}, indent=1) + "\n")
            print(f"done {name} -> {h}", flush=True)
        if all(done.values()):
            write_report(hashes)
            h = commit_push(
                "Paper B R10b: consolidated REPORT.md (B1–B4 bootstrap CIs + table fills).",
                ["results/paperB/r10b/REPORT.md", "results/paperB/r10b/_watch_state.json"],
            )
            hashes["REPORT"] = h
            write_report(hashes)  # refresh REPORT hash line
            commit_push(
                "Paper B R10b: REPORT.md commit-hash self-update.",
                ["results/paperB/r10b/REPORT.md"],
            )
            print("ALL R10b items finalized", flush=True)
            return
        time.sleep(60)
    raise SystemExit("timeout waiting for B1–B4")


if __name__ == "__main__":
    main()
