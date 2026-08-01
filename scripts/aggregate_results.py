#!/usr/bin/env python3
"""Aggregate run metrics into a full table (per-run + mean+-std)."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate summary/leakage json into a full report table.")
    p.add_argument(
        "--summary_inputs",
        type=str,
        default="",
        help="Comma-separated summary.json paths.",
    )
    p.add_argument(
        "--leakage_inputs",
        type=str,
        default="",
        help="Comma-separated leakage.json paths (same order as summary_inputs).",
    )
    p.add_argument(
        "--inputs",
        type=str,
        default="",
        help="Backward-compatible alias for --summary_inputs.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="experiment",
        help="Group label for this aggregation row.",
    )
    p.add_argument(
        "--output_md",
        type=Path,
        default=Path("results/aggregate_result.md"),
        help="Markdown output path.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/aggregate_result.json"),
        help="JSON output path.",
    )
    p.add_argument(
        "--append_csv",
        type=Path,
        default=None,
        help="Optional CSV file to append one aggregated row: Method,Acc,Bal Acc,ECE,Leakage",
    )
    return p.parse_args()


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def fmt(v_mean, v_std):
    return "{:.4f} +- {:.4f}".format(v_mean, v_std)


def parse_paths(text):
    return [Path(x.strip()) for x in str(text).split(",") if x.strip()]


def load_json_rows(paths):
    rows = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError("Missing file: {}".format(p))
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def metric_mean_std(rows, key):
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return None
    return mean_std(vals)


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    args = parse_args()
    summary_paths = parse_paths(args.summary_inputs or args.inputs)
    leakage_paths = parse_paths(args.leakage_inputs)
    if not summary_paths and not leakage_paths:
        raise ValueError("Provide --summary_inputs and/or --leakage_inputs.")

    summary_rows = load_json_rows(summary_paths) if summary_paths else []
    leakage_rows = load_json_rows(leakage_paths) if leakage_paths else []
    n = max(len(summary_rows), len(leakage_rows))
    if summary_rows and leakage_rows and len(summary_rows) != len(leakage_rows):
        raise ValueError("summary_inputs and leakage_inputs must have same length/order.")

    headers = [
        "run",
        "seed",
        "id_acc",
        "id_bal_acc",
        "id_ece",
        "maha_auroc",
        "maha_fpr95",
        "z_lesion_probe",
        "z_lesion_probe_std",
    ]
    run_rows = []
    merged_rows = []
    for i in range(n):
        s = summary_rows[i] if i < len(summary_rows) else {}
        l = leakage_rows[i] if i < len(leakage_rows) else {}
        merged = {}
        merged.update(s)
        merged.update(l)
        merged_rows.append(merged)
        run_rows.append(
            [
                str(merged.get("run_name", "run_{}".format(i + 1))),
                str(merged.get("seed", "-")),
                "{:.4f}".format(float(merged["id_acc"])) if "id_acc" in merged else "-",
                "{:.4f}".format(float(merged["id_balanced_acc"])) if "id_balanced_acc" in merged else "-",
                "{:.4f}".format(float(merged["id_ece"])) if "id_ece" in merged else "-",
                "{:.4f}".format(float(merged["ood_maha_auroc"])) if "ood_maha_auroc" in merged else "-",
                "{:.4f}".format(float(merged["ood_maha_fpr95"])) if "ood_maha_fpr95" in merged else "-",
                "{:.4f}".format(float(merged["z_lesion_acc_mean"])) if "z_lesion_acc_mean" in merged else "-",
                "{:.4f}".format(float(merged["z_lesion_acc_std"])) if "z_lesion_acc_std" in merged else "-",
            ]
        )

    agg_keys = [
        ("id_acc", "id_acc"),
        ("id_balanced_acc", "id_bal_acc"),
        ("id_ece", "id_ece"),
        ("ood_maha_auroc", "maha_auroc"),
        ("ood_maha_fpr95", "maha_fpr95"),
        ("z_lesion_acc_mean", "z_lesion_probe"),
        ("z_lesion_acc_std", "z_lesion_probe_std"),
    ]
    summary_out = {"label": args.label, "n_runs": n, "runs": merged_rows, "aggregate": {}}
    agg_row = ["{} (mean+-std)".format(args.label), "-", "-", "-", "-", "-", "-", "-", "-"]
    for src_key, _disp in agg_keys:
        ms = metric_mean_std(merged_rows, src_key)
        if ms is None:
            continue
        m, s = ms
        summary_out["aggregate"][src_key] = {"mean": m, "std": s}
        col_idx = {
            "id_acc": 2,
            "id_balanced_acc": 3,
            "id_ece": 4,
            "ood_maha_auroc": 5,
            "ood_maha_fpr95": 6,
            "z_lesion_acc_mean": 7,
            "z_lesion_acc_std": 8,
        }[src_key]
        agg_row[col_idx] = fmt(m, s)
        print("{:<22} {}".format(src_key, fmt(m, s)))

    md = []
    md.append("## Aggregated Results: {} (n={})".format(args.label, n))
    md.append("")
    md.append(md_table(headers, run_rows + [agg_row]))
    md.append("")
    md_text = "\n".join(md) + "\n"

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text, encoding="utf-8")
    args.output_json.write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8")
    print("Saved markdown: {}".format(args.output_md))
    print("Saved json: {}".format(args.output_json))

    if args.append_csv is not None:
        agg = summary_out["aggregate"]
        def _pair(key):
            if key not in agg:
                return "-"
            return fmt(float(agg[key]["mean"]), float(agg[key]["std"]))

        csv_row = "{},{},{},{},{}".format(
            "{} (mean +- std, n={})".format(args.label, n),
            _pair("id_acc"),
            _pair("id_balanced_acc"),
            _pair("id_ece"),
            _pair("z_lesion_acc_mean"),
        )
        need_header = not args.append_csv.exists()
        args.append_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.append_csv.open("a", encoding="utf-8") as f:
            if need_header:
                f.write("Method,Acc,Bal Acc,ECE,Leakage\n")
            f.write(csv_row + "\n")
        print("Appended CSV row: {}".format(args.append_csv))


if __name__ == "__main__":
    main()

