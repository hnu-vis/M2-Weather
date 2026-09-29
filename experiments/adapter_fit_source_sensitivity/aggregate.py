#!/usr/bin/env python3
"""Aggregate completed residual-source sensitivity runs without inventing missing values."""

from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "adapter_fit_source_sensitivity"
OUT = ARTIFACTS / "aggregate"
VARIABLES = ("T", "WS", "RH", "P")
VARIANTS = ("base", "in_sample_full", "in_sample_matched", "held_out")


def mean(values):
    return statistics.fmean(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def read_runs(datasets):
    runs = []
    for path in sorted(ARTIFACTS.glob("*/*/seed_*/sensitivity/summary.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        dataset_name = payload["dataset"].replace("M3", "")
        if dataset_name in datasets:
            runs.append(payload)
    return runs


def flatten(runs):
    rows = []
    for run in runs:
        base = run["test_metrics"]["base"]
        for variant in VARIANTS:
            metric = run["test_metrics"][variant]
            row = {
                "dataset": run["dataset"], "backbone": run["backbone"],
                "adapter_mode": run["adapter_mode"], "seed": run["seed"], "variant": variant,
                "selected_ridge": run["variants"].get(variant, {}).get("selected_ridge", ""),
                "backbone_training_seconds": run.get("backbone_training_seconds", ""),
                "adapter_analysis_seconds": run.get("adapter_analysis_seconds", ""),
            }
            if variant != "base":
                detail = run["variants"][variant]
                row.update({
                    "fit_normalized_mse": detail["fit_normalized_mse"],
                    "fit_base_normalized_mse": detail["fit_base_normalized_mse"],
                    "adapter_validation_normalized_mse": detail["adapter_validation_normalized_mse"],
                    "adapter_validation_base_normalized_mse": detail["adapter_validation_base_normalized_mse"],
                    **detail["coefficient_norms"],
                })
            for key in ("normalized_mse", "normalized_mae", "mse", "mae"):
                row[key] = metric[key]
                row[f"{key}_change_percent_vs_base"] = 100 * (metric[key] / base[key] - 1)
            for idx, variable in enumerate(run["variable_names"]):
                for key in ("normalized_mse", "normalized_mae", "mse", "mae"):
                    row[f"{variable}_{key}"] = metric[f"{key}_by_variable"][idx]
            rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["backbone"], row["adapter_mode"], row["variant"])].append(row)
    output = []
    metrics = ["normalized_mse", "normalized_mae", "mse", "mae"]
    metrics += [f"{variable}_{kind}" for variable in VARIABLES for kind in ("normalized_mse", "normalized_mae", "mse", "mae")]
    for key, values in sorted(groups.items()):
        row = dict(zip(("dataset", "backbone", "adapter_mode", "variant"), key))
        row["n_seeds"] = len(values)
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            row[f"{metric}_mean"] = mean(samples)
            row[f"{metric}_sample_std"] = sample_std(samples)
        base_rows = groups.get((key[0], key[1], key[2], "base"), [])
        if key[3] != "base" and len(base_rows) == len(values):
            base_by_seed = {entry["seed"]: entry for entry in base_rows}
            for metric in ("normalized_mse", "normalized_mae", "mse", "mae"):
                changes = [100 * (float(value[metric]) / float(base_by_seed[value["seed"]][metric]) - 1) for value in values]
                row[f"{metric}_change_percent_vs_base_mean"] = mean(changes)
                row[f"{metric}_change_percent_vs_base_sample_std"] = sample_std(changes)
        output.append(row)
    return output


def paired(rows):
    indexed = {(r["dataset"], r["backbone"], r["seed"], r["variant"]): r for r in rows}
    per_seed = []
    for dataset, backbone, seed, variant in sorted(indexed):
        if variant != "held_out":
            continue
        held = indexed[(dataset, backbone, seed, "held_out")]
        matched = indexed[(dataset, backbone, seed, "in_sample_matched")]
        row = {"dataset": dataset, "backbone": backbone, "seed": seed,
               "difference_definition": "held_out_minus_in_sample_matched"}
        for metric in ("normalized_mse", "normalized_mae", "mse", "mae"):
            row[f"{metric}_difference"] = float(held[metric]) - float(matched[metric])
            row[f"{metric}_relative_difference_percent"] = 100 * (float(held[metric]) / float(matched[metric]) - 1)
        per_seed.append(row)
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row["dataset"], row["backbone"])].append(row)
    summary = []
    for key, values in sorted(groups.items()):
        row = {"dataset": key[0], "backbone": key[1], "n_seeds": len(values),
               "difference_definition": "held_out_minus_in_sample_matched"}
        for metric in ("normalized_mse", "normalized_mae", "mse", "mae"):
            for suffix in ("difference", "relative_difference_percent"):
                samples = [float(value[f"{metric}_{suffix}"]) for value in values]
                row[f"{metric}_{suffix}_mean"] = mean(samples)
                row[f"{metric}_{suffix}_sample_std"] = sample_std(samples)
        summary.append(row)
    return per_seed, summary


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_status(runs, datasets):
    present = {(run["dataset"].replace("M3", ""), run["backbone"], int(run["seed"])) for run in runs}
    expected = [(d, m, s) for d in datasets for m in ("PatchTST", "iTransformer", "S2Transformer") for s in (2024, 2025, 2026)]
    missing = [item for item in expected if item not in present]
    text = ["# Adapter residual-fit source sensitivity status", "", f"Completed: {len(present)}/{len(expected)} runs.", ""]
    if missing:
        text += ["No paper conclusion is generated while requested runs are incomplete.", "", "Missing runs:", ""]
        text += [f"- {dataset} / {model} / seed {seed}" for dataset, model, seed in missing]
    else:
        text += ["All requested runs are complete. See `paper_analysis_zh_en.md` for result-grounded analysis."]
    (OUT / "status.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=("France", "Global"), default=["France"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = read_runs(set(args.datasets))
    rows = flatten(runs)
    summary = summarize(rows)
    paired_seed, paired_summary = paired(rows)
    write_csv(OUT / "per_seed.csv", rows)
    write_csv(OUT / "summary_mean_sample_std.csv", summary)
    write_csv(OUT / "paired_matched_vs_held_out_per_seed.csv", paired_seed)
    write_csv(OUT / "paired_matched_vs_held_out_summary.csv", paired_summary)
    missing = write_status(runs, args.datasets)
    manifest = {"datasets": args.datasets, "completed_runs": len(runs), "missing": missing, "source_summaries": [r["_path"] for r in runs]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
