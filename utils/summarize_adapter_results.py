"""Aggregate per-model adapter JSON files into paper-friendly tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from utils.run_all_adapters import DEFAULT_ORDER, EXPERIMENTS, OUTPUT_ROOT


CATEGORY = {
    "DLinear": "SS", "PatchTST": "SS", "Moirai": "SS",
    "Sundial": "SS", "TimeMoE": "SS", "Timer": "SS",
    "CDPNet": "MS", "STELLA": "MS", "Corrformer": "MS", "EasyST": "MS",
    "Autoformer": "SM", "DUET": "SM", "TQNet": "SM",
    "TimerXL": "SM", "iTransformer": "SM", "HiSTGNN": "MM",
}


def _results():
    rows = []
    for model in DEFAULT_ORDER:
        mode = EXPERIMENTS[model][0]
        path = OUTPUT_ROOT / "results" / f"{model}_{mode}_closed_form" / "summary.json"
        if not path.is_file():
            rows.append({"model": model, "category": CATEGORY[model], "mode": mode})
            continue
        result = json.loads(path.read_text())
        rows.append(
            {
                "model": model,
                "category": CATEGORY[model],
                "mode": mode,
                **result,
            }
        )
    return rows


def main():
    rows = _results()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.csv"
    fields = [
        "model", "category", "mode", "baseline_mse", "adapter_mse",
        "mse_change_percent", "baseline_mae", "adapter_mae",
        "mae_change_percent", "baseline_normalized_mse",
        "adapter_normalized_mse", "normalized_mse_change_percent",
        "selected_ridge", "fit_stride", "trainable_parameters", "status",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if "adapter" not in row:
                writer.writerow(
                    {"model": row["model"], "category": row["category"],
                     "mode": row["mode"], "status": "pending"}
                )
                continue
            writer.writerow(
                {
                    "model": row["model"], "category": row["category"],
                    "mode": row["mode"],
                    "baseline_mse": row["baseline"]["mse"],
                    "adapter_mse": row["adapter"]["mse"],
                    "mse_change_percent": row["adapter"]["mse_change_percent"],
                    "baseline_mae": row["baseline"]["mae"],
                    "adapter_mae": row["adapter"]["mae"],
                    "mae_change_percent": row["adapter"]["mae_change_percent"],
                    "baseline_normalized_mse": row["baseline"]["normalized_mse"],
                    "adapter_normalized_mse": row["adapter"]["normalized_mse"],
                    "normalized_mse_change_percent": row["adapter"]["normalized_mse_change_percent"],
                    "selected_ridge": row["selected_ridge"],
                    "fit_stride": row["fit_stride"],
                    "trainable_parameters": row["trainable_parameters"],
                    "status": "complete",
                }
            )

    variable_path = OUTPUT_ROOT / "per_variable.csv"
    variable_fields = [
        "model", "category", "mode", "variable", "baseline_mse",
        "adapter_mse", "mse_change_percent", "baseline_mae", "adapter_mae",
        "mae_change_percent",
    ]
    with variable_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=variable_fields)
        writer.writeheader()
        for row in rows:
            if "adapter" not in row:
                continue
            for index, variable in enumerate(row["variable_names"]):
                base_mse = row["baseline"]["mse_by_variable"][index]
                adapter_mse = row["adapter"]["mse_by_variable"][index]
                base_mae = row["baseline"]["mae_by_variable"][index]
                adapter_mae = row["adapter"]["mae_by_variable"][index]
                writer.writerow(
                    {
                        "model": row["model"], "category": row["category"],
                        "mode": row["mode"], "variable": variable,
                        "baseline_mse": base_mse, "adapter_mse": adapter_mse,
                        "mse_change_percent": 100.0 * (adapter_mse / base_mse - 1.0),
                        "baseline_mae": base_mae, "adapter_mae": adapter_mae,
                        "mae_change_percent": 100.0 * (adapter_mae / base_mae - 1.0),
                    }
                )

    markdown = [
        "# All-baseline Interaction Adapter Results", "",
        "All backbones are frozen. Only the output interaction adapter is fitted; "
        "the test set contains all 7,773 windows.", "",
        "## Per-model results", "",
        "`Norm. MSE` is the dimensionless training objective. `Raw MSE/MAE` "
        "are inverse-transformed to the physical scales of u, v, T and RH.", "",
        "| Category | Backbone | Mode | Norm. MSE: base → adapter | Δ norm. MSE | Raw MSE: base → adapter | Δ raw MSE | Raw MAE: base → adapter | Δ raw MAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "adapter" not in row:
            markdown.append(
                f"| {row['category']} | {row['model']} | {row['mode']} | "
                "pending | — | pending | — | pending | — |"
            )
            continue
        markdown.append(
            f"| {row['category']} | {row['model']} | {row['mode']} | "
            f"{row['baseline']['normalized_mse']:.4f} → "
            f"{row['adapter']['normalized_mse']:.4f} | "
            f"{row['adapter']['normalized_mse_change_percent']:+.2f}% | "
            f"{row['baseline']['mse']:.4f} → {row['adapter']['mse']:.4f} | "
            f"{row['adapter']['mse_change_percent']:+.2f}% | "
            f"{row['baseline']['mae']:.4f} → {row['adapter']['mae']:.4f} | "
            f"{row['adapter']['mae_change_percent']:+.2f}% |"
        )

    aggregates = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if "adapter" not in row:
            continue
        for metric in (
            "normalized_mse_change_percent", "mse_change_percent",
            "mae_change_percent",
        ):
            aggregates[row["category"]][metric].append(row["adapter"][metric])
    markdown.extend([
        "", "## Category-level mean relative change", "",
        "The following is the unweighted mean across models in each category.", "",
        "| Category | Models | Δ norm. MSE | Δ raw MSE | Δ raw MAE |",
        "|---|---:|---:|---:|---:|",
    ])
    for category in ("SS", "MS", "SM", "MM"):
        values = aggregates[category]
        if not values:
            continue
        count = len(values["mse_change_percent"])
        mean = lambda key: sum(values[key]) / len(values[key])
        markdown.append(
            f"| {category} | {count} | "
            f"{mean('normalized_mse_change_percent'):+.2f}% | "
            f"{mean('mse_change_percent'):+.2f}% | "
            f"{mean('mae_change_percent'):+.2f}% |"
        )

    markdown.extend([
        "", "## Interpretation notes", "",
        "- Raw overall MSE averages squared errors with different units "
        "((m/s)^2, °C^2 and percentage-point^2). It is useful as a compact "
        "engineering summary, but is not a dimensionally homogeneous score.",
        "- Sundial improves the normalized objective and raw u/v/T errors, but "
        "its RH error increases enough to make raw overall MSE/MAE worse.",
        "- HiSTGNN is already an MM model. Its small gain is a control showing "
        "less missing interaction structure than SS/MS/SM backbones.",
        "- TimerXL is evaluated from the final fine-tuned checkpoint. The older "
        "raw record (MSE 63.8501) accidentally evaluated only the public "
        "pretrained weights; the corrected frozen-backbone baseline is 40.1409.",
    ])
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(markdown) + "\n")
    print(summary_path)
    print(variable_path)
    print(OUTPUT_ROOT / "summary.md")


if __name__ == "__main__":
    main()
