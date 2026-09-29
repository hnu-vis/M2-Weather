"""Paired significance tests for Table 3 normalized MSE results.

Each genuinely repeated model is compared across matched seeds 2024--2026 using
a two-sided paired t-test on normalized overall MSE.  Foundation-model outputs
that were deterministically reused across seeds are reported as non-estimable
rather than counted as independent replicates.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "significance_analysis"
SEEDS = (2024, 2025, 2026)
DATASETS = {
    "France": ("m3_france_twsrhp", "M3France"),
    "Europe": ("m3_europe_twsrhp_fast_s6", "M3Europe"),
    "Global": ("m3_global_twsrhp_fast_s6", "M3Global"),
}
MODELS = (
    ("SSSV", "DLinear"),
    ("SSSV", "xPatch"),
    ("SSSV", "PatchTST"),
    ("SSSV", "Timer"),
    ("SSSV", "TimeMoE"),
    ("SSMV", "iTransformer"),
    ("SSMV", "TQNet"),
    ("SSMV", "DUET"),
    ("SSMV", "Moirai"),
    ("SSMV", "TimerXL"),
    ("MSSV", "Corrformer"),
    ("MSSV", "S2Transformer"),
    ("MSSV", "CDPNet"),
    ("MSSV", "EasyST"),
    ("MSSV", "STELLA"),
    ("MSMV", "HiSTGNN"),
)
DETERMINISTIC_REUSE = {"Timer", "TimeMoE", "Moirai"}
NO_ADAPTER = {"HiSTGNN"}


def one_match(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {description}, found {len(paths)}: {paths}")
    return paths[0]


def baseline_path(record: Path, data_name: str, model: str, seed: int) -> Path:
    return one_match(
        sorted((record / "baselines" / "results").glob(
            f"*_{model}_{data_name}_*seed{seed}_0/metrics_by_variable.npz"
        )),
        f"baseline result for {data_name}/{model}/seed{seed}",
    )


def adapter_path(record: Path, model: str, seed: int) -> Path:
    return one_match(
        sorted((record / "adapters" / f"seed{seed}" / "results").glob(
            f"{model}_*_closed_form/metrics_by_variable.npz"
        )),
        f"adapter result for {record.name}/{model}/seed{seed}",
    )


def normalized_mse(path: Path) -> float:
    with np.load(path) as values:
        return float(values["normalized_overall_mse"])


def holm(values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    count = len(values)
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    rows: list[dict] = []
    for dataset, (record_name, data_name) in DATASETS.items():
        record = ROOT / "experiment_records" / record_name
        for paradigm, model in MODELS:
            base_paths = [baseline_path(record, data_name, model, seed) for seed in SEEDS]
            base = np.array([normalized_mse(path) for path in base_paths])
            if model in NO_ADAPTER:
                rows.append({
                    "dataset": dataset,
                    "paradigm": paradigm,
                    "model": model,
                    "n_independent_pairs": 0,
                    "base_mean": float(base.mean()),
                    "base_sd": float(base.std(ddof=1)),
                    "adapter_mean": None,
                    "adapter_sd": None,
                    "mean_paired_difference": None,
                    "relative_change_percent": None,
                    "all_observed_pairs_improve": None,
                    "seed_values_base": base.tolist(),
                    "seed_values_adapter": None,
                    "test": "two-sided paired t-test",
                    "t_statistic": None,
                    "p_raw": None,
                    "p_holm_dataset": None,
                    "p_holm_all": None,
                    "mean_difference_ci95_low": None,
                    "mean_difference_ci95_high": None,
                    "cohen_dz": None,
                    "estimable": False,
                    "reason": "HiSTGNN is reported without an adapter; no paired contrast",
                })
                continue
            adapter_paths = [adapter_path(record, model, seed) for seed in SEEDS]
            adapter = np.array([normalized_mse(path) for path in adapter_paths])
            difference = adapter - base
            row = {
                "dataset": dataset,
                "paradigm": paradigm,
                "model": model,
                "n_independent_pairs": 1 if model in DETERMINISTIC_REUSE else len(SEEDS),
                "base_mean": float(base.mean()),
                "base_sd": float(base.std(ddof=1)),
                "adapter_mean": float(adapter.mean()),
                "adapter_sd": float(adapter.std(ddof=1)),
                "mean_paired_difference": float(difference.mean()),
                "relative_change_percent": float(
                    100.0 * (adapter.mean() - base.mean()) / base.mean()
                ),
                "all_observed_pairs_improve": bool(np.all(difference < 0)),
                "seed_values_base": base.tolist(),
                "seed_values_adapter": adapter.tolist(),
                "test": "two-sided paired t-test",
            }
            if model in DETERMINISTIC_REUSE:
                row.update({
                    "t_statistic": None,
                    "p_raw": None,
                    "p_holm_dataset": None,
                    "p_holm_all": None,
                    "mean_difference_ci95_low": None,
                    "mean_difference_ci95_high": None,
                    "cohen_dz": None,
                    "estimable": False,
                    "reason": "deterministic prediction reused across seeds; only one independent pair",
                })
            else:
                result = stats.ttest_rel(adapter, base, alternative="two-sided")
                difference_sd = float(difference.std(ddof=1))
                standard_error = difference_sd / math.sqrt(len(SEEDS))
                critical = float(stats.t.ppf(0.975, df=len(SEEDS) - 1))
                row.update({
                    "t_statistic": float(result.statistic),
                    "p_raw": float(result.pvalue),
                    "p_holm_dataset": None,
                    "p_holm_all": None,
                    "mean_difference_ci95_low": float(difference.mean() - critical * standard_error),
                    "mean_difference_ci95_high": float(difference.mean() + critical * standard_error),
                    "cohen_dz": float(difference.mean() / difference_sd),
                    "estimable": True,
                    "reason": "",
                })
            rows.append(row)

    for dataset in DATASETS:
        indices = [
            index for index, row in enumerate(rows)
            if row["dataset"] == dataset and row["estimable"]
        ]
        corrected = holm([rows[index]["p_raw"] for index in indices])
        for index, value in zip(indices, corrected):
            rows[index]["p_holm_dataset"] = value
    indices = [index for index, row in enumerate(rows) if row["estimable"]]
    corrected = holm([rows[index]["p_raw"] for index in indices])
    for index, value in zip(indices, corrected):
        rows[index]["p_holm_all"] = value

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "table3_paired_tests.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    flat_fields = [
        "dataset", "paradigm", "model", "n_independent_pairs",
        "base_mean", "base_sd", "adapter_mean", "adapter_sd",
        "mean_paired_difference", "relative_change_percent",
        "mean_difference_ci95_low", "mean_difference_ci95_high",
        "cohen_dz", "t_statistic", "p_raw", "p_holm_dataset", "p_holm_all",
        "all_observed_pairs_improve", "estimable", "test", "reason",
    ]
    with (OUTPUT / "table3_paired_tests.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in flat_fields} for row in rows)

    lookup = {(row["dataset"], row["model"]): row for row in rows}
    wide_fields = ["paradigm", "model"]
    for dataset in DATASETS:
        wide_fields.extend([
            f"{dataset}_p_raw", f"{dataset}_p_holm",
            f"{dataset}_significant_holm_0.05",
        ])
    wide_rows = []
    for paradigm, model in MODELS:
        row = {"paradigm": paradigm, "model": model}
        for dataset in DATASETS:
            source = lookup[dataset, model]
            row[f"{dataset}_p_raw"] = source["p_raw"]
            row[f"{dataset}_p_holm"] = source["p_holm_dataset"]
            row[f"{dataset}_significant_holm_0.05"] = (
                None if not source["estimable"]
                else source["p_holm_dataset"] < 0.05
            )
        wide_rows.append(row)
    with (OUTPUT / "table3_pvalues_wide.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)

    def p_text(value: float | None) -> str:
        if value is None:
            return "--"
        if value < 0.001:
            return "$<.001$"
        return f"{value:.3f}".lstrip("0")

    latex = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{France} & \multicolumn{2}{c}{Europe} & \multicolumn{2}{c}{Global} \\",
        r"Paradigm & Model & $p$ & $p_{\mathrm{Holm}}$ & $p$ & $p_{\mathrm{Holm}}$ & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    previous = None
    for paradigm, model in MODELS:
        label = paradigm if paradigm != previous else ""
        values = []
        for dataset in DATASETS:
            source = lookup[dataset, model]
            values.extend([p_text(source["p_raw"]), p_text(source["p_holm_dataset"])])
        latex.append(" & ".join([label, model, *values]) + r" \\")
        previous = paradigm
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUTPUT / "table3_pvalues.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    summary = {
        dataset: {
            "estimable_tests": sum(
                row["dataset"] == dataset and row["estimable"] for row in rows
            ),
            "raw_p_below_0_05": sum(
                row["dataset"] == dataset and row["estimable"]
                and row["p_raw"] < 0.05 for row in rows
            ),
            "holm_dataset_below_0_05": sum(
                row["dataset"] == dataset and row["estimable"]
                and row["p_holm_dataset"] < 0.05 for row in rows
            ),
        }
        for dataset in DATASETS
    }
    summary["all_datasets"] = {
        "estimable_tests": len(indices),
        "holm_all_below_0_05": sum(rows[index]["p_holm_all"] < 0.05 for index in indices),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
