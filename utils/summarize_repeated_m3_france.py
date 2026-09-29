"""Aggregate repeated M3 baseline and Adapter metrics in both scales."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("T", "WS", "RH", "P")
SEEDS = (2024, 2025, 2026)
GROUPS = {
    "SS": ("DLinear", "PatchTST", "Timer", "TimeMoE", "Moirai"),
    "MS": ("CDPNet", "STELLA", "Corrformer", "EasyST", "S2Transformer"),
    "SM": ("DUET", "iTransformer", "TQNet", "TimerXL", "xPatch"),
    "MM": ("HiSTGNN",),
}
MODES = {
    model: mode
    for mode, models in {
        "both": (*GROUPS["SS"], *GROUPS["MM"]),
        "variable": GROUPS["MS"],
        "station": GROUPS["SM"],
    }.items()
    for model in models
}
METRICS = ("mse", "mae", "normalized_mse", "normalized_mae")
DATASETS = {
    "M3France": ("M3 France", "m3_france_twsrhp"),
    "M3Europe": ("M3 Europe", "m3_europe_twsrhp"),
    "M3Global": ("M3 Global", "m3_global_twsrhp"),
}
MODEL_COUNT = sum(len(models) for models in GROUPS.values())


def _one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one result matching {root / pattern}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _baseline(
    root: Path, model: str, seed: int, data_name: str = "M3France"
) -> dict[str, object]:
    path = _one_match(
        root / "results",
        f"*_{model}_{data_name}_*seed{seed}_0/metrics_by_variable.npz",
    )
    payload = np.load(path)
    names = tuple(str(value) for value in payload["variable_names"])
    if names != VARIABLES:
        raise ValueError(f"Unexpected variable order in {path}: {names}")
    result: dict[str, object] = {}
    for metric in METRICS:
        values = np.asarray(payload[metric], dtype=np.float64)
        result[metric] = float(values.mean())
        result[f"{metric}_by_variable"] = values.tolist()
    return result


def _adapter(root: Path, model: str, seed: int) -> dict[str, object]:
    path = (
        root / f"seed{seed}" / "results"
        / f"{model}_{MODES[model]}_closed_form" / "summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["variable_names"]) != VARIABLES:
        raise ValueError(
            f"Unexpected variable order in {path}: {payload['variable_names']}"
        )
    return payload["adapter"]


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"runs": len(runs)}
    for metric in METRICS:
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        by_variable = np.asarray(
            [run[f"{metric}_by_variable"] for run in runs], dtype=np.float64
        )
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_variance"] = float(values.var(ddof=1))
        result[f"{metric}_std"] = float(values.std(ddof=1))
        for statistic, function in (
            ("mean", lambda value: value.mean(axis=0)),
            ("variance", lambda value: value.var(axis=0, ddof=1)),
            ("std", lambda value: value.std(axis=0, ddof=1)),
        ):
            result[f"{metric}_by_variable_{statistic}"] = function(
                by_variable
            ).tolist()
    return result


def _flat_row(group, model, method, aggregate):
    row = {
        "group": group,
        "model": model,
        "method": method,
        "runs": aggregate["runs"],
    }
    for metric in METRICS:
        for statistic in ("mean", "variance", "std"):
            row[f"{metric}_{statistic}"] = aggregate[f"{metric}_{statistic}"]
        for index, variable in enumerate(VARIABLES):
            for statistic in ("mean", "variance", "std"):
                row[f"{variable}_{metric}_{statistic}"] = aggregate[
                    f"{metric}_by_variable_{statistic}"
                ][index]
    return row


def _mean_variance_std(row, key):
    return (
        f"{row[key + '_mean']:.4f} / {row[key + '_variance']:.4f} / "
        f"{row[key + '_std']:.4f}"
    )


def _markdown(rows, dataset_label="M3 France"):
    lines = [
        f"# {dataset_label} 三次重复实验",
        "",
        "变量顺序为 T、WS、RH、P；所有单元格均为 mean / variance / std，"
        "方差和标准差使用样本统计量（ddof=1）。",
        "",
        "## Overall：原始尺度与归一化尺度",
        "",
        "| 范式 | 模型 | 方法 | Raw MSE | Raw MAE | Normalized MSE | Normalized MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['model']} | {row['method']} | "
            f"{_mean_variance_std(row, 'mse')} | "
            f"{_mean_variance_std(row, 'mae')} | "
            f"{_mean_variance_std(row, 'normalized_mse')} | "
            f"{_mean_variance_std(row, 'normalized_mae')} |"
        )
    for scale, prefix in (("原始尺度", ""), ("归一化尺度", "normalized_")):
        lines.extend(
            [
                "",
                f"## {scale}逐变量结果",
                "",
                "| 范式 | 模型 | 方法 | T MSE | WS MSE | RH MSE | P MSE | T MAE | WS MAE | RH MAE | P MAE |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            mse_values = " | ".join(
                _mean_variance_std(row, f"{variable}_{prefix}mse")
                for variable in VARIABLES
            )
            mae_values = " | ".join(
                _mean_variance_std(row, f"{variable}_{prefix}mae")
                for variable in VARIABLES
            )
            lines.append(
                f"| {row['group']} | {row['model']} | {row['method']} | "
                f"{mse_values} | {mae_values} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", choices=tuple(DATASETS), default="M3France")
    parser.add_argument(
        "--dataset-label", default=None,
        help="human-readable dataset name used in the Markdown title",
    )
    parser.add_argument(
        "--seeds", default="2024,2025,2026",
        help="comma- or space-separated repeated-run seeds",
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=None,
    )
    parser.add_argument(
        "--adapter-root", type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=None,
    )
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--adapter-fit-stride", type=int, default=None)
    parser.add_argument("--skip-epoch-test", action="store_true")
    parser.add_argument("--deterministic-reuse", default="")
    parser.add_argument("--batch-size-multipliers", default="")
    args = parser.parse_args()
    dataset_label, record_stem = DATASETS[args.data_name]
    dataset_label = args.dataset_label or dataset_label
    seeds = tuple(
        int(value) for value in args.seeds.replace(",", " ").split()
    )
    if len(seeds) < 2:
        parser.error("--seeds must contain at least two seeds for sample statistics")
    if len(seeds) != len(set(seeds)):
        parser.error("--seeds must not contain duplicates")
    experiment_root = PROJECT_ROOT / "experiment_records" / record_stem
    baseline_root = args.baseline_root or experiment_root / "baselines"
    adapter_root = args.adapter_root or experiment_root / "adapters"
    output_root = args.output_root or experiment_root / "summary"

    records, rows = [], []
    for group, models in GROUPS.items():
        for model in models:
            baseline_runs = [
                _baseline(baseline_root, model, seed, args.data_name)
                for seed in seeds
            ]
            adapter_runs = [_adapter(adapter_root, model, seed) for seed in seeds]
            for method, runs in (("Baseline", baseline_runs), ("Adapter", adapter_runs)):
                aggregate = _aggregate(runs)
                records.append(
                    {
                        "group": group,
                        "model": model,
                        "method": method,
                        "seeds": list(seeds),
                        "per_seed": dict(zip(map(str, seeds), runs)),
                        "aggregate": aggregate,
                    }
                )
                rows.append(_flat_row(group, model, method, aggregate))

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.data_name,
        "dataset_label": dataset_label,
        "variables": list(VARIABLES),
        "seeds": list(seeds),
        "baseline_runs": MODEL_COUNT * len(seeds),
        "adapter_runs": MODEL_COUNT * len(seeds),
        "aggregate_rows": len(rows),
        "statistics": "sample variance and standard deviation (ddof=1)",
        "experiment_config": {
            "train_stride": args.train_stride,
            "validation_stride": 1,
            "test_stride": 1,
            "skip_epoch_test": args.skip_epoch_test,
            "adapter_fit_stride": args.adapter_fit_stride,
            "deterministic_reuse": [
                item.strip() for item in args.deterministic_reuse.split(",")
                if item.strip()
            ],
            "batch_size_multipliers": args.batch_size_multipliers,
        },
        "records": records,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "summary.md").write_text(
        _markdown(rows, dataset_label), encoding="utf-8"
    )
    print(
        f"Wrote {len(rows)} aggregate rows from "
        f"{MODEL_COUNT * len(seeds)} baseline and adapter runs each to "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
