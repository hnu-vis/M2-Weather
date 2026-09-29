"""Aggregate three-seed French baseline and interaction-adapter results.

All values consumed here are already in the physical/original scale.  Existing
models reuse the completed seed-2024 run and add seeds 2025/2026; the three new
models take all seeds from ``french_repeated``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("u", "v", "T", "RH")
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
NEW_MODELS = {"EasyST", "S2Transformer", "xPatch"}
SEEDS = (2024, 2025, 2026)


def _one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one result matching {root / pattern}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _metric_npz(path: Path) -> dict[str, object]:
    payload = np.load(path)
    names = tuple(str(value) for value in payload["variable_names"])
    if names != VARIABLES:
        raise ValueError(f"Unexpected variable order in {path}: {names}")
    mse = np.asarray(payload["mse"], dtype=np.float64)
    mae = np.asarray(payload["mae"], dtype=np.float64)
    return {
        "mse": float(mse.mean()),
        "mae": float(mae.mean()),
        "mse_by_variable": mse.tolist(),
        "mae_by_variable": mae.tolist(),
    }


def _baseline(args, model: str, seed: int) -> dict[str, object]:
    if seed == 2024 and model not in NEW_MODELS:
        root = (
            args.timerxl_legacy_root
            if model == "TimerXL"
            else args.baseline_legacy_root
        )
    else:
        root = args.baseline_root / "results"
    path = _one_match(
        root,
        f"*_{model}_French_*seed{seed}_0/metrics_by_variable.npz",
    )
    return _metric_npz(path)


def _adapter(args, model: str, seed: int) -> dict[str, object]:
    mode = MODES[model]
    if seed == 2024 and model not in NEW_MODELS:
        path = (
            args.adapter_legacy_root
            / f"{model}_{mode}_closed_form"
            / "summary.json"
        )
    else:
        path = (
            args.adapter_root
            / f"seed{seed}"
            / "results"
            / f"{model}_{mode}_closed_form"
            / "summary.json"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["adapter"]


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"runs": len(runs)}
    for metric in ("mse", "mae"):
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_variance"] = float(values.var(ddof=1))
        result[f"{metric}_std"] = float(values.std(ddof=1))
        by_variable = np.asarray(
            [run[f"{metric}_by_variable"] for run in runs], dtype=np.float64
        )
        result[f"{metric}_by_variable_mean"] = by_variable.mean(axis=0).tolist()
        result[f"{metric}_by_variable_variance"] = by_variable.var(
            axis=0, ddof=1
        ).tolist()
        result[f"{metric}_by_variable_std"] = by_variable.std(
            axis=0, ddof=1
        ).tolist()
    return result


def _flat_row(group, model, method, aggregate):
    row = {
        "group": group,
        "model": model,
        "method": method,
        "runs": aggregate["runs"],
        "mse_mean": aggregate["mse_mean"],
        "mse_variance": aggregate["mse_variance"],
        "mse_std": aggregate["mse_std"],
        "mae_mean": aggregate["mae_mean"],
        "mae_variance": aggregate["mae_variance"],
        "mae_std": aggregate["mae_std"],
    }
    for metric in ("mse", "mae"):
        for index, variable in enumerate(VARIABLES):
            for statistic in ("mean", "variance", "std"):
                row[f"{variable}_{metric}_{statistic}"] = aggregate[
                    f"{metric}_by_variable_{statistic}"
                ][index]
    return row


def _markdown(rows):
    lines = [
        "# 法国数据集三次重复实验（原始尺度）",
        "",
        "方差与标准差均采用样本统计量（ddof=1）；表中为 mean ± std。",
        "",
        "| 范式 | 模型 | 方法 | Overall MSE | Overall MAE | u MSE | v MSE | T MSE | RH MSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = lambda key: f"{row[key + '_mean']:.4f} ± {row[key + '_std']:.4f}"
        lines.append(
            "| {group} | {model} | {method} | {mse} | {mae} | {u} | {v} | {t} | {rh} |".format(
                group=row["group"], model=row["model"], method=row["method"],
                mse=value("mse"), mae=value("mae"), u=value("u_mse"),
                v=value("v_mse"), t=value("T_mse"), rh=value("RH_mse"),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root", type=Path,
        default=PROJECT_ROOT / "experiment_records/french_repeated/baselines",
    )
    parser.add_argument(
        "--adapter-root", type=Path,
        default=PROJECT_ROOT / "experiment_records/french_repeated/adapters",
    )
    parser.add_argument(
        "--baseline-legacy-root", type=Path,
        default=PROJECT_ROOT / "experiment_records/forecasting_uvtrh_original_scale/results",
    )
    parser.add_argument(
        "--timerxl-legacy-root", type=Path,
        default=PROJECT_ROOT / "experiment_records/timerxl_corrected_original_scale/results",
    )
    parser.add_argument(
        "--adapter-legacy-root", type=Path,
        default=PROJECT_ROOT / "experiment_records/adapters_all/results",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "experiment_records/french_repeated/summary",
    )
    args = parser.parse_args()

    records = []
    rows = []
    for group, models in GROUPS.items():
        for model in models:
            baseline_runs = [_baseline(args, model, seed) for seed in SEEDS]
            adapter_runs = [_adapter(args, model, seed) for seed in SEEDS]
            baseline = _aggregate(baseline_runs)
            adapter = _aggregate(adapter_runs)
            for method, runs, aggregate in (
                ("Baseline", baseline_runs, baseline),
                ("Adapter", adapter_runs, adapter),
            ):
                records.append(
                    {
                        "group": group,
                        "model": model,
                        "method": method,
                        "seeds": list(SEEDS),
                        "per_seed": dict(zip(map(str, SEEDS), runs)),
                        "aggregate": aggregate,
                    }
                )
                rows.append(_flat_row(group, model, method, aggregate))

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "summary.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} aggregate rows to {args.output_root}")


if __name__ == "__main__":
    main()
