"""Leakage-safe residual-fit source sensitivity for the linear adapter."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from data_provider.data_factory import data_dict
from utils.fit_interaction_adapter import (
    DEFAULT_RIDGES,
    _backbone_prediction,
    _collect_statistics,
    _features_for_variable,
    _interaction_inputs,
    _loader,
    _solve,
    _unwrapped,
    _validation_mse,
)


FIT_ROLES = {
    "in_sample_full": "backbone_train",
    "in_sample_matched": "in_sample_matched",
    "held_out": "calibration_holdout",
}


class _Metrics:
    def __init__(self, variable_names, scale):
        self.variable_names = list(variable_names)
        self.scale = torch.as_tensor(scale, dtype=torch.float64)
        n = len(self.variable_names)
        self.sq = torch.zeros(n, dtype=torch.float64)
        self.ab = torch.zeros(n, dtype=torch.float64)
        self.raw_sq = torch.zeros(n, dtype=torch.float64)
        self.raw_ab = torch.zeros(n, dtype=torch.float64)
        self.count = torch.zeros(n, dtype=torch.float64)

    def add(self, prediction, target):
        error = (prediction - target).double().cpu()
        finite = torch.isfinite(error)
        error = torch.where(finite, error, 0.0)
        axes = (0, 1, 2)
        self.sq += error.square().sum(axes)
        self.ab += error.abs().sum(axes)
        raw = error * self.scale
        self.raw_sq += raw.square().sum(axes)
        self.raw_ab += raw.abs().sum(axes)
        self.count += finite.sum(axes)

    def result(self):
        count = self.count.clamp_min(1)
        return {
            "normalized_mse": float(self.sq.sum() / count.sum()),
            "normalized_mae": float(self.ab.sum() / count.sum()),
            "normalized_mse_by_variable": (self.sq / count).tolist(),
            "normalized_mae_by_variable": (self.ab / count).tolist(),
            "mse": float(self.raw_sq.sum() / count.sum()),
            "mae": float(self.raw_ab.sum() / count.sum()),
            "mse_by_variable": (self.raw_sq / count).tolist(),
            "mae_by_variable": (self.raw_ab / count).tolist(),
            "count_by_variable": self.count.tolist(),
        }


def _make_dataset(args, role):
    cls = data_dict[args.data]
    flag = "test" if role == "test" else ("val" if "validation" in role else "train")
    return cls(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        freq=args.freq,
        split_role=role,
    )


def _base_mse(statistics):
    _, _, y2 = statistics.normalized()
    return float((y2 * statistics.count).sum() / statistics.count.sum().clamp_min(1))


def _coefficient_norms(coefficients, adapter):
    cursor = 0
    result = {"total_l2": float(torch.linalg.vector_norm(coefficients))}
    if adapter.use_calibration:
        result["calibration_scale_l2"] = float(torch.linalg.vector_norm(coefficients[..., 0]))
        result["calibration_bias_l2"] = float(torch.linalg.vector_norm(coefficients[..., 1]))
        cursor = 2
    if adapter.mode in {"variable", "both"}:
        width = adapter.num_variables - 1
        result["variable_interaction_l2"] = float(
            torch.linalg.vector_norm(coefficients[..., cursor:cursor + width])
        )
        cursor += width
    if adapter.mode in {"station", "both"}:
        result["station_interaction_l2"] = float(
            torch.linalg.vector_norm(coefficients[..., cursor:cursor + 2])
        )
    return result


def _adapter_features(adapter, baseline):
    variable_normalized, station_difference = _interaction_inputs(adapter, baseline)
    return [
        _features_for_variable(
            adapter, baseline, output_variable,
            variable_normalized, station_difference,
        )
        for output_variable in range(adapter.num_variables)
    ]


def _adapt_with_coefficients(baseline, features_by_variable, coefficients):
    corrections = []
    coefficients = coefficients.to(device=baseline.device, dtype=baseline.dtype)
    for output_variable, features in enumerate(features_by_variable):
        corrections.append(
            torch.einsum("bhnp,hp->bhn", features, coefficients[:, output_variable])
        )
    return baseline + torch.stack(corrections, dim=-1)


def _evaluate_all(wrapper, dataset, loader, coefficients, device, args):
    metrics = {
        name: _Metrics(dataset.variable_names, dataset.state_std)
        for name in ["base", *coefficients]
    }
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        baseline, target = _backbone_prediction(wrapper, batch, device, args)
        metrics["base"].add(baseline, target)
        with torch.no_grad():
            features = _adapter_features(wrapper.adapter, baseline)
            for name, beta in coefficients.items():
                metrics[name].add(
                    _adapt_with_coefficients(baseline, features, beta), target
                )
        if batch_index == 1 or batch_index == len(loader) or batch_index % max(1, len(loader) // 10) == 0:
            elapsed = time.perf_counter() - started
            eta = elapsed / batch_index * (len(loader) - batch_index)
            print(
                f"SENSITIVITY_TEST_PROGRESS batch={batch_index}/{len(loader)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return {name: accumulator.result() for name, accumulator in metrics.items()}


def _interval_record(dataset, sampled_origins=None):
    start, end = dataset.split_start, dataset.split_end
    origins = len(dataset) if sampled_origins is None else int(sampled_origins)
    return {
        "role": dataset.split_role,
        "raw_start_index": start,
        "raw_end_index_exclusive": end,
        "raw_start_time": str(dataset.times[start]),
        "raw_end_time_exclusive": str(dataset.times[end - 1] + np.timedelta64(1, "h")),
        "first_forecast_time": str(dataset.times[start + dataset.seq_len]),
        "last_forecast_time": str(dataset.times[end - dataset.pred_len]),
        "raw_hour_count": end - start,
        "valid_prediction_origins": len(dataset),
        "sampled_prediction_origins": origins,
        "stations": dataset.num_stations,
        "M_origins_times_stations": origins * dataset.num_stations,
        "standardization_start_index": dataset.statistics_start,
        "standardization_end_index_exclusive": dataset.statistics_end,
    }


def _write_flat_csv(summary, path):
    rows = []
    base = summary["test_metrics"]["base"]
    for name in ["base", *FIT_ROLES]:
        metric = summary["test_metrics"][name]
        row = {
            "dataset": summary["dataset"], "backbone": summary["backbone"],
            "adapter_mode": summary["adapter_mode"], "seed": summary["seed"],
            "variant": name,
            "selected_ridge": summary["variants"].get(name, {}).get("selected_ridge", ""),
        }
        for key in ("normalized_mse", "normalized_mae", "mse", "mae"):
            row[key] = metric[key]
            row[f"{key}_change_percent_vs_base"] = 100.0 * (metric[key] / base[key] - 1.0)
        for index, variable in enumerate(summary["variable_names"]):
            for key in ("normalized_mse", "normalized_mae", "mse", "mae"):
                row[f"{variable}_{key}"] = metric[f"{key}_by_variable"][index]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_source_sensitivity(args, Exp):
    if args.m3_split_profile != "adapter_sensitivity":
        raise ValueError("--adapter_source_sensitivity requires --m3_split_profile adapter_sensitivity")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    experiment = Exp(args)
    wrapper = _unwrapped(experiment.model)
    wrapper.eval()
    device = experiment.device
    fit_stride = max(1, int(args.adapter_fit_stride))
    datasets = {
        role: _make_dataset(args, role)
        for role in [
            "backbone_train", "in_sample_matched", "calibration_holdout",
            "backbone_validation", "adapter_validation", "test",
        ]
    }

    # Audit invariants before any model inference.
    ordered = ["backbone_train", "calibration_holdout", "backbone_validation", "adapter_validation", "test"]
    for left, right in zip(ordered, ordered[1:]):
        if datasets[left].split_end > datasets[right].split_start:
            raise RuntimeError(f"Raw-time leakage between {left} and {right}.")
    expected_stats_end = datasets["backbone_train"].split_end
    if any(ds.statistics_end != expected_stats_end for ds in datasets.values()):
        raise RuntimeError("Normalization statistics are not frozen to backbone-train.")

    loaders = {role: _loader(ds, args, fit_stride) for role, ds in datasets.items() if role != "test"}
    loaders["test"] = _loader(datasets["test"], args, 1)
    statistics = {}
    accumulation_seconds = {}
    for role in ["backbone_train", "in_sample_matched", "calibration_holdout", "adapter_validation"]:
        started = time.perf_counter()
        statistics[role] = _collect_statistics(
            wrapper, datasets[role], loaders[role], device, args, role
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        accumulation_seconds[role] = time.perf_counter() - started

    candidates = tuple(float(value) for value in (args.adapter_ridge or DEFAULT_RIDGES))
    variants = {}
    selected_coefficients = {}
    for variant, role in FIT_ROLES.items():
        solve_started = time.perf_counter()
        trials = []
        for ridge in candidates:
            coefficients = _solve(statistics[role], ridge)
            trials.append({
                "ridge": ridge,
                "adapter_validation_normalized_mse": _validation_mse(
                    statistics["adapter_validation"], coefficients
                ),
                "coefficients": coefficients,
            })
        best = min(trials, key=lambda trial: trial["adapter_validation_normalized_mse"])
        beta = best.pop("coefficients")
        selected_coefficients[variant] = beta
        variants[variant] = {
            "fit_role": role,
            "selected_ridge": best["ridge"],
            "fit_normalized_mse": _validation_mse(statistics[role], beta),
            "fit_base_normalized_mse": _base_mse(statistics[role]),
            "adapter_validation_normalized_mse": best["adapter_validation_normalized_mse"],
            "adapter_validation_base_normalized_mse": _base_mse(statistics["adapter_validation"]),
            "coefficient_norms": _coefficient_norms(beta, wrapper.adapter),
            "ridge_candidates": [
                {
                    "ridge": trial["ridge"],
                    "adapter_validation_normalized_mse": trial["adapter_validation_normalized_mse"],
                }
                for trial in trials
            ],
            "fit_accumulation_seconds": accumulation_seconds[role],
            "solve_seconds": time.perf_counter() - solve_started,
        }
        variants[variant]["fit_total_seconds_including_shared_validation"] = (
            variants[variant]["fit_accumulation_seconds"]
            + accumulation_seconds["adapter_validation"]
            + variants[variant]["solve_seconds"]
        )

    shared_ridge = variants["held_out"]["selected_ridge"]
    shared_coefficients = {
        f"{variant}_shared_ridge": _solve(statistics[role], shared_ridge)
        for variant, role in FIT_ROLES.items()
    }
    test_started = time.perf_counter()
    test_metrics = _evaluate_all(
        wrapper, datasets["test"], loaders["test"],
        {**selected_coefficients, **shared_coefficients}, device, args,
    )
    test_seconds = time.perf_counter() - test_started
    for name, metric in test_metrics.items():
        if name == "base":
            continue
        base = test_metrics["base"]
        for key in ("normalized_mse", "normalized_mae", "mse", "mae"):
            metric[f"{key}_change_percent_vs_base"] = 100.0 * (metric[key] / base[key] - 1.0)

    output = Path(args.results)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "format_version": 1,
        "dataset": args.data,
        "backbone": args.backbone_model,
        "adapter_mode": args.adapter_mode,
        "seed": args.seed,
        "checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "split_profile": args.m3_split_profile,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "fit_stride": fit_stride,
        "ridge_candidates": list(candidates),
        "shared_ridge_control": {"ridge": shared_ridge, "source": "held_out selected ridge"},
        "variable_names": list(datasets["test"].variable_names),
        "intervals": {role: _interval_record(ds, len(loaders[role].dataset)) for role, ds in datasets.items()},
        "leakage_checks": {
            "ordered_raw_intervals_are_disjoint": True,
            "all_windows_are_generated_inside_each_raw_interval": True,
            "normalization_uses_only_backbone_train": True,
            "test_used_after_ridge_selection_only": True,
            "matched_rule": "latest contiguous raw-time block inside backbone-train with holdout length",
        },
        "accumulation_seconds": accumulation_seconds,
        "adapter_validation_accumulation_seconds": accumulation_seconds["adapter_validation"],
        "test_evaluation_seconds": test_seconds,
        "variants": variants,
        "test_metrics": test_metrics,
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output / "summary.json")
    _write_flat_csv(summary, output / "metrics.csv")
    torch.save(
        {
            "selected": selected_coefficients,
            "shared_ridge": shared_coefficients,
            "metadata": {"ridge": {name: value["selected_ridge"] for name, value in variants.items()}},
        },
        output / "coefficients.pt",
    )
    print("ADAPTER_SOURCE_SENSITIVITY_RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
