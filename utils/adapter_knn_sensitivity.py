"""Closed-form sensitivity sweep over geographic kNN graph sizes.

Each dataset/backbone process evaluates all requested K values while performing
the frozen-backbone forward pass only once per batch and split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from models.InteractionAdapter import (
    PredictionInteractionAdapter,
    _geographic_adjacency,
)
from utils.fit_interaction_adapter import (
    DEFAULT_RIDGES,
    MetricAccumulator,
    SufficientStatistics,
    _atomic_torch_save,
    _backbone_prediction,
    _feature_count,
    _features_for_variable,
    _interaction_inputs,
    _load_coefficients,
    _loader,
    _solve,
    _unwrapped,
    _validation_mse,
)
from utils.station_coordinates import load_station_coordinates


def _paths(args, k: int):
    name = f"{args.backbone_model}_{args.adapter_mode}_closed_form"
    result_dir = Path(args.results) / f"k{k}" / name
    checkpoint_dir = Path(args.checkpoints) / f"k{k}" / name
    return result_dir, checkpoint_dir


def _all_complete(args, k_values):
    return all((_paths(args, k)[0] / "summary.json").is_file() for k in k_values)


def _make_adapters(args, wrapper, device, k_values):
    if args.adapter_mode not in {"station", "both"}:
        raise ValueError("KNN sensitivity requires a station or both Adapter mode.")
    coordinates = load_station_coordinates(
        args.root_path,
        getattr(args, "station_coords_path", None),
        expected_nodes=int(args.num_nodes),
    )
    adapters = {}
    for k in k_values:
        adjacency = _geographic_adjacency(coordinates, k)
        adapter = PredictionInteractionAdapter(
            pred_len=args.pred_len,
            num_variables=args.enc_in,
            mode=args.adapter_mode,
            adjacency=adjacency,
            use_calibration=bool(args.adapter_calibration),
        ).to(device)
        adapter.eval()
        adapters[k] = adapter
    del wrapper
    return adapters


def _collect_multi(wrapper, adapters, loader, device, args, split):
    statistics = {
        k: SufficientStatistics(
            args.pred_len, adapter.num_variables, _feature_count(adapter)
        )
        for k, adapter in adapters.items()
    }
    started = time.time()
    total = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        prediction, target = _backbone_prediction(wrapper, batch, device, args)
        residual = target - prediction
        for k, adapter in adapters.items():
            variable_normalized, station_difference = _interaction_inputs(
                adapter, prediction
            )
            for output_variable in range(adapter.num_variables):
                features = _features_for_variable(
                    adapter,
                    prediction,
                    output_variable,
                    variable_normalized,
                    station_difference,
                )
                statistics[k].add(
                    output_variable, features, residual[..., output_variable]
                )
        if (
            batch_index == 1
            or batch_index % max(1, total // 10) == 0
            or batch_index == total
        ):
            elapsed = time.time() - started
            eta = elapsed / batch_index * (total - batch_index)
            print(
                f"KNN_SWEEP_PROGRESS split={split} batch={batch_index}/{total} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return statistics


def _select_coefficients(args, train_statistics, validation_statistics):
    candidates = args.adapter_ridge or DEFAULT_RIDGES
    selected = {}
    for k in train_statistics:
        trials = []
        for ridge in candidates:
            coefficients = _solve(train_statistics[k], ridge)
            validation_mse = _validation_mse(
                validation_statistics[k], coefficients
            )
            trials.append((validation_mse, float(ridge), coefficients))
            print(
                f"KNN_SWEEP_RIDGE k={k} ridge={ridge:g} "
                f"val_mse={validation_mse:.9f}",
                flush=True,
            )
        selected[k] = min(trials, key=lambda item: item[0])
    return selected


def _evaluate_multi(wrapper, adapters, dataset, loader, device, args):
    metrics = {
        k: MetricAccumulator(adapter.num_variables, dataset.state_std)
        for k, adapter in adapters.items()
    }
    started = time.time()
    total = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        baseline, target = _backbone_prediction(wrapper, batch, device, args)
        with torch.no_grad():
            for k, adapter in adapters.items():
                metrics[k].add(baseline, adapter(baseline), target)
        if (
            batch_index == 1
            or batch_index % max(1, total // 10) == 0
            or batch_index == total
        ):
            elapsed = time.time() - started
            eta = elapsed / batch_index * (total - batch_index)
            print(
                f"KNN_SWEEP_TEST batch={batch_index}/{total} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return {k: accumulator.result() for k, accumulator in metrics.items()}


def _save(args, adapters, selected, results, k_values, timing):
    for k in k_values:
        validation_mse, selected_ridge, _ = selected[k]
        result = results[k]
        result.update(
            {
                "dataset": args.data,
                "seed": int(args.seed),
                "backbone_model": args.backbone_model,
                "adapter_mode": args.adapter_mode,
                "use_calibration": bool(adapters[k].use_calibration),
                "adapter_k_neighbors": int(k),
                "knn_values_in_shared_run": list(k_values),
                "shared_backbone_predictions": True,
                "fit_stride": int(args.adapter_fit_stride),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size or args.batch_size),
                "selected_ridge": selected_ridge,
                "validation_mse": validation_mse,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in adapters[k].parameters()
                ),
                "variable_names": list(args.state_variables or ("T", "WS", "RH", "P")),
                "timing": timing,
            }
        )
        result_dir, checkpoint_dir = _paths(args, k)
        result_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(
            {
                "format_version": 1,
                "backbone_model": args.backbone_model,
                "adapter_mode": args.adapter_mode,
                "use_calibration": bool(adapters[k].use_calibration),
                "adapter_k_neighbors": int(k),
                "selected_ridge": selected_ridge,
                "adapter_state_dict": adapters[k].state_dict(),
            },
            checkpoint_dir / "checkpoint.pth",
        )
        (result_dir / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        np.save(
            result_dir / "metrics.npy",
            np.asarray([result["adapter"]["mae"], result["adapter"]["mse"]]),
        )
        np.savez(
            result_dir / "metrics_by_variable.npz",
            variable_names=np.asarray(result["variable_names"]),
            mse=np.asarray(result["adapter"]["mse_by_variable"]),
            mae=np.asarray(result["adapter"]["mae_by_variable"]),
            normalized_mse=np.asarray(
                result["adapter"]["normalized_mse_by_variable"]
            ),
            normalized_mae=np.asarray(
                result["adapter"]["normalized_mae_by_variable"]
            ),
            normalized_overall_mse=np.asarray(
                result["adapter"]["normalized_mse"]
            ),
            normalized_overall_mae=np.asarray(
                result["adapter"]["normalized_mae"]
            ),
        )
        print("KNN_SWEEP_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


def run_knn_sensitivity(args, Exp):
    k_values = tuple(dict.fromkeys(int(k) for k in args.adapter_knn_values))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("--adapter_knn_values must contain positive integers.")
    if args.resume and _all_complete(args, k_values):
        print(
            f"KNN_SWEEP_SKIP backbone={args.backbone_model} "
            f"completed_k={','.join(map(str, k_values))}",
            flush=True,
        )
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    experiment = Exp(args)
    wrapper = _unwrapped(experiment.model)
    if wrapper.__class__.__module__ != "models.InteractionAdapter":
        raise TypeError("KNN sensitivity requires --model InteractionAdapter.")
    wrapper.eval()
    device = experiment.device
    adapters = _make_adapters(args, wrapper, device, k_values)

    train_data, _ = experiment._get_data("train")
    validation_data, _ = experiment._get_data("val")
    test_data, _ = experiment._get_data("test")
    fit_stride = max(1, int(args.adapter_fit_stride))
    train_loader = _loader(train_data, args, fit_stride)
    validation_loader = _loader(validation_data, args, fit_stride)
    test_loader = _loader(test_data, args, 1)
    print(
        f"KNN_SWEEP_START dataset={args.data} backbone={args.backbone_model} "
        f"mode={args.adapter_mode} k={','.join(map(str, k_values))} "
        f"train_windows={len(train_loader.dataset)} "
        f"val_windows={len(validation_loader.dataset)} "
        f"test_windows={len(test_loader.dataset)} stride={fit_stride}",
        flush=True,
    )

    started = time.perf_counter()
    train_started = time.perf_counter()
    train_statistics = _collect_multi(
        wrapper, adapters, train_loader, device, args, "train"
    )
    train_seconds = time.perf_counter() - train_started
    validation_started = time.perf_counter()
    validation_statistics = _collect_multi(
        wrapper, adapters, validation_loader, device, args, "validation"
    )
    validation_seconds = time.perf_counter() - validation_started
    solve_started = time.perf_counter()
    selected = _select_coefficients(
        args, train_statistics, validation_statistics
    )
    for k, (_, _, coefficients) in selected.items():
        _load_coefficients(adapters[k], coefficients.to(device))
    solve_seconds = time.perf_counter() - solve_started
    test_started = time.perf_counter()
    results = _evaluate_multi(
        wrapper, adapters, test_data, test_loader, device, args
    )
    test_seconds = time.perf_counter() - test_started
    timing = {
        "train_statistics_seconds": train_seconds,
        "validation_statistics_seconds": validation_seconds,
        "solve_seconds": solve_seconds,
        "test_seconds": test_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    _save(args, adapters, selected, results, k_values, timing)
