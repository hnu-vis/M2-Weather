"""Streaming closed-form fitting for the linear prediction adapter.

The frozen backbone is evaluated once per split.  Because every adapter branch
is linear in its trainable parameters, sufficient statistics are enough to
solve the MSE objective exactly; repeated backbone inference is unnecessary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


DEFAULT_RIDGES = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _unwrapped(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _loader(dataset, args, stride):
    indices = range(0, len(dataset), max(1, int(stride)))
    # Adapter fitting freezes the backbone and never builds an autograd graph,
    # so it can use the same larger batch as validation/test.
    batch_size = getattr(args, "eval_batch_size", None) or args.batch_size
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def _move_batch(batch, device, pred_len, label_len):
    batch_x, batch_y, batch_x_mark, batch_y_mark, forecast_start = batch
    batch_x = batch_x.float().to(device, non_blocking=True)
    batch_y = batch_y.float().to(device, non_blocking=True)
    batch_x_mark = batch_x_mark.float().to(device, non_blocking=True)
    batch_y_mark = batch_y_mark.float().to(device, non_blocking=True)
    forecast_start = forecast_start.to(device, dtype=torch.long, non_blocking=True)
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, ...])
    dec_inp = torch.cat([batch_y[:, :label_len, ...], dec_inp], dim=1)
    target = batch_y[:, -pred_len:, ...]
    return batch_x, target, batch_x_mark, batch_y_mark, forecast_start, dec_inp


def _backbone_prediction(wrapper, batch, device, args):
    values = _move_batch(batch, device, args.pred_len, args.label_len)
    batch_x, target, batch_x_mark, batch_y_mark, forecast_start, dec_inp = values
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=args.use_amp and device.type == "cuda"
    ):
        prediction = wrapper.forecast_backbone(
            batch_x, batch_x_mark, dec_inp, batch_y_mark, forecast_start
        )
    prediction = prediction.float()
    target = target.float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Backbone prediction {tuple(prediction.shape)} and target "
            f"{tuple(target.shape)} do not match."
        )
    return prediction, target


def _interaction_inputs(adapter, prediction):
    variable_normalized = None
    station_difference = None
    if adapter.mode in {"variable", "both"}:
        variable_normalized = adapter.variable_norm(prediction)
    if adapter.mode in {"station", "both"}:
        station_difference = adapter.station_differences(prediction)
    return variable_normalized, station_difference


def _features_for_variable(
    adapter, prediction, output_variable, variable_normalized, station_difference
):
    features = []
    if adapter.use_calibration:
        features.extend(
            [prediction[..., output_variable], torch.ones_like(prediction[..., 0])]
        )
    if variable_normalized is not None:
        features.extend(
            variable_normalized[..., index]
            for index in range(adapter.num_variables)
            if index != output_variable
        )
    if station_difference is not None:
        features.extend(
            [
                station_difference[..., output_variable, 0],
                station_difference[..., output_variable, 1],
            ]
        )
    return torch.stack(features, dim=-1)


def _feature_count(adapter):
    count = 2 if adapter.use_calibration else 0
    if adapter.mode in {"variable", "both"}:
        count += adapter.num_variables - 1
    if adapter.mode in {"station", "both"}:
        count += 2
    return count


class SufficientStatistics:
    def __init__(self, horizon, variables, features):
        self.xtx = torch.zeros(horizon, variables, features, features, dtype=torch.float64)
        self.xty = torch.zeros(horizon, variables, features, dtype=torch.float64)
        self.y2 = torch.zeros(horizon, variables, dtype=torch.float64)
        self.count = torch.zeros(horizon, variables, dtype=torch.float64)

    def add(self, output_variable, features, residual):
        finite = torch.isfinite(residual) & torch.isfinite(features).all(dim=-1)
        clean_features = torch.where(finite[..., None], features, 0.0)
        clean_residual = torch.where(finite, residual, 0.0)
        self.xtx[:, output_variable] += torch.einsum(
            "bhnp,bhnq->hpq", clean_features, clean_features
        ).double().cpu()
        self.xty[:, output_variable] += torch.einsum(
            "bhnp,bhn->hp", clean_features, clean_residual
        ).double().cpu()
        self.y2[:, output_variable] += clean_residual.square().sum((0, 2)).double().cpu()
        self.count[:, output_variable] += finite.sum((0, 2)).double().cpu()

    def normalized(self):
        count = self.count.clamp_min(1.0)
        return (
            self.xtx / count[..., None, None],
            self.xty / count[..., None],
            self.y2 / count,
        )


def _collect_statistics(wrapper, dataset, loader, device, args, split):
    adapter = wrapper.adapter
    statistics = SufficientStatistics(
        args.pred_len, adapter.num_variables, _feature_count(adapter)
    )
    started = time.time()
    total = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        prediction, target = _backbone_prediction(wrapper, batch, device, args)
        variable_normalized, station_difference = _interaction_inputs(
            adapter, prediction
        )
        residual = target - prediction
        for output_variable in range(adapter.num_variables):
            features = _features_for_variable(
                adapter, prediction, output_variable,
                variable_normalized, station_difference,
            )
            statistics.add(output_variable, features, residual[..., output_variable])
        if batch_index == 1 or batch_index % max(1, total // 10) == 0 or batch_index == total:
            elapsed = time.time() - started
            eta = elapsed / batch_index * (total - batch_index)
            print(
                f"ADAPTER_FIT_PROGRESS split={split} batch={batch_index}/{total} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return statistics


def _solve(statistics, ridge):
    xtx, xty, _ = statistics.normalized()
    features = xtx.shape[-1]
    penalty = torch.eye(features, dtype=torch.float64)
    # Do not shrink the intercept.
    if features >= 2:
        penalty[1, 1] = 0.0
    matrix = xtx + float(ridge) * penalty
    try:
        return torch.linalg.solve(matrix, xty.unsqueeze(-1)).squeeze(-1)
    except torch.linalg.LinAlgError:
        return torch.linalg.lstsq(matrix, xty.unsqueeze(-1)).solution.squeeze(-1)


def _validation_mse(statistics, coefficients):
    xtx, xty, y2 = statistics.normalized()
    linear = torch.einsum("hvp,hvp->hv", coefficients, xty)
    quadratic = torch.einsum("hvp,hvpq,hvq->hv", coefficients, xtx, coefficients)
    mse = y2 - 2.0 * linear + quadratic
    weights = statistics.count
    return float((mse * weights).sum() / weights.sum().clamp_min(1.0))


def _load_coefficients(adapter, coefficients):
    cursor = 0
    with torch.no_grad():
        if adapter.use_calibration:
            adapter.calibration_scale.copy_(coefficients[..., cursor].float())
            adapter.calibration_bias.copy_(coefficients[..., cursor + 1].float())
            cursor += 2
        if adapter.mode in {"variable", "both"}:
            adapter.variable_weight.zero_()
            for output_variable in range(adapter.num_variables):
                for input_variable in range(adapter.num_variables):
                    if input_variable == output_variable:
                        continue
                    adapter.variable_weight[:, input_variable, output_variable].copy_(
                        coefficients[:, output_variable, cursor].float()
                    )
                    cursor += 1
                cursor -= adapter.num_variables - 1
            cursor += adapter.num_variables - 1
        if adapter.mode in {"station", "both"}:
            adapter.station_weight.copy_(coefficients[..., cursor:cursor + 2].float())


class MetricAccumulator:
    def __init__(self, variables, scale):
        self.variables = variables
        self.scale = torch.as_tensor(scale, dtype=torch.float64)
        self.base_sq = torch.zeros(variables, dtype=torch.float64)
        self.base_abs = torch.zeros(variables, dtype=torch.float64)
        self.adapter_sq = torch.zeros(variables, dtype=torch.float64)
        self.adapter_abs = torch.zeros(variables, dtype=torch.float64)
        self.count = torch.zeros(variables, dtype=torch.float64)
        self.normalized_base_sq = torch.zeros(variables, dtype=torch.float64)
        self.normalized_base_abs = torch.zeros(variables, dtype=torch.float64)
        self.normalized_adapter_sq = torch.zeros(variables, dtype=torch.float64)
        self.normalized_adapter_abs = torch.zeros(variables, dtype=torch.float64)

    def add(self, baseline, adapted, target):
        base_error = (baseline - target).double().cpu()
        adapter_error = (adapted - target).double().cpu()
        finite = torch.isfinite(base_error) & torch.isfinite(adapter_error)
        base_error = torch.where(finite, base_error, 0.0)
        adapter_error = torch.where(finite, adapter_error, 0.0)
        axes = (0, 1, 2)
        self.normalized_base_sq += base_error.square().sum(axes)
        self.normalized_base_abs += base_error.abs().sum(axes)
        self.normalized_adapter_sq += adapter_error.square().sum(axes)
        self.normalized_adapter_abs += adapter_error.abs().sum(axes)
        raw_base = base_error * self.scale
        raw_adapter = adapter_error * self.scale
        self.base_sq += raw_base.square().sum(axes)
        self.base_abs += raw_base.abs().sum(axes)
        self.adapter_sq += raw_adapter.square().sum(axes)
        self.adapter_abs += raw_adapter.abs().sum(axes)
        self.count += finite.sum(axes)

    def result(self):
        count = self.count.clamp_min(1.0)
        result = {
            "baseline": {
                "mse": (self.base_sq.sum() / count.sum()).item(),
                "mae": (self.base_abs.sum() / count.sum()).item(),
                "normalized_mse": (
                    self.normalized_base_sq.sum() / count.sum()
                ).item(),
                "normalized_mae": (
                    self.normalized_base_abs.sum() / count.sum()
                ).item(),
                "mse_by_variable": (self.base_sq / count).tolist(),
                "mae_by_variable": (self.base_abs / count).tolist(),
                "normalized_mse_by_variable": (
                    self.normalized_base_sq / count
                ).tolist(),
                "normalized_mae_by_variable": (
                    self.normalized_base_abs / count
                ).tolist(),
            },
            "adapter": {
                "mse": (self.adapter_sq.sum() / count.sum()).item(),
                "mae": (self.adapter_abs.sum() / count.sum()).item(),
                "normalized_mse": (
                    self.normalized_adapter_sq.sum() / count.sum()
                ).item(),
                "normalized_mae": (
                    self.normalized_adapter_abs.sum() / count.sum()
                ).item(),
                "mse_by_variable": (self.adapter_sq / count).tolist(),
                "mae_by_variable": (self.adapter_abs / count).tolist(),
                "normalized_mse_by_variable": (
                    self.normalized_adapter_sq / count
                ).tolist(),
                "normalized_mae_by_variable": (
                    self.normalized_adapter_abs / count
                ).tolist(),
            },
        }
        for metric in ("mse", "mae", "normalized_mse", "normalized_mae"):
            base = result["baseline"][metric]
            adapted = result["adapter"][metric]
            result["adapter"][f"{metric}_change_percent"] = 100.0 * (adapted / base - 1.0)
        return result


def _evaluate(wrapper, dataset, loader, device, args):
    metrics = MetricAccumulator(wrapper.adapter.num_variables, dataset.state_std)
    started = time.time()
    total = len(loader)
    from utils.timemoe_prediction_cache import open_test_cache
    prediction_cache = open_test_cache(args, dataset, wrapper.backbone)
    for batch_index, batch in enumerate(loader, start=1):
        cached = prediction_cache.get(batch) if prediction_cache is not None else None
        if cached is None:
            baseline, target = _backbone_prediction(wrapper, batch, device, args)
            if prediction_cache is not None:
                prediction_cache.put(batch, baseline)
        else:
            baseline = torch.from_numpy(cached).to(device)
            target = batch[1][:, -args.pred_len:].float().to(device)
        with torch.no_grad():
            adapted = wrapper.adapter(baseline)
        metrics.add(baseline, adapted, target)
        if batch_index == 1 or batch_index % max(1, total // 10) == 0 or batch_index == total:
            elapsed = time.time() - started
            eta = elapsed / batch_index * (total - batch_index)
            print(
                f"ADAPTER_TEST_PROGRESS batch={batch_index}/{total} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    if prediction_cache is not None:
        prediction_cache.report()
    return metrics.result()


def _atomic_torch_save(payload, path):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit_interaction_adapter(args, Exp):
    """Fit, validate and test one adapter experiment."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Sufficient statistics should not use TF32-reduced mantissas.
        torch.backends.cuda.matmul.allow_tf32 = False

    experiment_kind = "_interaction_only" if not args.adapter_calibration else ""
    experiment_name = (
        f"{args.backbone_model}_{args.adapter_mode}{experiment_kind}_closed_form"
    )
    checkpoint_dir = Path(args.checkpoints) / experiment_name
    result_dir = Path(args.results) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.json"
    if args.resume and summary_path.is_file():
        print(f"ADAPTER_FIT_SKIP completed={summary_path}")
        print(summary_path.read_text())
        return

    experiment = Exp(args)
    wrapper = _unwrapped(experiment.model)
    if wrapper.__class__.__module__ != "models.InteractionAdapter":
        raise TypeError("--adapter_closed_form requires --model InteractionAdapter.")
    wrapper.eval()
    device = experiment.device

    train_data, _ = experiment._get_data("train")
    val_data, _ = experiment._get_data("val")
    test_data, _ = experiment._get_data("test")
    fit_stride = max(1, int(args.adapter_fit_stride))
    train_loader = _loader(train_data, args, fit_stride)
    val_loader = _loader(val_data, args, fit_stride)
    test_loader = _loader(test_data, args, 1)
    print(
        f"ADAPTER_FIT_START backbone={args.backbone_model} mode={args.adapter_mode} "
        f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)} "
        f"test_windows={len(test_loader.dataset)} stride={fit_stride}",
        flush=True,
    )

    efficiency_profile = os.environ.get("ADAPTER_EFFICIENCY_PROFILE") == "1"
    if efficiency_profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    train_stats = _collect_statistics(
        wrapper, train_data, train_loader, device, args, "train"
    )
    if efficiency_profile and device.type == "cuda":
        torch.cuda.synchronize(device)
    train_accumulation_seconds = time.perf_counter() - fit_started
    validation_started = time.perf_counter()
    val_stats = _collect_statistics(
        wrapper, val_data, val_loader, device, args, "val"
    )
    if efficiency_profile and device.type == "cuda":
        torch.cuda.synchronize(device)
    validation_accumulation_seconds = time.perf_counter() - validation_started
    solve_started = time.perf_counter()
    candidates = args.adapter_ridge or DEFAULT_RIDGES
    candidate_results = []
    for ridge in candidates:
        coefficients = _solve(train_stats, ridge)
        validation_mse = _validation_mse(val_stats, coefficients)
        candidate_results.append((validation_mse, float(ridge), coefficients))
        print(f"ADAPTER_RIDGE ridge={ridge:g} val_mse={validation_mse:.9f}")
    validation_mse, selected_ridge, coefficients = min(
        candidate_results, key=lambda item: item[0]
    )
    _load_coefficients(wrapper.adapter, coefficients.to(device))
    if efficiency_profile and device.type == "cuda":
        torch.cuda.synchronize(device)
    solve_seconds = time.perf_counter() - solve_started
    full_training_seconds = time.perf_counter() - fit_started
    if efficiency_profile:
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 2 ** 20
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 2 ** 20
        else:
            peak_memory_mb = 0.0
            peak_reserved_mb = 0.0
        profile = {
            "backbone_model": args.backbone_model,
            "adapter_mode": args.adapter_mode,
            "batch_size": args.batch_size,
            "fit_stride": fit_stride,
            "train_batches": len(train_loader),
            "validation_batches": len(val_loader),
            "feature_accumulation_seconds": train_accumulation_seconds,
            "validation_accumulation_seconds": validation_accumulation_seconds,
            "solve_seconds": solve_seconds,
            "full_training_seconds": full_training_seconds,
            "peak_memory_mb": peak_memory_mb,
            "peak_reserved_mb": peak_reserved_mb,
        }
        print("ADAPTER_EFFICIENCY_RESULT " + json.dumps(profile), flush=True)
        profile_path = os.environ.get("ADAPTER_EFFICIENCY_PROFILE_PATH")
        if profile_path:
            Path(profile_path).write_text(
                json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        if os.environ.get("ADAPTER_EFFICIENCY_PROFILE_ONLY") == "1":
            return
    result = _evaluate(wrapper, test_data, test_loader, device, args)
    result.update(
        {
            "backbone_model": args.backbone_model,
            "adapter_mode": args.adapter_mode,
            "use_calibration": bool(wrapper.adapter.use_calibration),
            "fit_stride": fit_stride,
            "selected_ridge": selected_ridge,
            "validation_mse": validation_mse,
            "trainable_parameters": sum(
                parameter.numel() for parameter in wrapper.adapter.parameters()
            ),
            "variable_names": list(test_data.variable_names),
        }
    )

    _atomic_torch_save(
        {
            "format_version": 1,
            "backbone_model": args.backbone_model,
            "adapter_mode": args.adapter_mode,
            "use_calibration": bool(wrapper.adapter.use_calibration),
            "selected_ridge": selected_ridge,
            "adapter_state_dict": wrapper.adapter.state_dict(),
        },
        checkpoint_dir / "checkpoint.pth",
    )
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
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
        normalized_overall_mse=np.asarray(result["adapter"]["normalized_mse"]),
        normalized_overall_mae=np.asarray(result["adapter"]["normalized_mae"]),
    )
    with open(args.result_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print("ADAPTER_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
