"""Run the closed-form interaction adapter on every retained M3 baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "forecasting"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiment_records" / "french_repeated" / "adapters"
)
DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT / "experiment_records" / "french_repeated" / "baselines"
)


EXPERIMENTS = {
    # SS: neither station nor variable interaction is present.
    "DLinear": ("both", "long_term_forecast_French_UVTRH_48_72_DLinear_French_ftM_sl48_ll0_pl72_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0", 1),
    "PatchTST": ("both", "long_term_forecast_French_UVTRH_48_72_PatchTST_French_ftM_sl48_ll0_pl72_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_Exp_seed2024_0", 1),
    "Moirai": ("both", None, 6),
    "TimeMoE": ("both", None, 6),
    "Timer": ("both", None, 6),
    # MS: supplement the station model with cross-variable interaction.
    "CDPNet": ("variable", "long_term_forecast_French_UVTRH_48_72_CDPNet_French_ftM_sl48_ll0_pl72_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0", 1),
    "STELLA": ("variable", "long_term_forecast_French_UVTRH_48_72_STELLA_French_ftM_sl48_ll0_pl72_dm32_nh8_el2_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0", 1),
    "Corrformer": ("variable", "long_term_forecast_French_UVTRH_48_72_Corrformer_French_ftM_sl48_ll24_pl72_dm256_nh8_el2_dl1_df512_fc1_ebtimeF_Exp_seed2024_0", 1),
    "EasyST": ("variable", None, 1),
    "S2Transformer": ("variable", None, 1),
    # SM: supplement the within-station multivariate model with station mixing.
    "DUET": ("station", "long_term_forecast_French_UVTRH_48_72_DUET_French_ftM_sl48_ll0_pl72_dm256_nh8_el2_dl1_df512_fc1_ebtimeF_Exp_seed2024_0", 1),
    "TQNet": ("station", "long_term_forecast_French_UVTRH_48_72_TQNet_French_ftM_sl48_ll0_pl72_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0", 1),
    "TimerXL": ("station", "long_term_forecast_French_UVTRH_48_72_TimerXL_French_ftM_sl48_ll0_pl72_dm1024_nh8_el8_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0", 1),
    "iTransformer": ("station", "long_term_forecast_French_UVTRH_48_72_iTransformer_French_ftM_sl48_ll0_pl72_dm256_nh8_el4_dl1_df512_fc3_ebtimeF_Exp_seed2024_0", 1),
    "xPatch": ("station", None, 1),
    # Existing MM baseline: test whether both residual interactions still help.
    "HiSTGNN": ("both", "long_term_forecast_French_UVTRH_48_72_HiSTGNN_French_ftM_sl48_ll0_pl72_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0", 1),
}

DEFAULT_ORDER = (
    "DLinear", "xPatch", "TQNet", "STELLA", "EasyST", "DUET", "iTransformer",
    "PatchTST", "CDPNet", "TimerXL", "HiSTGNN", "S2Transformer",
    "Corrformer", "Timer", "TimeMoE", "Moirai",
)

FOUNDATION_MODELS = {"Moirai", "TimeMoE", "Timer"}
FOUNDATION_BATCH_SIZE = {"Moirai": 64, "TimeMoE": 5}
OOM_PATTERN = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error:\s+out of memory|"
    r"CUBLAS_STATUS_ALLOC_FAILED|cuDNN error:\s+CUDNN_STATUS_ALLOC_FAILED|"
    r"canUse32BitIndexMath",
    re.IGNORECASE,
)


def _baseline_arguments(model, script_root=SCRIPT_ROOT, script_suffix="French"):
    script_path = Path(script_root) / f"{model}_{script_suffix}.sh"
    source = script_path.read_text()
    model_match = re.search(r"^model_name=(\S+)", source, re.MULTILINE)
    if not model_match:
        raise ValueError(f"Cannot find model_name in {script_path}")
    source = source.replace("\\\n", " ")
    command_match = re.search(r"python\s+-u\s+run\.py\s+(.+)$", source)
    if not command_match:
        raise ValueError(f"Cannot parse run.py command in {script_path}")
    return [
        model if token == "$model_name" else token
        for token in shlex.split(command_match.group(1))
    ]


def _replace_value(tokens, flag, value):
    if flag in tokens:
        index = tokens.index(flag)
        tokens[index + 1] = str(value)
    else:
        tokens.extend([flag, str(value)])


def _find_checkpoint(baseline_root, model, seed, data_name="French"):
    matches = sorted(
        (Path(baseline_root) / "checkpoints").glob(
            f"*_{model}_{data_name}_*seed{seed}_0/checkpoint.pth"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {model} seed {seed} checkpoint under "
            f"{Path(baseline_root) / 'checkpoints'}, found {len(matches)}: "
            f"{matches}"
        )
    return matches[0].resolve()


def _baseline_metrics(baseline_root, model, seed, data_name):
    return sorted(
        (Path(baseline_root) / "results").glob(
            f"*_{model}_{data_name}_*seed{seed}_0/metrics.npy"
        )
    )


def _adapter_metrics(output_root, model, mode, seed, data_name,
                     adapter_model_id_prefix, interaction_only=False):
    del data_name, adapter_model_id_prefix
    experiment_kind = "_interaction_only" if interaction_only else ""
    metrics = (
        Path(output_root) / f"seed{seed}" / "results"
        / f"{model}_{mode}{experiment_kind}_closed_form" / "metrics.npy"
    )
    return [metrics] if metrics.is_file() else []


def _write_status_log(log_root, model, seed, status, **details):
    model_log_root = Path(log_root)
    model_log_root.mkdir(parents=True, exist_ok=True)
    path = model_log_root / f"{model}.seed{seed}.{status.lower()}.log"
    lines = [
        f"ADAPTER_{status}",
        f"model={model}",
        f"seed={seed}",
        *(f"{key}={value}" for key, value in details.items()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _remove_status_log(log_root, model, seed, status):
    path = Path(log_root) / f"{model}.seed{seed}.{status.lower()}.log"
    path.unlink(missing_ok=True)


def _command(
    model, seed, python_bin, baseline_root, output_root,
    script_root=SCRIPT_ROOT, script_suffix="French", data_name="French",
    adapter_model_id_prefix="French_UVTRH_48_72",
    num_nodes=133, station_coords_path="./dataset/French/station_coords.npy",
    batch_size_overrides=None, require_checkpoint=True,
    adapter_fit_stride=None, batch_size_multipliers=None,
    batch_retry_divisor=1, interaction_only=False,
):
    mode, _, stride = EXPERIMENTS[model]
    if adapter_fit_stride is not None:
        stride = adapter_fit_stride
    seed_output_root = Path(output_root) / f"seed{seed}"
    tokens = _baseline_arguments(model, script_root, script_suffix)
    _replace_value(tokens, "--is_training", 1)
    _replace_value(tokens, "--model", "InteractionAdapter")
    _replace_value(
        tokens, "--model_id", f"{adapter_model_id_prefix}_{model}_{mode}Adapter"
    )
    _replace_value(tokens, "--seed", seed)
    _replace_value(tokens, "--num_nodes", num_nodes)
    if batch_size_overrides is None:
        batch_size_overrides = FOUNDATION_BATCH_SIZE
    if model in batch_size_overrides:
        _replace_value(tokens, "--batch_size", batch_size_overrides[model])
    if batch_size_multipliers is None:
        batch_size_multipliers = {}
    batch_index = tokens.index("--batch_size") + 1
    base_batch_size = int(tokens[batch_index])
    multiplier = batch_size_multipliers.get(model, 1)
    expanded_batch_size = base_batch_size * multiplier
    tokens[batch_index] = str(
        max(1, expanded_batch_size // max(1, batch_retry_divisor))
    )
    if multiplier > 1:
        if "--num_workers" in tokens:
            worker_index = tokens.index("--num_workers") + 1
            tokens[worker_index] = str(min(int(tokens[worker_index]), 2))
        else:
            tokens.extend(["--num_workers", "2"])
    _replace_value(tokens, "--checkpoints", seed_output_root / "checkpoints")
    _replace_value(tokens, "--results", seed_output_root / "results")
    _replace_value(tokens, "--test_results", seed_output_root / "test_results")
    _replace_value(
        tokens, "--result_file", seed_output_root / "result_long_term_forecast.jsonl"
    )
    if "--station_coords_path" not in tokens:
        tokens.extend(["--station_coords_path", station_coords_path])
    tokens.extend(
        [
            "--backbone_model", model,
            "--adapter_mode", mode,
            "--adapter_fit_stride", str(stride),
            "--adapter_closed_form",
        ]
    )
    if interaction_only:
        tokens.append("--no_adapter_calibration")
    if model not in FOUNDATION_MODELS:
        if require_checkpoint:
            checkpoint = _find_checkpoint(baseline_root, model, seed, data_name)
        else:
            checkpoint = (
                Path(baseline_root) / "checkpoints"
                / f"<completed-{model}-{data_name}-seed{seed}-checkpoint.pth>"
            )
        tokens.extend(["--backbone_checkpoint", str(checkpoint)])
    if "--resume" not in tokens:
        tokens.append("--resume")
    return [str(python_bin), "-u", "run.py", *map(str, tokens)]


def _parse_seeds(value):
    seeds = tuple(int(item) for item in value.replace(",", " ").split())
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _parse_seed_overrides(value):
    overrides = {}
    for assignment in value.split(";"):
        assignment = assignment.strip()
        if not assignment:
            continue
        model, seeds = assignment.split("=", 1)
        overrides[model.strip()] = _parse_seeds(seeds)
    return overrides


def _parse_batch_size_overrides(value):
    """Parse ``MODEL=SIZE;MODEL=SIZE``; an empty value preserves script sizes."""
    overrides = {}
    for assignment in value.split(";"):
        assignment = assignment.strip()
        if not assignment:
            continue
        if "=" not in assignment:
            raise ValueError(
                f"Invalid batch-size override {assignment!r}; expected MODEL=SIZE."
            )
        model, size = assignment.split("=", 1)
        model = model.strip()
        size = int(size)
        if size <= 0:
            raise ValueError(f"Batch size for {model} must be positive, got {size}.")
        overrides[model] = size
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seeds", default="2025,2026")
    parser.add_argument(
        "--seed-overrides",
        default="EasyST=2024,2025,2026;S2Transformer=2024,2025,2026;"
        "xPatch=2024,2025,2026",
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--log-root", type=Path, default=None,
        help="central Adapter log root; defaults to OUTPUT_ROOT/logs",
    )
    parser.add_argument("--script-root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--script-suffix", default="French")
    parser.add_argument("--data-name", default="French")
    parser.add_argument("--adapter-model-id-prefix", default="French_UVTRH_48_72")
    parser.add_argument("--num-nodes", type=int, default=133)
    parser.add_argument(
        "--station-coords-path", default="./dataset/French/station_coords.npy"
    )
    parser.add_argument(
        "--python-bin", default="/root/miniconda3/envs/tslib/bin/python"
    )
    parser.add_argument(
        "--batch-size-overrides", default="Moirai=64;TimeMoE=5",
        help=(
            "semicolon-separated MODEL=SIZE overrides; pass an empty string to "
            "retain each baseline script's memory-tested batch size"
        ),
    )
    parser.add_argument(
        "--batch-size-multipliers", default="",
        help="semicolon-separated MODEL=FACTOR batch expansion mapping",
    )
    parser.add_argument(
        "--fit-stride", type=int, default=None,
        help="override the train/validation fitting stride for every Adapter",
    )
    parser.add_argument("--max-oom-retries", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument(
        "--only-if-baseline-complete", action="store_true",
        help="defer jobs whose baseline final-test metrics do not exist yet",
    )
    parser.add_argument(
        "--no-skip-completed", action="store_true",
        help="rerun adapters even when their final-test metrics already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print all adapter commands without requiring checkpoints or GPUs",
    )
    parser.add_argument(
        "--interaction-only", action="store_true",
        help="disable the calibration scale/bias branch",
    )
    args = parser.parse_args()
    log_root = args.log_root or (args.output_root / "logs")
    log_root.mkdir(parents=True, exist_ok=True)
    models = args.models or list(DEFAULT_ORDER)
    unknown = sorted(set(models).difference(EXPERIMENTS))
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    if args.poll_interval < 0:
        raise ValueError("--poll-interval must be non-negative.")
    if args.fit_stride is not None and args.fit_stride <= 0:
        raise ValueError("--fit-stride must be positive.")
    if args.max_oom_retries < 0:
        raise ValueError("--max-oom-retries must be non-negative.")

    seeds = _parse_seeds(args.seeds)
    seed_overrides = _parse_seed_overrides(args.seed_overrides)
    unknown_overrides = sorted(set(seed_overrides).difference(EXPERIMENTS))
    if unknown_overrides:
        parser.error(f"unknown seed override model(s): {', '.join(unknown_overrides)}")
    batch_size_overrides = _parse_batch_size_overrides(args.batch_size_overrides)
    batch_size_multipliers = _parse_batch_size_overrides(
        args.batch_size_multipliers
    )
    unknown_batch_overrides = sorted(
        (set(batch_size_overrides) | set(batch_size_multipliers)).difference(
            EXPERIMENTS
        )
    )
    if unknown_batch_overrides:
        parser.error(
            "unknown batch-size override model(s): "
            + ", ".join(unknown_batch_overrides)
        )

    requested = [
        (model, seed)
        for model in models
        for seed in seed_overrides.get(model, seeds)
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate model/seed adapter jobs were requested.")
    pending = []
    for model, seed in requested:
        mode = EXPERIMENTS[model][0]
        completed = _adapter_metrics(
            args.output_root, model, mode, seed, args.data_name,
            args.adapter_model_id_prefix,
            interaction_only=args.interaction_only,
        )
        if completed and not args.no_skip_completed:
            _remove_status_log(log_root, model, seed, "BASELINE_PENDING")
            status_log = _write_status_log(
                log_root, model, seed, "COMPLETED_SKIP", metrics=completed[0]
            )
            print(
                f"ADAPTER_BATCH_SKIP model={model} seed={seed} "
                f"reason=completed metrics={completed[0]} log={status_log}",
                flush=True,
            )
            continue
        baseline_metrics = _baseline_metrics(
            args.baseline_root, model, seed, args.data_name
        )
        if args.only_if_baseline_complete and not baseline_metrics:
            status_log = _write_status_log(
                log_root, model, seed, "BASELINE_PENDING",
                reason="baseline_not_complete",
            )
            print(
                f"ADAPTER_BATCH_DEFER model={model} seed={seed} "
                f"reason=baseline_not_complete log={status_log}", flush=True,
            )
            continue
        _remove_status_log(log_root, model, seed, "BASELINE_PENDING")
        pending.append((model, seed))
    running = {}
    retry_counts = {}
    failures = []
    print("ADAPTER_BATCH_ORDER " + " -> ".join(models), flush=True)
    print(
        "ADAPTER_BATCH_JOBS "
        + " -> ".join(f"{model}:seed{seed}" for model, seed in pending),
        flush=True,
    )
    print(
        f"ADAPTER_BATCH_JOB_COUNT requested={len(requested)} pending={len(pending)}",
        flush=True,
    )
    if args.dry_run:
        for index, (model, seed) in enumerate(pending, start=1):
            command = _command(
                model, seed, args.python_bin, args.baseline_root,
                args.output_root, args.script_root, args.script_suffix,
                args.data_name, args.adapter_model_id_prefix,
                args.num_nodes, args.station_coords_path,
                batch_size_overrides=batch_size_overrides,
                require_checkpoint=False,
                adapter_fit_stride=args.fit_stride,
                batch_size_multipliers=batch_size_multipliers,
                interaction_only=args.interaction_only,
            )
            print(
                f"ADAPTER_BATCH_DRY_RUN index={index}/{len(pending)} "
                f"model={model} seed={seed} command={shlex.join(command)}",
                flush=True,
            )
        return

    while pending or running:
        free_gpus = [gpu for gpu in gpus if gpu not in running]
        while pending and free_gpus:
            model, seed = pending.pop(0)
            gpu = free_gpus.pop(0)
            key = (model, seed)
            attempt = retry_counts.get(key, 0) + 1
            log_path = log_root / (
                f"{model}.seed{seed}.try{attempt}.gpu{gpu}.log"
            )
            command = _command(
                model, seed, args.python_bin, args.baseline_root,
                args.output_root, args.script_root, args.script_suffix,
                args.data_name, args.adapter_model_id_prefix,
                args.num_nodes, args.station_coords_path,
                batch_size_overrides=batch_size_overrides,
                adapter_fit_stride=args.fit_stride,
                batch_size_multipliers=batch_size_multipliers,
                interaction_only=args.interaction_only,
                batch_retry_divisor=2 ** (attempt - 1),
            )
            handle = open(log_path, "a", buffering=1)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = (
                model, seed, process, handle, log_path, time.time(), attempt
            )
            print(
                f"ADAPTER_BATCH_START model={model} seed={seed} gpu={gpu} "
                f"pid={process.pid} attempt={attempt} "
                f"log={log_path}", flush=True,
            )
        time.sleep(args.poll_interval)
        for gpu, item in list(running.items()):
            model, seed, process, handle, log_path, started, attempt = item
            status = process.poll()
            if status is None:
                continue
            handle.close()
            elapsed = time.time() - started
            with open(log_path, "a", encoding="utf-8") as status_handle:
                status_handle.write(
                    f"\nADAPTER_BATCH_DONE status={status} elapsed={elapsed:.1f}s\n"
                )
            print(
                f"ADAPTER_BATCH_DONE model={model} seed={seed} gpu={gpu} "
                f"status={status} "
                f"elapsed={elapsed:.1f}s", flush=True,
            )
            if status != 0:
                log_text = log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if (
                    OOM_PATTERN.search(log_text)
                    and attempt <= args.max_oom_retries
                ):
                    retry_counts[(model, seed)] = attempt
                    pending.append((model, seed))
                    print(
                        f"ADAPTER_BATCH_RETRY model={model} seed={seed} "
                        f"reason=oom next_attempt={attempt + 1}",
                        flush=True,
                    )
                else:
                    failures.append((model, seed, status, str(log_path)))
            del running[gpu]
    if failures:
        for model, seed, status, log_path in failures:
            print(
                f"ADAPTER_BATCH_FAIL {model} seed={seed} status={status} "
                f"log={log_path}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
