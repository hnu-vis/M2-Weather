"""Schedule the France/Global KNN Adapter sensitivity sweep on six GPUs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_all_adapters import (
    EXPERIMENTS,
    FOUNDATION_BATCH_SIZE,
    OOM_PATTERN,
    _command,
    _replace_value,
)


OUTPUT_ROOT = ROOT / "experiment_records" / "adapter_knn_sensitivity"
LOG_ROOT = ROOT / "logs" / "adapter_knn_sensitivity"
STATUS_PATH = OUTPUT_ROOT / "status.json"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
PYTHON = "/root/miniconda3/envs/tslib/bin/python"
MODELS = ("iTransformer", "TQNet", "DUET", "Moirai", "TimerXL")
K_VALUES = (2, 4, 8, 16)
SEED = 2024
GPUS = tuple(str(index) for index in range(6))
MAX_OOM_RETRIES = 3

EVAL_BATCH_SIZE = {
    "France": {
        "iTransformer": 512,
        "TQNet": 1024,
        "DUET": 1024,
        "Moirai": 64,
        "TimerXL": 64,
    },
    "Global": {
        "iTransformer": 32,
        "TQNet": 128,
        "DUET": 64,
        "Moirai": 32,
        "TimerXL": 8,
    },
}

DATASETS = {
    "France": {
        "script_root": ROOT / "scripts" / "forecasting_m3_france",
        "script_suffix": "M3France",
        "data_name": "M3France",
        "prefix": "M3France_TWSRHP_48_72_KNNSensitivity",
        "num_nodes": 176,
        "coords": "./dataset/M3/France/station_coords.npy",
        "baseline_root": ROOT / "experiment_records" / "m3_france_twsrhp" / "baselines",
        "fit_stride": None,
    },
    "Global": {
        "script_root": ROOT / "scripts" / "forecasting_m3_global",
        "script_suffix": "M3Global",
        "data_name": "M3Global",
        "prefix": "M3Global_TWSRHP_48_72_KNNSensitivity",
        "num_nodes": 2504,
        "coords": "./dataset/M3/Global/station_coords.npy",
        "baseline_root": ROOT / "experiment_records" / "m3_global_twsrhp_fast_s6" / "baselines",
        "fit_stride": 6,
    },
}


def result_path(dataset: str, model: str, k: int) -> Path:
    mode = EXPERIMENTS[model][0]
    return (
        OUTPUT_ROOT / dataset / f"seed{SEED}" / "results" / f"k{k}"
        / f"{model}_{mode}_closed_form" / "summary.json"
    )


def complete(dataset: str, model: str) -> bool:
    return all(result_path(dataset, model, k).is_file() for k in K_VALUES)


def build_command(dataset: str, model: str, attempt: int):
    config = DATASETS[dataset]
    command = _command(
        model=model,
        seed=SEED,
        python_bin=PYTHON,
        baseline_root=config["baseline_root"],
        output_root=OUTPUT_ROOT / dataset,
        script_root=config["script_root"],
        script_suffix=config["script_suffix"],
        data_name=config["data_name"],
        adapter_model_id_prefix=config["prefix"],
        num_nodes=config["num_nodes"],
        station_coords_path=config["coords"],
        batch_size_overrides=FOUNDATION_BATCH_SIZE,
        adapter_fit_stride=config["fit_stride"],
        batch_retry_divisor=2 ** (attempt - 1),
    )
    tokens = list(command)
    tokens.remove("--adapter_closed_form")
    _replace_value(tokens, "--adapter_k_neighbors", K_VALUES[0])
    # Reduce the actual evaluation batch on OOM retries as well.  The shared
    # command builder only scales the foundation-model training batch, while
    # this sweep overrides eval_batch_size explicitly below.
    eval_batch_size = max(
        1, EVAL_BATCH_SIZE[dataset][model] // (2 ** (attempt - 1))
    )
    _replace_value(tokens, "--eval_batch_size", eval_batch_size)
    tokens.extend(
        ["--adapter_knn_sensitivity", "--adapter_knn_values", *map(str, K_VALUES)]
    )
    return tokens


def write_status(pending, running, completed, failures, started):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "models": list(MODELS),
        "datasets": list(DATASETS),
        "k_values": list(K_VALUES),
        "gpus": list(GPUS),
        "shared_backbone_predictions": True,
        "started_unix": started,
        "updated_unix": time.time(),
        "pending": [f"{d}:{m}" for d, m in pending],
        "running": {
            gpu: {
                "dataset": item[0],
                "model": item[1],
                "pid": item[2].pid,
                "attempt": item[6],
                "log": str(item[4]),
                "elapsed_seconds": time.time() - item[5],
            }
            for gpu, item in running.items()
        },
        "completed": [f"{d}:{m}" for d, m in completed],
        "failures": failures,
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS_PATH)


def write_summary():
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            for k in K_VALUES:
                path = result_path(dataset, model, k)
                if not path.is_file():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "adapter_mode": result["adapter_mode"],
                        "seed": result["seed"],
                        "k": k,
                        "selected_ridge": result["selected_ridge"],
                        "validation_mse": result["validation_mse"],
                        "normalized_mse": result["adapter"]["normalized_mse"],
                        "normalized_mae": result["adapter"]["normalized_mae"],
                        "raw_mse": result["adapter"]["mse"],
                        "raw_mae": result["adapter"]["mae"],
                        "normalized_mse_change_percent": result["adapter"][
                            "normalized_mse_change_percent"
                        ],
                        "fit_stride": result["fit_stride"],
                        "summary_path": str(path),
                    }
                )
    SUMMARY_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if records:
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Interleave scales so both are represented from the first scheduling wave.
    requested = [(dataset, model) for model in MODELS for dataset in ("Global", "France")]
    completed = [job for job in requested if complete(*job)]
    pending = [job for job in requested if job not in completed]
    running = {}
    failures = []
    attempts = {}
    started = time.time()
    print(
        "KNN_SCHEDULER_START jobs={} completed={} gpus={} k={}".format(
            len(requested), len(completed), ",".join(GPUS), ",".join(map(str, K_VALUES))
        ),
        flush=True,
    )
    write_status(pending, running, completed, failures, started)

    while pending or running:
        free_gpus = [gpu for gpu in GPUS if gpu not in running]
        while pending and free_gpus:
            dataset, model = pending.pop(0)
            gpu = free_gpus.pop(0)
            key = (dataset, model)
            attempt = attempts.get(key, 0) + 1
            log_path = LOG_ROOT / (
                f"{dataset}.{model}.seed{SEED}.try{attempt}.gpu{gpu}.log"
            )
            command = build_command(dataset, model, attempt)
            handle = log_path.open("w", buffering=1, encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = (
                dataset, model, process, handle, log_path, time.time(), attempt
            )
            print(
                f"KNN_JOB_START dataset={dataset} model={model} gpu={gpu} "
                f"pid={process.pid} attempt={attempt} log={log_path}",
                flush=True,
            )
        write_status(pending, running, completed, failures, started)
        time.sleep(5)
        for gpu, item in list(running.items()):
            dataset, model, process, handle, log_path, job_started, attempt = item
            status = process.poll()
            if status is None:
                continue
            handle.close()
            elapsed = time.time() - job_started
            print(
                f"KNN_JOB_DONE dataset={dataset} model={model} gpu={gpu} "
                f"status={status} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if status == 0 and complete(dataset, model):
                completed.append((dataset, model))
            else:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                if OOM_PATTERN.search(log_text) and attempt <= MAX_OOM_RETRIES:
                    attempts[(dataset, model)] = attempt
                    pending.append((dataset, model))
                    print(
                        f"KNN_JOB_RETRY dataset={dataset} model={model} "
                        f"reason=oom next_attempt={attempt + 1}",
                        flush=True,
                    )
                else:
                    failures.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "status": status,
                            "attempt": attempt,
                            "log": str(log_path),
                        }
                    )
            del running[gpu]
        write_status(pending, running, completed, failures, started)

    write_summary()
    if failures:
        print("KNN_SCHEDULER_FAILED " + json.dumps(failures), flush=True)
        raise SystemExit(1)
    print(
        f"KNN_SCHEDULER_COMPLETE jobs={len(completed)} "
        f"summary={SUMMARY_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
