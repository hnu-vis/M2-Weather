"""Run the France lookback-window sweep on all available GPUs.

The forecast horizon stays at 72.  The completed seed-2024/lookback-48 runs are
reused, while lookbacks 24, 72, and 96 are trained once in an isolated output
tree.  HiSTGNN jobs are launched first so that the long tail overlaps all of
the shorter baselines.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_all_adapters import _baseline_arguments, _replace_value


PYTHON = Path("/root/miniconda3/envs/tslib/bin/python")
SCRIPT_ROOT = ROOT / "scripts" / "forecasting_m3_france"
OUTPUT_ROOT = ROOT / "experiment_records" / "m3_france_lookback_sweep"
EXISTING_ROOT = ROOT / "experiment_records" / "m3_france_twsrhp" / "baselines"
LOG_ROOT = OUTPUT_ROOT / "logs"
SEED = 2024
PRED_LEN = 72
NEW_LOOKBACKS = (24, 72, 96)
ALL_LOOKBACKS = (24, 48, 72, 96)
MODELS = (
    "PatchTST",
    "xPatch",
    "STELLA",
    "S2Transformer",
    "DUET",
    "iTransformer",
    "HiSTGNN",
)
CATEGORIES = {
    "PatchTST": "SSSV",
    "xPatch": "SSSV",
    "STELLA": "MSSV",
    "S2Transformer": "MSSV",
    "DUET": "SSMV",
    "iTransformer": "SSMV",
    "HiSTGNN": "MSMV",
}


def metric_matches(root: Path, model: str, lookback: int) -> list[Path]:
    return sorted(
        (root / "results").glob(
            f"*_{model}_M3France_*_sl{lookback}_*seed{SEED}_0/metrics_by_variable.npz"
        )
    )


def metric_path(model: str, lookback: int) -> Path | None:
    root = EXISTING_ROOT if lookback == 48 else OUTPUT_ROOT
    matches = metric_matches(root, model, lookback)
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one result for {model}/L={lookback}, found {matches}"
        )
    return matches[0] if matches else None


def command(model: str, lookback: int, batch_divisor: int = 1) -> list[str]:
    tokens = _baseline_arguments(model, SCRIPT_ROOT, "M3France")
    _replace_value(tokens, "--seed", SEED)
    _replace_value(tokens, "--seq_len", lookback)
    _replace_value(tokens, "--pred_len", PRED_LEN)
    _replace_value(tokens, "--model_id", f"M3France_TWSRHP_{lookback}_{PRED_LEN}")
    _replace_value(tokens, "--num_workers", 2)
    _replace_value(tokens, "--checkpoints", OUTPUT_ROOT / "checkpoints")
    _replace_value(tokens, "--results", OUTPUT_ROOT / "results")
    _replace_value(tokens, "--test_results", OUTPUT_ROOT / "test_results")
    _replace_value(tokens, "--result_file", OUTPUT_ROOT / "result_long_term_forecast.txt")
    batch_index = tokens.index("--batch_size") + 1
    tokens[batch_index] = str(max(1, int(tokens[batch_index]) // batch_divisor))
    if "--skip_epoch_test" not in tokens:
        tokens.append("--skip_epoch_test")
    return [str(PYTHON), "-u", "run.py", *map(str, tokens)]


def write_manifest() -> None:
    payload = {
        "dataset": "M3France",
        "seed": SEED,
        "forecast_length": PRED_LEN,
        "lookback_lengths": list(ALL_LOOKBACKS),
        "models": [
            {"model": model, "category": CATEGORIES[model]} for model in MODELS
        ],
        "reused_lookback_48_root": str(EXISTING_ROOT),
        "new_output_root": str(OUTPUT_ROOT),
        "per_epoch_test_disabled": True,
        "final_test_enabled": True,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_status(states: dict[str, dict], started_at: float) -> None:
    payload = {
        "manager_pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started_at, 1),
        "counts": {
            status: sum(item["status"] == status for item in states.values())
            for status in ("reused", "pending", "running", "completed", "failed")
        },
        "jobs": states,
    }
    temporary = OUTPUT_ROOT / "status.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(OUTPUT_ROOT / "status.json")


def text_values(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def write_summary() -> None:
    rows: list[dict] = []
    for model in MODELS:
        for lookback in ALL_LOOKBACKS:
            path = metric_path(model, lookback)
            if path is None:
                continue
            with np.load(path) as metrics:
                names = text_values(metrics["variable_names"])
                mse = np.asarray(metrics["normalized_mse"], dtype=float)
                mae = np.asarray(metrics["normalized_mae"], dtype=float)
                row = {
                    "category": CATEGORIES[model],
                    "model": model,
                    "lookback": lookback,
                    "forecast_length": PRED_LEN,
                    "seed": SEED,
                    "normalized_mse": float(metrics["normalized_overall_mse"]),
                    "normalized_mae": float(metrics["normalized_overall_mae"]),
                    "metrics_path": str(path),
                }
                for name, value in zip(names, mse):
                    row[f"{name}_normalized_mse"] = float(value)
                for name, value in zip(names, mae):
                    row[f"{name}_normalized_mae"] = float(value)
                rows.append(row)
    rows.sort(key=lambda row: (row["category"], row["model"], row["lookback"]))
    (OUTPUT_ROOT / "lookback_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (OUTPUT_ROOT / "lookback_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    write_manifest()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    states: dict[str, dict] = {}
    for model in MODELS:
        path = metric_path(model, 48)
        if path is None:
            raise FileNotFoundError(f"Missing reusable lookback-48 result for {model}")
        states[f"{model}.L48"] = {
            "model": model,
            "category": CATEGORIES[model],
            "lookback": 48,
            "status": "reused",
            "metrics_path": str(path),
        }

    jobs = [("HiSTGNN", length, 1) for length in NEW_LOOKBACKS]
    jobs += [
        (model, length, 1)
        for model in MODELS
        if model != "HiSTGNN"
        for length in NEW_LOOKBACKS
    ]
    for model, length, _ in jobs:
        key = f"{model}.L{length}"
        existing = metric_path(model, length)
        states[key] = {
            "model": model,
            "category": CATEGORIES[model],
            "lookback": length,
            "status": "completed" if existing else "pending",
            **({"metrics_path": str(existing)} if existing else {}),
        }
    jobs = [job for job in jobs if metric_path(job[0], job[1]) is None]
    write_summary()
    write_status(states, started_at)

    active: dict[str, tuple[subprocess.Popen, object, str, int, int, float, Path]] = {}
    failures: list[str] = []
    gpus = [str(index) for index in range(6)]
    while jobs or active:
        for gpu in gpus:
            if gpu in active or not jobs:
                continue
            model, lookback, divisor = jobs.pop(0)
            key = f"{model}.L{lookback}"
            attempt = int(round(np.log2(divisor))) + 1
            log_path = LOG_ROOT / f"{model}.L{lookback}.try{attempt}.gpu{gpu}.log"
            log = log_path.open("w", encoding="utf-8")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
            )
            process = subprocess.Popen(
                command(model, lookback, divisor),
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            launched = time.time()
            active[gpu] = (
                process, log, model, lookback, divisor, launched, log_path
            )
            states[key].update(
                status="running",
                gpu=int(gpu),
                pid=process.pid,
                batch_divisor=divisor,
                log_path=str(log_path),
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            print(
                f"SWEEP_START model={model} lookback={lookback} gpu={gpu} "
                f"pid={process.pid} batch_divisor={divisor}",
                flush=True,
            )
            write_status(states, started_at)

        for gpu, item in list(active.items()):
            process, log, model, lookback, divisor, launched, log_path = item
            returncode = process.poll()
            if returncode is None:
                continue
            log.close()
            del active[gpu]
            key = f"{model}.L{lookback}"
            elapsed = round(time.time() - launched, 1)
            output = log_path.read_text(encoding="utf-8", errors="ignore")
            oom = bool(
                re.search(
                    r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|"
                    r"canUse32BitIndexMath",
                    output,
                    re.IGNORECASE,
                )
            )
            if returncode and oom and divisor < 64:
                jobs.insert(0, (model, lookback, divisor * 2))
                states[key].update(
                    status="pending",
                    last_returncode=returncode,
                    last_elapsed_seconds=elapsed,
                    retry_batch_divisor=divisor * 2,
                )
                print(
                    f"SWEEP_RETRY model={model} lookback={lookback} "
                    f"batch_divisor={divisor * 2}",
                    flush=True,
                )
            elif returncode:
                failures.append(key)
                states[key].update(
                    status="failed",
                    returncode=returncode,
                    elapsed_seconds=elapsed,
                )
                print(
                    f"SWEEP_FAIL model={model} lookback={lookback} "
                    f"returncode={returncode} log={log_path}",
                    flush=True,
                )
            else:
                result = metric_path(model, lookback)
                if result is None:
                    failures.append(key)
                    states[key].update(
                        status="failed",
                        returncode=0,
                        elapsed_seconds=elapsed,
                        error="process succeeded but metrics file is missing",
                    )
                    print(f"SWEEP_METRICS_MISSING {key}", flush=True)
                else:
                    states[key].update(
                        status="completed",
                        returncode=0,
                        elapsed_seconds=elapsed,
                        metrics_path=str(result),
                        finished_at=time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    )
                    print(
                        f"SWEEP_DONE model={model} lookback={lookback} "
                        f"gpu={gpu} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            write_summary()
            write_status(states, started_at)
        if active:
            time.sleep(5)

    write_summary()
    write_status(states, started_at)
    if failures:
        raise SystemExit(f"Sweep failures: {failures}")
    print("SWEEP_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
