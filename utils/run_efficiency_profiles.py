"""Run isolated one-epoch baseline and closed-form Adapter efficiency profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_all_adapters import (
    _baseline_arguments,
    _command,
    _replace_value,
)


PYTHON = "/root/miniconda3/envs/tslib/bin/python"
MODELS = (
    "PatchTST", "xPatch", "CDPNet", "Corrformer",
    "DUET", "iTransformer", "HiSTGNN",
)
EUROPE_MULTIPLIERS = {
    "DUET": 3,
    "iTransformer": 2,
}


def baseline_command(dataset: str, model: str, divisor: int) -> list[str]:
    script_root = ROOT / "scripts" / f"forecasting_m3_{dataset.lower()}"
    data_name = f"M3{dataset}"
    tokens = _baseline_arguments(model, script_root, data_name)
    _replace_value(tokens, "--seed", 2024)
    _replace_value(tokens, "--train_stride", 1 if dataset == "France" else 6)
    _replace_value(tokens, "--profile_steps", 999999999)
    if dataset == "Europe":
        index = tokens.index("--batch_size") + 1
        batch_size = int(tokens[index]) * EUROPE_MULTIPLIERS.get(model, 1)
        tokens[index] = str(max(1, batch_size // divisor))
        _replace_value(tokens, "--num_workers", 2)
    elif divisor > 1:
        index = tokens.index("--batch_size") + 1
        tokens[index] = str(max(1, int(tokens[index]) // divisor))
    return [PYTHON, "-u", "run.py", *tokens]


def adapter_command(dataset: str) -> list[str]:
    is_france = dataset == "France"
    record = ROOT / "experiment_records" / (
        "m3_france_twsrhp" if is_france else "m3_europe_twsrhp_fast_s6"
    )
    output = ROOT / "paper" / "efficiency_analysis" / dataset / "adapter_work"
    multipliers = {} if is_france else {"DLinear": 3}
    return _command(
        "DLinear",
        2024,
        PYTHON,
        record / "baselines",
        output,
        script_root=ROOT / "scripts" / f"forecasting_m3_{dataset.lower()}",
        script_suffix=f"M3{dataset}",
        data_name=f"M3{dataset}",
        adapter_model_id_prefix=f"M3{dataset}_TWSRHP_48_72",
        num_nodes=176 if is_france else 983,
        station_coords_path=f"./dataset/M3/{dataset}/station_coords.npy",
        batch_size_overrides={},
        adapter_fit_stride=1 if is_france else 6,
        batch_size_multipliers=multipliers,
    )


def run_baselines(gpus: list[str]) -> None:
    output = ROOT / "paper" / "efficiency_analysis"
    queue = [
        (dataset, model, 1)
        for dataset in ("Europe", "France")
        for model in MODELS
    ]
    # Put the known long jobs first so they overlap the shorter profiles.
    priority = {("Europe", "HiSTGNN"): 0, ("Europe", "Corrformer"): 1,
                ("France", "Corrformer"): 2, ("France", "HiSTGNN"): 3}
    queue.sort(key=lambda item: priority.get(item[:2], 10))
    active: dict[str, tuple[subprocess.Popen, str, str, object, int]] = {}
    failures = []
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            dataset, model, divisor = queue.pop(0)
            folder = output / dataset / "baseline"
            folder.mkdir(parents=True, exist_ok=True)
            log_path = folder / f"{model}.seed2024.log"
            log = log_path.open("w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            command = baseline_command(dataset, model, divisor)
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, dataset, model, log, divisor)
            print("PROFILE_START", dataset, model, "gpu", gpu,
                  "divisor", divisor, flush=True)
        for gpu, item in list(active.items()):
            process, dataset, model, log, divisor = item
            if process.poll() is None:
                continue
            log.close()
            del active[gpu]
            path = output / dataset / "baseline" / f"{model}.seed2024.log"
            text = path.read_text(errors="ignore")
            if process.returncode and (
                "CUDA out of memory" in text or "OutOfMemoryError" in text
            ) and divisor < 64:
                queue.insert(0, (dataset, model, divisor * 2))
                print("PROFILE_RETRY", dataset, model,
                      "divisor", divisor * 2, flush=True)
            elif process.returncode:
                failures.append((dataset, model, process.returncode))
                print("PROFILE_FAIL", dataset, model,
                      process.returncode, flush=True)
            else:
                match = re.search(r"PROFILE_RESULT .+", text)
                print(match.group(0) if match else
                      f"PROFILE_MISSING {dataset} {model}", flush=True)
        if active:
            time.sleep(2)
    if failures:
        raise SystemExit(f"Baseline profile failures: {failures}")


def run_adapter(dataset: str, gpu: str) -> None:
    folder = ROOT / "paper" / "efficiency_analysis" / dataset / "adapter"
    folder.mkdir(parents=True, exist_ok=True)
    profile_path = folder / "DLinear_both_seed2024.json"
    log_path = folder / "DLinear_both_seed2024.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        ADAPTER_EFFICIENCY_PROFILE="1",
        ADAPTER_EFFICIENCY_PROFILE_ONLY="1",
        ADAPTER_EFFICIENCY_PROFILE_PATH=str(profile_path),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            adapter_command(dataset), cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise SystemExit(f"Adapter profile failed for {dataset}; see {log_path}")
    print("ADAPTER_PROFILE_DONE", dataset,
          json.loads(profile_path.read_text()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "adapter"))
    parser.add_argument("--dataset", choices=("France", "Europe"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args()
    if args.action == "baseline":
        run_baselines([item for item in args.gpus.split(",") if item])
    else:
        if not args.dataset:
            parser.error("--dataset is required for adapter")
        run_adapter(args.dataset, args.gpu)
