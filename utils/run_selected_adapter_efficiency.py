"""Profile six frozen backbones with the full two-branch Adapter at batch size 2."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_all_adapters import _command, _replace_value


PYTHON = "/root/miniconda3/envs/tslib/bin/python"
OUTPUT = ROOT / "paper" / "efficiency_analysis" / "batch2_epoch"
MODELS = ("PatchTST", "xPatch", "CDPNet", "Corrformer", "DUET", "iTransformer")
RUN_ORDER = (
    ("France", "CDPNet"), ("Europe", "CDPNet"),
    ("France", "Corrformer"), ("Europe", "Corrformer"),
    ("France", "DUET"), ("France", "iTransformer"),
    ("Europe", "DUET"), ("Europe", "iTransformer"),
)


def build_job(dataset: str, model: str):
    is_france = dataset == "France"
    record = ROOT / "experiment_records" / (
        "m3_france_twsrhp" if is_france else "m3_europe_twsrhp_fast_s6"
    )
    work = OUTPUT / dataset / f"{model.lower()}_both_adapter_work"
    folder = OUTPUT / dataset / "adapter"
    folder.mkdir(parents=True, exist_ok=True)
    profile = folder / f"{model}_both_seed2024_batch2.json"
    log = folder / f"{model}_both_seed2024_batch2.log"
    command = _command(
        model, 2024, PYTHON, record / "baselines", work,
        script_root=ROOT / "scripts" / f"forecasting_m3_{dataset.lower()}",
        script_suffix=f"M3{dataset}", data_name=f"M3{dataset}",
        adapter_model_id_prefix=f"M3{dataset}_TWSRHP_48_72",
        num_nodes=176 if is_france else 983,
        station_coords_path=f"./dataset/M3/{dataset}/station_coords.npy",
        batch_size_overrides={model: 2},
        adapter_fit_stride=1 if is_france else 6,
        batch_size_multipliers={},
    )
    _replace_value(command, "--adapter_mode", "both")
    env = {
        "ADAPTER_EFFICIENCY_PROFILE": "1",
        "ADAPTER_EFFICIENCY_PROFILE_ONLY": "1",
        "ADAPTER_EFFICIENCY_PROFILE_PATH": str(profile),
    }
    return command, log, profile, env


def write_status(active, pending, finished, failures):
    payload = {
        "active": {gpu: {"dataset": item[1], "model": item[2], "pid": item[0].pid}
                   for gpu, item in active.items()},
        "pending": [{"dataset": d, "model": m} for d, m in pending],
        "finished": finished, "failures": failures,
    }
    path = OUTPUT / "adapter_all_models_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def summarize():
    rows = []
    for dataset in ("France", "Europe"):
        for model in MODELS:
            path = OUTPUT / dataset / "adapter" / f"{model}_both_seed2024_batch2.json"
            item = json.loads(path.read_text())
            item = {"dataset": dataset, "model": model, **item}
            item["peak_memory_gib"] = item["peak_memory_mb"] / 1024
            item["peak_reserved_gib"] = item["peak_reserved_mb"] / 1024
            item["profile_path"] = str(path.relative_to(ROOT))
            rows.append(item)
    with (OUTPUT / "adapter_all_models_batch2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUTPUT / "adapter_all_models_batch2.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )


def main():
    gpus = ("0", "1", "2", "3", "4", "5")
    pending = list(RUN_ORDER)
    active = {}
    finished, failures = [], []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            dataset, model = pending.pop(0)
            command, log_path, profile, extra_env = build_job(dataset, model)
            handle = log_path.open("w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", **extra_env)
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (process, dataset, model, handle, log_path, profile)
            print("START", gpu, dataset, model, process.pid, flush=True)
        write_status(active, pending, finished, failures)
        time.sleep(3)
        for gpu, item in list(active.items()):
            process, dataset, model, handle, log_path, profile = item
            if process.poll() is None:
                continue
            handle.close()
            record = {"dataset": dataset, "model": model, "gpu": gpu,
                      "returncode": process.returncode,
                      "log": str(log_path.relative_to(ROOT))}
            if process.returncode == 0 and profile.is_file():
                finished.append(record)
                print("DONE", gpu, dataset, model, flush=True)
            else:
                failures.append(record)
                print("FAIL", gpu, dataset, model, process.returncode, flush=True)
            del active[gpu]
    write_status(active, pending, finished, failures)
    if failures:
        raise SystemExit(f"Adapter profile failures: {failures}")
    summarize()
    print("ALL_COMPLETE", len(finished), flush=True)


if __name__ == "__main__":
    main()
