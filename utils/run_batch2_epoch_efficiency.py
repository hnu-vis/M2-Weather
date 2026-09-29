"""Run full-epoch batch-2 efficiency profiles on six GPUs and summarize them."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_all_adapters import _replace_value
from utils.run_efficiency_profiles import adapter_command, baseline_command


OUTPUT = ROOT / "paper" / "efficiency_analysis" / "batch2_epoch"
GPUS = ("0", "1", "2", "3", "4", "5")
BATCH_SIZE = 2
SEED = 2024
PARADIGMS = {
    "PatchTST": "SSSV", "xPatch": "SSSV",
    "CDPNet": "MSSV", "Corrformer": "MSSV",
    "DUET": "SSMV", "iTransformer": "SSMV",
    "HiSTGNN": "MSMV",
}

# Long jobs start first so the dynamic queue has the shortest makespan.
BASELINE_ORDER = (
    ("France", "CDPNet"),
    ("France", "HiSTGNN"),
    ("Europe", "HiSTGNN"),
    ("France", "Corrformer"),
    ("Europe", "Corrformer"),
    ("Europe", "CDPNet"),
    ("France", "iTransformer"),
    ("France", "DUET"),
    ("France", "xPatch"),
    ("France", "PatchTST"),
    ("Europe", "xPatch"),
    ("Europe", "DUET"),
    ("Europe", "iTransformer"),
    ("Europe", "PatchTST"),
)
JOBS = [("baseline", *item) for item in BASELINE_ORDER] + [
    ("adapter", "France", "DLinear"),
    ("adapter", "Europe", "DLinear"),
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def baseline_job(dataset: str, model: str):
    command = baseline_command(dataset, model, divisor=1)
    _replace_value(command, "--batch_size", BATCH_SIZE)
    _replace_value(command, "--num_workers", 0)
    _replace_value(command, "--profile_warmup_steps", 0)
    _replace_value(command, "--profile_steps", 999999999)
    log = OUTPUT / dataset / "baseline" / f"{model}.seed{SEED}.batch2_epoch.log"
    return command, log, {}


def adapter_job(dataset: str):
    command = adapter_command(dataset)
    work = OUTPUT / dataset / "adapter_work"
    _replace_value(command, "--batch_size", BATCH_SIZE)
    _replace_value(command, "--checkpoints", work / "checkpoints")
    _replace_value(command, "--results", work / "results")
    folder = OUTPUT / dataset / "adapter"
    profile = folder / f"DLinear_both_seed{SEED}_batch2.json"
    log = folder / f"DLinear_both_seed{SEED}_batch2.log"
    env = {
        "ADAPTER_EFFICIENCY_PROFILE": "1",
        "ADAPTER_EFFICIENCY_PROFILE_ONLY": "1",
        "ADAPTER_EFFICIENCY_PROFILE_PATH": str(profile),
    }
    return command, log, env


def write_status(started, active, pending, finished, failures):
    atomic_json(OUTPUT / "status.json", {
        "started_at": started,
        "updated_at": now(),
        "batch_size": BATCH_SIZE,
        "active": {
            gpu: {"kind": v[1], "dataset": v[2], "model": v[3],
                  "pid": v[0].pid, "started_at": v[6]}
            for gpu, v in active.items()
        },
        "pending": [dict(kind=j[0], dataset=j[1], model=j[2]) for j in pending],
        "finished": finished,
        "failures": failures,
    })


def summarize(failures):
    baseline_pattern = re.compile(
        r"PROFILE_RESULT model=(\S+) batch_size=(\d+) "
        r"warmup_steps=(\d+) steps=(\d+) elapsed_seconds=([0-9.]+) "
        r"seconds_per_step=([0-9.]+) peak_memory_mb=([0-9.]+) "
        r"peak_reserved_mb=([0-9.]+)"
    )
    baseline = []
    for dataset in ("France", "Europe"):
        for model in PARADIGMS:
            path = OUTPUT / dataset / "baseline" / f"{model}.seed{SEED}.batch2_epoch.log"
            if not path.is_file():
                continue
            match = baseline_pattern.search(path.read_text(errors="ignore"))
            if not match:
                continue
            got_model, batch, warmup, steps, elapsed, per_step, peak, reserved = match.groups()
            peak, reserved = float(peak), float(reserved)
            baseline.append({
                "dataset": dataset, "paradigm": PARADIGMS[model], "model": got_model,
                "seed": SEED, "batch_size": int(batch),
                "train_stride": 1 if dataset == "France" else 6,
                "epoch_steps": int(steps), "epoch_time_seconds": float(elapsed),
                "epoch_time_minutes": float(elapsed) / 60,
                "seconds_per_iteration": float(per_step),
                "peak_allocated_mb": peak, "peak_allocated_gib": peak / 1024,
                "peak_reserved_mb": reserved, "peak_reserved_gib": reserved / 1024,
                "log": str(path.relative_to(ROOT)),
            })
    adapters = []
    for dataset in ("France", "Europe"):
        path = OUTPUT / dataset / "adapter" / f"DLinear_both_seed{SEED}_batch2.json"
        if path.is_file():
            item = json.loads(path.read_text())
            item["dataset"] = dataset
            item["peak_memory_gib"] = item["peak_memory_mb"] / 1024
            item["peak_reserved_gib"] = item["peak_reserved_mb"] / 1024
            adapters.append(item)
    if baseline:
        with (OUTPUT / "baseline_batch2_epoch.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(baseline[0]))
            writer.writeheader(); writer.writerows(baseline)
    if adapters:
        with (OUTPUT / "adapter_dlinear_batch2.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(adapters[0]))
            writer.writeheader(); writer.writerows(adapters)
    payload = {
        "hardware": "NVIDIA GeForce RTX 4080 SUPER 32760 MiB",
        "protocol": {"batch_size": BATCH_SIZE, "baseline": "one complete training epoch",
                     "adapter": "complete closed-form fit using DLinear backbone"},
        "baseline": baseline, "adapter": adapters, "failures": failures,
    }
    atomic_json(OUTPUT / "results.json", payload)
    lines = ["# Batch-size-2 full-epoch efficiency", "",
             "All baselines use batch size 2 and num_workers=0. Baseline time covers one complete training epoch (forward, loss, backward, and optimizer update) and excludes validation/test. Peak memory covers that epoch.", "",
             "| Dataset | Paradigm | Model | Steps | Epoch time | Time / iter | Peak allocated | Peak reserved |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for x in baseline:
        lines.append(
            f"| {x['dataset']} | {x['paradigm']} | {x['model']} | {x['epoch_steps']} | "
            f"{x['epoch_time_minutes']:.2f} min | {x['seconds_per_iteration']:.4f} s | "
            f"{x['peak_allocated_gib']:.2f} GiB | {x['peak_reserved_gib']:.2f} GiB |"
        )
    lines += ["", "## Adapter: DLinear + full both-branch Adapter, batch size 2", "",
              "| Dataset | Train accumulation | Validation accumulation | Solve | Full fit | Peak allocated | Peak reserved |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for x in adapters:
        lines.append(
            f"| {x['dataset']} | {x['feature_accumulation_seconds']:.3f} s | "
            f"{x['validation_accumulation_seconds']:.3f} s | {x['solve_seconds']:.3f} s | "
            f"{x['full_training_seconds']:.3f} s | {x['peak_memory_gib']:.2f} GiB | "
            f"{x['peak_reserved_gib']:.2f} GiB |"
        )
    if failures:
        lines += ["", "## Failures", "", "```json", json.dumps(failures, indent=2), "```"]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = now()
    pending = list(JOBS)
    active = {}
    finished, failures = [], []
    while pending or active:
        for gpu in GPUS:
            if gpu in active or not pending:
                continue
            kind, dataset, model = pending.pop(0)
            if kind == "baseline":
                command, log_path, extra_env = baseline_job(dataset, model)
            else:
                command, log_path, extra_env = adapter_job(dataset)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", **extra_env)
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            active[gpu] = (process, kind, dataset, model, log_handle, log_path, now())
            print("START", gpu, kind, dataset, model, process.pid, flush=True)
        write_status(started, active, pending, finished, failures)
        time.sleep(5)
        for gpu, item in list(active.items()):
            process, kind, dataset, model, handle, log_path, job_started = item
            if process.poll() is None:
                continue
            handle.close()
            record = {"kind": kind, "dataset": dataset, "model": model,
                      "gpu": gpu, "returncode": process.returncode,
                      "started_at": job_started, "finished_at": now(),
                      "log": str(log_path.relative_to(ROOT))}
            (finished if process.returncode == 0 else failures).append(record)
            del active[gpu]
            print("DONE" if process.returncode == 0 else "FAIL",
                  gpu, kind, dataset, model, process.returncode, flush=True)
    summarize(failures)
    write_status(started, active, pending, finished, failures)
    atomic_json(OUTPUT / "complete.json", {"completed_at": now(),
                "successful_jobs": len(finished), "failures": failures})
    print("ALL_COMPLETE", len(finished), "failures", len(failures), flush=True)


if __name__ == "__main__":
    main()
