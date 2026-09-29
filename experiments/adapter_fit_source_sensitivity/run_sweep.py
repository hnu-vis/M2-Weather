#!/usr/bin/env python3
"""Run controlled backbone training and residual-source Adapter fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "adapter_fit_source_sensitivity"
SEEDS = (2024, 2025, 2026)

DATASETS = {
    "France": {
        "data": "M3France", "root": "dataset/M3/France",
        "file": "m3_france_france_q0q1_weather5_1h_2017_2021.npy",
        "nodes": 176, "train_batches": {"PatchTST": 128, "iTransformer": 128, "S2Transformer": 384},
        "patience": 10,
    },
    "Global": {
        "data": "M3Global", "root": "dataset/M3/Global",
        "file": "m3_global_global_q0_weather5_1h_2017_2021.npy",
        "nodes": 2504, "train_batches": {"PatchTST": 8, "iTransformer": 8, "S2Transformer": 24},
        "patience": 5,
    },
}

MODELS = {
    "PatchTST": {
        "mode": "both",
        "args": ["--patch_len", "16", "--e_layers", "3", "--n_heads", "16", "--d_model", "128", "--d_ff", "256", "--dropout", "0.2", "--pct_start", "0.3"],
    },
    "iTransformer": {
        "mode": "station",
        "args": ["--e_layers", "4", "--d_layers", "1", "--factor", "3", "--d_model", "256", "--n_heads", "8", "--d_ff", "512", "--dropout", "0.1"],
    },
    "S2Transformer": {
        "mode": "variable",
        "args": ["--s2_group_counts", "16", "4", "--e_layers", "2", "--n_heads", "8", "--d_model", "256", "--d_ff", "512", "--dropout", "0.1", "--use_amp", "--weight_decay", "0.0001", "--clip_grad", "5"],
    },
}


def common_args(dataset, model, seed):
    cfg = DATASETS[dataset]
    batch = cfg["train_batches"][model]
    args = [
        "--task_name", "long_term_forecast", "--root_path", str(ROOT / cfg["root"]),
        "--data_path", cfg["file"], "--model_id", f"{cfg['data']}_AdapterSource_48_72",
        "--data", cfg["data"], "--features", "M", "--seq_len", "48", "--label_len", "0",
        "--pred_len", "72", "--enc_in", "4", "--dec_in", "4", "--c_out", "4",
        "--num_nodes", str(cfg["nodes"]), "--batch_size", str(batch),
        "--eval_batch_size", str(batch), "--num_workers", "4", "--seed", str(seed),
        "--m3_split_profile", "adapter_sensitivity", "--learning_rate", "0.0001",
        "--lradj", "type1", "--des", "AdapterSourceSensitivity", "--gpu", "0",
        "--deterministic",
    ]
    if model == "S2Transformer":
        args += ["--station_coords_path", str(ROOT / cfg["root"] / "station_coords.npy")]
        if dataset == "Global":
            index = args.index("--s2_group_counts") if "--s2_group_counts" in args else -1
    return args


def job_paths(dataset, model, seed):
    base = ARTIFACT_ROOT / dataset / model / f"seed_{seed}"
    return {
        "base": base, "checkpoints": base / "checkpoints", "backbone_results": base / "backbone_results",
        "sensitivity": base / "sensitivity", "logs": base / "logs", "status": base / "status.json",
    }


def run_logged(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return process.returncode, time.perf_counter() - started


def one_job(dataset, model, seed, gpu):
    paths = job_paths(dataset, model, seed)
    for key in ("base", "checkpoints", "backbone_results", "sensitivity", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if (paths["sensitivity"] / "summary.json").is_file():
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cfg = DATASETS[dataset]
    model_args = list(MODELS[model]["args"])
    if model == "S2Transformer" and dataset == "Global":
        pos = model_args.index("--s2_group_counts")
        model_args[pos + 1:pos + 3] = ["224", "56"]
    base = common_args(dataset, model, seed) + model_args
    train_command = [sys.executable, "-u", "run.py", "--is_training", "1", "--model", model,
                     "--train_epochs", "30", "--patience", str(cfg["patience"]), "--train_stride", "1",
                     "--skip_epoch_test", "--skip_final_test", "--resume",
                     "--checkpoints", str(paths["checkpoints"]), "--results", str(paths["backbone_results"])] + base
    code, training_seconds = run_logged(train_command, paths["logs"] / "backbone.log", env)
    if code:
        paths["status"].write_text(json.dumps({"stage": "backbone_failed", "returncode": code}, indent=2))
        return code
    checkpoints = sorted(paths["checkpoints"].glob("*/checkpoint.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one best checkpoint in {paths['checkpoints']}, got {checkpoints}")
    adapter_command = [
        sys.executable, "-u", "run.py", "--is_training", "0", "--model", "InteractionAdapter",
        "--backbone_model", model, "--backbone_checkpoint", str(checkpoints[0]),
        "--adapter_mode", MODELS[model]["mode"], "--adapter_source_sensitivity",
        "--adapter_fit_stride", "1", "--results", str(paths["sensitivity"]),
        "--checkpoints", str(paths["base"] / "adapter_checkpoints"),
    ] + base
    code, sensitivity_seconds = run_logged(adapter_command, paths["logs"] / "adapter.log", env)
    status = {
        "dataset": dataset, "model": model, "seed": seed, "gpu": gpu,
        "stage": "complete" if code == 0 else "adapter_failed", "returncode": code,
        "backbone_training_seconds": training_seconds,
        "adapter_analysis_seconds": sensitivity_seconds,
        "checkpoint": str(checkpoints[0]),
    }
    paths["status"].write_text(json.dumps(status, indent=2) + "\n")
    summary_path = paths["sensitivity"] / "summary.json"
    if code == 0 and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["backbone_training_seconds"] = training_seconds
        summary["adapter_analysis_seconds"] = sensitivity_seconds
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["France"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    jobs = [(dataset, model, seed) for dataset in args.datasets for model in MODELS for seed in SEEDS]
    active = []
    failures = []
    while jobs or active:
        used_gpus = {entry[0][3] for entry in active}
        free_gpus = [gpu for gpu in gpus if gpu not in used_gpus]
        while jobs and free_gpus:
            dataset, model, seed = jobs.pop(0)
            gpu = free_gpus.pop(0)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", dataset, model, str(seed), str(gpu)]
            active.append(((dataset, model, seed, gpu), subprocess.Popen(command, cwd=ROOT)))
        time.sleep(5)
        remaining = []
        for job, process in active:
            code = process.poll()
            if code is None:
                remaining.append((job, process))
            elif code:
                failures.append((job[:3], code))
        active = remaining
    if failures:
        raise SystemExit(f"Failed jobs: {failures}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        index = sys.argv.index("--worker")
        dataset, model, seed, gpu = sys.argv[index + 1:index + 5]
        raise SystemExit(one_job(dataset, model, int(seed), int(gpu)))
    main()
