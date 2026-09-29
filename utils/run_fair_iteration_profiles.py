"""Profile baseline iteration time and memory with a shared batch size."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_efficiency_profiles import PYTHON, ROOT, baseline_command
from utils.run_all_adapters import _replace_value


MODELS = (
    "PatchTST", "xPatch", "CDPNet", "Corrformer",
    "DUET", "iTransformer", "HiSTGNN",
)
DATASETS = ("France", "Europe")
BATCH_SIZE = 2
WARMUP_STEPS = 3
PROFILE_STEPS = 20


def command(dataset: str, model: str) -> list[str]:
    tokens = baseline_command(dataset, model, divisor=1)
    _replace_value(tokens, "--batch_size", BATCH_SIZE)
    _replace_value(tokens, "--num_workers", 0)
    _replace_value(tokens, "--profile_warmup_steps", WARMUP_STEPS)
    _replace_value(tokens, "--profile_steps", PROFILE_STEPS)
    return tokens


def main() -> None:
    gpus = ["0", "1", "2", "3", "4", "5"]
    queue = [(dataset, model) for dataset in DATASETS for model in MODELS]
    output = ROOT / "paper" / "efficiency_analysis"
    active = {}
    failures = []
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            dataset, model = queue.pop(0)
            log_path = output / dataset / "baseline" / (
                f"{model}.seed2024.batch2_iter.log"
            )
            log = log_path.open("w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            process = subprocess.Popen(
                command(dataset, model), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
            active[gpu] = (process, dataset, model, log, log_path)
            print("ITER_PROFILE_START", dataset, model, "gpu", gpu, flush=True)
        for gpu, item in list(active.items()):
            process, dataset, model, log, log_path = item
            if process.poll() is None:
                continue
            log.close()
            del active[gpu]
            text = log_path.read_text(errors="ignore")
            match = re.search(r"PROFILE_RESULT .+", text)
            if process.returncode or not match:
                failures.append((dataset, model, process.returncode, str(log_path)))
                print("ITER_PROFILE_FAIL", dataset, model,
                      process.returncode, flush=True)
            else:
                print(dataset, match.group(0), flush=True)
        if active:
            time.sleep(1)
    if failures:
        raise SystemExit(f"Iteration profile failures: {failures}")


if __name__ == "__main__":
    main()
