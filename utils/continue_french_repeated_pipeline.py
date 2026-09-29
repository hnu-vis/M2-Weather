"""Continue with adapters and aggregation after an active baseline batch exits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-pid", type=int, required=True)
    parser.add_argument(
        "--batch-log", type=Path,
        default=PROJECT_ROOT / "logs/french_repeated_baselines/batch.log",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument(
        "--python-bin", default="/root/miniconda3/envs/tslib/bin/python"
    )
    args = parser.parse_args()

    print(f"PIPELINE_WAIT baseline_pid={args.batch_pid}", flush=True)
    while _alive(args.batch_pid):
        time.sleep(max(1, args.poll_interval))

    text = args.batch_log.read_text(encoding="utf-8", errors="replace")
    completions = re.findall(r"All scripts finished\. failures=(\d+)", text)
    if not completions:
        raise RuntimeError(
            f"Baseline process exited without a completion marker: {args.batch_log}"
        )
    failures = int(completions[-1])
    if failures:
        raise RuntimeError(
            f"Baseline stage had {failures} final failure(s); adapters were not started."
        )

    print("PIPELINE_START stage=adapters", flush=True)
    subprocess.run(
        [
            args.python_bin, "-u", "utils/run_all_adapters.py",
            "--seeds", "2025,2026",
            "--seed-overrides",
            "EasyST=2024,2025,2026;S2Transformer=2024,2025,2026;"
            "xPatch=2024,2025,2026",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("PIPELINE_START stage=summary", flush=True)
    subprocess.run(
        [args.python_bin, "-u", "utils/summarize_repeated_french.py"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("PIPELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
