"""Write a compact, continuously refreshable M3 experiment progress table."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys


MODELS = (
    "DLinear", "xPatch", "TQNet", "STELLA", "EasyST", "DUET",
    "iTransformer", "PatchTST", "CDPNet", "TimerXL", "S2Transformer",
    "Corrformer", "Timer", "TimeMoE", "Moirai", "HiSTGNN",
)
MODES = {
    "DLinear": "both", "PatchTST": "both", "Timer": "both",
    "TimeMoE": "both", "Moirai": "both", "HiSTGNN": "both",
    "CDPNet": "variable", "STELLA": "variable", "Corrformer": "variable",
    "EasyST": "variable", "S2Transformer": "variable", "DUET": "station",
    "iTransformer": "station", "TQNet": "station", "TimerXL": "station",
    "xPatch": "station",
}
DATASETS = {
    "Europe": ("M3Europe", "m3_europe_twsrhp_fast_s6"),
    "Global": ("M3Global", "m3_global_twsrhp_fast_s6"),
}


def any_match(root: Path, pattern: str) -> bool:
    return root.is_dir() and next(root.glob(pattern), None) is not None


def option(tokens: list[str], name: str) -> str | None:
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError):
        return None


def running_jobs() -> tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]:
    """Inspect live run.py commands; duplicated DataLoader workers collapse in sets."""
    baselines: set[tuple[str, str, int]] = set()
    adapters: set[tuple[str, str, int]] = set()
    for command_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = command_path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        tokens = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        if not any(Path(token).name == "run.py" for token in tokens):
            continue
        data_name = option(tokens, "--data")
        seed_text = option(tokens, "--seed")
        model = option(tokens, "--model")
        if data_name not in {"M3Europe", "M3Global"} or seed_text is None:
            continue
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        if model == "InteractionAdapter" or "--adapter_closed_form" in tokens:
            backbone = option(tokens, "--backbone_model")
            if backbone:
                adapters.add((data_name, backbone, seed))
        elif model:
            baselines.add((data_name, model, seed))
    return baselines, adapters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument(
        "--chain-needed", nargs=2, metavar=("DATASET", "MODELS"),
        help="exit 0 when any job in a comma-separated model chain is unfinished",
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    seeds = tuple(int(item) for item in args.seeds.replace(",", " ").split())
    log_root = project / "logs"
    running_baselines, running_adapters = running_jobs()
    rows = []
    for dataset, (data_name, record_stem) in DATASETS.items():
        record_root = project / "experiment_records" / record_stem
        for model in MODELS:
            for seed in seeds:
                baseline_pattern = f"*_{model}_{data_name}_*seed{seed}_0/metrics.npy"
                baseline_done = any_match(
                    record_root / "baselines" / "results", baseline_pattern
                )
                last_checkpoint = any_match(
                    record_root / "baselines" / "checkpoints",
                    f"*_{model}_{data_name}_*seed{seed}_0/last_checkpoint.pth",
                )
                adapter_metric = (
                    record_root / "adapters" / f"seed{seed}" / "results"
                    / f"{model}_{MODES[model]}_closed_form" / "metrics.npy"
                )
                baseline_status = (
                    "completed" if baseline_done
                    else "running" if (data_name, model, seed) in running_baselines
                    else "resumable" if last_checkpoint
                    else "pending"
                )
                adapter_status = (
                    "completed" if adapter_metric.is_file()
                    else "running" if (data_name, model, seed) in running_adapters
                    else "ready" if baseline_done
                    else "pending"
                )
                baseline_log_dir = log_root / dataset / "baseline"
                adapter_log_dir = log_root / dataset / "adapter"
                if baseline_done:
                    baseline_log_dir.mkdir(parents=True, exist_ok=True)
                    skip_log = baseline_log_dir / (
                        f"{model}_{data_name}.seed{seed}.completed_skip.log"
                    )
                    if not skip_log.exists():
                        metric = next(
                            (record_root / "baselines" / "results").glob(
                                baseline_pattern
                            )
                        )
                        skip_log.write_text(
                            "BASELINE_COMPLETED_SKIP\n"
                            f"model={model}\nseed={seed}\nmetrics={metric}\n",
                            encoding="utf-8",
                        )
                if adapter_metric.is_file():
                    adapter_log_dir.mkdir(parents=True, exist_ok=True)
                    skip_log = adapter_log_dir / (
                        f"{model}.seed{seed}.completed_skip.log"
                    )
                    if not skip_log.exists():
                        skip_log.write_text(
                            "ADAPTER_COMPLETED_SKIP\n"
                            f"model={model}\nseed={seed}\nmetrics={adapter_metric}\n",
                            encoding="utf-8",
                        )
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "baseline": baseline_status,
                    "adapter": adapter_status,
                    "baseline_log_dir": str(baseline_log_dir),
                    "adapter_log_dir": str(adapter_log_dir),
                })
    log_root.mkdir(parents=True, exist_ok=True)
    output = log_root / "progress.csv"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        handle.write(f"# updated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    temporary.replace(output)
    if args.chain_needed:
        dataset, model_text = args.chain_needed
        models = set(model_text.replace(",", " ").split())
        if dataset not in DATASETS or not models.issubset(MODELS):
            raise SystemExit(2)
        selected = [
            row for row in rows
            if row["dataset"] == dataset and row["model"] in models
        ]
        needed = any(
            row["baseline"] != "completed" or row["adapter"] != "completed"
            for row in selected
        )
        sys.exit(0 if needed else 10)


if __name__ == "__main__":
    main()
