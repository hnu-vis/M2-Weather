"""Materialize seed records for deterministic pretrained-model inference.

Timer and TimeMoE are frozen, greedy, and independent of the experiment seed.
The canonical seed is evaluated once; its immutable metrics (and fitted Adapter)
are copied to the requested seed paths with explicit provenance metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ADAPTER_MODES = {"Timer": "both", "TimeMoE": "both"}


def _one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source matching {root / pattern}, found {matches}"
        )
    return matches[0]


def _copytree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _reuse_baseline(
    root: Path, data_name: str, model: str, source_seed: int, target_seed: int
) -> dict[str, object]:
    source = _one_match(
        root / "results", f"*_{model}_{data_name}_*seed{source_seed}_0"
    )
    old_suffix = f"seed{source_seed}_0"
    if old_suffix not in source.name:
        raise ValueError(f"Cannot replace seed suffix in {source}")
    destination = source.with_name(
        source.name.replace(old_suffix, f"seed{target_seed}_0", 1)
    )
    _copytree(source, destination)
    provenance = {
        "reuse_type": "deterministic_pretrained_inference",
        "model": model,
        "data_name": data_name,
        "source_seed": source_seed,
        "materialized_seed": target_seed,
        "source": str(source),
    }
    (destination / "deterministic_reuse.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def _reuse_adapter(
    root: Path, data_name: str, model: str, source_seed: int, target_seed: int
) -> dict[str, object]:
    mode = ADAPTER_MODES[model]
    relative_result = Path("results") / f"{model}_{mode}_closed_form"
    relative_checkpoint = Path("checkpoints") / f"{model}_{mode}_closed_form"
    source_root = root / f"seed{source_seed}"
    destination_root = root / f"seed{target_seed}"
    source_result = source_root / relative_result
    if not (source_result / "summary.json").is_file():
        raise FileNotFoundError(source_result / "summary.json")
    _copytree(source_result, destination_root / relative_result)
    source_checkpoint = source_root / relative_checkpoint
    if source_checkpoint.is_dir():
        _copytree(source_checkpoint, destination_root / relative_checkpoint)

    provenance = {
        "reuse_type": "deterministic_pretrained_adapter",
        "model": model,
        "data_name": data_name,
        "source_seed": source_seed,
        "materialized_seed": target_seed,
        "source": str(source_result),
    }
    summary_path = destination_root / relative_result / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["deterministic_reuse"] = provenance
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("baseline", "adapter"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--models", nargs="+", default=("Timer", "TimeMoE"))
    parser.add_argument("--source-seed", type=int, default=2024)
    parser.add_argument("--target-seeds", nargs="+", type=int, default=(2025, 2026))
    args = parser.parse_args()

    if args.source_seed in args.target_seeds:
        parser.error("--target-seeds must not contain --source-seed")
    unknown = sorted(set(args.models).difference(ADAPTER_MODES))
    if unknown:
        parser.error("unsupported deterministic model(s): " + ", ".join(unknown))

    reuse = _reuse_baseline if args.kind == "baseline" else _reuse_adapter
    records = [
        reuse(
            args.root, args.data_name, model, args.source_seed, target_seed
        )
        for model in args.models
        for target_seed in args.target_seeds
    ]
    manifest_path = args.root / f"deterministic_{args.kind}_reuse.json"
    manifest_path.write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"DETERMINISTIC_REUSE kind={args.kind} records={len(records)} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
