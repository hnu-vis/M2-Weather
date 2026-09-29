"""GPU-aware scheduler for forecasting experiment shell scripts."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, TextIO

BATCH_ENTRYPOINTS = {"forecast_batch.sh"}
OOM_PATTERN = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA error:\s+out of memory|"
    r"CUBLAS_STATUS_ALLOC_FAILED|cuDNN error:\s+CUDNN_STATUS_ALLOC_FAILED|"
    r"canUse32BitIndexMath",
    re.IGNORECASE,
)
SUMMARY_FIELDS = (
    "script", "seed", "gpu", "pid", "status", "attempt", "start_time", "end_time", "log_file"
)

# Estimated completion order for the French benchmark.  The first entries are
# cheap trainable baselines; pretrained full-test inference is left until the
# end. EasyST follows STELLA because it loads the matching frozen teacher.
SCRIPT_PRIORITY = (
    "DLinear_French",
    "xPatch_French",
    "TQNet_French",
    "STELLA_French",
    "EasyST_French",
    "DUET_French",
    "iTransformer_French",
    "PatchTST_French",
    "CDPNet_French",
    "TimerXL_French",
    "HiSTGNN_French",
    "S2Transformer_French",
    "Corrformer_French",
    "Timer_French",
    "TimeMoE_French",
    "Moirai_French",
)
SCRIPT_PRIORITY_INDEX = {
    dataset_stem: index
    for index, stem in enumerate(SCRIPT_PRIORITY)
    for dataset_stem in (
        stem,
        stem.replace("_French", "_M3France"),
        stem.replace("_French", "_M3Europe"),
        stem.replace("_French", "_M3Global"),
    )
}

# A dependency is resolved per random seed. EasyST distils a frozen STELLA
# teacher, so launching it before the matching teacher checkpoint exists would
# either fail or (worse) use the wrong seed.
SCRIPT_DEPENDENCIES = {
    "EasyST_French": "STELLA_French",
    "EasyST_M3France": "STELLA_M3France",
    "EasyST_M3Europe": "STELLA_M3Europe",
    "EasyST_M3Global": "STELLA_M3Global",
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_label(is_training: str) -> str:
    return f"is_training-{is_training or 'keep'}"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


def env_seeds() -> tuple[int, ...]:
    value = os.environ.get("SEEDS", os.environ.get("SEED", "2021"))
    parts = value.replace(",", " ").split()
    if not parts:
        raise ValueError("SEEDS must contain at least one integer")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"SEEDS must contain integers, got: {value}") from exc


def env_gpu_ids() -> tuple[str, ...] | None:
    value = os.environ.get("GPU_IDS", "").strip()
    if not value:
        return None
    gpu_ids = tuple(item for item in value.replace(",", " ").split() if item)
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU_IDS contains duplicates: {value}")
    if not all(item.isdigit() for item in gpu_ids):
        raise ValueError(f"GPU_IDS must contain numeric GPU indices: {value}")
    return gpu_ids


def env_seed_overrides() -> dict[str, tuple[int, ...]]:
    """Parse ``STEM=seed,seed;STEM=seed`` per-script seed policies."""
    value = os.environ.get("SEED_OVERRIDES", "").strip()
    if not value:
        return {}
    overrides: dict[str, tuple[int, ...]] = {}
    for assignment in value.split(";"):
        assignment = assignment.strip()
        if not assignment:
            continue
        if "=" not in assignment:
            raise ValueError(
                "SEED_OVERRIDES entries must use STEM=seed,seed, got: "
                f"{assignment}"
            )
        stem, seed_text = (part.strip() for part in assignment.split("=", 1))
        if not stem:
            raise ValueError("SEED_OVERRIDES contains an empty script stem")
        try:
            seeds = tuple(
                int(item) for item in seed_text.replace(",", " ").split()
            )
        except ValueError as exc:
            raise ValueError(
                f"SEED_OVERRIDES contains non-integer seeds: {assignment}"
            ) from exc
        if not seeds:
            raise ValueError(f"SEED_OVERRIDES has no seeds for {stem}")
        overrides[stem.removesuffix(".sh")] = seeds
    return overrides


def env_positive_int_map(name: str) -> dict[str, int]:
    """Parse a ``KEY=INTEGER;KEY=INTEGER`` environment mapping."""
    value = os.environ.get(name, "").strip()
    if not value:
        return {}
    result: dict[str, int] = {}
    for assignment in value.split(";"):
        assignment = assignment.strip()
        if not assignment:
            continue
        if "=" not in assignment:
            raise ValueError(f"{name} entries must use KEY=INTEGER: {assignment}")
        key, raw_value = (part.strip() for part in assignment.split("=", 1))
        if not key or key in result:
            raise ValueError(f"{name} contains an empty or duplicate key: {key!r}")
        try:
            parsed = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} value must be an integer: {assignment}") from exc
        if parsed <= 0:
            raise ValueError(f"{name} value must be positive: {assignment}")
        result[key] = parsed
    return result


@dataclass(frozen=True)
class Config:
    project_root: Path
    script_dir: Path
    log_root: Path
    job_tmpdir: Path
    gpu_memory_free_mb: int
    gpu_util_max: int
    max_jobs_per_gpu: int
    max_oom_retries: int
    poll_interval: int
    dry_run: bool
    is_training: str
    seeds: tuple[int, ...]
    seed_overrides: dict[str, tuple[int, ...]]
    record_root: str
    checkpoint_root: str
    inverse: bool
    train_stride: int
    skip_epoch_test: bool
    batch_size_multipliers: dict[str, int]
    skip_completed: bool
    gpu_ids: tuple[str, ...] | None

    @classmethod
    def from_environment(cls) -> "Config":
        default_root = Path(__file__).resolve().parents[1]
        project_root = Path(os.environ.get("PROJECT_ROOT", default_root)).resolve()
        is_training = os.environ.get("IS_TRAINING", "")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_root = Path(os.environ.get(
            "LOG_ROOT",
            project_root / "logs" / "forecast_batch"
            / f"{run_id}_{run_label(is_training)}",
        )).resolve()
        dry_run = os.environ.get("DRY_RUN", "0")
        if dry_run not in {"0", "1"}:
            raise ValueError(f"DRY_RUN must be 0 or 1, got: {dry_run}")
        inverse = os.environ.get("INVERSE", "0")
        if inverse not in {"0", "1"}:
            raise ValueError(f"INVERSE must be 0 or 1, got: {inverse}")
        skip_epoch_test = os.environ.get("SKIP_EPOCH_TEST", "0")
        if skip_epoch_test not in {"0", "1"}:
            raise ValueError(
                f"SKIP_EPOCH_TEST must be 0 or 1, got: {skip_epoch_test}"
            )
        skip_completed = os.environ.get("SKIP_COMPLETED", "1")
        if skip_completed not in {"0", "1"}:
            raise ValueError(f"SKIP_COMPLETED must be 0 or 1, got: {skip_completed}")
        record_root = os.environ.get(
            "RECORD_ROOT", "./experiment_records/forecasting"
        )
        config = cls(
            project_root=project_root,
            script_dir=Path(os.environ.get("SCRIPT_DIR", project_root / "scripts" / "forecasting")).resolve(),
            log_root=log_root,
            job_tmpdir=Path(os.environ.get("JOB_TMPDIR", "/mnt/sdb/containers/wmy/tmp")),
            gpu_memory_free_mb=env_int("GPU_MEMORY_FREE_MB", 5000),
            gpu_util_max=env_int("GPU_UTIL_MAX", 100),
            max_jobs_per_gpu=env_int("MAX_JOBS_PER_GPU", 1),
            max_oom_retries=env_int("MAX_OOM_RETRIES", 3),
            poll_interval=env_int("POLL_INTERVAL", 60),
            dry_run=dry_run == "1",
            is_training=is_training,
            seeds=env_seeds(),
            seed_overrides=env_seed_overrides(),
            record_root=record_root,
            checkpoint_root=os.environ.get("CHECKPOINT_ROOT", record_root),
            inverse=inverse == "1",
            train_stride=env_int("TRAIN_STRIDE", 1),
            skip_epoch_test=skip_epoch_test == "1",
            batch_size_multipliers=env_positive_int_map(
                "BATCH_SIZE_MULTIPLIERS"
            ),
            skip_completed=skip_completed == "1",
            gpu_ids=env_gpu_ids(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_jobs_per_gpu <= 0:
            raise ValueError("MAX_JOBS_PER_GPU must be a positive integer")
        if self.max_oom_retries < 0:
            raise ValueError("MAX_OOM_RETRIES must be a non-negative integer")
        if self.poll_interval < 0 or self.gpu_memory_free_mb < 0:
            raise ValueError("POLL_INTERVAL and GPU_MEMORY_FREE_MB must be non-negative")
        if not 0 <= self.gpu_util_max <= 100:
            raise ValueError("GPU_UTIL_MAX must be between 0 and 100")
        if self.is_training not in {"", "0", "1"}:
            raise ValueError(f"IS_TRAINING must be 0, 1, or empty, got: {self.is_training}")
        if self.train_stride <= 0:
            raise ValueError("TRAIN_STRIDE must be a positive integer")


@dataclass
class RunningJob:
    process: subprocess.Popen[bytes]
    script: Path
    seed: int
    gpu: str
    log_file: Path
    log_handle: BinaryIO
    start_time: str
    attempt: int
    temporary_script: Path


class BatchRunner:
    def __init__(
        self,
        config: Config,
        patterns: list[str],
        summary_fields: tuple[str, ...] = SUMMARY_FIELDS,
    ) -> None:
        self.config = config
        self.patterns = patterns
        self.summary_fields = summary_fields
        log_prefix = os.environ.get("LOG_PREFIX", "").strip()
        if log_prefix and ("/" in log_prefix or "\\" in log_prefix):
            raise ValueError(f"LOG_PREFIX must be a filename prefix: {log_prefix}")
        write_batch_metadata = os.environ.get("WRITE_BATCH_METADATA", "1")
        if write_batch_metadata not in {"0", "1"}:
            raise ValueError(
                f"WRITE_BATCH_METADATA must be 0 or 1: {write_batch_metadata}"
            )
        self.write_batch_metadata = write_batch_metadata == "1"
        self.batch_log = config.log_root / (
            f"{log_prefix}.batch.log" if log_prefix else "batch.log"
        )
        self.summary_path = config.log_root / (
            f"{log_prefix}.summary.tsv" if log_prefix else "summary.tsv"
        )
        self.scripts: list[Path] = []
        self.running: dict[int, RunningJob] = {}
        self.gpu_job_counts: dict[str, int] = {}
        self.retry_counts: dict[tuple[Path, int], int] = {}
        self.completed_jobs: set[tuple[str, int]] = set()
        self.failed_jobs: set[tuple[str, int]] = set()
        self.failures = 0
        self.summary_handle: TextIO | None = None
        self.summary_writer: csv.DictWriter | None = None

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        if self.write_batch_metadata:
            with self.batch_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def prepare(self) -> None:
        self.config.log_root.mkdir(parents=True, exist_ok=True)
        self.config.job_tmpdir.mkdir(parents=True, exist_ok=True)
        for directory, label in (
            (self.config.log_root, "log directory"),
            (self.config.job_tmpdir, "job temp directory"),
        ):
            if not os.access(directory, os.W_OK):
                raise RuntimeError(f"{label} is not writable: {directory}")
        if not self.config.script_dir.is_dir():
            raise RuntimeError(f"script directory not found: {self.config.script_dir}")
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi not found; cannot detect idle GPUs")
        patterns = self.patterns or ["*.sh"]
        matched_scripts = {
            path.resolve()
            for pattern in patterns
            for path in self.config.script_dir.rglob(pattern)
            if path.is_file() and path.name not in BATCH_ENTRYPOINTS
        }
        self.scripts = sorted(
            matched_scripts,
            key=lambda path: (
                SCRIPT_PRIORITY_INDEX.get(path.stem, len(SCRIPT_PRIORITY)),
                path.name,
            ),
        )
        if not self.scripts:
            raise RuntimeError("no scripts matched")
        if self.write_batch_metadata:
            self.summary_handle = self.summary_path.open(
                "w", encoding="utf-8", newline=""
            )
            self.summary_writer = csv.DictWriter(
                self.summary_handle, fieldnames=self.summary_fields, delimiter="\t"
            )
            self.summary_writer.writeheader()
            self.summary_handle.flush()

    def write_summary(self, **row: object) -> None:
        if self.summary_writer is None or self.summary_handle is None:
            return
        self.summary_writer.writerow(row)
        self.summary_handle.flush()

    def available_gpus(self) -> list[str]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        available: list[str] = []
        for line in result.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != 3:
                continue
            gpu, free_memory, utilization = values
            if self.config.gpu_ids is not None and gpu not in self.config.gpu_ids:
                continue
            try:
                free_memory_value, utilization_value = int(free_memory), int(utilization)
            except ValueError:
                continue
            current = self.gpu_job_counts.get(gpu, 0)
            if (current < self.config.max_jobs_per_gpu
                    and free_memory_value >= self.config.gpu_memory_free_mb
                    and utilization_value <= self.config.gpu_util_max):
                available.extend([gpu] * (self.config.max_jobs_per_gpu - current))
        return available

    def override_output_paths(self, source: str) -> str:
        root = self.config.record_root.rstrip("/")
        checkpoint_root = self.config.checkpoint_root.rstrip("/")
        paths = (
            ("--checkpoints", f"{checkpoint_root}/checkpoints"),
            ("--results", f"{root}/results"),
            ("--test_results", f"{root}/test_results"),
            ("--result_file", f"{root}/result_long_term_forecast.txt"),
        )
        missing: list[tuple[str, str]] = []
        for option, path in paths:
            quoted_path = shlex.quote(path)
            source, count = re.subn(
                rf"^(\s*{re.escape(option)}\s+)\S+",
                lambda match, value=quoted_path: match.group(1) + value,
                source, count=1, flags=re.MULTILINE,
            )
            if count == 0:
                missing.append((option, quoted_path))
        if missing:
            anchor = re.search(r"^\s*--des\s+.*", source, re.MULTILINE)
            if anchor is None:
                raise ValueError("cannot insert output path arguments")
            insertion = "\n".join(
                f"  {option} {path} \\" for option, path in missing
            ) + "\n" + anchor.group(0)
            source = source[:anchor.start()] + insertion + source[anchor.end():]
        return source

    def create_temporary_script(
        self, script: Path, gpu: str, seed: int, attempt: int
    ) -> Path:
        source = script.read_text(encoding="utf-8")
        source, count = re.subn(
            r"^export CUDA_VISIBLE_DEVICES=.*$", f"export CUDA_VISIBLE_DEVICES={gpu}",
            source, count=1, flags=re.MULTILINE,
        )
        if count == 0:
            raise ValueError(f"missing CUDA_VISIBLE_DEVICES export in {script}")
        if self.config.is_training:
            source, count = re.subn(
                r"^(\s*--is_training\s+)\d+",
                lambda match: match.group(1) + self.config.is_training,
                source, count=1, flags=re.MULTILINE,
            )
            if count == 0:
                raise ValueError(f"missing --is_training argument in {script}")
        source, count = re.subn(
            r"^(\s*--seed\s+)-?\d+",
            lambda match: match.group(1) + str(seed),
            source, count=1, flags=re.MULTILINE,
        )
        if count == 0:
            anchor = re.search(r"^\s*--is_training\s+.*$", source, re.MULTILINE)
            if anchor is None:
                anchor = re.search(r"^\s*--mask_rate\s+.*$", source, re.MULTILINE)
            if anchor is None:
                raise ValueError(f"cannot insert --seed in {script}")
            insertion = anchor.group(0) + f"\n  --seed {seed} \\"
            source = source[:anchor.start()] + insertion + source[anchor.end():]
        source = self.override_output_paths(source)
        source, count = re.subn(
            r"^(\s*--train_stride\s+)\d+",
            lambda match: match.group(1) + str(self.config.train_stride),
            source, count=1, flags=re.MULTILINE,
        )
        if count == 0:
            anchor = re.search(r"^\s*--des\s+.*", source, re.MULTILINE)
            if anchor is None:
                raise ValueError("cannot insert --train_stride argument")
            insertion = (
                f"  --train_stride {self.config.train_stride} \\\n" + anchor.group(0)
            )
            source = source[:anchor.start()] + insertion + source[anchor.end():]
        if self.config.skip_epoch_test and not re.search(
            r"^\s*--skip_epoch_test(?:\s|\\|$)", source, re.MULTILINE
        ):
            anchor = re.search(r"^\s*--des\s+.*", source, re.MULTILINE)
            if anchor is None:
                raise ValueError("cannot insert --skip_epoch_test argument")
            insertion = "  --skip_epoch_test \\\n" + anchor.group(0)
            source = source[:anchor.start()] + insertion + source[anchor.end():]
        if self.config.inverse and not re.search(
            r"^\s*--inverse(?:\s|\\|$)", source, re.MULTILINE
        ):
            anchor = re.search(r"^\s*--des\s+.*", source, re.MULTILINE)
            if anchor is None:
                raise ValueError("cannot insert --inverse argument")
            insertion = "  --inverse \\\n" + anchor.group(0)
            source = source[:anchor.start()] + insertion + source[anchor.end():]
        model_name = script.stem.split("_", 1)[0]
        multiplier = self.config.batch_size_multipliers.get(
            script.stem, self.config.batch_size_multipliers.get(model_name, 1)
        )
        if multiplier > 1:
            match = re.search(
                r"^(\s*--batch_size\s+)(\d+)", source, re.MULTILINE
            )
            if match is None:
                raise ValueError(f"missing --batch_size argument in {script}")
            original_batch = int(match.group(2))
            expanded_batch = original_batch * multiplier
            source = (
                source[:match.start(2)] + str(expanded_batch)
                + source[match.end(2):]
            )
            worker_match = re.search(
                r"^(\s*--num_workers\s+)(\d+)", source, re.MULTILINE
            )
            if worker_match is not None:
                current_workers = int(worker_match.group(2))
                expanded_workers = min(current_workers, 2)
                source = (
                    source[:worker_match.start(2)] + str(expanded_workers)
                    + source[worker_match.end(2):]
                )
            else:
                anchor = re.search(r"^\s*--des\s+.*", source, re.MULTILINE)
                if anchor is None:
                    raise ValueError("cannot insert --num_workers argument")
                insertion = "  --num_workers 2 \\\n" + anchor.group(0)
                source = (
                    source[:anchor.start()] + insertion + source[anchor.end():]
                )
            if attempt == 1:
                self.log(
                    f"AUTO-BATCH script={script.stem} "
                    f"batch_size={original_batch}->{expanded_batch} "
                    f"multiplier={multiplier}"
                )
        if attempt > 1:
            match = re.search(r"^(\s*--batch_size\s+)(\d+)", source, re.MULTILINE)
            if match is None:
                raise ValueError(f"missing --batch_size argument in {script}")
            original_batch = int(match.group(2))
            retry_batch = max(1, original_batch // (2 ** (attempt - 1)))
            source = source[:match.start(2)] + str(retry_batch) + source[match.end(2):]
            self.log(
                f"OOM backoff script={script.stem} attempt={attempt} "
                f"batch_size={original_batch}->{retry_batch}"
            )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"{script.stem}.",
            suffix=".sh",
            dir=self.config.job_tmpdir,
            delete=False,
        ) as handle:
            handle.write(source)
            return Path(handle.name)

    @staticmethod
    def is_oom_failure(log_file: Path) -> bool:
        try:
            return OOM_PATTERN.search(
                log_file.read_text(encoding="utf-8", errors="replace")
            ) is not None
        except FileNotFoundError:
            return False

    def launch(self, script: Path, seed: int, gpu: str) -> None:
        key = (script, seed)
        attempt = self.retry_counts.get(key, 0) + 1
        log_file = self.config.log_root / (
            f"{script.stem}.seed{seed}.try{attempt}.gpu{gpu}.log"
        )
        start_time = now()
        if self.config.dry_run:
            self.log(
                f"DRY-RUN script={script.stem} seed={seed} gpu={gpu} log={log_file}"
            )
            self.write_summary(
                script=script, seed=seed, gpu=gpu, pid="dry-run", status=0,
                attempt=attempt, start_time=start_time, end_time=now(),
                log_file=log_file,
            )
            self.completed_jobs.add((script.stem, seed))
            return
        try:
            temporary_script = self.create_temporary_script(
                script, gpu, seed, attempt
            )
        except (OSError, ValueError) as exc:
            self.failures += 1
            self.failed_jobs.add((script.stem, seed))
            self.log(
                f"FAIL  script={script.stem} seed={seed} gpu={gpu} "
                f"action=prepare_failed error={exc}"
            )
            self.write_summary(
                script=script, seed=seed, gpu=gpu, pid="", status="prepare_failed",
                attempt=attempt, start_time=start_time, end_time=now(),
                log_file=log_file,
            )
            return
        self.log(
            f"START script={script.stem} seed={seed} gpu={gpu} "
            f"attempt={attempt} log={log_file}"
        )
        log_handle = log_file.open("wb")
        env = os.environ.copy()
        env.update({
            "TMPDIR": str(self.config.job_tmpdir),
            "TEMP": str(self.config.job_tmpdir),
            "TMP": str(self.config.job_tmpdir),
        })
        process = subprocess.Popen(
            ["bash", str(temporary_script)], cwd=self.config.project_root, env=env,
            stdout=log_handle, stderr=subprocess.STDOUT,
        )
        self.running[process.pid] = RunningJob(
            process, script, seed, gpu, log_file, log_handle, start_time, attempt,
            temporary_script,
        )
        self.gpu_job_counts[gpu] = self.gpu_job_counts.get(gpu, 0) + 1
        self.write_summary(
            script=script, seed=seed, gpu=gpu, pid=process.pid, status="running",
            attempt=attempt, start_time=start_time, end_time="", log_file=log_file,
        )

    def refresh(self, queue: list[tuple[Path, int]]) -> None:
        for pid, job in list(self.running.items()):
            status = job.process.poll()
            if status is None:
                continue
            job.log_handle.close()
            job.temporary_script.unlink(missing_ok=True)
            del self.running[pid]
            current = self.gpu_job_counts[job.gpu]
            if current <= 1:
                del self.gpu_job_counts[job.gpu]
            else:
                self.gpu_job_counts[job.gpu] = current - 1
            key = (job.script, job.seed)
            final_status: int | str = status
            retries = self.retry_counts.get(key, 0)
            if status == 0:
                self.completed_jobs.add((job.script.stem, job.seed))
                self.log(
                    f"DONE  script={job.script.name} seed={job.seed} "
                    f"gpu={job.gpu} pid={pid} status=0"
                )
            elif self.is_oom_failure(job.log_file) and retries < self.config.max_oom_retries:
                retries += 1
                self.retry_counts[key] = retries
                queue.append(key)
                final_status = "oom_retry_queued"
                self.log(
                    f"RETRY script={job.script.name} seed={job.seed} reason=oom "
                    f"retry={retries}/{self.config.max_oom_retries} "
                    f"previous_log={job.log_file}"
                )
            else:
                self.failures += 1
                self.failed_jobs.add((job.script.stem, job.seed))
                self.log(
                    f"FAIL  script={job.script.name} seed={job.seed} gpu={job.gpu} "
                    f"pid={pid} status={status} action=final"
                )
            self.write_summary(
                script=job.script, seed=job.seed, gpu=job.gpu, pid=pid,
                status=final_status, attempt=job.attempt,
                start_time=job.start_time, end_time=now(), log_file=job.log_file,
            )

    def _dependency_checkpoint_exists(self, dependency: str, seed: int) -> bool:
        model, dataset_name = dependency.split("_", 1)
        roots = [
            Path(self.config.checkpoint_root),
            self.config.project_root / "experiment_records" / "forecasting_uvtrh",
        ]
        for root in roots:
            if not root.is_absolute():
                root = self.config.project_root / root
            checkpoint_dir = root.resolve() / "checkpoints"
            pattern = f"*_{model}_{dataset_name}_*seed{seed}_0/checkpoint.pth"
            if checkpoint_dir.is_dir() and any(checkpoint_dir.glob(pattern)):
                return True
        return False

    def dependency_state(self, script: Path, seed: int) -> str:
        """Return ready, wait, or failed for a script's same-seed dependency."""
        dependency = SCRIPT_DEPENDENCIES.get(script.stem)
        if dependency is None or self.config.dry_run:
            return "ready"
        key = (dependency, seed)
        if key in self.completed_jobs or self._dependency_checkpoint_exists(
            dependency, seed
        ):
            return "ready"
        if key in self.failed_jobs:
            return "failed"
        if dependency not in {candidate.stem for candidate in self.scripts}:
            return "failed"
        return "wait"

    def _completed_metrics(self, script: Path, seed: int) -> list[Path]:
        """Return final-test metric files for one baseline job."""
        try:
            model, dataset_name = script.stem.split("_", 1)
        except ValueError:
            return []
        root = Path(self.config.record_root)
        if not root.is_absolute():
            root = self.config.project_root / root
        return sorted(
            (root.resolve() / "results").glob(
                f"*_{model}_{dataset_name}_*seed{seed}_0/metrics.npy"
            )
        )

    def run(self) -> int:
        self.prepare()
        requested_queue = [
            (script, seed)
            for script in self.scripts
            for seed in self.config.seed_overrides.get(
                script.stem, self.config.seeds
            )
        ]
        queue: list[tuple[Path, int]] = []
        for script, seed in requested_queue:
            completed_metrics = self._completed_metrics(script, seed)
            if self.config.skip_completed and completed_metrics:
                self.completed_jobs.add((script.stem, seed))
                skip_log = self.config.log_root / (
                    f"{script.stem}.seed{seed}.completed_skip.log"
                )
                skip_log.write_text(
                    f"[{now()}] BASELINE_COMPLETED_SKIP\n"
                    f"script={script.stem}\nseed={seed}\n"
                    f"metrics={completed_metrics[0]}\n",
                    encoding="utf-8",
                )
                self.log(
                    f"SKIP  script={script.stem} seed={seed} reason=completed "
                    f"metrics={completed_metrics[0]}"
                )
                self.write_summary(
                    script=script, seed=seed, gpu="", pid="",
                    status="completed_skip", attempt=0,
                    start_time=now(), end_time=now(), log_file=skip_log,
                )
            else:
                queue.append((script, seed))
        next_index = 0
        self.log(f"Project root: {self.config.project_root}")
        self.log(f"Script dir:   {self.config.script_dir}")
        self.log(f"Log root:     {self.config.log_root}")
        self.log(f"Job tmpdir:   {self.config.job_tmpdir}")
        self.log(f"Record root: {self.config.record_root}")
        self.log(
            "GPU policy:   "
            f"free_memory>={self.config.gpu_memory_free_mb}MiB "
            f"utilization<={self.config.gpu_util_max}% "
            f"max_jobs_per_gpu={self.config.max_jobs_per_gpu} "
            f"max_oom_retries={self.config.max_oom_retries}"
        )
        self.log(
            "GPU pool:     "
            + (",".join(self.config.gpu_ids) if self.config.gpu_ids else "all")
        )
        seeds = ",".join(str(seed) for seed in self.config.seeds)
        self.log(
            f"Script opts:  is_training={self.config.is_training or 'keep'} "
            f"seeds={seeds} train_stride={self.config.train_stride} "
            f"skip_epoch_test={int(self.config.skip_epoch_test)} "
            f"skip_completed={int(self.config.skip_completed)}"
        )
        if self.config.batch_size_multipliers:
            self.log(
                "Batch multipliers: "
                + ";".join(
                    f"{key}={value}"
                    for key, value in self.config.batch_size_multipliers.items()
                )
            )
        if self.config.seed_overrides:
            override_text = "; ".join(
                f"{stem}={','.join(map(str, override_seeds))}"
                for stem, override_seeds in self.config.seed_overrides.items()
            )
            self.log(f"Seed overrides: {override_text}")
        self.log(
            f"Scripts:      {len(self.scripts)}; requested_jobs={len(requested_queue)}; "
            f"pending_jobs={len(queue)}"
        )
        self.log(
            "Dispatch order: " + " -> ".join(script.stem for script in self.scripts)
        )
        try:
            while next_index < len(queue) or self.running:
                self.refresh(queue)
                if next_index < len(queue):
                    available = self.available_gpus()
                    if not available:
                        self.log(
                            f"No GPU slot available; waiting {self.config.poll_interval}s. "
                            f"running={len(self.running)} "
                            f"remaining={len(queue) - next_index}"
                        )
                    else:
                        for gpu in available:
                            if next_index >= len(queue):
                                break
                            script, seed = queue[next_index]
                            dependency = SCRIPT_DEPENDENCIES.get(script.stem)
                            dependency_state = self.dependency_state(script, seed)
                            if dependency_state == "wait":
                                self.log(
                                    f"WAIT  script={script.stem} seed={seed} "
                                    f"dependency={dependency}"
                                )
                                break
                            if dependency_state == "failed":
                                self.failures += 1
                                self.failed_jobs.add((script.stem, seed))
                                self.log(
                                    f"FAIL  script={script.stem} seed={seed} "
                                    f"dependency={dependency} action=dependency_missing"
                                )
                                self.write_summary(
                                    script=script, seed=seed, gpu="", pid="",
                                    status="dependency_missing", attempt=0,
                                    start_time=now(), end_time=now(), log_file="",
                                )
                                next_index += 1
                                continue
                            self.launch(script, seed, gpu)
                            next_index += 1
                if next_index >= len(queue) and not self.running:
                    break
                time.sleep(self.config.poll_interval)
        except KeyboardInterrupt:
            self.log("Interrupted; terminating running jobs")
            for job in self.running.values():
                job.process.terminate()
            for job in self.running.values():
                job.process.wait()
                job.log_handle.close()
                job.temporary_script.unlink(missing_ok=True)
            return 130
        finally:
            if self.summary_handle is not None:
                self.summary_handle.close()
        summary = self.summary_path if self.write_batch_metadata else "disabled"
        self.log(f"All scripts finished. failures={self.failures} summary={summary}")
        return self.failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run forecasting scripts and assign jobs to idle GPUs."
    )
    parser.add_argument("script_glob", nargs="*",
                        help="recursive filename pattern, for example '*_Weather.sh'")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return BatchRunner(Config.from_environment(), args.script_glob).run()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
