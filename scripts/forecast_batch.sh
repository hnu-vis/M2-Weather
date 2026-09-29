#!/usr/bin/env bash
set -euo pipefail

# Override options through environment variables as needed.
export IS_TRAINING="${IS_TRAINING:-}"
export SEEDS="${SEEDS:-2024}"
export RECORD_ROOT="${RECORD_ROOT:-./experiment_records/forecasting}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RECORD_ROOT}"
export INVERSE="${INVERSE:-0}"
export JOB_TMPDIR="${JOB_TMPDIR:-${TMPDIR:-/tmp}/forecast_batch}"

export GPU_MEMORY_FREE_MB="${GPU_MEMORY_FREE_MB:-5000}"
export GPU_UTIL_MAX="${GPU_UTIL_MAX:-100}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

export MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-3}"
export POLL_INTERVAL="${POLL_INTERVAL:-60}"
export DRY_RUN="${DRY_RUN:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT

# Optional arguments are recursive shell filename globs, e.g. '*_Weather.sh'.
exec "${PYTHON_BIN:-python}" "$PROJECT_ROOT/utils/forecast_batch.py" "$@"
