#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSLIB_PYTHON="${TSLIB_PYTHON:-/root/miniconda3/envs/tslib/bin/python}"
export PATH="$(dirname "$TSLIB_PYTHON"):$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PROJECT_ROOT
export SCRIPT_DIR="$PROJECT_ROOT/scripts/forecasting_m3_france"
export LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/m3_france_twsrhp/baselines}"
export JOB_TMPDIR="${JOB_TMPDIR:-/tmp/m3_france_twsrhp_jobs}"
export SEEDS="2024,2025,2026"
export SEED_OVERRIDES=""
export RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/experiment_records/m3_france_twsrhp/baselines}"
export CHECKPOINT_ROOT="$RECORD_ROOT"
export INVERSE=1
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export GPU_MEMORY_FREE_MB="${GPU_MEMORY_FREE_MB:-5000}"
export GPU_UTIL_MAX="${GPU_UTIL_MAX:-100}"
export MAX_JOBS_PER_GPU=1
export MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-3}"

"$TSLIB_PYTHON" "$PROJECT_ROOT/utils/forecast_batch.py" '*_M3France.sh'

"$TSLIB_PYTHON" -u "$PROJECT_ROOT/utils/run_all_adapters.py" \
  --gpus 0,1 \
  --seeds 2024,2025,2026 \
  --seed-overrides '' \
  --script-root "$PROJECT_ROOT/scripts/forecasting_m3_france" \
  --script-suffix M3France \
  --data-name M3France \
  --adapter-model-id-prefix M3France_TWSRHP_48_72 \
  --num-nodes 176 \
  --station-coords-path ./dataset/M3/France/station_coords.npy \
  --baseline-root "$PROJECT_ROOT/experiment_records/m3_france_twsrhp/baselines" \
  --output-root "$PROJECT_ROOT/experiment_records/m3_france_twsrhp/adapters" \
  --python-bin "$TSLIB_PYTHON"

"$TSLIB_PYTHON" -u "$PROJECT_ROOT/utils/summarize_repeated_m3_france.py"
