#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSLIB_PYTHON="${TSLIB_PYTHON:-/root/miniconda3/envs/tslib/bin/python}"
export PATH="$(dirname "$TSLIB_PYTHON"):$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_VARIANT="${RUN_VARIANT:-}"
RECORD_STEM="m3_europe_twsrhp"
if [[ -n "$RUN_VARIANT" ]]; then
  RECORD_STEM="${RECORD_STEM}_${RUN_VARIANT}"
fi

export PROJECT_ROOT
export SCRIPT_DIR="$PROJECT_ROOT/scripts/forecasting_m3_europe"
BASELINE_LOG_ROOT="${BASELINE_LOG_ROOT:-$PROJECT_ROOT/logs/Europe/baseline}"
export JOB_TMPDIR="${JOB_TMPDIR:-/tmp/${RECORD_STEM}_jobs}"
export SEEDS="${SEEDS:-2024,2025,2026}"
REUSE_DETERMINISTIC="${REUSE_DETERMINISTIC:-0}"
if [[ "$REUSE_DETERMINISTIC" == "1" ]]; then
  export SEED_OVERRIDES="${SEED_OVERRIDES:-Timer_M3Europe=2024;TimeMoE_M3Europe=2024}"
  ADAPTER_SEED_OVERRIDES="${ADAPTER_SEED_OVERRIDES:-Timer=2024;TimeMoE=2024}"
else
  export SEED_OVERRIDES="${SEED_OVERRIDES:-}"
  ADAPTER_SEED_OVERRIDES="${ADAPTER_SEED_OVERRIDES:-}"
fi
export TRAIN_STRIDE="${TRAIN_STRIDE:-1}"
export SKIP_EPOCH_TEST="${SKIP_EPOCH_TEST:-0}"
export BATCH_SIZE_MULTIPLIERS="${BATCH_SIZE_MULTIPLIERS:-}"
export RECORD_ROOT="${RECORD_ROOT:-$PROJECT_ROOT/experiment_records/$RECORD_STEM/baselines}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RECORD_ROOT}"
export INVERSE="${INVERSE:-1}"
export POLL_INTERVAL="${POLL_INTERVAL:-5}"
export GPU_MEMORY_FREE_MB="${GPU_MEMORY_FREE_MB:-5000}"
export GPU_UTIL_MAX="${GPU_UTIL_MAX:-100}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-3}"
export WRITE_BATCH_METADATA="${WRITE_BATCH_METADATA:-0}"

DATA_ROOT="$PROJECT_ROOT/dataset/M3/Europe"
for required_file in \
  "$DATA_ROOT/m3_europe_europe_q0q1_weather5_1h_2017_2021.npy" \
  "$DATA_ROOT/station_coords.npy" \
  "$DATA_ROOT/time.npy"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required M3 Europe file: $required_file" >&2
    exit 1
  fi
done

ADAPTER_ROOT="${ADAPTER_ROOT:-$PROJECT_ROOT/experiment_records/$RECORD_STEM/adapters}"
ADAPTER_LOG_ROOT="${ADAPTER_LOG_ROOT:-$PROJECT_ROOT/logs/Europe/adapter}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$PROJECT_ROOT/experiment_records/$RECORD_STEM/summary}"
ADAPTER_GPUS="${ADAPTER_GPUS:-0,1}"
ADAPTER_BATCH_SIZE_OVERRIDES="${ADAPTER_BATCH_SIZE_OVERRIDES:-}"
ADAPTER_BATCH_SIZE_MULTIPLIERS="${ADAPTER_BATCH_SIZE_MULTIPLIERS:-}"
ADAPTER_FIT_STRIDE="${ADAPTER_FIT_STRIDE:-1}"
RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
PRECOMPLETED_ADAPTERS_FIRST="${PRECOMPLETED_ADAPTERS_FIRST:-0}"
PREPASS_ONLY="${PREPASS_ONLY:-0}"
MODEL_ORDER="${MODELS:-DLinear,xPatch,TQNet,STELLA,EasyST,DUET,iTransformer,PatchTST,CDPNet,TimerXL,S2Transformer,Corrformer,Timer,TimeMoE,Moirai,HiSTGNN}"
read -r -a MODELS_TO_RUN <<< "${MODEL_ORDER//,/ }"

adapter_dry_run=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  adapter_dry_run+=(--dry-run)
fi

reuse_if_ready() {
  local kind="$1"
  local model="$2"
  if [[ "$REUSE_DETERMINISTIC" != "1" || "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  if [[ "$kind" == "baseline" ]]; then
    if ! compgen -G "$RECORD_ROOT/results/*_${model}_M3Europe_*seed2024_0/metrics.npy" >/dev/null; then
      return
    fi
    "$TSLIB_PYTHON" -u "$PROJECT_ROOT/utils/reuse_deterministic_results.py" baseline \
      --root "$RECORD_ROOT" --data-name M3Europe \
      --models "$model" --source-seed 2024 --target-seeds 2025 2026
  elif [[ -f "$ADAPTER_ROOT/seed2024/results/${model}_both_closed_form/summary.json" ]]; then
    "$TSLIB_PYTHON" -u "$PROJECT_ROOT/utils/reuse_deterministic_results.py" adapter \
      --root "$ADAPTER_ROOT" --data-name M3Europe \
      --models "$model" --source-seed 2024 --target-seeds 2025 2026
  fi
}

run_adapters() {
  "$TSLIB_PYTHON" -u "$PROJECT_ROOT/utils/run_all_adapters.py" "$@" \
    --gpus "$ADAPTER_GPUS" \
    --seeds "$SEEDS" \
    --seed-overrides "$ADAPTER_SEED_OVERRIDES" \
    --script-root "$PROJECT_ROOT/scripts/forecasting_m3_europe" \
    --script-suffix M3Europe \
    --data-name M3Europe \
    --adapter-model-id-prefix M3Europe_TWSRHP_48_72 \
    --num-nodes 983 \
    --station-coords-path ./dataset/M3/Europe/station_coords.npy \
    --baseline-root "$CHECKPOINT_ROOT" \
    --output-root "$ADAPTER_ROOT" \
    --log-root "$ADAPTER_LOG_ROOT" \
    --batch-size-overrides "$ADAPTER_BATCH_SIZE_OVERRIDES" \
    --batch-size-multipliers "$ADAPTER_BATCH_SIZE_MULTIPLIERS" \
    --fit-stride "$ADAPTER_FIT_STRIDE" \
    --max-oom-retries "$MAX_OOM_RETRIES" \
    --only-if-baseline-complete \
    --python-bin "$TSLIB_PYTHON" \
    "${adapter_dry_run[@]}"
}

if [[ "$RUN_EXPERIMENTS" == "1" ]]; then
  if [[ "$PRECOMPLETED_ADAPTERS_FIRST" == "1" ]]; then
    reuse_if_ready baseline Timer
    reuse_if_ready baseline TimeMoE
    echo "INTERLEAVED_PREPASS dataset=M3Europe models=${MODELS_TO_RUN[*]}"
    run_adapters "${MODELS_TO_RUN[@]}"
    reuse_if_ready adapter Timer
    reuse_if_ready adapter TimeMoE
  fi

  if [[ "$PREPASS_ONLY" != "1" ]]; then
   for model in "${MODELS_TO_RUN[@]}"; do
    echo "INTERLEAVED_MODEL_START dataset=M3Europe model=$model"
    LOG_ROOT="$BASELINE_LOG_ROOT" LOG_PREFIX="$model" \
      "$TSLIB_PYTHON" "$PROJECT_ROOT/utils/forecast_batch.py" "${model}_M3Europe.sh"
    if [[ "$model" == "Timer" || "$model" == "TimeMoE" ]]; then
      reuse_if_ready baseline "$model"
    fi
    run_adapters "$model"
    if [[ "$model" == "Timer" || "$model" == "TimeMoE" ]]; then
      reuse_if_ready adapter "$model"
    fi
    echo "INTERLEAVED_MODEL_DONE dataset=M3Europe model=$model"
   done
  fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: skipping result aggregation because no jobs were executed."
  exit 0
fi

if [[ "$RUN_SUMMARY" != "1" ]]; then
  exit 0
fi

summary_flags=()
if [[ "$SKIP_EPOCH_TEST" == "1" ]]; then
  summary_flags+=(--skip-epoch-test)
fi
if [[ "$REUSE_DETERMINISTIC" == "1" ]]; then
  summary_flags+=(--deterministic-reuse Timer,TimeMoE)
fi

"$TSLIB_PYTHON" -u "$PROJECT_ROOT/utils/summarize_repeated_m3_france.py" \
  --data-name M3Europe \
  --seeds "$SEEDS" \
  --baseline-root "$RECORD_ROOT" \
  --adapter-root "$ADAPTER_ROOT" \
  --output-root "$SUMMARY_ROOT" \
  --train-stride "$TRAIN_STRIDE" \
  --adapter-fit-stride "$ADAPTER_FIT_STRIDE" \
  --batch-size-multipliers "$BATCH_SIZE_MULTIPLIERS" \
  "${summary_flags[@]}"
