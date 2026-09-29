#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSLIB_PYTHON="${TSLIB_PYTHON:-/root/miniconda3/envs/tslib/bin/python}"

export RUN_VARIANT="fast_s6"
export TRAIN_STRIDE="6"
export SKIP_EPOCH_TEST="1"
export ADAPTER_FIT_STRIDE="6"
export REUSE_DETERMINISTIC="1"
export ADAPTER_GPUS="${ADAPTER_GPUS:-0,1,2,3,4,5}"

# Dataset-specific multipliers are based on observed peak memory on the 32-GiB
# devices. Both schedulers retain automatic OOM backoff and halve an expanded
# batch after each OOM, so a resumed task can safely converge to a fitting size.
EUROPE_BATCH_SIZE_MULTIPLIERS="${EUROPE_BATCH_SIZE_MULTIPLIERS:-DLinear=6;TQNet=2;STELLA=8;DUET=3;iTransformer=2;Timer=32;TimeMoE=24;Moirai=16}"
GLOBAL_BATCH_SIZE_MULTIPLIERS="${GLOBAL_BATCH_SIZE_MULTIPLIERS:-DLinear=6;TQNet=2;STELLA=8;DUET=3;iTransformer=2;Timer=32;TimeMoE=24;Moirai=8}"
EUROPE_ADAPTER_BATCH_SIZE_MULTIPLIERS="${EUROPE_ADAPTER_BATCH_SIZE_MULTIPLIERS:-$EUROPE_BATCH_SIZE_MULTIPLIERS}"
GLOBAL_ADAPTER_BATCH_SIZE_MULTIPLIERS="${GLOBAL_ADAPTER_BATCH_SIZE_MULTIPLIERS:-$GLOBAL_BATCH_SIZE_MULTIPLIERS}"

NON_HISTGNN_MODELS="DLinear,xPatch,TQNet,STELLA,EasyST,DUET,iTransformer,PatchTST,CDPNet,TimerXL,S2Transformer,Corrformer,Timer,TimeMoE,Moirai"

update_progress() {
  "$TSLIB_PYTHON" "$PROJECT_ROOT/utils/write_m3_progress.py" \
    --project-root "$PROJECT_ROOT" --seeds "${SEEDS:-2024,2025,2026}"
}

progress_monitor() {
  while true; do
    update_progress || true
    sleep 30
  done
}

progress_monitor &
progress_monitor_pid=$!
trap 'kill "$progress_monitor_pid" 2>/dev/null || true' EXIT

filter_pipeline_output() {
  awk '
    /^ADAPTER_BATCH_(START|DONE|RETRY|FAIL)/ ||
    (/^ADAPTER_BATCH_JOB_COUNT/ && $0 !~ /pending=0$/) ||
    /^ERROR:|^Traceback/ ||
    /^\[[0-9-]+ [0-9:]+\] (START|DONE|RETRY|FAIL)/ {
      print
      fflush()
    }
  '
}

# Each pool item is an atomic per-model chain: all requested baseline seeds,
# then all corresponding Adapter seeds. STELLA and EasyST share one chain so the
# teacher is guaranteed to finish before its dependent student starts.
PAIRED_CHAINS=(
  "DLinear" "xPatch" "TQNet" "STELLA,EasyST" "DUET" "iTransformer"
  "PatchTST" "CDPNet" "TimerXL" "S2Transformer"
)

run_chain() {
  local dataset="$1" models="$2" gpu="$3" status batch_map adapter_batch_map
  if [[ "$dataset" == "Europe" ]]; then
    batch_map="$EUROPE_BATCH_SIZE_MULTIPLIERS"
    adapter_batch_map="$EUROPE_ADAPTER_BATCH_SIZE_MULTIPLIERS"
  else
    batch_map="$GLOBAL_BATCH_SIZE_MULTIPLIERS"
    adapter_batch_map="$GLOBAL_ADAPTER_BATCH_SIZE_MULTIPLIERS"
  fi
  set +e
  "$TSLIB_PYTHON" "$PROJECT_ROOT/utils/write_m3_progress.py" \
    --project-root "$PROJECT_ROOT" --seeds "${SEEDS:-2024,2025,2026}" \
    --chain-needed "$dataset" "$models"
  status=$?
  set -e
  if (( status == 10 )); then
    return 0
  elif (( status != 0 )); then
    echo "GPU_POOL_CHAIN_CHECK_FAIL dataset=$dataset models=$models status=$status"
    return "$status"
  fi
  echo "GPU_POOL_CHAIN_START dataset=$dataset models=$models gpu=$gpu"
  set +e
  GPU_IDS="$gpu" ADAPTER_GPUS="$gpu" MODELS="$models" \
    BATCH_SIZE_MULTIPLIERS="$batch_map" \
    ADAPTER_BATCH_SIZE_MULTIPLIERS="$adapter_batch_map" \
    PRECOMPLETED_ADAPTERS_FIRST=0 PREPASS_ONLY=0 RUN_SUMMARY=0 \
    bash "$PROJECT_ROOT/scripts/run_m3_${dataset,,}_repeated.sh" 2>&1 \
      | filter_pipeline_output
  status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    echo "GPU_POOL_CHAIN_FAIL dataset=$dataset models=$models gpu=$gpu status=$status"
    return "$status"
  fi
  echo "GPU_POOL_CHAIN_DONE dataset=$dataset models=$models gpu=$gpu"
}

run_prepass() {
  local dataset="$1" status batch_map adapter_batch_map
  if [[ "$dataset" == "Europe" ]]; then
    batch_map="$EUROPE_BATCH_SIZE_MULTIPLIERS"
    adapter_batch_map="$EUROPE_ADAPTER_BATCH_SIZE_MULTIPLIERS"
  else
    batch_map="$GLOBAL_BATCH_SIZE_MULTIPLIERS"
    adapter_batch_map="$GLOBAL_ADAPTER_BATCH_SIZE_MULTIPLIERS"
  fi
  echo "PIPELINE_PREPASS_START dataset=$dataset"
  set +e
  MODELS="$NON_HISTGNN_MODELS" PRECOMPLETED_ADAPTERS_FIRST=1 \
    BATCH_SIZE_MULTIPLIERS="$batch_map" \
    ADAPTER_BATCH_SIZE_MULTIPLIERS="$adapter_batch_map" \
    PREPASS_ONLY=1 RUN_SUMMARY=0 \
    bash "$PROJECT_ROOT/scripts/run_m3_${dataset,,}_repeated.sh" 2>&1 \
      | filter_pipeline_output
  status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    echo "PIPELINE_PREPASS_FAIL dataset=$dataset status=$status"
    return "$status"
  fi
  echo "PIPELINE_PREPASS_DONE dataset=$dataset"
}

run_dynamic_chains() {
  local dataset="$1"
  local -a pools=("0,1,2" "3,4,5")
  local -A pid_to_pool=()
  local -A pid_to_models=()
  local next_index=0 active=0 failures=0
  local pool models pid finished_pid status

  # Keep two independent three-GPU workers fed from one queue. Unlike the old
  # fixed pairs, the first worker that finishes immediately receives the next
  # model chain, so it never waits for the slower worker in the same round.
  echo "GPU_POOL_DYNAMIC_START dataset=$dataset chains=${#PAIRED_CHAINS[@]} pools=${pools[*]}"
  for pool in "${pools[@]}"; do
    if (( next_index >= ${#PAIRED_CHAINS[@]} )); then
      break
    fi
    models="${PAIRED_CHAINS[$next_index]}"
    echo "GPU_POOL_DYNAMIC_DISPATCH dataset=$dataset models=$models gpu=$pool"
    run_chain "$dataset" "$models" "$pool" &
    pid=$!
    pid_to_pool["$pid"]="$pool"
    pid_to_models["$pid"]="$models"
    next_index=$((next_index + 1))
    active=$((active + 1))
  done

  while (( active > 0 )); do
    finished_pid=""
    set +e
    wait -n -p finished_pid "${!pid_to_pool[@]}"
    status=$?
    set -e
    if [[ -z "$finished_pid" ]]; then
      echo "GPU_POOL_DYNAMIC_WAIT_FAIL dataset=$dataset active=$active" >&2
      return 1
    fi

    pool="${pid_to_pool[$finished_pid]}"
    models="${pid_to_models[$finished_pid]}"
    unset 'pid_to_pool[$finished_pid]' 'pid_to_models[$finished_pid]'
    active=$((active - 1))
    echo "GPU_POOL_DYNAMIC_RELEASE dataset=$dataset models=$models gpu=$pool status=$status"
    if (( status != 0 )); then
      failures=$((failures + 1))
    fi
    update_progress || true

    if (( next_index < ${#PAIRED_CHAINS[@]} )); then
      models="${PAIRED_CHAINS[$next_index]}"
      echo "GPU_POOL_DYNAMIC_DISPATCH dataset=$dataset models=$models gpu=$pool"
      run_chain "$dataset" "$models" "$pool" &
      pid=$!
      pid_to_pool["$pid"]="$pool"
      pid_to_models["$pid"]="$models"
      next_index=$((next_index + 1))
      active=$((active + 1))
    fi
  done

  echo "GPU_POOL_DYNAMIC_DONE dataset=$dataset failures=$failures"
  (( failures == 0 ))
}

wait_parallel_chains() {
  local dataset="$1"; shift
  local item model gpus pid status failures=0
  local -a pids=()
  for item in "$@"; do
    IFS='|' read -r model gpus <<< "$item"
    run_chain "$dataset" "$model" "$gpus" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    set +e
    wait "$pid"; status=$?
    set -e
    if (( status != 0 )); then
      failures=$((failures + 1))
    fi
  done
  update_progress || true
  (( failures == 0 ))
}

# Dataset barrier 1: Europe completes every non-HiSTGNN baseline and Adapter.
echo "PIPELINE_PHASE_START dataset=Europe scope=non_HiSTGNN"
run_prepass Europe
run_dynamic_chains Europe
wait_parallel_chains Europe \
  "Corrformer|0,1" "Moirai|2,3,4" "Timer,TimeMoE|5"
echo "PIPELINE_PHASE_DONE dataset=Europe scope=non_HiSTGNN"

# Dataset barrier 2: Global cannot start until the Europe pool above is done.
echo "PIPELINE_PHASE_START dataset=Global scope=non_HiSTGNN"
run_prepass Global
run_dynamic_chains Global
wait_parallel_chains Global \
  "Corrformer|0,1" "Moirai|2,3" "Timer|4" "TimeMoE|5"
echo "PIPELINE_PHASE_DONE dataset=Global scope=non_HiSTGNN"

# The legacy six-GPU launcher retains this preferred final phase, but there is
# no longer a hard result barrier; the work-queue launcher treats HiSTGNN as a
# normal per-seed job and may backfill it earlier.
echo "PIPELINE_PHASE_START dataset=Europe,Global scope=HiSTGNN"
run_chain Europe HiSTGNN "0,1,2" &
hist_europe_pid=$!
run_chain Global HiSTGNN "3,4,5" &
hist_global_pid=$!
wait "$hist_europe_pid"
wait "$hist_global_pid"
echo "PIPELINE_PHASE_DONE dataset=Europe,Global scope=HiSTGNN"
update_progress

RUN_EXPERIMENTS=0 RUN_SUMMARY=1 \
  BATCH_SIZE_MULTIPLIERS="$EUROPE_BATCH_SIZE_MULTIPLIERS" \
  ADAPTER_BATCH_SIZE_MULTIPLIERS="$EUROPE_ADAPTER_BATCH_SIZE_MULTIPLIERS" \
  bash "$PROJECT_ROOT/scripts/run_m3_europe_repeated.sh"
RUN_EXPERIMENTS=0 RUN_SUMMARY=1 \
  BATCH_SIZE_MULTIPLIERS="$GLOBAL_BATCH_SIZE_MULTIPLIERS" \
  ADAPTER_BATCH_SIZE_MULTIPLIERS="$GLOBAL_ADAPTER_BATCH_SIZE_MULTIPLIERS" \
  bash "$PROJECT_ROOT/scripts/run_m3_global_repeated.sh"
