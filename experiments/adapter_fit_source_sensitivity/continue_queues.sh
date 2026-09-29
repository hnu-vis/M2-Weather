#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/M3_benchmark
PY=/root/miniconda3/envs/tslib/bin/python
RUNNER="$ROOT/experiments/adapter_fit_source_sensitivity/run_sweep.py"
QUEUE_DIR="$ROOT/artifacts/adapter_fit_source_sensitivity/queues"
mkdir -p "$QUEUE_DIR"

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

run_worker() {
  "$PY" -u "$RUNNER" --worker "$1" "$2" "$3" "$4"
}

queue_gpu0() {
  wait_for_pid 256402
  run_worker Global PatchTST 2024 0
  run_worker Global S2Transformer 2024 0
}

queue_gpu1() {
  wait_for_pid 256403
  run_worker Global PatchTST 2025 1
  run_worker Global S2Transformer 2025 1
}

queue_gpu2() {
  wait_for_pid 256404
  run_worker Global PatchTST 2026 2
  run_worker Global S2Transformer 2026 2
}

queue_gpu3() {
  wait_for_pid 256405
  run_worker France S2Transformer 2024 3
  run_worker Global iTransformer 2024 3
}

queue_gpu4() {
  wait_for_pid 256406
  run_worker France S2Transformer 2025 4
  run_worker Global iTransformer 2025 4
}

queue_gpu5() {
  wait_for_pid 256407
  run_worker France S2Transformer 2026 5
  run_worker Global iTransformer 2026 5
}

if [[ $# -eq 1 ]]; then
  "queue_gpu${1}"
  exit 0
fi

for gpu in 0 1 2 3 4 5; do
  nohup setsid "$0" "$gpu" >"$QUEUE_DIR/gpu${gpu}.log" 2>&1 </dev/null &
  echo "$!" >"$QUEUE_DIR/gpu${gpu}.pid"
done
