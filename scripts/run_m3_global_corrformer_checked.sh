#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
: "${GPU_IDS:?A single exclusively reserved GPU is required}"
: "${SEEDS:?A single seed is required}"
if [[ "$GPU_IDS" == *,* || "$SEEDS" == *,* ]]; then
  echo 'Corrformer checked worker requires exactly one GPU and seed' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"${TSLIB_PYTHON:-/root/miniconda3/envs/tslib/bin/python}" -u utils/check_corrformer_global_memory.py \
  --report "logs/pipeline/corrformer_preflight_seed${SEEDS}.json"
exec bash scripts/run_m3_global_repeated.sh
