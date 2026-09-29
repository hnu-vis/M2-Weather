#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the two dataset schedulers sequential: each baseline scheduler already
# fills all visible GPUs, so concurrent schedulers could race and oversubscribe
# a device.  Adapter fitting can safely use all six devices after each
# dataset's baseline stage is complete.
export ADAPTER_GPUS="${ADAPTER_GPUS:-0,1,2,3,4,5}"

bash "$PROJECT_ROOT/scripts/run_m3_europe_repeated.sh"
bash "$PROJECT_ROOT/scripts/run_m3_global_repeated.sh"
