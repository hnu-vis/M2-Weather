#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec "${TSLIB_PYTHON:-/root/miniconda3/envs/tslib/bin/python}" -u utils/schedule_m3_work_queue.py "$@"
