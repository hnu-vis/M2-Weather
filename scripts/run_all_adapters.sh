#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec "${PYTHON_BIN:-/root/miniconda3/envs/tslib/bin/python}" \
  utils/run_all_adapters.py "$@"
