#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/cluster.local.sh" ]]; then
    source "${SCRIPT_DIR}/cluster.local.sh"
fi

exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/submit_direct_array_per_gpu.py" "$@"
