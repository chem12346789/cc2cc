#!/usr/bin/env bash

REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${SCRIPT_DIR}/cluster.local.sh" ]]; then
    source "${SCRIPT_DIR}/cluster.local.sh"
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
