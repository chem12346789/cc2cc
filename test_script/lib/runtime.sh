#!/usr/bin/env bash

TEST_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_SCRIPT_DIR}/.." && pwd)"

if [[ -f "${TEST_SCRIPT_DIR}/cluster.local.sh" ]]; then
    source "${TEST_SCRIPT_DIR}/cluster.local.sh"
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"
