#!/bin/bash

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/combine_runs.py"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/combine_runs_$(date +%Y%m%d_%H%M%S).log"

# ── Environment ──────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-env

# ── Run ──────────────────────────────────────────────────────────────────────
echo "============================================"
echo "Starting combine_runs"
echo "Time   : $(date)"
echo "Host   : $(hostname)"
echo "Script : ${PYTHON_SCRIPT}"
echo "Log    : ${LOG_FILE}"
echo "============================================"

python "${PYTHON_SCRIPT}" 2>&1 | tee "${LOG_FILE}"

echo "============================================"
echo "Finished combine_runs"
echo "Time : $(date)"
echo "============================================"