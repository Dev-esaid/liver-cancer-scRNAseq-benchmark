#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ts()  { date +"%Y-%m-%d %H:%M:%S"; }
die() { echo "[$(ts)] [ERROR] $*" >&2; exit 1; }

wait_jobs () {
  local label="$1"; shift
  local -a pids=("$@")
  echo
  echo "[$(ts)] ---- Waiting for ${label} ----"
  local fail=0
  local pid
  for pid in "${pids[@]}"; do
    if wait "${pid}"; then
      :
    else
      fail=$((fail+1))
    fi
  done
  if [[ "${fail}" -gt 0 ]]; then
    die "${label}: ${fail} job(s) failed"
  fi
  echo "[$(ts)] [OK] ${label} finished successfully"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/one_ti_method_across_integrations.sh"
[[ -f "${RUNNER}" ]] || die "Runner script not found: ${RUNNER}"

LOG_DIR="/data1/esraa/Thesis-Project/Results/coupled_benchmark"
LOG="${LOG_DIR}/four_methods_run.log"
mkdir -p "${LOG_DIR}"
printf '\n[%s] [RESTART] run_four_methods_two_batches.sh — PID %s\n' "$(ts)" "$$" >> "${LOG}"
exec > >(tee -a "${LOG}") 2>&1
echo "[$(ts)] Logging to: ${LOG}"

# Edit these four methods to the exact TI methods you want
TI_METHODS=(
  "tscan"
  "cellrank"
  "monocle3"
  "slingshot"
)

run_batch() {
  local task_id="$1"
  local batch_name="$2"

  echo
  echo "######################################################################"
  echo "[$(ts)] START ${batch_name} (${task_id})"
  echo "######################################################################"
  echo

  local pids=()
  local method
  for method in "${TI_METHODS[@]}"; do
    echo "[$(ts)] Launching ${method} for ${task_id}"
    bash "${RUNNER}" "${method}" "${task_id}" &
    pids+=($!)
  done

  wait_jobs "${batch_name} (${task_id})" "${pids[@]}"

  echo
  echo "######################################################################"
  echo "[$(ts)] DONE ${batch_name} (${task_id})"
  echo "######################################################################"
  echo
}

run_batch "task_1" "batch1"
run_batch "task_2" "batch2"

echo "[$(ts)] ALL BATCHES COMPLETED"