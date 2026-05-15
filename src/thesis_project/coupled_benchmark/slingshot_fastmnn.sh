#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ts()       { date +"%Y-%m-%d %H:%M:%S"; }
die()      { echo "[$(ts)] [ERROR] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
need_file(){ [[ -f "$1" ]] || die "File not found: $1"; }
need_dir() { [[ -d "$1" ]] || die "Directory not found: $1"; }
mkdirp()   { mkdir -p "$1" || die "Failed to create directory: $1"; }

print_sep() {
  echo "======================================================================"
}

print_batch_sep() {
  echo "######################################################################"
}

print_run_start() {
  local ti_method="$1"
  local task_id="$2"
  local task_name="$3"
  local integrated_name="$4"
  local batch_label="$5"
  local item_idx="$6"
  local total_items="$7"
  local input_h5ad="$8"
  local out_dir="$9"
  local root_group="${10}"
  local prior_json="${11}"

  echo
  print_sep
  echo "[$(ts)] START RUN"
  echo "  TI method        : ${ti_method}"
  echo "  Task ID          : ${task_id}"
  echo "  Task name        : ${task_name}"
  echo "  Integration data : ${integrated_name}"
  echo "  Batch            : ${batch_label}"
  echo "  Item             : ${item_idx}/${total_items}"
  echo "  Input adata      : ${input_h5ad}"
  echo "  Output dir       : ${out_dir}"
  echo "  Root group       : ${root_group}"
  echo "  Prior JSON       : ${prior_json}"
  print_sep
  echo
}

print_run_done() {
  local ti_method="$1"
  local task_id="$2"
  local integrated_name="$3"

  echo
  print_sep
  echo "[$(ts)] DONE"
  echo "  TI method        : ${ti_method}"
  echo "  Task ID          : ${task_id}"
  echo "  Integration data : ${integrated_name}"
  print_sep
  echo
}

print_batch_start() {
  local batch_id="$1"
  local start_idx="$2"
  local end_idx="$3"
  local total_items="$4"

  echo
  print_batch_sep
  echo "[$(ts)] STARTING BATCH ${batch_id}"
  echo "  Items            : ${start_idx}-${end_idx} / ${total_items}"
  print_batch_sep
  echo
}

print_batch_done() {
  local batch_id="$1"

  echo
  print_batch_sep
  echo "[$(ts)] FINISHED BATCH ${batch_id}"
  print_batch_sep
  echo
}

on_err() {
  local code=$?
  echo
  print_sep >&2
  echo "[$(ts)] FAILED" >&2
  echo "  Exit code        : ${code}" >&2
  echo "  Last command     : ${BASH_COMMAND}" >&2
  print_sep >&2
  echo >&2
  exit "${code}"
}
trap on_err ERR

wait_jobs () {
  local label="$1"; shift
  local -a pids=("$@")
  echo
  echo "[$(ts)] ---- Waiting for ${label} jobs ----"
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
    echo "[$(ts)] [ERROR] ${label}: ${fail} job(s) failed."
    exit 1
  else
    echo "[$(ts)] [OK] ${label}: all jobs finished."
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

METHODS_REGISTRY="/data1/esraa/Thesis-Project/src/thesis_project/Trajectory_Inference/TI_methods_runs/methods.sh"
need_file "${METHODS_REGISTRY}"
# shellcheck source=/dev/null
source "${METHODS_REGISTRY}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"

# Hard-coded run settings
TI_METHOD="slingshot"

BASE_INTEGRATION_DIR="/data1/esraa/Thesis-Project/Results/Integration/full_hvg_seed0/runs"
need_dir "${BASE_INTEGRATION_DIR}"

PRIORS_ROOT="${PRIORS_ROOT:-/data1/esraa/Thesis-Project/src/thesis_project/coupled_benchmark/priors_registry}"
need_dir "${PRIORS_ROOT}"

DATASET_NAME="integrated_liver_atlas_9datasets"
GROUP_KEY="cell_subtype_L2"
INCLUDE_KEY="cell_subtype_L2"
EXCLUDE_KEY="cell_subtype_L2"
REPLACE_LABELS_JSON="{}"

N_NEIGHBORS=20
N_PCS=30
N_BOOTSTRAPS=20
BOOTSTRAP_FRAC=0.8
BOOTSTRAP_MIN_PER_GROUP=10
BOOTSTRAP_SEED=0
SEED=0

SKIP_NORMALIZE="true"
SKIP_LOG1P="true"
SCALE="false"
USE_HVG="false"
N_TOP_GENES=3000

EXPORT_OBS_KEYS="dataset,major_celltype_l1,cell_subtype_L2,donor_id,tumor_status,technology,cancer_type"

PARALLEL_BATCH_SIZE=1

CELLRANK_KERNEL="cytotrace"
CELLRANK_CYTOTRACE_LAYER="counts"
CELLRANK_CYTOTRACE_USE_RAW="false"
CELLRANK_CYTOTRACE_N_GENES=200
CELLRANK_CYTOTRACE_AGGREGATION="mean"
CELLRANK_THRESHOLD_SCHEME="hard"
CELLRANK_FRAC_TO_KEEP=0.3
CELLRANK_B=10.0
CELLRANK_NU=0.5
CELLRANK_N_JOBS=1
CELLRANK_GRAPH_MODE="mst"
CELLRANK_SYMMETRIZE="mean"
CELLRANK_GRAPH_DIRECTED="true"
CELLRANK_ORIENT_TO_ROOT="true"

TI_METHOD_PY="$(method_py_path "${TI_METHOD}")"
[[ -n "${TI_METHOD_PY}" ]] || die "Unknown TI method: ${TI_METHOD}"
need_file "${TI_METHOD_PY}"

need_cmd "${PYTHON_BIN}"
need_cmd "${RSCRIPT}"
need_file "${METHOD_R_SLINGSHOT}"

find_integrated_h5ad() {
  local short_name="$1"
  local path=""

  case "${short_name}" in
    fastmnn)
      path="${BASE_INTEGRATION_DIR}/fastmnn_full_hvg_seed0/adata_fastmnn_fastmnn_full_hvg_seed0.h5ad"
      ;;
    *)
      die "Unknown integration short name: ${short_name}"
      ;;
  esac

  need_file "${path}"
  echo "${path}"
}

configure_task() {
  local task_id="$1"

  case "${task_id}" in
    task_1)
      TASK_NAME="task1_Monocyte_macrophage_TAM"
      PRIOR_JSON_NAME="task1_Monocyte_macrophage_TAM.json"
      INCLUDE_VALUES="Tissue Monocyte,CD14+ Monocyte,FCGR3A+ Monocyte,Monocyte-derived Macrophage,Transitional Macrophage,Inflammatory Macrophage,MARCO+ Macrophage,TREM2+ Macrophage,C1QA+ Macrophage,MMP9+ Macrophage,VEGFA+ Macrophage,THBS1+ Macrophage,HLA-II+ Macrophage"
      EXCLUDE_VALUES=""
      ROOT_GROUP="Tissue Monocyte"
      ;;
    task_2)
      TASK_NAME="task2_CD8_Tcell_differentiation"
      PRIOR_JSON_NAME="task2_CD8_Tcell_differentiation.json"
      INCLUDE_VALUES="Naive T,CD8 T (SELL+),CD8 T (GZMK+),CD8 T (CX3CR1+),CD8 T (KLRD1+),Cytotoxic T,Intermediate T,Memory/Activated T,Central Memory T,Effector Memory T,Tissue-resident Memory T,CD8 T (PDCD1+)"
      EXCLUDE_VALUES=""
      ROOT_GROUP="Naive T"
      ;;
    *)
      die "TASK_ID must be one of: task_1, task_2"
      ;;
  esac

  PRIOR_JSON_PATH="${PRIORS_ROOT}/${DATASET_NAME}/${PRIOR_JSON_NAME}"
  need_file "${PRIOR_JSON_PATH}"
  OUT_BASE="/data1/esraa/Thesis-Project/Results/coupled_benchmark/${task_id}/${TI_METHOD}"
  mkdirp "${OUT_BASE}"
}

run_one() {
  local task_id="$1"
  local integrated_name="$2"
  local batch_label="$3"
  local item_idx="$4"
  local total_items="$5"

  configure_task "${task_id}"

  local input_h5ad
  input_h5ad="$(find_integrated_h5ad "${integrated_name}")"

  local out_dir="${OUT_BASE}/${integrated_name}"
  mkdirp "${out_dir}"

  local log_file="${out_dir}/run.log"
  local cmd_file="${out_dir}/cmd.txt"

  print_run_start \
    "${TI_METHOD}" \
    "${task_id}" \
    "${TASK_NAME}" \
    "${integrated_name}" \
    "${batch_label}" \
    "${item_idx}" \
    "${total_items}" \
    "${input_h5ad}" \
    "${out_dir}" \
    "${ROOT_GROUP}" \
    "${PRIOR_JSON_PATH}"

  local cmd=(
    "${PYTHON_BIN}" "${TI_METHOD_PY}"
    --method      "${TI_METHOD}"
    --dataset     "${DATASET_NAME}"
    --task        "${TASK_NAME}"
    --adata       "${input_h5ad}"
    --run-dir     "${out_dir}"
    --priors-root "${PRIORS_ROOT}"
    --include-key    "${INCLUDE_KEY}"
    --include-values "${INCLUDE_VALUES}"
    --group-key      "${GROUP_KEY}"
    --root-group     "${ROOT_GROUP}"
    --replace-labels-json "${REPLACE_LABELS_JSON}"
    --n-neighbors "${N_NEIGHBORS}"
    --n-pcs       "${N_PCS}"
    --n-bootstrap             "${N_BOOTSTRAPS}"
    --bootstrap-frac          "${BOOTSTRAP_FRAC}"
    --bootstrap-min-per-group "${BOOTSTRAP_MIN_PER_GROUP}"
    --bootstrap-seed          "${BOOTSTRAP_SEED}"
    --export-obs-keys "${EXPORT_OBS_KEYS}"
    --random-state "${SEED}"
  )

  if [[ -n "${EXCLUDE_VALUES}" ]]; then
    cmd+=( --exclude-key "${EXCLUDE_KEY}" --exclude-values "${EXCLUDE_VALUES}" )
  fi

  if [[ "${SKIP_NORMALIZE}" == "true" ]]; then cmd+=( --no-normalize ); fi
  if [[ "${SKIP_LOG1P}" == "true" ]]; then cmd+=( --no-log1p ); fi
  if [[ "${SCALE}" == "true" ]]; then cmd+=( --scale ); fi
  if [[ "${USE_HVG}" == "true" ]]; then cmd+=( --hvg-subset --n-top-genes "${N_TOP_GENES}" ); fi

  local -a extras=()
  mapfile -t extras < <(method_extra_args "${TI_METHOD}")
  if [[ "${#extras[@]}" -gt 0 ]]; then
    cmd+=( "${extras[@]}" )
  fi

  {
    echo "# $(ts)"
    printf "%q " "${cmd[@]}"
    echo
  } > "${cmd_file}"

  echo "[$(ts)] Running ${TI_METHOD} on ${integrated_name} (task=${TASK_NAME})"
  echo "[$(ts)] [LOG] ${log_file}"
  echo "[$(ts)] [CMD]"
  printf ' %q' "${cmd[@]}"
  echo
  echo

  "${cmd[@]}" |& sed "s/^/[${TI_METHOD}|${task_id}|${integrated_name}] /" | tee "${log_file}"

  print_run_done "${TI_METHOD}" "${task_id}" "${integrated_name}"
}

run_task() {
  local task_id="$1"

  local -a INTEGRATED_METHODS=("fastmnn")
  local pids=()
  local count=0
  local batch_idx=1
  local total_items="${#INTEGRATED_METHODS[@]}"

  configure_task "${task_id}"

  echo "============================================================"
  echo "[$(ts)] [START]"
  echo "  TI method        : ${TI_METHOD}"
  echo "  Task ID          : ${task_id}"
  echo "  Task name        : ${TASK_NAME}"
  echo "  Priors root      : ${PRIORS_ROOT}"
  echo "  Prior JSON       : ${PRIOR_JSON_PATH}"
  echo "  Output base      : ${OUT_BASE}"
  echo "  Parallel batch   : ${PARALLEL_BATCH_SIZE}"
  echo "  Total adatas     : ${#INTEGRATED_METHODS[@]}"
  echo "============================================================"

  for integrated_name in "${INTEGRATED_METHODS[@]}"; do
    count=$((count+1))

    batch_slot=$(( (count - 1) % PARALLEL_BATCH_SIZE + 1 ))
    if [[ "${batch_slot}" -eq 1 ]]; then
      batch_start_idx="${count}"
      batch_end_idx=$(( count + PARALLEL_BATCH_SIZE - 1 ))
      if (( batch_end_idx > total_items )); then
        batch_end_idx="${total_items}"
      fi
      print_batch_start "${batch_idx}" "${batch_start_idx}" "${batch_end_idx}" "${total_items}"
    fi

    run_one "${task_id}" "${integrated_name}" "${batch_idx}" "${count}" "${total_items}" &
    pids+=($!)

    if (( count % PARALLEL_BATCH_SIZE == 0 )); then
      wait_jobs "task ${task_id} batch ${batch_idx}" "${pids[@]}"
      print_batch_done "${batch_idx}"
      pids=()
      batch_idx=$((batch_idx+1))
    fi
  done

  if (( ${#pids[@]} > 0 )); then
    wait_jobs "task ${task_id} batch ${batch_idx}" "${pids[@]}"
    print_batch_done "${batch_idx}"
  fi

  echo "============================================================"
  echo "[$(ts)] [TASK DONE]"
  echo "  TI method   : ${TI_METHOD}"
  echo "  Task        : ${task_id}"
  echo "  Results     : ${OUT_BASE}/<integration_method>/"
  echo "============================================================"
}

run_task "task_1"
run_task "task_2"

echo "============================================================"
echo "[$(ts)] [ALL DONE]"
echo "  TI method   : ${TI_METHOD}"
echo "  Integration : fastmnn"
echo "  Tasks       : task_1, task_2"
echo "============================================================"