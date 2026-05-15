#!/usr/bin/env bash
# ==============================================================
# run_GSE140228_all_methods_all_tasks.sh
# ==============================================================
set -euo pipefail
IFS=$'\n\t'

ts()       { date +"%Y-%m-%d %H:%M:%S"; }
die()      { echo "[$(ts)] [ERROR] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
need_file(){ [[ -f "$1" ]] || die "File not found: $1"; }
need_dir() { [[ -d "$1" ]] || die "Directory not found: $1"; }
mkdirp()   { mkdir -p "$1" || die "Failed to create directory: $1"; }

on_err() {
  local code=$?
  echo "[$(ts)] [ERROR] Exit ${code}. Last: ${BASH_COMMAND}" >&2
  exit "${code}"
}
trap on_err ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

METHODS_REGISTRY="${SCRIPT_DIR}/../methods.sh"
need_file "${METHODS_REGISTRY}"
# shellcheck source=/dev/null
source "${METHODS_REGISTRY}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"

DATASET_NAME="GSE140228"

INPUT_H5AD="${INPUT_H5AD:-/data1/esraa/Thesis-Project/Data/Processed_data/post_norm_log1p/adata_140228.h5ad}"

OUT_DATASET_BASE="${OUT_DATASET_BASE:-/data1/esraa/Thesis-Project/Results/Trajectory_Inference/${DATASET_NAME}}"

# ------------------------------------------------------------------
# PRIORS REGISTRY ROOT:
# should contain: <PRIORS_ROOT>/GSE140228/<task_key>.json
# ------------------------------------------------------------------
PRIORS_ROOT="${PRIORS_ROOT:-/data1/esraa/Thesis-Project/src/thesis_project/Trajectory_Inference/priors_registry}"

INCLUDE_KEY="celltype_sub"
EXCLUDE_KEY="celltype_sub"
GROUP_KEY="celltype_sub"

# No label replacement needed by default for GSE140228
REPLACE_LABELS_JSON="${REPLACE_LABELS_JSON:-}"

N_NEIGHBORS="${N_NEIGHBORS:-20}"
N_PCS="${N_PCS:-30}"

N_TOP_GENES="${N_TOP_GENES:-3000}"
SCALE="${SCALE:-false}"
USE_HVG="${USE_HVG:-true}"

# ------------------------------------------------------------------
# PREPROCESSED INPUT SWITCH
#   true  -> input already normalized/log1p; skip those steps
#   false -> input is raw counts; allow normalize/log1p in pipeline
# ------------------------------------------------------------------
PREPROCESSED_INPUT="${PREPROCESSED_INPUT:-true}"
if [[ "${PREPROCESSED_INPUT}" == "true" ]]; then
  SKIP_NORMALIZE="true"
  SKIP_LOG1P="true"
else
  SKIP_NORMALIZE="false"
  SKIP_LOG1P="false"
fi

N_BOOTSTRAPS="${N_BOOTSTRAPS:-20}"
BOOTSTRAP_FRAC="${BOOTSTRAP_FRAC:-0.8}"
BOOTSTRAP_MIN_PER_GROUP="${BOOTSTRAP_MIN_PER_GROUP:-10}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-0}"

SEED="${SEED:-0}"

EXPORT_OBS_KEYS="${EXPORT_OBS_KEYS:-Donor,Tissue,Sample,Histology,Tissue_sub,celltype_global,Platform}"

# -------------------------
# VIA defaults (consumed by methods.sh -> via extra args)
# -------------------------
VIA_REP_KEY="X_pca"
VIA_REP_DIMS=""  # leave empty to not pass --rep-dims
VIA_KNN=30
VIA_DISTANCE="l2"
VIA_CLUSTER_GRAPH_PRUNING=0.15
VIA_EDGEPRUNING_CLUSTERING_RESOLUTION=0.15
VIA_NUM_THREADS=1
VIA_GROUP_GRAPH_WEIGHT="similarity"
VIA_PRESERVE_DISCONNECTED="true"

# -------------------------
# CellRank defaults
# -------------------------
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

need_cmd "${PYTHON_BIN}"
need_cmd "${RSCRIPT}"
need_file "${INPUT_H5AD}"
need_dir  "${PRIORS_ROOT}"
need_dir  "${PRIORS_ROOT}/${DATASET_NAME}"
mkdirp "${OUT_DATASET_BASE}"

for m in "${METHODS[@]}"; do
  mp="$(method_py_path "${m}")"
  [[ -n "${mp}" ]] || die "Unknown method in METHODS array: ${m}"
  need_file "${mp}"
done

# Only check R scripts if you actually run those methods
need_file "${METHOD_R_TSCAN}"
need_file "${METHOD_R_SLINGSHOT}"
need_file "${METHOD_R_SCORPIUS}"
need_file "${METHOD_R_MONOCLE3}"

touch "${OUT_DATASET_BASE}/.write_test" 2>/dev/null || die "Output directory not writable: ${OUT_DATASET_BASE}"
rm -f "${OUT_DATASET_BASE}/.write_test"

echo "============================================================"
echo "[$(ts)] [START] TI all-methods / immune-lineage tasks runner"
echo "  Dataset    : ${DATASET_NAME}"
echo "  Input h5ad : ${INPUT_H5AD}"
echo "  Output base: ${OUT_DATASET_BASE}"
echo "  Priors root: ${PRIORS_ROOT}"
echo "  Python     : ${PYTHON_BIN}"
echo "  Rscript    : ${RSCRIPT}"
echo "  Host       : $(uname -a)"
echo "  Methods    : ${METHODS[*]}"
echo "============================================================"
echo

# ------------------------------------------------------------------
# IMPORTANT:
# TASK_KEYS must match priors filenames exactly:
#   ${PRIORS_ROOT}/${DATASET_NAME}/${TASK_KEY}.json
# ------------------------------------------------------------------
TASK_KEYS=(
 "task3_myeloid_monocyte_to_TAM"
 "task4_CD8_exhaustion"
 "task5_CD4_differentiation"
  "task6_NK_maturation"
)

TASK_LABELS=(
  "Monocyte-to-TAM Differentiation"
  "CD8 T Cell Exhaustion Continuum"
  "CD4 T Cell Differentiation"
  "NK Cell Maturation"
)

TASK_INCLUDE=(
  "Mono-C1-CD14,Mono-C2-FCGR3A,Mφ-C5-VCAN,Mφ-C1-THBS1,Mφ-C2-C1QA,Mφ-C3-APOE,Mφ-C4-GPX3,Mφ-C6-MARCO"
  "CD4/CD8-C1-CCR7,CD8-C5-SELL,CD8-C3-IL7R,CD8-C6-GZMK,CD8-C7-KLRD1,CD8-C8-PDCD1,CD8-C4-CX3CR1"
  "CD4-C5-TCF7,CD4/CD8-C1-CCR7,CD4-C4-IL7R,CD4-C3-ANXA1,CD4-C6-CXCL13,CD4-C7-FOXP3"
  "NK-C2-SELL,NK-C5-CD69,NK-C1-FCGR3A,NK-C7-CD160,NK-C3-IFNG"
)

TASK_EXCLUDE=(
  ""
  "CD8-C9-SLC4A10,CD4/CD8-C2-MKI67"
  "CD4/CD8-C2-MKI67"
  ""
)

TASK_ROOT=(
  "Mono-C1-CD14"
  "CD4/CD8-C1-CCR7"
  "CD4-C5-TCF7"
  "NK-C2-SELL"
)

for t in "${TASK_KEYS[@]}"; do
  need_file "${PRIORS_ROOT}/${DATASET_NAME}/${t}.json"
done

run_one() {
  local method="$1"
  local task_key="$2"
  local task_label="$3"
  local include_values="$4"
  local exclude_values="$5"
  local root_group="$6"

  [[ -n "${method}" ]] || die "run_one: method is empty"
  [[ -n "${task_key}" ]] || die "run_one: task_key is empty"
  [[ -n "${task_label}" ]] || die "run_one: task_label is empty"
  [[ -n "${include_values}" ]] || die "run_one: include_values is empty for ${task_key}"
  [[ -n "${root_group}" ]] || die "run_one: root_group is empty for ${task_key}"

  local method_py
  method_py="$(method_py_path "${method}")"
  [[ -n "${method_py}" ]] || die "No script registered for method: ${method}"

  local priors_file="${PRIORS_ROOT}/${DATASET_NAME}/${task_key}.json"
  need_file "${priors_file}"

  local out_dir="${OUT_DATASET_BASE}/${method}/${task_key}"
  mkdirp "${out_dir}"

  local log_file="${out_dir}/run.log"
  local cmd_file="${out_dir}/cmd.txt"

  echo "------------------------------------------------------------"
  echo "[$(ts)] [RUN] dataset=${DATASET_NAME} method=${method} task=${task_key}"
  echo "  Label      : ${task_label}"
  echo "  Output     : ${out_dir}"
  echo "  Root group : ${root_group}"
  echo "  Include    : ${include_values}"
  echo "  Exclude    : ${exclude_values:-<none>}"
  echo "------------------------------------------------------------"

  local cmd=(
    "${PYTHON_BIN}" "${method_py}"
    --method      "${method}"
    --dataset     "${DATASET_NAME}"
    --task        "${task_key}"
    --adata       "${INPUT_H5AD}"
    --run-dir     "${out_dir}"
    --priors-root "${PRIORS_ROOT}"
    --include-key    "${INCLUDE_KEY}"
    --include-values "${include_values}"
    --group-key      "${GROUP_KEY}"
    --root-group     "${root_group}"
    --n-neighbors "${N_NEIGHBORS}"
    --n-pcs       "${N_PCS}"
    --n-bootstrap             "${N_BOOTSTRAPS}"
    --bootstrap-frac          "${BOOTSTRAP_FRAC}"
    --bootstrap-min-per-group "${BOOTSTRAP_MIN_PER_GROUP}"
    --bootstrap-seed          "${BOOTSTRAP_SEED}"
    --export-obs-keys "${EXPORT_OBS_KEYS}"
    --random-state "${SEED}"
  )

  if [[ -n "${REPLACE_LABELS_JSON}" ]]; then
    cmd+=( --replace-labels-json "${REPLACE_LABELS_JSON}" )
  fi

  if [[ -n "${exclude_values}" ]]; then
    cmd+=( --exclude-key "${EXCLUDE_KEY}" --exclude-values "${exclude_values}" )
  fi

  if [[ "${SKIP_NORMALIZE}" == "true" ]]; then
    cmd+=( --no-normalize )
  fi
  if [[ "${SKIP_LOG1P}" == "true" ]]; then
    cmd+=( --no-log1p )
  fi
  if [[ "${SCALE}" == "true" ]]; then
    cmd+=( --scale )
  fi
  if [[ "${USE_HVG}" == "true" ]]; then
    cmd+=( --hvg-subset --n-top-genes "${N_TOP_GENES}" )
  fi

  local -a extras=()
  mapfile -t extras < <(method_extra_args "${method}")
  if [[ "${#extras[@]}" -gt 0 ]]; then
    cmd+=( "${extras[@]}" )
  fi

  {
    echo "# $(ts)"
    printf "%q " "${cmd[@]}"
    echo
  } > "${cmd_file}"

  echo "[$(ts)] [INFO] Logging to: ${log_file}"
  "${cmd[@]}" |& tee "${log_file}"

  echo "[$(ts)] [DONE] ${method} :: ${task_key}"
  echo
}

for method in "${METHODS[@]}"; do
  for idx in "${!TASK_KEYS[@]}"; do
    run_one \
      "${method}" \
      "${TASK_KEYS[$idx]}" \
      "${TASK_LABELS[$idx]}" \
      "${TASK_INCLUDE[$idx]}" \
      "${TASK_EXCLUDE[$idx]}" \
      "${TASK_ROOT[$idx]}"
  done
done

echo "============================================================"
echo "[$(ts)] [ALL DONE]"
echo "  Outputs written under: ${OUT_DATASET_BASE}/<method>/<task>/"
echo "============================================================"