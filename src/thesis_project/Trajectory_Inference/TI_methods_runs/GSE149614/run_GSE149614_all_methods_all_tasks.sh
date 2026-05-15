#!/usr/bin/env bash
# ==============================================================
# run_GSE149614_all_methods_all_tasks.sh
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

PYTHON_BIN="${PYTHON_BIN:-/data1/esraa/miniconda3/envs/thesis-env/bin/python}"
RSCRIPT="${RSCRIPT:-Rscript}"

DATASET_NAME="GSE149614"
INPUT_H5AD="/data1/esraa/Thesis-Project/Data/Processed_data/post_norm_log1p/adata_149614.h5ad"
OUT_DATASET_BASE="/data1/esraa/Thesis-Project/Results/Trajectory_Inference/${DATASET_NAME}"

# ✅ PRIORS REGISTRY ROOT (the directory that contains <dataset>/<task>.json)
# Based on your repo tree screenshot:
PRIORS_ROOT="${PRIORS_ROOT:-/data1/esraa/Thesis-Project/src/thesis_project/Trajectory_Inference/priors_registry}"

INCLUDE_KEY="cluster_annotation"
EXCLUDE_KEY="cluster_annotation"
GROUP_KEY="cluster_annotation"
REPLACE_LABELS_JSON="{\"patient-specific macropahge\":\"patient-specific macrophage\"}"

N_NEIGHBORS=20
N_PCS=30

N_TOP_GENES=3000
SKIP_NORMALIZE="true"
SKIP_LOG1P="true"
SCALE="false"
USE_HVG="true"

N_BOOTSTRAPS=20
BOOTSTRAP_FRAC=0.8
BOOTSTRAP_MIN_PER_GROUP=10
BOOTSTRAP_SEED=0

SEED=0

EXPORT_OBS_KEYS="site,patient"

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
# CellRank defaults (consumed by methods.sh -> cellrank extra args)
# NOTE: this adapter runs CellRank WITHOUT RNA velocity using CytoTRACEKernel.
# -------------------------
CELLRANK_KERNEL="cytotrace"
CELLRANK_CYTOTRACE_LAYER="counts"
CELLRANK_CYTOTRACE_USE_RAW="false"
CELLRANK_CYTOTRACE_N_GENES=200
CELLRANK_CYTOTRACE_AGGREGATION="mean"
CELLRANK_THRESHOLD_SCHEME="hard"   # hard|soft
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
mkdirp "${OUT_DATASET_BASE}"

for m in "${METHODS[@]}"; do
  mp="$(method_py_path "${m}")"
  [[ -n "${mp}" ]] || die "Unknown method in METHODS array: ${m}"
  need_file "${mp}"
done

# Only check R scripts if you actually run those methods (harmless if present)
need_file "${METHOD_R_TSCAN}"
need_file "${METHOD_R_SLINGSHOT}"
need_file "${METHOD_R_SCORPIUS}"
need_file "${METHOD_R_MONOCLE3}"

touch "${OUT_DATASET_BASE}/.write_test" 2>/dev/null || die "Output directory not writable: ${OUT_DATASET_BASE}"
rm -f "${OUT_DATASET_BASE}/.write_test"

echo "============================================================"
echo "[$(ts)] [START] TI all-methods / first-two-tasks runner"
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

TASK_NAMES=(
  "task1_macrophage_TAM"
  "task2_Tcell_continuum"
)

TASK_INCLUDE=(
  "monocyte-derived macrophage,TREM2+ macrophage,VEGFA+ macrophage,MMP9+ macrophage,MARCO+ macrophage,proliferative macrophage,patient-specific macrophage"
  "naïve T,central memory T,effector memory T,tissue-resident memory T,cytotoxic T lymphocytes,regulatory T,intermediate T,proliferative CD4 T,proliferative CD8 T"
)

TASK_EXCLUDE=(
  "CD1C+ conventional dendritic cell,IRF4+ plasmacytoid dendritic cell,mastocyte"
  ""
)

TASK_ROOT=(
  "monocyte-derived macrophage"
  "naïve T"
)

run_one() {
  local method="$1"
  local task_name="$2"
  local include_values="$3"
  local exclude_values="$4"
  local root_group="$5"

  [[ -n "${method}" ]] || die "run_one: method is empty"
  [[ -n "${task_name}" ]] || die "run_one: task_name is empty"
  [[ -n "${include_values}" ]] || die "run_one: include_values is empty for ${task_name}"
  [[ -n "${root_group}" ]] || die "run_one: root_group is empty for ${task_name}"

  local method_py
  method_py="$(method_py_path "${method}")"
  [[ -n "${method_py}" ]] || die "No script registered for method: ${method}"

  local out_dir="${OUT_DATASET_BASE}/${method}/${task_name}"
  mkdirp "${out_dir}"

  local log_file="${out_dir}/run.log"
  local cmd_file="${out_dir}/cmd.txt"

  echo "------------------------------------------------------------"
  echo "[$(ts)] [RUN] dataset=${DATASET_NAME} method=${method} task=${task_name}"
  echo "  Output     : ${out_dir}"
  echo "  Root group : ${root_group}"
  echo "  Include    : ${include_values}"
  echo "  Exclude    : ${exclude_values:-<none>}"
  echo "------------------------------------------------------------"

  local cmd=(
    "${PYTHON_BIN}" "${method_py}"
    --method      "${method}"
    --dataset     "${DATASET_NAME}"
    --task        "${task_name}"
    --adata       "${INPUT_H5AD}"
    --run-dir     "${out_dir}"

    # ✅ Load priors from registry
    --priors-root "${PRIORS_ROOT}"

    --include-key    "${INCLUDE_KEY}"
    --include-values "${include_values}"
    --group-key      "${GROUP_KEY}"
    --root-group     "${root_group}"
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

  if [[ -n "${exclude_values}" ]]; then
    cmd+=( --exclude-key "${EXCLUDE_KEY}" --exclude-values "${exclude_values}" )
  fi

  if [[ "${SKIP_NORMALIZE}" == "true" ]]; then cmd+=( --no-normalize ); fi
  if [[ "${SKIP_LOG1P}" == "true" ]]; then cmd+=( --no-log1p ); fi
  if [[ "${SCALE}" == "true" ]]; then cmd+=( --scale ); fi
  if [[ "${USE_HVG}" == "true" ]]; then cmd+=( --hvg-subset --n-top-genes "${N_TOP_GENES}" ); fi

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

  echo "[$(ts)] [DONE] ${method} :: ${task_name}"
  echo
}

for method in "${METHODS[@]}"; do
  for idx in "${!TASK_NAMES[@]}"; do
    run_one \
      "${method}" \
      "${TASK_NAMES[$idx]}" \
      "${TASK_INCLUDE[$idx]}" \
      "${TASK_EXCLUDE[$idx]}" \
      "${TASK_ROOT[$idx]}"
  done
done

echo "============================================================"
echo "[$(ts)] [ALL DONE]"
echo "  Outputs written under: ${OUT_DATASET_BASE}/<method>/<task>/"
echo "============================================================"