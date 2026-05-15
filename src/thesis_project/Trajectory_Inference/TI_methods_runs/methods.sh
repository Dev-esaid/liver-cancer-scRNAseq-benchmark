#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# methods.sh
# Registry of TI methods for one dataset launcher.
#
# IMPORTANT — IFS in the calling runner is set to $'\n\t'
# (space is NOT a field separator). method_extra_args() must
# emit each argument token on its OWN LINE.
# ============================================================

_METHODS_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_METHODS_SH_DIR}/../../../.." && pwd)"

TI_METHODS_DIR="${PROJECT_ROOT}/src/thesis_project/Trajectory_Inference/TI_methods"
TI_R_DIR="${TI_METHODS_DIR}/R"

# -------------------------
# Method script paths
# -------------------------
METHOD_PY_PAGA="${TI_METHODS_DIR}/paga.py"
METHOD_PY_VIA="${TI_METHODS_DIR}/via.py"
METHOD_PY_ELPIGRAPH="${TI_METHODS_DIR}/elpigraph.py"
METHOD_PY_CELLRANK="${TI_METHODS_DIR}/cellrank.py"
METHOD_PY_MONOCLE3="${TI_METHODS_DIR}/monocle3.py"
METHOD_PY_SLINGSHOT="${TI_METHODS_DIR}/slingshot.py"
METHOD_PY_SCORPIUS="${TI_METHODS_DIR}/scorpius.py"
METHOD_PY_TSCAN="${TI_METHODS_DIR}/tscan.py"

# -------------------------
# R scripts (sanity-check need_file calls in main runner)
# -------------------------

METHOD_R_MONOCLE3="${TI_R_DIR}/monocle3_run.R"
METHOD_R_SLINGSHOT="${TI_R_DIR}/slingshot_run.R"
METHOD_R_SCORPIUS="${TI_R_DIR}/scorpius_run.R"
METHOD_R_TSCAN="${TI_R_DIR}/tscan_run.R"

# -------------------------
# Ordered list of methods to run
# -------------------------
METHODS=(
# "paga"
# "via"
"cellrank" 
# "scorpius"
"tscan"
"monocle3"  
"slingshot"
#"elpigraph" 
)

method_py_path() {
  local m="$1"
  case "${m}" in
    paga)      echo "${METHOD_PY_PAGA}" ;;
    via)       echo "${METHOD_PY_VIA}" ;;
    elpigraph) echo "${METHOD_PY_ELPIGRAPH}" ;;
    cellrank)  echo "${METHOD_PY_CELLRANK}" ;;
    monocle3)  echo "${METHOD_PY_MONOCLE3}" ;;
    slingshot) echo "${METHOD_PY_SLINGSHOT}" ;;
    scorpius)  echo "${METHOD_PY_SCORPIUS}" ;;
    tscan)     echo "${METHOD_PY_TSCAN}" ;;
    *)         echo "" ;;
  esac
}

method_extra_args() {
  local m="$1"
  case "${m}" in

    paga)
      printf '%s\n' \
        "--paga-threshold" "${PAGA_THRESHOLD:-0.2}"
      ;;

    via)
      printf '%s\n' \
        "--rep-key" "${VIA_REP_KEY:-X_pca}" \
        "--via-knn" "${VIA_KNN:-30}" \
        "--via-distance" "${VIA_DISTANCE:-l2}" \
        "--via-cluster-graph-pruning" "${VIA_CLUSTER_GRAPH_PRUNING:-0.15}" \
        "--via-edgepruning-clustering-resolution" "${VIA_EDGEPRUNING_CLUSTERING_RESOLUTION:-0.15}" \
        "--via-num-threads" "${VIA_NUM_THREADS:-1}" \
        "--group-graph-weight" "${VIA_GROUP_GRAPH_WEIGHT:-similarity}"

      if [[ -n "${VIA_REP_DIMS:-}" ]]; then
        printf '%s\n' "--rep-dims" "${VIA_REP_DIMS}"
      fi

      if [[ "${VIA_PRESERVE_DISCONNECTED:-false}" == "true" ]]; then
        printf '%s\n' "--via-preserve-disconnected"
      fi
      ;;

    cellrank)
      # CellRank adapter uses CytoTRACEKernel (no RNA velocity).
      printf '%s\n' \
        "--cellrank-kernel"                "${CELLRANK_KERNEL:-cytotrace}" \
        "--cellrank-cytotrace-layer"       "${CELLRANK_CYTOTRACE_LAYER:-X}" \
        "--cellrank-cytotrace-n-genes"     "${CELLRANK_CYTOTRACE_N_GENES:-200}" \
        "--cellrank-cytotrace-aggregation" "${CELLRANK_CYTOTRACE_AGGREGATION:-mean}" \
        "--cellrank-threshold-scheme"      "${CELLRANK_THRESHOLD_SCHEME:-hard}" \
        "--cellrank-frac-to-keep"          "${CELLRANK_FRAC_TO_KEEP:-0.3}" \
        "--cellrank-b"                     "${CELLRANK_B:-10.0}" \
        "--cellrank-nu"                    "${CELLRANK_NU:-0.5}" \
        "--cellrank-n-jobs"                "${CELLRANK_N_JOBS:-1}" \
        "--cellrank-graph-mode"            "${CELLRANK_GRAPH_MODE:-mst}" \
        "--cellrank-symmetrize"            "${CELLRANK_SYMMETRIZE:-mean}"

      if [[ "${CELLRANK_CYTOTRACE_USE_RAW:-false}" == "true" ]]; then
        printf '%s\n' "--cellrank-cytotrace-use-raw"
      fi

      if [[ "${CELLRANK_ORIENT_TO_ROOT:-true}" == "true" ]]; then
        printf '%s\n' "--cellrank-orient-to-root"
      else
        printf '%s\n' "--cellrank-no-orient-to-root"
      fi

      if [[ "${CELLRANK_GRAPH_DIRECTED:-true}" == "true" ]]; then
        printf '%s\n' "--cellrank-graph-directed"
      else
        printf '%s\n' "--cellrank-graph-undirected"
      fi
      ;;

    elpigraph)
      printf '%s\n' \
        "--rep-key" "${ELPIGRAPH_REP_KEY:-X_pca}" \
        "--elpigraph-num-nodes" "${ELPIGRAPH_NUM_NODES:-50}" \
        "--elpigraph-lambda" "${ELPIGRAPH_LAMBDA:-0.01}" \
        "--elpigraph-mu" "${ELPIGRAPH_MU:-0.1}" \
        "--elpigraph-trimming-radius" "${ELPIGRAPH_TRIMMING_RADIUS:-inf}" \
        "--elpigraph-max-iter" "${ELPIGRAPH_MAX_ITER:-10}" \
        "--elpigraph-eps" "${ELPIGRAPH_EPS:-0.01}" \
        "--elpigraph-n-reps" "${ELPIGRAPH_N_REPS:-1}" \
        "--group-graph-weight" "${ELPIGRAPH_GROUP_GRAPH_WEIGHT:-unit}"
      ;;

    monocle3|slingshot|scorpius|tscan)
      printf '%s\n' "--rscript" "${RSCRIPT:-Rscript}"
      ;;

    *)
      : ;;
  esac
}