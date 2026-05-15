#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 3-BATCH INTEGRATION LAUNCHER
# - Batch 1: CPU methods
# - Batch 2: CPU methods
# - Batch 3: GPU methods running in parallel with CPU batches
# - Per-method logs + per-method output folders
# - LIGER gets BLAS/OpenMP safety override only for its subprocess
# ============================================================

# -----------------------
# PATHS (EDIT THESE)
# -----------------------
PROJECT_ROOT="/data1/esraa/Thesis-Project"
ENV_PY="/data1/esraa/miniconda3/envs/thesis-env/bin/python"
ATLAS_H5AD="${PROJECT_ROOT}/Data/Processed_data/post_HVG_intersection/concatenated_hvg.h5ad"
# ATLAS_H5AD="${PROJECT_ROOT}/Data/smoke_data/smoke_test.h5ad"

RUN_TAG="full_hvg_seed0"

OUTROOT="${PROJECT_ROOT}/Results/Integration/${RUN_TAG}"
RUNS_DIR="${OUTROOT}/runs"
LOGDIR="${OUTROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${RUNS_DIR}" "${LOGDIR}"

# -----------------------
# THREAD / GPU PLAN
# -----------------------
CPU_COMBAT=2
CPU_BBKN=2
CPU_HARMONY=2
CPU_SCANORAMA=2
CPU_LIGER=2
CPU_FASTMNN=2
CPU_SEURAT=2
CPU_MNN=2

GPU0="0"
GPU1="1"
CPU_SCVI=6
CPU_SCANVI=6
CPU_SCGEN=6

timestamp() { date "+%d %b, %Y %Z %H:%M:%S"; }

# -----------------------
# ENABLED METHODS
# -----------------------
BATCH1_METHODS=() # liger bbknn harmony)
BATCH2_METHODS=(scanorama combat bbknn) # fastmnn
BATCH3_METHODS=() #scvi scanvi scgen)

ALL_METHODS=("${BATCH1_METHODS[@]}" "${BATCH2_METHODS[@]}" "${BATCH3_METHODS[@]}")

# -----------------------
# BASIC CHECKS
# -----------------------
if [[ ! -f "${ATLAS_H5AD}" ]]; then
  echo "ERROR: ATLAS_H5AD not found: ${ATLAS_H5AD}"
  exit 1
fi

if [[ ! -x "${ENV_PY}" ]]; then
  echo "ERROR: Python not found or not executable: ${ENV_PY}"
  echo "Fix ENV_PY at the top of the script."
  exit 1
fi

# -----------------------
# UTILITIES
# -----------------------
wait_jobs () {
  local label="$1"; shift
  local -a pids=("$@")

  echo ""
  echo "---- Waiting for ${label} jobs ----"
  local fail=0
  for pid in "${pids[@]}"; do
    if wait "${pid}"; then
      :
    else
      fail=$((fail+1))
    fi
  done

  if [[ "${fail}" -gt 0 ]]; then
    echo "❌ ${label}: ${fail} job(s) FAILED."
    return 1
  else
    echo "✅ ${label}: all jobs finished OK."
    return 0
  fi
}

method_threads () {
  local method="$1"
  case "${method}" in
    combat)    echo "${CPU_COMBAT}" ;;
    bbknn)     echo "${CPU_BBKN}" ;;
    harmony)   echo "${CPU_HARMONY}" ;;
    scanorama) echo "${CPU_SCANORAMA}" ;;
    liger)     echo "${CPU_LIGER}" ;;
    fastmnn)   echo "${CPU_FASTMNN}" ;;
    seurat)    echo "${CPU_SEURAT}" ;;
    mnn)       echo "${CPU_MNN}" ;;
    scvi)      echo "${CPU_SCVI}" ;;
    scanvi)    echo "${CPU_SCANVI}" ;;
    scgen)     echo "${CPU_SCGEN}" ;;
    *)
      echo "ERROR: unknown method in method_threads(): ${method}" >&2
      exit 1
      ;;
  esac
}

method_gpu () {
  local method="$1"
  case "${method}" in
    scvi|scanvi) echo "${GPU0}" ;;
    scgen)       echo "${GPU1}" ;;
    *)           echo "" ;;
  esac
}

# -----------------------
# CORE PYTHON DISPATCH
# -----------------------
run_python_method () {
  local method="$1"
  local outdir="$2"
  local gpu="$3"
  local threads="$4"

  mkdir -p "${outdir}"

  export METHOD_THREADS="${threads}"

  if [[ "${method}" == "liger" ]]; then
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
  else
    export OMP_NUM_THREADS="${threads}"
    export OPENBLAS_NUM_THREADS="${threads}"
    export MKL_NUM_THREADS="${threads}"
    export NUMEXPR_NUM_THREADS="${threads}"
    export NUMBA_NUM_THREADS="${threads}"
  fi

  if [[ -n "${gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu}"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi

  "${ENV_PY}" - <<PY
import os
import scanpy as sc

method   = "${method}"
outdir   = "${outdir}"
run_tag  = "${RUN_TAG}"
atlas    = "${ATLAS_H5AD}"

print(f"=== RUNNING {method} ===")
print("  outdir:", outdir)
print("  run_tag:", run_tag)
print("  atlas:", atlas)
print("  CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("  METHOD_THREADS:", os.environ.get("METHOD_THREADS"))
print("  OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("  OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
print("  MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))

ad = sc.read_h5ad(atlas)

if method == "combat":
    from thesis_project.Integration.Integration_methods.combat import run, CombatConfig
    cfg = CombatConfig(run_tag=f"combat_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "bbknn":
    from thesis_project.Integration.Integration_methods.bbknn import run, BBKNNConfig
    cfg = BBKNNConfig(run_tag=f"bbknn_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "harmony":
    from thesis_project.Integration.Integration_methods.harmony import run, HarmonyConfig
    cfg = HarmonyConfig(run_tag=f"harmony_{run_tag}", seed=0, n_pcs=50)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "scanorama":
    from thesis_project.Integration.Integration_methods.scanorama import run, ScanoramaConfig
    cfg = ScanoramaConfig(run_tag=f"scanorama_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "liger":
    from thesis_project.Integration.Integration_methods.liger import run, LigerConfig
    cfg = LigerConfig(
        run_tag=f"liger_{run_tag}",
        seed=0,
        batch_key="dataset",
        label_key="major_celltype_l1",
        input_layer_raw="counts",
        require_hvg=True,
        hvg_key="highly_variable",
        max_hvgs=None,
        k_factors=50,
        lambda_reg=3.0,
        n_iters=50,
        n_cores=int(os.environ["METHOD_THREADS"]),
        align_method="centroidAlign",
        save_h5ad=True,
        save_rds=True,
    )
    run(ad, outdir=outdir, cfg=cfg)

elif method == "fastmnn":
    from thesis_project.Integration.Integration_methods.fastmnn import run_fastmnn_via_r, FastMNNConfig
    cfg = FastMNNConfig(run_tag=f"fastmnn_{run_tag}", seed=0)
    run_fastmnn_via_r(ad, outdir=outdir, cfg=cfg)

elif method == "seurat":
    from thesis_project.Integration.Integration_methods.seurat import run, SeuratConfig
    cfg = SeuratConfig(run_tag=f"seurat_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "mnn":
    from thesis_project.Integration.Integration_methods.mnn import run_mnncorrect_via_r, MNNConfig
    cfg = MNNConfig(run_tag=f"mnn_{run_tag}", seed=0)
    run_mnncorrect_via_r(ad, outdir=outdir, cfg=cfg)

elif method == "scvi":
    from thesis_project.Integration.Integration_methods.scvi import run, ScviConfig
    cfg = ScviConfig(run_tag=f"scvi_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "scanvi":
    from thesis_project.Integration.Integration_methods.scanvi import run, ScanviConfig
    cfg = ScanviConfig(run_tag=f"scanvi_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

elif method == "scgen":
    from thesis_project.Integration.Integration_methods.scgen import run, ScgenConfig
    cfg = ScgenConfig(run_tag=f"scgen_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

else:
    raise ValueError(f"Unknown method: {method}")

print(f"=== DONE {method} ===")
PY
}

run_one_method () {
  local method="$1"
  local gpu threads outdir log

  threads="$(method_threads "${method}")"
  gpu="$(method_gpu "${method}")"
  outdir="${RUNS_DIR}/${method}_${RUN_TAG}"
  log="${LOGDIR}/${method}.log"

  echo "==== ${method} START $(timestamp) ====" | tee -a "${log}"
  run_python_method "${method}" "${outdir}" "${gpu}" "${threads}" 2>&1 | tee -a "${log}"
}

run_batch_parallel () {
  local label="$1"; shift
  local methods=("$@")
  local pids=()
  local method

  echo ""
  echo "============================================================"
  echo "${label} START $(timestamp)"
  echo "Methods: ${methods[*]}"
  echo "============================================================"

  for method in "${methods[@]}"; do
    run_one_method "${method}" &
    pids+=($!)
  done

  wait_jobs "${label}" "${pids[@]}"
}

# -----------------------
# GPU BATCH: run in parallel with CPU batches
# Safe mode:
# - scvi then scanvi on GPU0
# - scgen on GPU1 in parallel
# -----------------------
run_batch3_gpu_safe () {
  echo ""
  echo "============================================================"
  echo "BATCH 3 (GPU, parallel with CPU batches) START $(timestamp)"
  echo "Methods: scvi scanvi scgen"
  echo "============================================================"

  run_one_method "scgen" &
  local pid_scgen=$!

  run_one_method "scvi"
  run_one_method "scanvi"

  if wait "${pid_scgen}"; then
    echo "✅ BATCH 3 GPU1 job finished OK."
  else
    echo "❌ BATCH 3 GPU1 job FAILED."
    return 1
  fi

  echo "✅ BATCH 3 (GPU) finished OK."
}

# ============================================================
# RUN PLAN
# ============================================================
echo "============================================================"
echo "3-BATCH INTEGRATION RUN"
echo "START: $(timestamp)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ATLAS_H5AD=${ATLAS_H5AD}"
echo "OUTROOT=${OUTROOT}"
echo ""
echo "Methods covered: ${#ALL_METHODS[@]}"
echo "Batch 1 (${#BATCH1_METHODS[@]}): ${BATCH1_METHODS[*]}"
echo "Batch 2 (${#BATCH2_METHODS[@]}): ${BATCH2_METHODS[*]}"
echo "Batch 3 (${#BATCH3_METHODS[@]}): ${BATCH3_METHODS[*]}"
echo ""
echo "Execution model:"
echo "  - Batch 2 runs on CPU only"
echo "  - Batch 1 is disabled"
echo "  - Batch 3 is disabled"
echo "============================================================"

CPU_STATUS=0

# Run Batch 2 only
if ! run_batch_parallel "BATCH 2 (CPU)" "${BATCH2_METHODS[@]}"; then
  CPU_STATUS=1
fi

echo "============================================================"
if [[ "${CPU_STATUS}" -eq 0 ]]; then
  echo "✅ ALL ENABLED METHODS FINISHED   $(timestamp)"
  echo "Results: ${RUNS_DIR}"
  echo "Logs   : ${LOGDIR}"
  echo "Covered methods: ${#ALL_METHODS[@]}"
  echo "============================================================"
else
  echo "❌ SOME METHODS FAILED   $(timestamp)"
  echo "Results: ${RUNS_DIR}"
  echo "Logs   : ${LOGDIR}"
  echo "Covered methods: ${#ALL_METHODS[@]}"
  echo "CPU_STATUS=${CPU_STATUS}"
  echo "============================================================"
  exit 1
fi