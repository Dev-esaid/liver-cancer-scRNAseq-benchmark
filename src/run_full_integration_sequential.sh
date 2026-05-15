#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FULL SEQUENTIAL LAUNCHER (11 integration methods)
# - Runs all methods one after another
# - No parallelization
# - Per-method logs + per-method output folders
# ============================================================

# -----------------------
# PATHS 
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
mkdir -p "${RUNS_DIR}" "${LOGDIR}"

# -----------------------
# THREAD / GPU PLAN
# -----------------------
# CPU threads per method
CPU_COMBAT=2
CPU_BBKN=2
CPU_HARMONY=2
CPU_SCANORAMA=3
CPU_FASTMNN=3
CPU_SEURAT=3
CPU_MNN=3
CPU_LIGER=3

# GPU methods
GPU0="0"
GPU1="1"
CPU_SCVI=8
CPU_SCANVI=8
CPU_SCGEN=8

timestamp() { date "+%d %b, %Y %Z %H:%M:%S"; }

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
# CORE PYTHON DISPATCH
# -----------------------
# method: combat / bbknn / harmony / scanorama / liger/  fastmnn / seurat / mnn / scvi / scanvi / scgen
# outdir: per-method results folder
# gpu   : "" for CPU-only, or "0"/"1" etc
# threads: OMP/MKL/BLAS threads
run_python_method () {
  local method="$1"
  local outdir="$2"
  local gpu="$3"
  local threads="$4"

  mkdir -p "${outdir}"

  export OMP_NUM_THREADS="${threads}"
  export MKL_NUM_THREADS="${threads}"
  export OPENBLAS_NUM_THREADS="${threads}"
  export NUMEXPR_NUM_THREADS="${threads}"
  export NUMBA_NUM_THREADS="${threads}"

  if [[ -n "${gpu}" ]]; then
    export CUDA_VISIBLE_DEVICES="${gpu}"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi

  "${ENV_PY}" - <<PY
import scanpy as sc

method = "${method}"
outdir = "${outdir}"
run_tag = "${RUN_TAG}"
atlas = "${ATLAS_H5AD}"

print(f"=== RUNNING {method} ===")
print("  outdir:", outdir)
print("  run_tag:", run_tag)
print("  atlas:", atlas)

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
elif method=="liger":
    from thesis_project.Integration.Integration_methods.liger import run, LigerConfig
    cfg = LigerConfig(run_tag=f"liger_{run_tag}", seed=0)
    run(ad, outdir=outdir, cfg=cfg)

else:
    raise ValueError(f"Unknown method: {method}")

print(f"=== DONE {method} ===")
PY
}

run_logged_method() {
  local method="$1"
  local outdir="$2"
  local gpu="$3"
  local threads="$4"
  local log="${LOGDIR}/${method}.log"

  echo "============================================================" | tee -a "${log}"
  echo "==== ${method} START $(timestamp) ====" | tee -a "${log}"
  echo "outdir=${outdir}" | tee -a "${log}"
  echo "gpu=${gpu:-CPU_ONLY}" | tee -a "${log}"
  echo "threads=${threads}" | tee -a "${log}"

  run_python_method "${method}" "${outdir}" "${gpu}" "${threads}" 2>&1 | tee -a "${log}"

  echo "==== ${method} DONE  $(timestamp) ====" | tee -a "${log}"
  echo "============================================================" | tee -a "${log}"
  echo ""
}

# ============================================================
# RUN PLAN
# ============================================================
echo "============================================================"
echo "FULL INTEGRATION SEQUENTIAL RUN"
echo "START: $(timestamp)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "ATLAS_H5AD=${ATLAS_H5AD}"
echo "OUTROOT=${OUTROOT}"
echo "============================================================"

# run_logged_method "combat"    "${RUNS_DIR}/combat_${RUN_TAG}"       ""       "${CPU_COMBAT}"
#run_logged_method "bbknn"     "${RUNS_DIR}/bbknn_${RUN_TAG}"        ""       "${CPU_BBKN}"
# run_logged_method "harmony"   "${RUNS_DIR}/harmony_${RUN_TAG}"      ""       "${CPU_HARMONY}"
#run_logged_method "scanorama" "${RUNS_DIR}/scanorama_${RUN_TAG}"    ""       "${CPU_SCANORAMA}"
#run_logged_method "fastmnn"   "${RUNS_DIR}/fastmnn_${RUN_TAG}"      ""       "${CPU_FASTMNN}"
run_logged_method "seurat"    "${RUNS_DIR}/seurat_${RUN_TAG}"       ""       "${CPU_SEURAT}"
# run_logged_method "mnn"       "${RUNS_DIR}/mnn_${RUN_TAG}"          ""       "${CPU_MNN}"
# run_logged_method "scvi"      "${RUNS_DIR}/scvi_${RUN_TAG}"         "${GPU0}" "${CPU_SCVI}"
# run_logged_method "scanvi"    "${RUNS_DIR}/scanvi_${RUN_TAG}"       "${GPU0}" "${CPU_SCANVI}"
# run_logged_method "scgen"     "${RUNS_DIR}/scgen_${RUN_TAG}"        "${GPU1}" "${CPU_SCGEN}"

echo "============================================================"
echo "ALL METHODS FINISHED   $(timestamp)"
echo "Results: ${RUNS_DIR}"
echo "Logs   : ${LOGDIR}"
echo "============================================================"