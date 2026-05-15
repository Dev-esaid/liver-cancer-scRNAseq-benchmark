"""Trajectory Inference Benchmarking: Method Runner (method_runner.py)

This runner executes one TI method on one (dataset, task) AnnData input, writes
canonical outputs, evaluates reference-free / weak-prior metrics, computes
bootstrap/subsample stability, and generates a publication-quality plot suite.

Plotting framework upgrade (2026-02-26, refined)
------------------------------------------------
Goal: automatically generate a full, consistent plot suite for EVERY task and method.

Plotting is fully isolated in: TI_benchmark/plotting.py
- method_runner.py always calls plotting.generate_plot_suite(...)
- Plot suite produces high-quality figures only (no legacy/duplicate plots).

Shared embedding per task/lineage
--------------------------------
Within a given run (dataset + task + lineage filtering), geometry is prepared
once and then reused for root selection, TI, metrics mirroring, stability,
and plotting.

Two geometry modes are supported transparently:
1) Standard single-dataset TI benchmarking (Chapter 4):
   - neighbors.prepare_geometry(...)
   - outputs: X_pca / connectivities / X_umap

2) Coupled integration × TI benchmarking (Chapter 6):
   - coupled_geometry.prepare_integrated_geometry(...)
   - outputs: X_integrated / X_pca (compat mirror) / connectivities / X_umap
   - routing is chosen automatically from the integrated input h5ad

No shared-geometry cache is written to disk; each run is self-contained.

Also fixes missing adata exports:
- adata/adata_post_preprocess.h5ad (always, after preprocessing)
- adata/adata_post_geometry.h5ad   (always, after geometry)
- adata/adata_with_ti_outputs.h5ad (on TI success; includes pseudotime/labels/probs)

Bootstrap correctness:
- Bootstraps start from already-preprocessed AnnData (fixed genes).
- We pass fixed_var_names into stability.run_stability to enforce identical genes.
- Canonical pseudotime stability definition handled in stability.py.

Bootstrap root correctness (Option A, 2026-02-28):
- Stability reselects root per replicate (when enabled) using root_selection.select_root(...)
  to avoid passing stale `uns["iroot"]` indices into smaller bootstraps.

Topology edge propagation (2026-03-06)
-------------------------------------
Some method wrappers write topology_edges.csv to disk but may forget to attach it
to TIOutput.edge_list. Since downstream stability and topology metrics rely on
TIOutput.edge_list, this runner includes a best-effort fallback:
  - If TIOutput.edge_list is None, try reading run_dir/tables/topology_edges.csv
    (or edges.csv) and validate it via io_schema.validate_edge_list.

Metrics gene panel policy (Option A, 2026-03-10)
-----------------------------------------------
TI methods / geometry typically run on HVGs for speed + robustness.
However, marker-based metrics (gene set scoring) should have access to a broader
(or full) gene panel to avoid biasing evaluation toward HVG selection.

This runner therefore:
  1) Keeps a full-gene AnnData after filtering (min_cells/min_counts, cell subset).
  2) Computes a log1p-normalized layer (adata.layers['log1p']) on that full gene set.
  3) Stores the full gene set into adata.raw with raw.X = log1p (for downstream access/export).
  4) Subsets the working AnnData to HVGs for geometry + TI (when requested).
  5) Runs metrics on the FULL gene panel AnnData (pre-HVG) using expression_layer='log1p'.
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("method_runner.py requires anndata to be installed.") from e

try:
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover
    sparse = None  # type: ignore

from .shared_types import TaskPriors, TIOutput
from . import io_schema
from . import metrics as metrics_mod
from . import neighbors as neighbors_mod
from . import coupled_geometry as coupled_geometry_mod
from . import preprocessing as preprocessing_mod
from . import priors as priors_mod
from . import root_selection as root_selection_mod
from . import stability as stability_mod
from . import plotting as plotting_mod
from .utils import (
    get_runtime_info,
    merge_json_shallow,
    set_global_seeds,
    validate_anndata_basic,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RunSpec
# =============================================================================


@dataclass
class RunSpec:
    method_name: str
    dataset_name: str
    task_name: str
    adata_path: str
    run_dir: str

    priors_path: Optional[str] = None
    priors_root: Optional[str] = None

    include_key: Optional[str] = None
    include_values: Optional[List[str]] = None
    exclude_key: Optional[str] = None
    exclude_values: Optional[List[str]] = None
    replace_labels_json: Optional[str] = None

    group_key: Optional[str] = None
    root_group: Optional[str] = None
    root_cell_id: Optional[str] = None

    expression_layer: Optional[str] = None
    batch_key: Optional[str] = None

    n_pcs: int = 30
    n_neighbors: int = 20

    n_bootstrap: int = 20
    bootstrap_frac: float = 0.8
    bootstrap_seed: int = 0
    bootstrap_stratify_by: Optional[str] = None
    bootstrap_min_per_group: int = 10
    skip_stability: bool = False

    random_state: int = 0

    # preprocessing
    min_cells: int = 3
    min_counts: int = 1
    n_top_genes: int = 3000
    hvg_flavor: str = "seurat"
    hvg_subset: bool = False
    target_sum: float = 1e4
    normalize: bool = True
    log1p: bool = True
    scale: bool = False

    color_keys: Optional[List[str]] = None
    export_obs_keys: Optional[List[str]] = None


# =============================================================================
# Small helpers
# =============================================================================


def load_adata(path: str) -> "ad.AnnData":
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AnnData file not found: {p}")
    logger.info(f"Loading adata from {p}")
    adata = ad.read_h5ad(str(p))
    logger.info(f"Loaded adata: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_priors(spec: RunSpec) -> TaskPriors:
    def _fill_group_key(p: TaskPriors) -> TaskPriors:
        if spec.group_key and getattr(p, "group_key", None) is None:
            return replace(p, group_key=spec.group_key)
        if spec.group_key and getattr(p, "group_key", None) and getattr(p, "group_key") != spec.group_key:
            logger.warning(
                f"group_key mismatch: priors.group_key='{getattr(p, 'group_key')}' vs "
                f"spec.group_key='{spec.group_key}'. Keeping priors.group_key."
            )
        return p

    def _minimal() -> TaskPriors:
        return priors_mod.build_minimal_priors(
            dataset=spec.dataset_name,
            task=spec.task_name,
            group_key=spec.group_key,
            root_group=spec.root_group,
            root_cell_id=spec.root_cell_id,
        )

    if spec.priors_path is not None:
        priors, res = priors_mod.load_task_priors(spec.priors_path, dataset=spec.dataset_name, task=spec.task_name)
        if res.status == "error":
            logger.warning(f"Failed to load priors from {spec.priors_path}: {res.reason}. Using minimal priors.")
            priors = _minimal()
        return _fill_group_key(priors)

    if spec.priors_root is not None:
        priors, res = priors_mod.load_task_priors_from_registry(spec.priors_root, spec.dataset_name, spec.task_name)
        if res.status == "error":
            logger.warning(f"Priors not found in registry: {res.reason}. Using minimal priors.")
            priors = _minimal()
        return _fill_group_key(priors)

    return _minimal()


def _safe_write_h5ad(adata: "ad.AnnData", path: Path) -> Tuple[str, Optional[str]]:
    """Best-effort .h5ad writer that never crashes the pipeline."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(str(path), compression="gzip")
        return "ok", None
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


def _preview_names(names: Any, *, n_head: int = 10, n_tail: int = 10) -> Dict[str, Any]:
    """Return a small head/tail preview of a name list/index for JSON logging."""
    try:
        lst = list(map(str, list(names)))
    except Exception:
        lst = []
    return {
        "n": int(len(lst)),
        "head": lst[:n_head],
        "tail": lst[-n_tail:] if len(lst) > n_tail else lst,
    }


def _var_overlap(current: pd.Index, fixed: pd.Index) -> Dict[str, int]:
    cur = pd.Index(current)
    fx = pd.Index(fixed)
    inter = cur.intersection(fx)
    return {
        "n_fixed": int(len(fx)),
        "n_current": int(len(cur)),
        "n_intersection": int(len(inter)),
        "n_fixed_only": int(len(fx.difference(cur))),
        "n_current_only": int(len(cur.difference(fx))),
    }


def _compute_log1p_from_counts(
    adata: "ad.AnnData",
    *,
    counts_layer: str = "counts",
    target_sum: float = 1e4,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Compute log1p(normalize_total(counts)) on full gene panel.

    Returns (X_log1p, report). Does NOT modify adata.

    Works for dense or sparse matrices.
    """
    report: Dict[str, Any] = {
        "status": "ok",
        "counts_source": None,
        "target_sum": float(target_sum),
        "used_sparse": False,
        "note": "X_log1p is log1p(normalize_total(counts)).",
    }

    X = None
    if counts_layer in adata.layers:
        X = adata.layers[counts_layer]
        report["counts_source"] = f"layers['{counts_layer}']"
    else:
        X = adata.X
        report["counts_source"] = "X"
        report["warning"] = f"counts layer '{counts_layer}' not found; using adata.X as counts."

    if X is None:
        return None, {"status": "error", "reason": "No expression matrix found (X is None)."}

    try:
        if sparse is not None and sparse.issparse(X):
            report["used_sparse"] = True
            lib = np.asarray(X.sum(axis=1)).ravel().astype(float)
            lib_safe = np.maximum(lib, 1e-12)
            scale = (float(target_sum) / lib_safe).astype(float)
            D = sparse.diags(scale)
            Xn = D @ X
            Xn = Xn.tocsr(copy=False)
            Xn.data = np.log1p(Xn.data)
            return Xn, report

        Xd = np.asarray(X, dtype=float)
        lib = Xd.sum(axis=1)
        lib_safe = np.maximum(lib, 1e-12)
        scale = (float(target_sum) / lib_safe)[:, None]
        Xn = Xd * scale
        Xn = np.log1p(Xn)
        return Xn, report

    except Exception as e:
        return None, {"status": "error", "reason": f"{type(e).__name__}: {e}"}


def _ensure_hvg_mask(
    adata: "ad.AnnData",
    *,
    n_top_genes: int,
    flavor: str,
    batch_key: Optional[str],
    run_config_path: Path,
) -> pd.Series:
    """Ensure adata.var['highly_variable'] exists; compute if missing."""
    if "highly_variable" in adata.var.columns:
        mask = adata.var["highly_variable"]
        if isinstance(mask, pd.Series) and mask.dtype == bool and int(mask.sum()) > 0:
            return mask

    try:
        import scanpy as sc  # type: ignore

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor=str(flavor),
            batch_key=batch_key,
            subset=False,
        )
        mask = adata.var.get("highly_variable")
        if mask is None:
            raise RuntimeError("scanpy did not set adata.var['highly_variable']")
        merge_json_shallow(
            run_config_path,
            {
                "preprocessing_post": {
                    "hvg_fallback": {
                        "status": "ok",
                        "method": "scanpy.pp.highly_variable_genes",
                        "n_top_genes": int(n_top_genes),
                        "flavor": str(flavor),
                        "batch_key": batch_key,
                    }
                }
            },
        )
        return pd.Series(mask.values.astype(bool), index=adata.var_names)

    except Exception as e:
        merge_json_shallow(
            run_config_path,
            {"preprocessing_post": {"hvg_fallback": {"status": "error", "error": f"{type(e).__name__}: {e}"}}},
        )
        return pd.Series(np.ones(adata.n_vars, dtype=bool), index=adata.var_names)


def _apply_scaling_if_requested(
    adata: "ad.AnnData",
    *,
    do_scale: bool,
    run_config_path: Path,
    max_value: float = 10.0,
) -> Dict[str, Any]:
    if not do_scale:
        return {"status": "skipped", "reason": "scale=False"}

    try:
        import scanpy as sc  # type: ignore

        sc.pp.scale(adata, zero_center=True, max_value=float(max_value))
        rep = {"status": "ok", "zero_center": True, "max_value": float(max_value), "tool": "scanpy.pp.scale"}
        merge_json_shallow(run_config_path, {"preprocessing_post": {"scale_hvg": rep}})
        return rep
    except Exception as e:
        rep = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        merge_json_shallow(run_config_path, {"preprocessing_post": {"scale_hvg": rep}})
        return rep


def _log_metrics_inputs(
    *,
    adata_metrics: "ad.AnnData",
    fixed_var_names: pd.Index,
    priors: TaskPriors,
    expression_layer_for_metrics: Optional[str],
    run_config_path: Path,
) -> None:
    """Log which gene panel and matrix will be used for metrics."""

    layer_present = bool(expression_layer_for_metrics and expression_layer_for_metrics in adata_metrics.layers)
    layer_shape = None
    if layer_present:
        try:
            layer_shape = list(map(int, adata_metrics.layers[expression_layer_for_metrics].shape))
        except Exception:
            layer_shape = None

    payload: Dict[str, Any] = {
        "n_cells": int(adata_metrics.n_obs),
        "n_genes_var_names": int(adata_metrics.n_vars),
        "var_names_preview": _preview_names(adata_metrics.var_names),
        "expression_layer": expression_layer_for_metrics,
        "layers_present": list(map(str, list(adata_metrics.layers.keys()))),
        "X_shape": list(map(int, adata_metrics.X.shape)) if getattr(adata_metrics, "X", None) is not None else None,
        "layer_shape": layer_shape,
        "layer_present": layer_present,
        "fixed_var_names_preview": _preview_names(fixed_var_names),
        "var_name_overlap_with_fixed": _var_overlap(adata_metrics.var_names, fixed_var_names),
    }

    coverage = None
    gene_sets = None
    for attr in ("marker_genes", "marker_program_genes", "marker_program_to_genes", "marker_gene_sets"):
        gene_sets = getattr(priors, attr, None)
        if isinstance(gene_sets, dict) and gene_sets:
            break
        gene_sets = None

    if gene_sets is not None:
        try:
            var_set = set(map(str, adata_metrics.var_names))
            per_prog = []
            total_req = 0
            total_present = 0
            for prog, genes in gene_sets.items():
                genes_list = list(map(str, list(genes)))
                total_req += len(genes_list)
                present = [g for g in genes_list if g in var_set]
                missing = [g for g in genes_list if g not in var_set]
                total_present += len(present)
                per_prog.append(
                    {
                        "program": str(prog),
                        "n_requested": int(len(genes_list)),
                        "n_present_in_var_names": int(len(present)),
                        "n_missing_in_var_names": int(len(missing)),
                        "missing_example": missing[:10],
                    }
                )
            coverage = {
                "n_programs_total": int(len(gene_sets)),
                "n_programs_logged": int(len(per_prog)),
                "total_genes_requested": int(total_req),
                "total_genes_present_in_var_names": int(total_present),
                "coverage_fraction": float(total_present / max(total_req, 1)),
                "per_program": per_prog,
                "note": "Coverage is computed against adata.var_names (this is what metrics.gene_set_score uses for indexing).",
            }
        except Exception as e:
            coverage = {"status": "error", "error": f"{type(e).__name__}: {e}"}

    if coverage is not None:
        payload["marker_program_coverage"] = coverage

    merge_json_shallow(run_config_path, {"metrics_inputs": payload})


def _write_exception_trace(
    *,
    paths: io_schema.RunPaths,
    stage: str,
    exc: BaseException,
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Write traceback to disk and return structured metadata for run_config logging."""
    tb = traceback.format_exc()
    safe_stage = str(stage).replace("/", "_").replace(" ", "_")
    filename = f"{safe_stage}_traceback.txt" if prefix is None else f"{prefix}_{safe_stage}_traceback.txt"
    tb_path = paths.logs_dir / filename
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tb_path, "w", encoding="utf-8") as fh:
        fh.write(tb)

    lines = tb.splitlines()
    return {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback_file": str(tb_path),
        "traceback_preview": lines[-20:],
    }


def _infer_integration_method_from_path(path: str) -> Optional[str]:
    """
    Infer the integration method name from the input h5ad path.
    Ordered to avoid substring collisions:
      scanvi before scvi, fastmnn before mnn.
    """
    s = str(path).lower()

    ordered = [
        "scanvi",
        "fastmnn",
        "scanorama",
        "harmony",
        "combat",
        "bbknn",
        "scgen",
        "scvi",
        "liger",
        "seurat",
        "mnn",
    ]

    for m in ordered:
        patterns = [
            f"/{m}_",
            f"/{m}/",
            f"_{m}_",
            f"adata_{m}",
            f"{m}.h5ad",
        ]
        if any(p in s for p in patterns):
            return m
    return None


def _detect_coupled_geometry_input(
    adata: "ad.AnnData",
    spec: RunSpec,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Decide whether this input h5ad is an integrated object for the coupled benchmark.

    Detection is intentionally conservative:
    - individual Chapter 4 inputs should continue down the standard geometry path
    - integrated Chapter 6 inputs should route through coupled_geometry

    Strong evidence of an integrated object:
    - obsm['X_emb'] exists
    - namespaced neighbors keys exist in uns/obsp
    - input path lives under an Integration results tree
    - integration method can be inferred from the filename/path
    """
    path_s = str(spec.adata_path).lower()

    has_x_emb = "X_emb" in adata.obsm
    has_namespaced_neighbors_uns = any(str(k).lower().startswith("neighbors_") for k in adata.uns.keys())
    has_namespaced_graph = any(
        ("connectivities" in str(k).lower() or "distances" in str(k).lower()) and "_" in str(k)
        for k in adata.obsp.keys()
    )

    in_integration_tree = "/results/integration/" in path_s or "/integration/" in path_s
    in_coupled_tree = "/results/coupled_benchmark/" in path_s or "/coupled_benchmark/" in path_s

    inferred_method = _infer_integration_method_from_path(path_s)

    strong_evidence = bool(
        has_x_emb or
        has_namespaced_neighbors_uns or
        has_namespaced_graph or
        in_integration_tree or
        in_coupled_tree
    )

    detection_report = {
        "path": str(spec.adata_path),
        "has_x_emb": bool(has_x_emb),
        "has_namespaced_neighbors_uns": bool(has_namespaced_neighbors_uns),
        "has_namespaced_graph": bool(has_namespaced_graph),
        "in_integration_tree": bool(in_integration_tree),
        "in_coupled_tree": bool(in_coupled_tree),
        "inferred_integration_method": inferred_method,
        "strong_evidence": bool(strong_evidence),
    }

    if strong_evidence and inferred_method is None:
        raise RuntimeError(
            "Input h5ad looks like an integrated object for the coupled benchmark "
            "but the integration method could not be inferred from the path. "
            f"Path: {spec.adata_path}"
        )

    is_coupled = bool(strong_evidence and inferred_method is not None)
    return is_coupled, inferred_method, detection_report


def _geometry_error_summary(
    *,
    rep_key: str = "X_pca",
    neighbors_key: str = "neighbors",
    connectivities_key: str = "connectivities",
    distances_key: str = "distances",
    umap_key: str = "X_umap",
) -> Dict[str, Any]:
    return {
        "pca": {"status": "error"},
        "neighbors": {"status": "error"},
        "umap": {"status": "error"},
        "geometry_keys": {
            "pca_key": rep_key,
            "neighbors_key": neighbors_key,
            "connectivities_key": connectivities_key,
            "distances_key": distances_key,
            "umap_key": umap_key,
        },
    }


# =============================================================================
# TIOutput standardization + fallbacks
# =============================================================================


def _standardize_ti_output(adata: "ad.AnnData", ti: TIOutput) -> Tuple[Optional[TIOutput], Dict[str, Any]]:
    """Validate + align TIOutput fields to the current adata."""
    report: Dict[str, Any] = {"status": "ok", "issues": []}

    if not isinstance(getattr(ti, "pseudotime", None), pd.Series):
        report["status"] = "error"
        report["issues"].append("pseudotime is not a pandas Series")
        return None, report

    if ti.pseudotime.index.has_duplicates:
        report["status"] = "error"
        report["issues"].append("pseudotime index has duplicates")
        return None, report

    try:
        pt = io_schema.align_series_to_obs(ti.pseudotime, adata.obs_names, name="pseudotime", allow_missing=False)
        pt = pd.to_numeric(pt, errors="coerce").astype(float)
    except Exception as e:
        report["status"] = "error"
        report["issues"].append(f"pseudotime alignment failed: {type(e).__name__}: {e}")
        return None, report

    if not np.isfinite(pt.values).all():
        report["status"] = "error"
        report["issues"].append("pseudotime contains non-finite values")
        return None, report

    if int(pd.Series(pt.values).nunique()) < 2:
        report["status"] = "error"
        report["issues"].append("pseudotime is constant (n_unique < 2)")
        return None, report

    ti.pseudotime = pd.Series(pt.values, index=adata.obs_names, name=ti.pseudotime.name or "pseudotime")

    if getattr(ti, "edge_list", None) is not None:
        try:
            df, rep = io_schema.validate_edge_list(ti.edge_list, allow_self_loops=False)
            ti.edge_list = df
            report["edge_list_validation"] = rep
        except Exception as e:
            report["issues"].append(f"edge_list invalid; dropped: {type(e).__name__}: {e}")
            ti.edge_list = None

    if getattr(ti, "branch_labels", None) is not None:
        try:
            bl = io_schema.align_series_to_obs(ti.branch_labels, adata.obs_names, name="branch_labels", allow_missing=True)
            ti.branch_labels = pd.Series(
                bl.astype("string").values,
                index=adata.obs_names,
                name=ti.branch_labels.name or "branch_labels",
            )
        except Exception as e:
            report["issues"].append(f"branch_labels invalid; dropped: {type(e).__name__}: {e}")
            ti.branch_labels = None

    if getattr(ti, "terminal_probabilities", None) is not None:
        try:
            tp = ti.terminal_probabilities
            if not isinstance(tp, pd.DataFrame):
                raise TypeError("terminal_probabilities not a DataFrame")
            if tp.index.has_duplicates:
                raise ValueError("terminal_probabilities index has duplicates")
            tp2 = tp.loc[tp.index.intersection(adata.obs_names)].copy().reindex(adata.obs_names)
            for c in tp2.columns:
                tp2[c] = pd.to_numeric(tp2[c], errors="coerce").astype(float)
            ti.terminal_probabilities = tp2
        except Exception as e:
            report["issues"].append(f"terminal_probabilities invalid; dropped: {type(e).__name__}: {e}")
            ti.terminal_probabilities = None

    if getattr(ti, "extras", None) is not None and not isinstance(ti.extras, dict):
        report["issues"].append("extras is not a dict; dropped")
        ti.extras = None

    return ti, report


def _maybe_load_topology_edges_from_disk(paths: Any) -> Optional[pd.DataFrame]:
    """Best-effort fallback to load topology edges written by a method wrapper."""
    candidates = [
        Path(paths.tables_dir) / "topology_edges.csv",
        Path(paths.tables_dir) / "edges.csv",
    ]
    for fp in candidates:
        try:
            if fp.exists() and fp.stat().st_size > 0:
                df = pd.read_csv(fp)
                if {"source", "target"}.issubset(df.columns):
                    return df
        except Exception:
            continue
    return None


def _write_placeholder_and_metrics(
    *,
    adata_for_outputs: "ad.AnnData",
    adata_for_metrics: "ad.AnnData",
    metrics_expression_layer: Optional[str],
    spec: RunSpec,
    run_dir: Path,
    priors: TaskPriors,
    connectivities: Any,
    run_config_path: Path,
    reason: str,
) -> Dict[str, Any]:
    """Write placeholder outputs and run metrics in a safe 'failed run' mode."""
    placeholder_pt = pd.Series(np.nan, index=adata_for_outputs.obs_names, name="pseudotime")

    io_schema.write_all_canonical_outputs(
        adata_for_outputs,
        pseudotime=placeholder_pt,
        run_dir=run_dir,
        group_key=getattr(priors, "group_key", None),
        edge_list=None,
        branch_labels=None,
        terminal_probabilities=None,
        allow_missing_pseudotime=True,
        require_any_finite_pseudotime=False,
        require_all_finite_pseudotime=False,
        allow_self_loops=False,
    )

    merge_json_shallow(
        run_config_path,
        {
            "ti_method": {
                "status": "error",
                "error_msg": str(reason),
            }
        },
    )

    return metrics_mod.evaluate_all_metrics(
        adata_for_metrics,
        TIOutput(
            pseudotime=placeholder_pt,
            method_name=spec.method_name,
            dataset_name=spec.dataset_name,
            task_name=spec.task_name,
        ),
        priors,
        expression_layer=metrics_expression_layer,
        connectivities=connectivities,
        out_dir=str(run_dir),
        run_config_path=str(run_config_path),
    )


# =============================================================================
# Main benchmark runner
# =============================================================================


def run_benchmark(
    spec: RunSpec,
    ti_runner: Callable[..., TIOutput],
    *,
    preprocessing_extra_kwargs: Optional[Dict[str, Any]] = None,
    runner_extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    run_dir = Path(spec.run_dir)
    paths = io_schema.RunPaths.from_run_dir(run_dir).mkdirs()
    run_config_path = paths.logs_dir / "run_config.json"

    seed_res = set_global_seeds(spec.random_state)

    merge_json_shallow(
        run_config_path,
        {
            "run_meta": {
                "method": spec.method_name,
                "dataset": spec.dataset_name,
                "task": spec.task_name,
                "run_dir": str(run_dir),
                "random_state": int(spec.random_state),
            },
            "runtime": get_runtime_info(),
            "seeding": seed_res.to_dict(),
            "run_spec": {
                "group_key": spec.group_key,
                "include_key": spec.include_key,
                "include_values": spec.include_values,
                "exclude_key": spec.exclude_key,
                "exclude_values": spec.exclude_values,
                "replace_labels_json": spec.replace_labels_json,
                "batch_key": spec.batch_key,
                "expression_layer": spec.expression_layer,
                "n_pcs": spec.n_pcs,
                "n_neighbors": spec.n_neighbors,
                "n_bootstrap": spec.n_bootstrap,
                "bootstrap_frac": spec.bootstrap_frac,
                "bootstrap_seed": spec.bootstrap_seed,
                "bootstrap_stratify_by": spec.bootstrap_stratify_by,
                "bootstrap_min_per_group": spec.bootstrap_min_per_group,
                "skip_stability": spec.skip_stability,
                "hvg_subset": spec.hvg_subset,
                "n_top_genes": spec.n_top_genes,
                "hvg_flavor": spec.hvg_flavor,
                "metrics_gene_panel": "full_genes_via_raw_log1p",
            },
        },
    )

    merge_json_shallow(
        run_config_path,
        {
            "ti_method": {
                "status": "not_started",
                "method_name": spec.method_name,
                "logs_dir": str(paths.logs_dir),
            }
        },
    )

    # 1) Load
    adata = load_adata(spec.adata_path)
    merge_json_shallow(run_config_path, {"data": {"n_cells_raw": int(adata.n_obs), "n_genes_raw": int(adata.n_vars)}})
    v = validate_anndata_basic(adata)
    merge_json_shallow(run_config_path, {"data_validation": v.to_dict()})

    # 2) Priors
    priors = load_priors(spec)
    priors_mod.log_priors_to_run_config(priors, run_config_path)

    # 3) Preprocess
    preprocess_kwargs: Dict[str, Any] = dict(
        min_cells=spec.min_cells,
        min_counts=spec.min_counts,
        n_top_genes=spec.n_top_genes,
        hvg_flavor=spec.hvg_flavor,
        hvg_subset=False,
        normalize=spec.normalize,
        log1p=spec.log1p,
        scale=False,
        target_sum=spec.target_sum,
        batch_key=spec.batch_key,
        run_config_path=run_config_path,
        tables_dir=str(paths.tables_dir),
        return_report=True,
        include_key=spec.include_key,
        include_values=spec.include_values,
        exclude_key=spec.exclude_key,
        exclude_values=spec.exclude_values,
        replace_labels_json=spec.replace_labels_json,
    )
    if preprocessing_extra_kwargs:
        preprocess_kwargs.update(preprocessing_extra_kwargs)

    adata_full, preprocess_report = preprocessing_mod.preprocess_adata(adata, **preprocess_kwargs)

    merge_json_shallow(
        run_config_path,
        {
            "data": {
                "n_cells_post_filtering": int(adata_full.n_obs),
                "n_genes_post_filtering": int(adata_full.n_vars),
            }
        },
    )

    # 3b) Build full-gene log1p layer and store in .raw for downstream access/export.
    X_log1p, logrep = _compute_log1p_from_counts(adata_full, counts_layer="counts", target_sum=spec.target_sum)
    if X_log1p is not None:
        adata_full.layers["log1p"] = X_log1p
        try:
            X_prev = adata_full.X
            adata_full.X = adata_full.layers["log1p"]
            adata_full.raw = adata_full
            adata_full.X = X_prev
            logrep["raw_set"] = True
        except Exception as e:
            logrep["raw_set"] = False
            logrep["raw_set_error"] = f"{type(e).__name__}: {e}"
    else:
        logrep["raw_set"] = False

    merge_json_shallow(run_config_path, {"metrics_gene_panel": {"raw_setup": logrep}})

    # 3c) Subset to HVGs for TI/geometry if requested.
    hvg_subset_applied = False
    n_hvg = None
    if spec.hvg_subset:
        hvg_mask = _ensure_hvg_mask(
            adata_full,
            n_top_genes=int(spec.n_top_genes),
            flavor=str(spec.hvg_flavor),
            batch_key=spec.batch_key,
            run_config_path=run_config_path,
        )
        n_hvg = int(hvg_mask.sum())
        adata = adata_full[:, hvg_mask.values].copy()
        hvg_subset_applied = True
    else:
        adata = adata_full
        n_hvg = int(getattr(adata_full.var.get("highly_variable"), "sum", lambda: 0)()) if "highly_variable" in adata_full.var.columns else 0

    merge_json_shallow(
        run_config_path,
        {
            "preprocessing_post": {
                "hvg_subset_runner": {
                    "requested": bool(spec.hvg_subset),
                    "applied": bool(hvg_subset_applied),
                    "n_genes_full": int(adata_full.n_vars),
                    "n_hvg": int(n_hvg) if n_hvg is not None else None,
                    "n_genes_working": int(adata.n_vars),
                }
            }
        },
    )

    _apply_scaling_if_requested(adata, do_scale=spec.scale, run_config_path=run_config_path, max_value=10.0)

    merge_json_shallow(
        run_config_path,
        {"data": {"n_cells_post_preprocess": int(adata.n_obs), "n_genes_post_preprocess": int(adata.n_vars)}},
    )

    st, msg = _safe_write_h5ad(adata, paths.adata_dir / "adata_post_preprocess.h5ad")
    merge_json_shallow(
        run_config_path,
        {
            "adata_exports": {
                "post_preprocess": {
                    "status": st,
                    "path": str(paths.adata_dir / "adata_post_preprocess.h5ad"),
                    "error": msg,
                }
            }
        },
    )

    # 4) Geometry routing: standard single-dataset OR coupled integration-aware
    color_keys = list(spec.color_keys or [])
    if getattr(priors, "group_key", None) and priors.group_key not in color_keys:
        color_keys.insert(0, priors.group_key)

    geometry_exception_meta: Dict[str, Any] = {}
    try:
        is_coupled_input, integration_method, detection_report = _detect_coupled_geometry_input(adata, spec)
        merge_json_shallow(
            run_config_path,
            {
                "geometry": {
                    "routing_detection": {
                        "mode": "coupled" if is_coupled_input else "standard",
                        **detection_report,
                    }
                }
            },
        )

        if is_coupled_input:
            logger.info(
                "Geometry routing: detected integrated coupled-benchmark input "
                "(integration_method=%s). Using coupled_geometry.prepare_integrated_geometry().",
                integration_method,
            )
            adata = coupled_geometry_mod.prepare_integrated_geometry(
                adata,
                integration_method=str(integration_method),
                n_pcs=int(spec.n_pcs),
                n_neighbors=int(spec.n_neighbors),
                n_umap_components=2,
                umap_min_dist=float(getattr(neighbors_mod, "DEFAULT_UMAP_MIN_DIST", 0.3)),
                umap_spread=float(getattr(neighbors_mod, "DEFAULT_UMAP_SPREAD", 1.0)),
                random_state=int(spec.random_state),
                run_config_path=run_config_path,
            )
            geometry_summary = coupled_geometry_mod.geometry_summary_from_adata(adata)
        else:
            logger.info(
                "Geometry routing: standard single-dataset input. "
                "Using neighbors.prepare_geometry()."
            )
            adata, geometry_summary = neighbors_mod.prepare_geometry(
                adata,
                n_pcs=spec.n_pcs,
                n_neighbors=spec.n_neighbors,
                rep_key="X_pca",
                neighbors_key_added=None,
                umap_key="X_umap",
                color_keys=None,
                figures_dir=None,
                random_state=spec.random_state,
                run_config_path=run_config_path,
            )

    except Exception as e:
        geometry_exception_meta = _write_exception_trace(paths=paths, stage="geometry", exc=e)
        merge_json_shallow(
            run_config_path,
            {
                "geometry": {
                    "status": "error",
                    "error_msg": f"{type(e).__name__}: {e}",
                    **geometry_exception_meta,
                }
            },
        )
        geometry_summary = _geometry_error_summary()

    st, msg = _safe_write_h5ad(adata, paths.adata_dir / "adata_post_geometry.h5ad")
    merge_json_shallow(
        run_config_path,
        {
            "adata_exports": {
                "post_geometry": {
                    "status": st,
                    "path": str(paths.adata_dir / "adata_post_geometry.h5ad"),
                    "error": msg,
                }
            }
        },
    )

    pca_status = geometry_summary.get("pca", {}).get("status") if isinstance(geometry_summary, dict) else None
    nn_status = geometry_summary.get("neighbors", {}).get("status") if isinstance(geometry_summary, dict) else None
    geometry_failed = pca_status == "error" or nn_status == "error"
    geometry_keys = geometry_summary.get("geometry_keys", {}) if isinstance(geometry_summary, dict) else {}

    rep_key = str(geometry_keys.get("pca_key", "X_pca"))
    neighbors_key = str(geometry_keys.get("neighbors_key", "neighbors"))
    connectivities_key = str(geometry_keys.get("connectivities_key", "connectivities"))
    umap_key = str(geometry_keys.get("umap_key", "X_umap"))

    # Build the AnnData used for metrics: FULL gene panel with log1p layer.
    adata_metrics = adata_full
    metrics_expression_layer = "log1p" if "log1p" in adata_metrics.layers else spec.expression_layer

    fixed_var_names = (
        adata_metrics.var_names.copy()
        if spec.method_name == "cellrank"
        else adata.var_names.copy()
    )

    try:
        if rep_key in adata.obsm:
            adata_metrics.obsm[rep_key] = adata.obsm[rep_key]
        if umap_key in adata.obsm:
            adata_metrics.obsm[umap_key] = adata.obsm[umap_key]
        if connectivities_key in adata.obsp:
            adata_metrics.obsp[connectivities_key] = adata.obsp[connectivities_key]
        if "distances" in adata.obsp and "distances" not in adata_metrics.obsp:
            adata_metrics.obsp["distances"] = adata.obsp["distances"]
        if neighbors_key in adata.uns:
            adata_metrics.uns[neighbors_key] = adata.uns[neighbors_key]
        if "neighbors" in adata.uns and "neighbors" not in adata_metrics.uns:
            adata_metrics.uns["neighbors"] = adata.uns["neighbors"]
        merge_json_shallow(
            run_config_path,
            {"metrics_gene_panel": {"geometry_mirror": {"status": "ok", "copied": ["obsm", "obsp", "uns"]}}},
        )
    except Exception as e:
        merge_json_shallow(
            run_config_path,
            {"metrics_gene_panel": {"geometry_mirror": {"status": "error", "error": f"{type(e).__name__}: {e}"}}},
        )

    _log_metrics_inputs(
        adata_metrics=adata_metrics,
        fixed_var_names=fixed_var_names,
        priors=priors,
        expression_layer_for_metrics=metrics_expression_layer,
        run_config_path=run_config_path,
    )

    if geometry_failed:
        root_cell_id = None
        root_summary = {"status": "skipped", "reason": "geometry_failed"}

        metric_results = _write_placeholder_and_metrics(
            adata_for_outputs=adata,
            adata_for_metrics=adata_metrics,
            metrics_expression_layer=metrics_expression_layer,
            spec=spec,
            run_dir=run_dir,
            priors=priors,
            connectivities=adata.obsp.get(connectivities_key),
            run_config_path=run_config_path,
            reason="geometry_failed",
        )
        plotting_mod.generate_plot_suite(
            adata,
            run_dir=run_dir,
            figures_dir=paths.figures_dir,
            umap_key=umap_key,
            group_key=getattr(priors, "group_key", None),
            color_keys=color_keys,
            ti=None,
            stability_manifest=None,
            run_config_path=run_config_path,
            canonical_group_order=list(spec.include_values or []),
        )
        elapsed = round(time.time() - t0, 3)
        merge_json_shallow(
            run_config_path,
            {"final": {"run_completed": True, "success": False, "reason": "geometry_failed", "total_elapsed_seconds": elapsed}},
        )
        return {
            "preprocess_report": preprocess_report,
            "geometry_summary": geometry_summary,
            "root_cell_id": root_cell_id,
            "root_summary": root_summary,
            "metric_results": metric_results,
            "stability_summary": None,
            "elapsed_seconds": elapsed,
        }

    # 5) Root selection
    root_cell_id, root_summary = root_selection_mod.select_root(
        adata,
        priors,
        cli_group_key=spec.group_key,
        cli_root_group=spec.root_group,
        cli_root_cell_id=spec.root_cell_id,
        rep_key=rep_key,
        layer=spec.expression_layer,
        random_state=spec.random_state,
        run_config_path=run_config_path,
    )

    # 6) TI method
    geometry_runner_kwargs: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "rep_key": rep_key,
        "neighbors_key": neighbors_key,
        "connectivities_key": connectivities_key,
        "umap_key": umap_key,
        "group_key": getattr(priors, "group_key", None),
        "cluster_key": getattr(priors, "group_key", None),
    }
    if runner_extra_kwargs:
        geometry_runner_kwargs.update(runner_extra_kwargs)

    ti_status = "ok"
    ti_error: Optional[str] = None
    ti_output_validated: Optional[TIOutput] = None
    validation_report: Dict[str, Any] = {}
    ti_extras: Optional[Dict[str, Any]] = None
    ti_exception_meta: Dict[str, Any] = {}

    t_ti = time.time()
    try:
        set_global_seeds(spec.random_state)
        adata_for_ti = adata_metrics if spec.method_name == "cellrank" else adata
        ti_output = ti_runner(
            adata_for_ti,
            root_cell_id,
            spec.random_state,
            bootstrap_index=None,
            **geometry_runner_kwargs,
        )
        ti_output_validated, validation_report = _standardize_ti_output(adata, ti_output)
        if ti_output_validated is None:
            ti_status = "error"
            ti_error = "; ".join(validation_report.get("issues", [])) or "invalid TIOutput"
        else:
            ti_extras = ti_output_validated.extras
    except Exception as e:
        ti_status = "error"
        ti_error = f"{type(e).__name__}: {e}"
        ti_exception_meta = _write_exception_trace(paths=paths, stage="ti_method", exc=e)

    t_ti_elapsed = time.time() - t_ti
    merge_json_shallow(
        run_config_path,
        {
            "ti_method": {
                "method_name": spec.method_name,
                "status": ti_status,
                "elapsed_seconds": round(t_ti_elapsed, 3),
                "error_msg": ti_error,
                "output_validation": validation_report,
                "extras": ti_extras,
                **ti_exception_meta,
            }
        },
    )

    if ti_output_validated is None:
        metric_results = _write_placeholder_and_metrics(
            adata_for_outputs=adata,
            adata_for_metrics=adata_metrics,
            metrics_expression_layer=metrics_expression_layer,
            spec=spec,
            run_dir=run_dir,
            priors=priors,
            connectivities=adata.obsp.get(connectivities_key),
            run_config_path=run_config_path,
            reason="ti_failed",
        )
        plotting_mod.generate_plot_suite(
            adata,
            run_dir=run_dir,
            figures_dir=paths.figures_dir,
            umap_key=umap_key,
            group_key=getattr(priors, "group_key", None),
            color_keys=color_keys,
            ti=None,
            stability_manifest=None,
            run_config_path=run_config_path,
            canonical_group_order=list(spec.include_values or []),
        )
        elapsed = round(time.time() - t0, 3)
        merge_json_shallow(
            run_config_path,
            {"final": {"run_completed": True, "success": False, "reason": "ti_failed", "total_elapsed_seconds": elapsed}},
        )
        return {
            "preprocess_report": preprocess_report,
            "geometry_summary": geometry_summary,
            "root_cell_id": root_cell_id,
            "root_summary": root_summary,
            "metric_results": metric_results,
            "stability_summary": None,
            "elapsed_seconds": elapsed,
        }

    if getattr(ti_output_validated, "method_name", None) is None:
        ti_output_validated.method_name = spec.method_name
    if getattr(ti_output_validated, "dataset_name", None) is None:
        ti_output_validated.dataset_name = spec.dataset_name
    if getattr(ti_output_validated, "task_name", None) is None:
        ti_output_validated.task_name = spec.task_name

    if getattr(ti_output_validated, "edge_list", None) is None:
        df = _maybe_load_topology_edges_from_disk(paths)
        if df is not None:
            try:
                df2, rep = io_schema.validate_edge_list(df, allow_self_loops=False)
                ti_output_validated.edge_list = df2
                merge_json_shallow(
                    run_config_path,
                    {"ti_method": {"edge_list_fallback": {"status": "ok", "source": "disk", "validation": rep}}},
                )
            except Exception as e:
                merge_json_shallow(
                    run_config_path,
                    {"ti_method": {"edge_list_fallback": {"status": "error", "error": f"{type(e).__name__}: {e}"}}},
                )

    # 7) Write canonical outputs
    schema_manifest = io_schema.write_all_canonical_outputs(
        adata,
        pseudotime=ti_output_validated.pseudotime,
        run_dir=run_dir,
        group_key=getattr(priors, "group_key", None),
        edge_list=getattr(ti_output_validated, "edge_list", None),
        branch_labels=getattr(ti_output_validated, "branch_labels", None),
        terminal_probabilities=getattr(ti_output_validated, "terminal_probabilities", None),
        allow_missing_pseudotime=False,
        require_any_finite_pseudotime=True,
        require_all_finite_pseudotime=True,
        allow_self_loops=False,
    )
    merge_json_shallow(
        run_config_path,
        {
            "io_schema": {
                "schema_version": io_schema.IO_SCHEMA_VERSION,
                "manifest_path": str(run_dir / io_schema.LOG_SCHEMA_MANIFEST),
                "n_artifacts": len(schema_manifest.artifacts),
                "artifact_keys": sorted(list(schema_manifest.artifacts.keys())),
            }
        },
    )

    # Save adata with TI outputs
    try:
        adata_ti = adata.copy()
        adata_ti.obs["ti_pseudotime"] = pd.to_numeric(
            ti_output_validated.pseudotime.reindex(adata_ti.obs_names),
            errors="coerce",
        ).astype(float)
        if getattr(ti_output_validated, "branch_labels", None) is not None:
            adata_ti.obs["ti_branch_labels"] = ti_output_validated.branch_labels.reindex(adata_ti.obs_names).astype("string")
        if getattr(ti_output_validated, "terminal_probabilities", None) is not None:
            tp = ti_output_validated.terminal_probabilities.reindex(adata_ti.obs_names)
            adata_ti.obsm["ti_terminal_probabilities"] = tp.to_numpy(dtype=float, copy=False)
            adata_ti.uns["ti_terminal_probability_columns"] = list(map(str, tp.columns.tolist()))
        st, msg = _safe_write_h5ad(adata_ti, paths.adata_dir / "adata_with_ti_outputs.h5ad")
        merge_json_shallow(
            run_config_path,
            {
                "adata_exports": {
                    "with_ti_outputs": {
                        "status": st,
                        "path": str(paths.adata_dir / "adata_with_ti_outputs.h5ad"),
                        "error": msg,
                    }
                }
            },
        )
    except Exception as e:
        save_meta = _write_exception_trace(paths=paths, stage="adata_with_ti_outputs_export", exc=e)
        merge_json_shallow(
            run_config_path,
            {
                "adata_exports": {
                    "with_ti_outputs": {
                        "status": "error",
                        "path": str(paths.adata_dir / "adata_with_ti_outputs.h5ad"),
                        "error": f"{type(e).__name__}: {e}",
                        **save_meta,
                    }
                }
            },
        )

    # 8) Metrics (FULL gene panel)
    connectivities = adata.obsp.get(connectivities_key)
    metric_results = metrics_mod.evaluate_all_metrics(
        adata_metrics,
        ti_output_validated,
        priors,
        expression_layer=metrics_expression_layer,
        connectivities=connectivities,
        out_dir=str(run_dir),
        run_config_path=str(run_config_path),
    )

    # 9) Stability
    stability_summary: Optional[Dict[str, Any]] = None
    if not spec.skip_stability:
        bootstrap_config = stability_mod.BootstrapConfig(
            n_replicates=spec.n_bootstrap,
            frac_cells=spec.bootstrap_frac,
            base_seed=spec.bootstrap_seed,
            stratify_by=spec.bootstrap_stratify_by or getattr(priors, "group_key", None),
            min_per_group=spec.bootstrap_min_per_group,
        )

        geom_cfg = stability_mod.GeometryConfig(
            n_pcs=int(spec.n_pcs),
            n_neighbors=int(spec.n_neighbors),
            rep_key=str(rep_key),
            neighbors_key_added=None,
            compute_umap=True,
            umap_key=str(umap_key),
            n_umap_components=2,
            umap_min_dist=float(getattr(neighbors_mod, "DEFAULT_UMAP_MIN_DIST", 0.3)),
            random_state=int(spec.random_state),
        )

        adata_for_stability = adata_metrics if spec.method_name == "cellrank" else adata

        try:
            _, stability_manifest = stability_mod.run_stability(
                adata_for_stability,
                ti_runner,
                root_cell_id,
                priors=priors,
                reselect_root_each_replicate=True,
                cli_group_key=spec.group_key,
                cli_root_group=spec.root_group,
                cli_root_cell_id=spec.root_cell_id,
                root_layer=spec.expression_layer,
                k_nearest_centroid=1,
                config=bootstrap_config,
                fixed_var_names=fixed_var_names,
                preprocessing_kwargs=None,
                geometry_config=geom_cfg,
                runner_kwargs=geometry_runner_kwargs,
                run_config_path=run_config_path,
                out_dir=str(run_dir),
            )
            stability_summary = stability_manifest.to_dict()
        except Exception as e:
            stab_meta = _write_exception_trace(paths=paths, stage="stability", exc=e)
            merge_json_shallow(
                run_config_path,
                {
                    "stability": {
                        "status": "error",
                        "error_msg": f"{type(e).__name__}: {e}",
                        **stab_meta,
                    }
                },
            )
            stability_summary = {
                "status": "error",
                "error_msg": f"{type(e).__name__}: {e}",
                **stab_meta,
            }

    # 10) Plot suite
    plotting_mod.generate_plot_suite(
        adata,
        run_dir=run_dir,
        figures_dir=paths.figures_dir,
        umap_key=umap_key,
        group_key=getattr(priors, "group_key", None),
        color_keys=color_keys,
        ti=ti_output_validated,
        stability_manifest=stability_summary,
        run_config_path=run_config_path,
        canonical_group_order=list(spec.include_values or []),
    )

    elapsed = round(time.time() - t0, 3)
    merge_json_shallow(run_config_path, {"final": {"run_completed": True, "success": True, "total_elapsed_seconds": elapsed}})

    return {
        "preprocess_report": preprocess_report,
        "geometry_summary": geometry_summary,
        "root_cell_id": root_cell_id,
        "root_summary": root_summary,
        "metric_results": metric_results,
        "stability_summary": stability_summary,
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# CLI helpers
# =============================================================================


def build_arg_parser(*, require_method: bool = True, default_method: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TI Benchmarking: run one (method, dataset, task).")

    if require_method:
        p.add_argument("--method", required=True, help="TI method name (e.g. paga, dpt, slingshot).")
    else:
        p.add_argument(
            "--method",
            required=False,
            default=default_method,
            help=f"TI method name (default: {default_method}).",
        )

    p.add_argument("--dataset", required=True, help="Dataset identifier.")
    p.add_argument("--task", required=True, help="Task identifier.")
    p.add_argument("--adata", required=True, help="Path to input .h5ad file.")
    p.add_argument("--run-dir", required=True, dest="run_dir", help="Output directory for this run.")

    p.add_argument("--priors-path", default=None, help="Direct path to a priors JSON/YAML file.")
    p.add_argument("--priors-root", default=None, help="Root directory of the priors registry.")

    p.add_argument("--include-key", dest="include_key", default=None)
    p.add_argument("--include-values", dest="include_values", default=None)
    p.add_argument("--exclude-key", dest="exclude_key", default=None)
    p.add_argument("--exclude-values", dest="exclude_values", default=None)
    p.add_argument("--replace-labels-json", dest="replace_labels_json", default=None)

    p.add_argument("--group-key", dest="group_key", default=None)
    p.add_argument("--root-group", dest="root_group", default=None)
    p.add_argument("--root-cell-id", dest="root_cell_id", default=None)
    p.add_argument("--expression-layer", dest="expression_layer", default=None)
    p.add_argument("--batch-key", dest="batch_key", default=None)

    p.add_argument("--n-pcs", dest="n_pcs", type=int, default=30)
    p.add_argument("--n-neighbors", dest="n_neighbors", type=int, default=20)

    p.add_argument("--n-bootstrap", dest="n_bootstrap", type=int, default=20)
    p.add_argument("--bootstrap-frac", dest="bootstrap_frac", type=float, default=0.8)
    p.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int, default=0)
    p.add_argument("--bootstrap-stratify-by", dest="bootstrap_stratify_by", default=None)
    p.add_argument("--bootstrap-min-per-group", dest="bootstrap_min_per_group", type=int, default=10)
    p.add_argument("--skip-stability", dest="skip_stability", action="store_true")

    p.add_argument("--random-state", dest="random_state", type=int, default=0)

    p.add_argument("--min-cells", dest="min_cells", type=int, default=3)
    p.add_argument("--min-counts", dest="min_counts", type=int, default=1)
    p.add_argument("--n-top-genes", dest="n_top_genes", type=int, default=3000)
    p.add_argument("--hvg-flavor", dest="hvg_flavor", default="seurat")
    p.add_argument("--hvg-subset", dest="hvg_subset", action="store_true")
    p.add_argument("--target-sum", dest="target_sum", type=float, default=1e4)
    p.add_argument("--no-normalize", dest="no_normalize", action="store_true")
    p.add_argument("--no-log1p", dest="no_log1p", action="store_true")
    p.add_argument("--scale", dest="scale", action="store_true")

    p.add_argument("--color-key", dest="color_keys", action="append", default=None)
    p.add_argument("--export-obs-keys", dest="export_obs_keys", default=None)

    return p


def args_to_run_spec(args: argparse.Namespace) -> RunSpec:
    def _csv_to_list(s: Optional[str]) -> Optional[List[str]]:
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts if parts else None

    return RunSpec(
        method_name=str(args.method),
        dataset_name=str(args.dataset),
        task_name=str(args.task),
        adata_path=str(args.adata),
        run_dir=str(args.run_dir),
        priors_path=args.priors_path,
        priors_root=args.priors_root,
        include_key=args.include_key,
        include_values=_csv_to_list(args.include_values),
        exclude_key=args.exclude_key,
        exclude_values=_csv_to_list(args.exclude_values),
        replace_labels_json=args.replace_labels_json,
        group_key=args.group_key,
        root_group=args.root_group,
        root_cell_id=args.root_cell_id,
        expression_layer=args.expression_layer,
        batch_key=args.batch_key,
        n_pcs=int(args.n_pcs),
        n_neighbors=int(args.n_neighbors),
        n_bootstrap=int(args.n_bootstrap),
        bootstrap_frac=float(args.bootstrap_frac),
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_stratify_by=args.bootstrap_stratify_by,
        bootstrap_min_per_group=int(args.bootstrap_min_per_group),
        skip_stability=bool(args.skip_stability),
        random_state=int(args.random_state),
        min_cells=int(args.min_cells),
        min_counts=int(args.min_counts),
        n_top_genes=int(args.n_top_genes),
        hvg_flavor=str(args.hvg_flavor),
        hvg_subset=bool(args.hvg_subset),
        target_sum=float(args.target_sum),
        normalize=not bool(args.no_normalize),
        log1p=not bool(args.no_log1p),
        scale=bool(args.scale),
        color_keys=list(args.color_keys) if args.color_keys else None,
        export_obs_keys=_csv_to_list(args.export_obs_keys),
    )


__all__ = [
    "RunSpec",
    "run_benchmark",
    "load_adata",
    "load_priors",
    "build_arg_parser",
    "args_to_run_spec",
]