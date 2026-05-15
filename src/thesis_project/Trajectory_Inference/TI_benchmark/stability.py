"""
Trajectory Inference Benchmarking: Bootstrap/Subsample Stability Analysis (stability.py)

Assess reproducibility of TI outputs across subsample (bootstrap-like) replicates.

This module focuses on *reference-free* stability signals:
- Does the method return similar pseudotime orderings under subsampling?
- Does it return similar topology edges under subsampling?
- Does it fail frequently (hard reliability number)?
- Are specific cells / clusters unstable in pseudotime assignment?
- Are branch-count features stable even when edge-level overlap is noisy?

Canonical stability definitions
------------------------------
1) Pseudotime stability:
   Pairwise Spearman correlation on the INTERSECTION of cells, using ABS(rho).

2) Topology stability:
   Edge Jaccard similarity restricted to COMMON NODES for a replicate pair.
   Robustness tweak: if both restricted edge sets are empty => similarity = 1.0.

Publication-level additions in this version
-------------------------------------------
- Explicit Bootstrap Failure Rate (BFR) as a scalar in the manifest.
- Per-cell pseudotime stability across replicates using SD and MAD (not CV).
- Branch number consistency: number of leaves (deg==1) and branchpoints (deg>=3)
  and their coefficients of variation (CV) across replicates.

Coupled benchmark routing update
--------------------------------
Geometry inside each replicate now supports two modes:

1) Standard single-dataset TI benchmarking (Chapter 4)
   -> neighbors.prepare_geometry(...)

2) Coupled integration × TI benchmarking (Chapter 6)
   -> coupled_geometry.prepare_integrated_geometry(...)

Detection is automatic by default. If the replicate AnnData looks like a
subsetted integrated object (e.g. contains X_integrated / X_emb or namespaced
integration artifacts), the replicate geometry is routed through
coupled_geometry. Otherwise the classic single-dataset neighbors path is used.

Implementation notes
--------------------
- If replicate preprocessing is performed, geometry MUST be recomputed on the
  replicate; otherwise embeddings/neighbors become stale.
- Root selection correctness under subsampling: optionally reselect root in each
  replicate via root_selection.select_root(...) and set adata_rep.uns["iroot"].
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("stability.py requires anndata to be installed.") from e

from .shared_types import TIOutput, TaskPriors
from . import root_selection as root_selection_mod
from . import coupled_geometry as coupled_geometry_mod
from .utils import (
    make_rng,
    merge_json_shallow,
    set_global_seeds,
    stratified_sample_indices,
    write_json,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class TIRunnerProtocol(Protocol):
    def __call__(
        self,
        adata: "ad.AnnData",
        root_cell_id: str,
        seed: int,
        *,
        bootstrap_index: Optional[int] = None,
        **kwargs: Any,
    ) -> TIOutput: ...


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class BootstrapConfig:
    """
    Bootstrap configuration for TI stability.

    Canonical field names (used by method_runner.py):
      - n_replicates
      - frac_cells
      - stratify_by
      - replace
      - base_seed
      - min_per_group

    Backward compat:
      - min_cells_per_group (deprecated alias)
    """

    n_replicates: int = 20
    frac_cells: float = 0.8

    # CANONICAL (what method_runner passes)
    min_per_group: int = 10

    stratify_by: Optional[str] = None
    replace: bool = False
    base_seed: int = 42

    # deprecated alias (kept so older code doesn't break)
    min_cells_per_group: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_cells_per_group is not None:
            self.min_per_group = int(self.min_cells_per_group)

        self.n_replicates = int(self.n_replicates)
        self.frac_cells = float(self.frac_cells)
        self.min_per_group = int(self.min_per_group)
        self.replace = bool(self.replace)
        self.base_seed = int(self.base_seed)

        if self.n_replicates < 1:
            raise ValueError("BootstrapConfig.n_replicates must be >= 1")
        if not (0.0 < self.frac_cells <= 1.0):
            raise ValueError("BootstrapConfig.frac_cells must be in (0, 1]")
        if self.min_per_group < 1:
            raise ValueError("BootstrapConfig.min_per_group must be >= 1")


@dataclass
class GeometryConfig:
    """
    Geometry parameters to recompute inside each bootstrap replicate.

    Defaults match the neighbors module as closely as possible.

    New coupled-benchmark fields:
      - geometry_mode:
          "auto"     -> detect from replicate AnnData
          "standard" -> always use neighbors.prepare_geometry
          "coupled"  -> always use coupled_geometry.prepare_integrated_geometry
      - coupled_integration_method:
          explicit integration method name for forced coupled routing
          (optional; auto-inferred if omitted)
    """

    n_pcs: int = 30
    n_neighbors: int = 15
    rep_key: str = "X_pca"
    neighbors_key_added: Optional[str] = None
    compute_umap: bool = True
    umap_key: str = "X_umap"
    n_umap_components: int = 2
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    random_state: int = 0

    # Coupled geometry routing
    geometry_mode: str = "auto"  # {"auto", "standard", "coupled"}
    coupled_integration_method: Optional[str] = None


# -----------------------------------------------------------------------------
# Replicate result + manifest
# -----------------------------------------------------------------------------
@dataclass
class ReplicateResult:
    replicate_index: int
    seed: int
    status: str  # ok | error

    n_cells: Optional[int] = None
    cell_ids: Optional[List[str]] = None

    # Root per replicate (Option A)
    root_cell_id: Optional[str] = None
    root_selection: Optional[Dict[str, Any]] = None

    pseudotime: Optional[pd.Series] = None
    edge_list: Optional[pd.DataFrame] = None

    error_msg: Optional[str] = None
    preprocessing_report: Optional[Dict[str, Any]] = None
    geometry_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "replicate_index": int(self.replicate_index),
            "seed": int(self.seed),
            "status": str(self.status),
            "n_cells": self.n_cells,
            "error_msg": self.error_msg,
        }
        if self.root_cell_id is not None:
            d["root_cell_id"] = str(self.root_cell_id)
        if self.root_selection is not None:
            d["root_selection"] = self.root_selection

        if self.preprocessing_report is not None:
            d["preprocessing"] = {
                "n_cells_input": self.preprocessing_report.get("n_cells_input"),
                "n_cells_output": self.preprocessing_report.get("n_cells_output"),
                "n_genes_input": self.preprocessing_report.get("n_genes_input"),
                "n_genes_output": self.preprocessing_report.get("n_genes_output"),
                "params": self.preprocessing_report.get("params"),
            }
        if self.geometry_report is not None:
            d["geometry"] = {
                "pca_status": self.geometry_report.get("pca", {}).get("status"),
                "neighbors_status": self.geometry_report.get("neighbors", {}).get("status"),
                "umap_status": self.geometry_report.get("umap", {}).get("status"),
                "geometry_keys": self.geometry_report.get("geometry_keys"),
                "routing": self.geometry_report.get("routing"),
            }
        return d


@dataclass
class StabilityManifest:
    n_replicates_requested: int
    n_replicates_ok: int
    bootstrap_failure_rate: float

    pseudotime_spearman_abs_mean: Optional[float]
    pseudotime_spearman_abs_std: Optional[float]
    pseudotime_spearman_n_pairs: Optional[int]

    pseudotime_spearman_rho_mean: Optional[float]
    pseudotime_spearman_rho_std: Optional[float]

    edge_jaccard_mean: Optional[float]
    edge_jaccard_std: Optional[float]
    edge_jaccard_n_pairs: Optional[int]
    edge_jaccard_mode: str

    topology_stats: Dict[str, Any]
    branch_stats: Dict[str, Any]
    per_cell_pseudotime_stability: Dict[str, Any]

    replicate_summaries: List[Dict[str, Any]]
    config: Dict[str, Any]

    pairwise_pseudotime_spearman_abs: Optional[List[float]] = field(default=None)
    pairwise_edge_jaccard: Optional[List[float]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_replicates_requested": self.n_replicates_requested,
            "n_replicates_ok": self.n_replicates_ok,
            "bootstrap_failure_rate": self.bootstrap_failure_rate,

            "pseudotime_spearman_abs_mean": self.pseudotime_spearman_abs_mean,
            "pseudotime_spearman_abs_std": self.pseudotime_spearman_abs_std,
            "pseudotime_spearman_n_pairs": self.pseudotime_spearman_n_pairs,

            "pseudotime_spearman_rho_mean": self.pseudotime_spearman_rho_mean,
            "pseudotime_spearman_rho_std": self.pseudotime_spearman_rho_std,

            "edge_jaccard_mean": self.edge_jaccard_mean,
            "edge_jaccard_std": self.edge_jaccard_std,
            "edge_jaccard_n_pairs": self.edge_jaccard_n_pairs,
            "edge_jaccard_mode": self.edge_jaccard_mode,

            "pairwise_pseudotime_spearman_abs": self.pairwise_pseudotime_spearman_abs,
            "pairwise_edge_jaccard": self.pairwise_edge_jaccard,

            "topology_stats": self.topology_stats,
            "branch_stats": self.branch_stats,
            "per_cell_pseudotime_stability": self.per_cell_pseudotime_stability,

            "replicate_summaries": self.replicate_summaries,
            "config": self.config,
        }


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------
def _edge_set_and_nodes(
    edge_list: Optional[pd.DataFrame],
    *,
    directed: bool = False,
) -> Tuple[frozenset, frozenset]:
    if edge_list is None or edge_list.empty:
        return frozenset(), frozenset()
    if not {"source", "target"}.issubset(edge_list.columns):
        return frozenset(), frozenset()

    edges: List[Tuple[str, str]] = []
    nodes: set = set()

    a_col = edge_list["source"].astype(str).values
    b_col = edge_list["target"].astype(str).values
    for a, b in zip(a_col, b_col):
        if a == b or a in ("", "nan") or b in ("", "nan"):
            continue
        e = (a, b) if directed else tuple(sorted((a, b)))
        edges.append(e)
        nodes.add(a)
        nodes.add(b)

    return frozenset(edges), frozenset(nodes)


def _node_degrees_undirected(edge_list: Optional[pd.DataFrame]) -> Dict[str, int]:
    if edge_list is None or edge_list.empty or (not {"source", "target"}.issubset(edge_list.columns)):
        return {}
    deg: Counter = Counter()
    for a, b in zip(edge_list["source"].astype(str).values, edge_list["target"].astype(str).values):
        if a == b or a in ("", "nan") or b in ("", "nan"):
            continue
        deg[a] += 1
        deg[b] += 1
    return dict(deg)


def jaccard(A: frozenset, B: frozenset) -> float:
    union = len(A | B)
    if union == 0:
        return 1.0
    return float(len(A & B) / union)


# -----------------------------------------------------------------------------
# Root state and sampling
# -----------------------------------------------------------------------------
def _clear_stale_root_state(adata_rep: "ad.AnnData") -> None:
    if isinstance(getattr(adata_rep, "uns", None), dict):
        adata_rep.uns.pop("iroot", None)


def _sample_indices(
    adata: "ad.AnnData",
    config: BootstrapConfig,
    replicate_index: int,
    *,
    force_include_cell_id: Optional[str] = None,
    force_include_group: Optional[Tuple[str, str]] = None,
) -> np.ndarray:
    seed = int(config.base_seed) + int(replicate_index)
    rng = make_rng(seed)

    if config.stratify_by is not None and config.stratify_by in adata.obs.columns:
        groups = adata.obs[config.stratify_by].astype(str).tolist()
    else:
        groups = ["all"] * int(adata.n_obs)

    idx = stratified_sample_indices(
        groups,
        frac=float(config.frac_cells),
        min_per_group=int(config.min_per_group),
        rng=rng,
        replace=bool(config.replace),
        strict_min=False,
        sort_indices=True,
    )

    if idx.size == 0:
        return idx

    if force_include_cell_id is not None and str(force_include_cell_id) in adata.obs_names:
        force_idx = int(adata.obs_names.get_loc(str(force_include_cell_id)))
        if force_idx not in idx:
            idx = idx.copy()
            idx[-1] = force_idx
            idx = np.unique(idx)
            idx = np.sort(idx)

    if force_include_group is not None:
        gk, gv = force_include_group
        if gk in adata.obs.columns:
            mask = adata.obs[gk].astype(str).values == str(gv)
            group_idx = np.where(mask)[0]
            if group_idx.size > 0:
                force_idx = int(group_idx.min())
                if force_idx not in idx:
                    idx = idx.copy()
                    idx[-1] = force_idx
                    idx = np.unique(idx)
                    idx = np.sort(idx)

    return idx


def _enforce_fixed_gene_set(
    adata_rep: "ad.AnnData",
    fixed_var_names: Optional[pd.Index],
) -> "ad.AnnData":
    if fixed_var_names is None:
        return adata_rep

    fixed = pd.Index([str(x) for x in fixed_var_names])
    rep_vars = pd.Index([str(x) for x in adata_rep.var_names])

    common = fixed.intersection(rep_vars)
    if len(common) != len(fixed):
        if len(common) < max(50, int(0.5 * len(fixed))):
            raise RuntimeError(
                f"Replicate gene set diverged unexpectedly: fixed={len(fixed)} vs common={len(common)}. "
                "Do not recompute HVGs inside bootstraps; start bootstraps from the already-preprocessed adata."
            )
        fixed = common

    return adata_rep[:, fixed].copy()


def _ensure_rep_key_available_for_root(adata_rep: "ad.AnnData", rep_key: str) -> None:
    if rep_key == "X":
        return
    if rep_key in getattr(adata_rep, "obsm", {}):
        return
    raise RuntimeError(
        f"Root reselection requires embedding '{rep_key}' to exist in adata_rep.obsm. "
        "Provide geometry_config to recompute PCA/geometry inside each replicate, "
        "or set GeometryConfig.rep_key to a key that exists."
    )


# -----------------------------------------------------------------------------
# Coupled-geometry detection inside replicates
# -----------------------------------------------------------------------------
_COUPLED_METHOD_ORDER: List[str] = [
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


def _infer_integration_method_from_context(
    adata_rep: "ad.AnnData",
    *,
    geometry_config: Optional[GeometryConfig] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Infer integration method for coupled routing from:
      1) explicit config/runner kwargs
      2) run_dir path
      3) obsm/uns/obsp key names
    """
    if geometry_config is not None and getattr(geometry_config, "coupled_integration_method", None):
        return str(geometry_config.coupled_integration_method).lower().strip()

    if runner_kwargs:
        for key in ("integration_method", "coupled_integration_method"):
            if runner_kwargs.get(key) is not None:
                return str(runner_kwargs[key]).lower().strip()

    haystacks: List[str] = []
    if runner_kwargs and runner_kwargs.get("run_dir") is not None:
        haystacks.append(str(runner_kwargs["run_dir"]).lower())

    haystacks.extend([str(k).lower() for k in adata_rep.obsm.keys()])
    haystacks.extend([str(k).lower() for k in adata_rep.uns.keys()])
    haystacks.extend([str(k).lower() for k in adata_rep.obsp.keys()])

    text = " || ".join(haystacks)

    for method in _COUPLED_METHOD_ORDER:
        if method in text:
            return method

    return None


def _detect_coupled_geometry_input(
    adata_rep: "ad.AnnData",
    *,
    geometry_config: Optional[GeometryConfig] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Decide whether a replicate should route through coupled_geometry.
    """
    requested_mode = getattr(geometry_config, "geometry_mode", "auto") if geometry_config is not None else "auto"
    requested_mode = str(requested_mode).lower().strip()

    if requested_mode not in {"auto", "standard", "coupled"}:
        raise ValueError(
            f"GeometryConfig.geometry_mode must be one of "
            f"{{'auto', 'standard', 'coupled'}}, got {requested_mode!r}."
        )

    if requested_mode == "standard":
        return False, None, {
            "requested_mode": requested_mode,
            "detected_mode": "standard",
            "reason": "geometry_mode='standard'",
        }

    has_x_integrated = "X_integrated" in adata_rep.obsm
    has_x_emb = "X_emb" in adata_rep.obsm
    has_namespaced_neighbors_uns = any(str(k).lower().startswith("neighbors_") for k in adata_rep.uns.keys())
    has_namespaced_graph = any(
        (("connectivities" in str(k).lower()) or ("distances" in str(k).lower())) and "_" in str(k)
        for k in adata_rep.obsp.keys()
    )

    inferred_method = _infer_integration_method_from_context(
        adata_rep,
        geometry_config=geometry_config,
        runner_kwargs=runner_kwargs,
    )

    strong_evidence = bool(
        has_x_integrated or
        has_x_emb or
        has_namespaced_neighbors_uns or
        has_namespaced_graph
    )

    report = {
        "requested_mode": requested_mode,
        "has_x_integrated": bool(has_x_integrated),
        "has_x_emb": bool(has_x_emb),
        "has_namespaced_neighbors_uns": bool(has_namespaced_neighbors_uns),
        "has_namespaced_graph": bool(has_namespaced_graph),
        "inferred_integration_method": inferred_method,
        "strong_evidence": bool(strong_evidence),
    }

    if requested_mode == "coupled":
        if inferred_method is None:
            raise RuntimeError(
                "geometry_mode='coupled' was requested, but the integration method "
                "could not be inferred for the replicate."
            )
        report["detected_mode"] = "coupled"
        report["reason"] = "geometry_mode='coupled'"
        return True, inferred_method, report

    # auto mode
    if strong_evidence and inferred_method is None:
        raise RuntimeError(
            "Replicate AnnData looks like a coupled integrated object, but the "
            "integration method could not be inferred automatically."
        )

    is_coupled = bool(strong_evidence and inferred_method is not None)
    report["detected_mode"] = "coupled" if is_coupled else "standard"
    report["reason"] = "auto_detection"
    return is_coupled, inferred_method, report


# -----------------------------------------------------------------------------
# Run a single replicate
# -----------------------------------------------------------------------------
def _run_one_replicate(
    adata: "ad.AnnData",
    runner: TIRunnerProtocol,
    root_cell_id: str,
    config: BootstrapConfig,
    replicate_index: int,
    *,
    priors: Optional[TaskPriors] = None,
    reselect_root_each_replicate: bool = False,
    cli_group_key: Optional[str] = None,
    cli_root_group: Optional[str] = None,
    cli_root_cell_id: Optional[str] = None,
    root_layer: Optional[str] = None,
    k_nearest_centroid: int = 1,
    fixed_var_names: Optional[pd.Index] = None,
    preprocessing_kwargs: Optional[Dict[str, Any]] = None,
    geometry_config: Optional[GeometryConfig] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
) -> ReplicateResult:
    seed = int(config.base_seed) + int(replicate_index)

    try:
        explicit_root_id: Optional[str] = None
        if cli_root_cell_id is not None:
            explicit_root_id = str(cli_root_cell_id)
        elif priors is not None and getattr(priors, "root_cell_id", None) is not None:
            explicit_root_id = str(getattr(priors, "root_cell_id"))

        force_include_group: Optional[Tuple[str, str]] = None
        if priors is not None:
            effective_root_group = (cli_root_group or getattr(priors, "root_group", None))
            effective_group_key = (
                cli_group_key
                or getattr(priors, "root_group_key", None)
                or getattr(priors, "group_key", None)
            )
            if effective_root_group is not None and effective_group_key is not None:
                force_include_group = (str(effective_group_key), str(effective_root_group))

        idx = _sample_indices(
            adata,
            config,
            replicate_index,
            force_include_cell_id=explicit_root_id,
            force_include_group=force_include_group if explicit_root_id is None else None,
        )
        if idx.size == 0:
            return ReplicateResult(
                replicate_index=replicate_index,
                seed=seed,
                status="error",
                error_msg="subsample sampling returned 0 cells",
            )

        adata_rep = adata[idx].copy()
        adata_rep = _enforce_fixed_gene_set(adata_rep, fixed_var_names)
        n_cells = int(adata_rep.n_obs)

        _clear_stale_root_state(adata_rep)

        preproc_report: Optional[Dict[str, Any]] = None
        if preprocessing_kwargs is not None:
            from .preprocessing import preprocess_adata  # local import

            adata_rep, preproc_report = preprocess_adata(
                adata_rep,
                **{
                    k: v
                    for k, v in preprocessing_kwargs.items()
                    if k not in ("run_config_path", "tables_dir", "return_report")
                },
                run_config_path=None,
                tables_dir=None,
                return_report=True,
            )
            adata_rep = _enforce_fixed_gene_set(adata_rep, fixed_var_names)

        geom_report: Optional[Dict[str, Any]] = None
        if geometry_config is not None:
            is_coupled_geom, integration_method, routing_report = _detect_coupled_geometry_input(
                adata_rep,
                geometry_config=geometry_config,
                runner_kwargs=runner_kwargs,
            )

            if is_coupled_geom:
                adata_rep = coupled_geometry_mod.prepare_integrated_geometry(
                    adata_rep,
                    integration_method=str(integration_method),
                    n_pcs=int(geometry_config.n_pcs),
                    n_neighbors=int(geometry_config.n_neighbors),
                    n_umap_components=int(geometry_config.n_umap_components),
                    umap_min_dist=float(geometry_config.umap_min_dist),
                    umap_spread=float(geometry_config.umap_spread),
                    random_state=int(geometry_config.random_state),
                    run_config_path=None,
                )
                geom_report = coupled_geometry_mod.geometry_summary_from_adata(adata_rep)
                geom_report["routing"] = routing_report
            else:
                from . import neighbors as neighbors_mod  # local import

                adata_rep, geom_report = neighbors_mod.prepare_geometry(
                    adata_rep,
                    n_pcs=int(geometry_config.n_pcs),
                    n_neighbors=int(geometry_config.n_neighbors),
                    rep_key=str(geometry_config.rep_key),
                    neighbors_key_added=geometry_config.neighbors_key_added,
                    umap_key=str(geometry_config.umap_key),
                    n_umap_components=int(geometry_config.n_umap_components),
                    umap_min_dist=float(geometry_config.umap_min_dist),
                    umap_spread=float(geometry_config.umap_spread),
                    color_keys=None,
                    figures_dir=None,
                    random_state=int(geometry_config.random_state),
                    run_config_path=None,
                )
                geom_report["routing"] = routing_report

            if (
                geom_report.get("pca", {}).get("status") == "error"
                or geom_report.get("neighbors", {}).get("status") == "error"
            ):
                return ReplicateResult(
                    replicate_index=replicate_index,
                    seed=seed,
                    status="error",
                    n_cells=n_cells,
                    cell_ids=list(adata_rep.obs_names),
                    error_msg="geometry_failed_in_replicate",
                    preprocessing_report=preproc_report,
                    geometry_report=geom_report,
                )

        root_rep: str
        root_sel_summary: Optional[Dict[str, Any]] = None

        if reselect_root_each_replicate:
            if priors is None:
                raise ValueError("reselect_root_each_replicate=True requires `priors` to be provided.")

            if geom_report is not None and "geometry_keys" in geom_report:
                rep_key_for_root = str(geom_report["geometry_keys"].get("pca_key", "X_pca"))
            elif geometry_config is not None:
                rep_key_for_root = str(geometry_config.rep_key)
            else:
                rep_key_for_root = "X_pca"

            _ensure_rep_key_available_for_root(adata_rep, rep_key_for_root)

            root_rep, root_sel_summary = root_selection_mod.select_root(
                adata_rep,
                priors,
                cli_group_key=cli_group_key,
                cli_root_group=cli_root_group,
                cli_root_cell_id=cli_root_cell_id,
                rep_key=rep_key_for_root,
                layer=root_layer,
                k_nearest_centroid=int(k_nearest_centroid),
                random_state=int(seed),
                run_config_path=None,
            )
        else:
            root_rep = root_cell_id if root_cell_id in adata_rep.obs_names else str(adata_rep.obs_names[0])

        _clear_stale_root_state(adata_rep)
        if root_rep in adata_rep.obs_names:
            adata_rep.uns["iroot"] = int(adata_rep.obs_names.get_loc(root_rep))
        else:
            root_rep = str(adata_rep.obs_names[0])
            adata_rep.uns["iroot"] = 0

        set_global_seeds(seed)
        ti_output = runner(
            adata_rep,
            root_rep,
            seed,
            bootstrap_index=replicate_index,
            **(runner_kwargs or {}),
        )

        return ReplicateResult(
            replicate_index=replicate_index,
            seed=seed,
            status="ok",
            n_cells=n_cells,
            cell_ids=list(adata_rep.obs_names),
            root_cell_id=root_rep,
            root_selection=root_sel_summary,
            pseudotime=ti_output.pseudotime,
            edge_list=ti_output.edge_list,
            preprocessing_report=preproc_report,
            geometry_report=geom_report,
        )

    except Exception as e:
        return ReplicateResult(
            replicate_index=replicate_index,
            seed=seed,
            status="error",
            error_msg=f"{type(e).__name__}: {e}",
        )


# -----------------------------------------------------------------------------
# Pairwise stability metrics
# -----------------------------------------------------------------------------
def _pairwise_pseudotime_spearman_abs(replicates: List[ReplicateResult]) -> Tuple[List[float], int]:
    ok_reps = [r for r in replicates if r.status == "ok" and r.pseudotime is not None]
    abs_rhos: List[float] = []

    for i in range(len(ok_reps)):
        for j in range(i + 1, len(ok_reps)):
            a = ok_reps[i].pseudotime
            b = ok_reps[j].pseudotime
            assert a is not None and b is not None

            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue

            ax = pd.to_numeric(a.loc[common], errors="coerce").astype(float).values
            bx = pd.to_numeric(b.loc[common], errors="coerce").astype(float).values
            m = np.isfinite(ax) & np.isfinite(bx)
            if int(m.sum()) < 10:
                continue

            rho, _ = stats.spearmanr(ax[m], bx[m])
            if np.isfinite(rho):
                abs_rhos.append(float(abs(rho)))

    return abs_rhos, len(abs_rhos)


def _pairwise_edge_jaccard_common_nodes(replicates: List[ReplicateResult]) -> Tuple[List[float], int]:
    ok_reps = [r for r in replicates if r.status == "ok"]
    edge_sets: List[frozenset] = []
    node_sets: List[frozenset] = []

    for r in ok_reps:
        E, N = _edge_set_and_nodes(r.edge_list, directed=False)
        edge_sets.append(E)
        node_sets.append(N)

    vals: List[float] = []
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            common_nodes = node_sets[i] & node_sets[j]
            if not common_nodes:
                continue

            Ei = frozenset(e for e in edge_sets[i] if (e[0] in common_nodes and e[1] in common_nodes))
            Ej = frozenset(e for e in edge_sets[j] if (e[0] in common_nodes and e[1] in common_nodes))

            if not Ei and not Ej:
                vals.append(1.0)
                continue

            vals.append(jaccard(Ei, Ej))

    return vals, len(vals)


# -----------------------------------------------------------------------------
# Descriptive topology stats + branch consistency
# -----------------------------------------------------------------------------
def _stats_basic(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": int(a.min()),
        "max": int(a.max()),
        "n": int(len(vals)),
    }


def _stats_with_cv(vals: List[int]) -> Dict[str, Any]:
    s = _stats_basic(vals)
    if s["n"] == 0 or s["mean"] in (None, 0.0):
        s["cv"] = None
    else:
        mean = float(s["mean"])
        std = float(s["std"]) if s["std"] is not None else np.nan
        s["cv"] = float(std / mean) if (np.isfinite(std) and mean > 0) else None
    return s


def _topology_stats(replicates: List[ReplicateResult]) -> Dict[str, Any]:
    n_nodes_list: List[int] = []
    n_edges_list: List[int] = []

    for r in replicates:
        if r.status != "ok" or r.edge_list is None:
            continue
        E, N = _edge_set_and_nodes(r.edge_list, directed=False)
        n_nodes_list.append(len(N))
        n_edges_list.append(len(E))

    return {"n_nodes": _stats_basic(n_nodes_list), "n_edges": _stats_basic(n_edges_list)}


def _branch_stats(replicates: List[ReplicateResult]) -> Dict[str, Any]:
    leaves: List[int] = []
    branchpoints: List[int] = []

    for r in replicates:
        if r.status != "ok":
            continue
        deg = _node_degrees_undirected(r.edge_list)
        if not deg:
            leaves.append(0)
            branchpoints.append(0)
            continue

        leaves.append(int(sum(1 for d in deg.values() if d == 1)))
        branchpoints.append(int(sum(1 for d in deg.values() if d >= 3)))

    return {
        "n_leaves": _stats_with_cv(leaves),
        "n_branchpoints": _stats_with_cv(branchpoints),
    }


# -----------------------------------------------------------------------------
# Per-cell pseudotime stability (SD / MAD across replicates)
# -----------------------------------------------------------------------------
def _rescale_0_1_series(pt: pd.Series) -> pd.Series:
    x = pd.to_numeric(pt, errors="coerce").astype(float)
    v = x.values
    m = np.isfinite(v)
    if int(m.sum()) < 2:
        return x * np.nan
    lo = float(np.nanmin(v[m]))
    hi = float(np.nanmax(v[m]))
    if np.isclose(lo, hi):
        return x * np.nan
    return (x - lo) / (hi - lo)


def _align_to_reference(rep_pt01: pd.Series, ref_pt01: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    common = rep_pt01.index.intersection(ref_pt01.index)
    if len(common) < 10:
        return rep_pt01, {"status": "skipped", "reason": f"too few common cells (n_common={len(common)})"}

    a = rep_pt01.loc[common].to_numpy(dtype=float)
    b = ref_pt01.loc[common].to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 10:
        return rep_pt01, {"status": "skipped", "reason": f"too few finite common cells (n_finite={int(m.sum())})"}

    rho, _ = stats.spearmanr(a[m], b[m])
    if not np.isfinite(rho):
        return rep_pt01, {"status": "skipped", "reason": "spearman rho is non-finite"}

    flipped = bool(rho < 0)
    if flipped:
        rep_pt01 = 1.0 - rep_pt01

    return rep_pt01, {"status": "ok", "rho_to_ref": float(rho), "flipped": flipped}


def _per_cell_pseudotime_stability(
    adata: "ad.AnnData",
    replicates: List[ReplicateResult],
    *,
    priors: Optional[TaskPriors] = None,
    min_reps_per_cell: int = 3,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    ok = [r for r in replicates if r.status == "ok" and r.pseudotime is not None]
    if len(ok) < 2:
        return ({"status": "skipped", "reason": "fewer than 2 successful replicates"}, None)

    min_reps_per_cell = int(min_reps_per_cell)
    if min_reps_per_cell < 2:
        min_reps_per_cell = 2

    ref_pt01 = _rescale_0_1_series(ok[0].pseudotime)  # type: ignore[arg-type]
    if ref_pt01.isna().all():
        return ({"status": "skipped", "reason": "reference replicate pseudotime could not be rescaled"}, None)

    aligned_pts: List[pd.Series] = []
    align_reports: List[Dict[str, Any]] = []

    for r in ok:
        pt01 = _rescale_0_1_series(r.pseudotime)  # type: ignore[arg-type]
        rep = {"replicate_index": int(r.replicate_index), "status": "ok"}

        pt01_aligned, rep_align = _align_to_reference(pt01, ref_pt01)
        rep.update(rep_align)
        aligned_pts.append(pt01_aligned.rename(f"rep{r.replicate_index}"))
        align_reports.append(rep)

    mat = pd.concat(aligned_pts, axis=1, join="outer")

    n_obs = mat.notna().sum(axis=1).astype(int)
    eligible = n_obs >= int(min_reps_per_cell)
    n_eligible = int(eligible.sum())
    if n_eligible < 10:
        return (
            {
                "status": "skipped",
                "reason": f"too few cells eligible for per-cell stability (n_eligible={n_eligible})",
                "min_reps_per_cell": int(min_reps_per_cell),
                "n_replicates_ok": int(len(ok)),
            },
            None,
        )

    mat_el = mat.loc[eligible]
    arr = mat_el.to_numpy(dtype=float)

    sd = np.nanstd(arr, axis=1, ddof=0)
    med = np.nanmedian(arr, axis=1)
    mad = np.nanmedian(np.abs(arr - med[:, None]), axis=1)

    per_cell = pd.DataFrame(
        {
            "n_replicates": n_obs.loc[eligible].values,
            "pt_sd": sd,
            "pt_mad": mad,
        },
        index=mat_el.index.astype(str),
    )

    per_cluster: Optional[Dict[str, Any]] = None
    group_key = None
    if priors is not None:
        group_key = getattr(priors, "group_key", None) or getattr(priors, "root_group_key", None)

    if group_key is not None and group_key in adata.obs.columns:
        labels = adata.obs[group_key].astype(str)
        labels = labels.reindex(per_cell.index)
        tmp = per_cell.copy()
        tmp["cluster"] = labels.values
        g = tmp.groupby("cluster", sort=True)
        per_cluster = {
            "group_key": str(group_key),
            "n_clusters": int(g.ngroups),
            "median_sd_by_cluster": g["pt_sd"].median().sort_values().to_dict(),
            "n_cells_by_cluster": g.size().astype(int).to_dict(),
        }

    summary = {
        "status": "ok",
        "min_reps_per_cell": int(min_reps_per_cell),
        "n_replicates_ok": int(len(ok)),
        "n_cells_eligible": int(n_eligible),
        "sd_median": float(np.nanmedian(sd)),
        "sd_p90": float(np.nanpercentile(sd, 90)),
        "mad_median": float(np.nanmedian(mad)),
        "mad_p90": float(np.nanpercentile(mad, 90)),
        "alignment_reports": align_reports,
    }
    if per_cluster is not None:
        summary["per_cluster"] = per_cluster

    return summary, per_cell


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def run_stability(
    adata: "ad.AnnData",
    runner: TIRunnerProtocol,
    root_cell_id: str,
    *,
    priors: Optional[TaskPriors] = None,
    reselect_root_each_replicate: bool = False,
    cli_group_key: Optional[str] = None,
    cli_root_group: Optional[str] = None,
    cli_root_cell_id: Optional[str] = None,
    root_layer: Optional[str] = None,
    k_nearest_centroid: int = 1,
    config: Optional[BootstrapConfig] = None,
    fixed_var_names: Optional[Union[pd.Index, List[str]]] = None,
    preprocessing_kwargs: Optional[Dict[str, Any]] = None,
    geometry_config: Optional[GeometryConfig] = None,
    runner_kwargs: Optional[Dict[str, Any]] = None,
    run_config_path: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    min_reps_per_cell: int = 3,
    write_per_cell_table: bool = False,
) -> Tuple[pd.DataFrame, StabilityManifest]:
    """
    Run bootstrap/subsample stability evaluation.

    Returns:
      - summary_df: per-replicate status summary
      - manifest: StabilityManifest with aggregate stability metrics
    """
    if config is None:
        config = BootstrapConfig()

    fixed_idx: Optional[pd.Index] = None
    if fixed_var_names is not None:
        fixed_idx = pd.Index([str(x) for x in list(fixed_var_names)])

    replicates: List[ReplicateResult] = []
    for i in range(int(config.n_replicates)):
        rep = _run_one_replicate(
            adata,
            runner,
            root_cell_id,
            config,
            i,
            priors=priors,
            reselect_root_each_replicate=bool(reselect_root_each_replicate),
            cli_group_key=cli_group_key,
            cli_root_group=cli_root_group,
            cli_root_cell_id=cli_root_cell_id,
            root_layer=root_layer,
            k_nearest_centroid=int(k_nearest_centroid),
            fixed_var_names=fixed_idx,
            preprocessing_kwargs=preprocessing_kwargs,
            geometry_config=geometry_config,
            runner_kwargs=runner_kwargs,
        )
        replicates.append(rep)

    n_ok = int(sum(1 for r in replicates if r.status == "ok"))
    n_req = int(config.n_replicates)
    bfr = float((n_req - n_ok) / max(1, n_req))

    abs_rhos, n_rho_pairs = _pairwise_pseudotime_spearman_abs(replicates)
    jacs, n_jac_pairs = _pairwise_edge_jaccard_common_nodes(replicates)
    topo = _topology_stats(replicates)
    branch = _branch_stats(replicates)

    abs_mean = float(np.mean(abs_rhos)) if abs_rhos else None
    abs_std = float(np.std(abs_rhos)) if abs_rhos else None

    per_cell_summary, per_cell_df = _per_cell_pseudotime_stability(
        adata,
        replicates,
        priors=priors,
        min_reps_per_cell=int(min_reps_per_cell),
    )

    manifest = StabilityManifest(
        n_replicates_requested=n_req,
        n_replicates_ok=n_ok,
        bootstrap_failure_rate=bfr,

        pseudotime_spearman_abs_mean=abs_mean,
        pseudotime_spearman_abs_std=abs_std,
        pseudotime_spearman_n_pairs=int(n_rho_pairs),

        pseudotime_spearman_rho_mean=abs_mean,
        pseudotime_spearman_rho_std=abs_std,

        edge_jaccard_mean=float(np.mean(jacs)) if jacs else None,
        edge_jaccard_std=float(np.std(jacs)) if jacs else None,
        edge_jaccard_n_pairs=int(n_jac_pairs),
        edge_jaccard_mode="restricted_to_common_nodes",

        pairwise_pseudotime_spearman_abs=abs_rhos if abs_rhos else None,
        pairwise_edge_jaccard=jacs if jacs else None,

        topology_stats=topo,
        branch_stats=branch,
        per_cell_pseudotime_stability=per_cell_summary,

        replicate_summaries=[r.to_dict() for r in replicates],
        config={
            "n_replicates": n_req,
            "frac_cells": float(config.frac_cells),
            "min_per_group": int(config.min_per_group),
            "stratify_by": config.stratify_by,
            "replace": bool(config.replace),
            "base_seed": int(config.base_seed),
            "geometry_config": (geometry_config.__dict__ if geometry_config is not None else None),
            "fixed_gene_set": (list(fixed_idx) if fixed_idx is not None else None),
            "pseudotime_stability_definition": "spearman_on_cell_intersection_abs_rho",
            "edge_jaccard_definition": "jaccard_on_edges_restricted_to_common_nodes",
            "root_reselection": {
                "enabled": bool(reselect_root_each_replicate),
                "cli_group_key": cli_group_key,
                "cli_root_group": cli_root_group,
                "cli_root_cell_id": cli_root_cell_id,
                "root_layer": root_layer,
                "k_nearest_centroid": int(k_nearest_centroid),
            },
            "per_cell_stability": {
                "min_reps_per_cell": int(min_reps_per_cell),
                "write_per_cell_table": bool(write_per_cell_table),
            },
        },
    )

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"stability": manifest.to_dict()})

    rows: List[Dict[str, Any]] = []
    for r in replicates:
        rows.append(
            {
                "replicate_index": int(r.replicate_index),
                "seed": int(r.seed),
                "status": str(r.status),
                "n_cells": r.n_cells,
                "root_cell_id": r.root_cell_id,
                "error_msg": r.error_msg,
            }
        )
    summary_df = pd.DataFrame(rows)

    if out_dir is not None:
        od = Path(out_dir)
        (od / "tables").mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(od / "tables" / "stability_replicates.csv", index=False)
        write_json(
            od / "tables" / "stability_manifest.json",
            manifest.to_dict(),
            indent=2,
            sort_keys=False,
            atomic=True,
        )
        pd.DataFrame({"pseudotime_spearman_abs_rho": abs_rhos}).to_csv(
            od / "tables" / "stability_pairwise_pseudotime_abs_rho.csv",
            index=False,
        )
        pd.DataFrame({"edge_jaccard_common_nodes": jacs}).to_csv(
            od / "tables" / "stability_pairwise_edge_jaccard_common_nodes.csv",
            index=False,
        )

        if bool(write_per_cell_table) and per_cell_df is not None:
            per_cell_df.to_csv(od / "tables" / "stability_per_cell_pseudotime.csv", index=True)

    return summary_df, manifest


__all__ = [
    "TIRunnerProtocol",
    "BootstrapConfig",
    "GeometryConfig",
    "ReplicateResult",
    "StabilityManifest",
    "run_stability",
    "jaccard",
]