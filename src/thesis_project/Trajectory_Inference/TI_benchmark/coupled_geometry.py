"""
coupled_geometry.py
===================
Geometry preparation for the coupled integration × TI benchmark (Chapter 6).

Two-route design
----------------
Integration methods differ in what constitutes their authoritative corrected
output. This module routes each method through the appropriate path so that
all downstream TI methods receive a consistent set of geometry slots:

    obsm["X_integrated"]   <- authoritative corrected representation (honest name)
    obsm["X_pca"]          <- compatibility mirror for legacy TI wrappers
    obsp["connectivities"] <- kNN graph built fresh from X_integrated on task cells
    obsp["distances"]      <- kNN distances
    uns["neighbors"]       <- neighbors metadata
    obsm["X_umap"]         <- 2-D UMAP for visualisation

Route A — Expression-correction methods: ComBat, MNN
    Authoritative output: corrected adata.X (continuous expression values).
    Action: compute_pca from corrected X -> compute_neighbors -> compute_umap.
    Rationale: ComBat/MNN correct the expression matrix itself. PCA computed
    from that corrected matrix IS the integration output, not a pre-existing
    embedding. Recomputing it faithfully propagates the correction.

Route B — Embedding/latent methods: all others
    Authoritative output: obsm["X_emb"].
    Action: copy X_emb -> X_integrated, mirror to X_pca, then
    compute_neighbors from X_integrated -> compute_umap.
    Methods: scVI, scANVI, scGen, Harmony, fastMNN, Scanorama, LIGER,
             Seurat RPCA, BBKNN.
    Note on BBKNN: X_emb is a diffmap coordinate projection of the
    corrected graph -- a valid and stable representation for neighbour
    recomputation on the lineage subset.

Critical design principle
--------------------------
In BOTH routes, the neighbourhood graph is ALWAYS recomputed fresh on the
lineage-task cell subset. The full-atlas graphs stored in the integration
h5ad files are built on ~299,891 cells and are not valid neighbourhood
representations for lineage populations of a few thousand cells.

Key naming
----------
"X_integrated" is used instead of "X_pca" for the authoritative
representation because most Route B embeddings are not PCA. "X_pca"
is provided only as a compatibility mirror for TI wrappers that expect it.

The rep_key passed downstream by method_runner is "X_integrated",
so TI methods and root selection receive the honest, semantically
correct key name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import numpy as np

try:
    import anndata as ad
except ImportError as e:
    raise ImportError("coupled_geometry.py requires anndata.") from e

from . import neighbors as neighbors_mod
from .utils import merge_json_shallow

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Route registries
# ─────────────────────────────────────────────────────────────────────────────

# Methods whose authoritative output is the corrected expression matrix (adata.X).
# PCA is recomputed from that matrix in the coupled benchmark.
ROUTE_A_METHODS: Set[str] = {"combat", "mnn"}

# Methods whose authoritative output is obsm["X_emb"].
ROUTE_B_METHODS: Set[str] = {
    "scvi", "scanvi", "scgen",
    "harmony", "fastmnn", "scanorama",
    "liger", "seurat", "bbknn",
}

# Canonical keys written by this module
INTEGRATED_REP_KEY = "X_integrated"   # authoritative corrected representation
LEGACY_PCA_KEY     = "X_pca"          # compatibility mirror for TI wrappers


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_x_emb(adata: "ad.AnnData", method: str) -> np.ndarray:
    """
    Confirm obsm["X_emb"] is present, 2-D, cell-aligned, and fully finite.
    Returns the embedding as a float32 numpy array.
    """
    if "X_emb" not in adata.obsm:
        raise RuntimeError(
            f"coupled_geometry [{method}]: 'X_emb' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}. "
            "Ensure the integration pipeline set ad.obsm['X_emb'] before saving."
        )

    emb = np.asarray(adata.obsm["X_emb"], dtype=np.float32)

    if emb.ndim != 2:
        raise RuntimeError(
            f"coupled_geometry [{method}]: X_emb must be 2-D, got shape {emb.shape}."
        )

    if emb.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"coupled_geometry [{method}]: X_emb has {emb.shape[0]} rows but "
            f"adata.n_obs={adata.n_obs}. Cell count mismatch after lineage subsetting."
        )

    n_bad = int(np.sum(~np.isfinite(emb)))
    if n_bad > 0:
        raise RuntimeError(
            f"coupled_geometry [{method}]: X_emb contains {n_bad} non-finite "
            "value(s) (NaN or Inf). Check the integration output."
        )

    return emb


def _validate_x_for_pca(adata: "ad.AnnData", method: str) -> None:
    """
    Confirm adata.X is suitable for PCA recomputation (Route A).
    Raises RuntimeError on hard failures; logs a warning on soft failures.
    """
    if adata.X is None:
        raise RuntimeError(
            f"coupled_geometry [{method}]: adata.X is None. "
            "Route A methods require a corrected expression matrix in X."
        )

    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        sample = np.asarray(
            X.data[:min(10000, X.data.size)], dtype=np.float32
        )
    else:
        arr = np.asarray(X, dtype=np.float32)
        sample = arr.ravel()[:10000]

    if not np.isfinite(sample).all():
        raise RuntimeError(
            f"coupled_geometry [{method}]: adata.X contains non-finite values. "
            "Cannot compute PCA on corrupted expression matrix."
        )

    xmax = float(sample.max())
    if xmax > 1e4:
        logger.warning(
            "coupled_geometry [%s]: adata.X max=%.1f is very large. "
            "Ensure X contains corrected expression values, not raw counts.",
            method, xmax,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Route A — expression-correction methods
# ─────────────────────────────────────────────────────────────────────────────

def _route_a(
    adata: "ad.AnnData",
    method: str,
    *,
    n_pcs: int,
    random_state: int,
    run_config_path: Optional[Union[str, Path]],
    log_payload: Dict[str, Any],
) -> "ad.AnnData":
    """
    Route A: recompute PCA from corrected adata.X.

    Writes:
        obsm["X_pca"]         <- PCA from corrected X (scanpy canonical)
        obsm["X_integrated"]  <- same array (honest alias)
    Graph and UMAP are built by the shared step that follows.
    """
    _validate_x_for_pca(adata, method)

    logger.info(
        "coupled_geometry [%s]: Route A — recomputing PCA from corrected X "
        "(n_pcs=%d, n_obs=%d)",
        method, n_pcs, adata.n_obs,
    )

    adata, res_pca = neighbors_mod.compute_pca(
        adata,
        n_comps=n_pcs,
        rep_key="X_pca",
        use_highly_variable=True,
        random_state=random_state,
        run_config_path=run_config_path,
    )
    if res_pca.status == "error":
        raise RuntimeError(
            f"coupled_geometry [{method}]: PCA failed — {res_pca.reason}"
        )

    # Copy to canonical integrated key
    adata.obsm[INTEGRATED_REP_KEY] = adata.obsm["X_pca"].copy()

    log_payload.update({
        "route":              "A",
        "pca_source":         "corrected_adata.X",
        "n_pcs_requested":    int(n_pcs),
        "n_pcs_actual":       int(res_pca.details.get("n_comps_actual", n_pcs)),
        "integrated_rep_key": INTEGRATED_REP_KEY,
        "integrated_shape":   list(adata.obsm[INTEGRATED_REP_KEY].shape),
    })

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Route B — embedding/latent methods
# ─────────────────────────────────────────────────────────────────────────────

def _route_b(
    adata: "ad.AnnData",
    method: str,
    *,
    log_payload: Dict[str, Any],
) -> "ad.AnnData":
    """
    Route B: copy X_emb into canonical keys, skip PCA entirely.

    Writes:
        obsm["X_integrated"]  <- copy of X_emb (authoritative)
        obsm["X_pca"]         <- same copy (compatibility mirror)
    Graph and UMAP are built by the shared step that follows.
    """
    emb = _validate_x_emb(adata, method)

    logger.info(
        "coupled_geometry [%s]: Route B — using X_emb as corrected "
        "representation (shape=%s, range=(%.3f, %.3f))",
        method, emb.shape, float(emb.min()), float(emb.max()),
    )

    adata.obsm[INTEGRATED_REP_KEY] = emb.copy()
    adata.obsm[LEGACY_PCA_KEY]     = emb.copy()

    log_payload.update({
        "route":              "B",
        "embedding_source":   "obsm['X_emb']",
        "embedding_shape":    list(emb.shape),
        "embedding_range":    [round(float(emb.min()), 4),
                               round(float(emb.max()), 4)],
        "integrated_rep_key": INTEGRATED_REP_KEY,
    })

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Shared: build graph and UMAP from X_integrated
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph_and_umap(
    adata: "ad.AnnData",
    method: str,
    *,
    n_neighbors: int,
    n_umap_components: int,
    umap_min_dist: float,
    umap_spread: float,
    random_state: int,
    run_config_path: Optional[Union[str, Path]],
    log_payload: Dict[str, Any],
) -> "ad.AnnData":
    """
    Build kNN graph from X_integrated and compute UMAP.
    Called by both routes after the corrected representation is in place.
    Always operates on the cell subset already in adata -- never the full atlas.
    """
    logger.info(
        "coupled_geometry [%s]: building kNN graph from '%s' "
        "(n_neighbors=%d, n_obs=%d)",
        method, INTEGRATED_REP_KEY, n_neighbors, adata.n_obs,
    )

    adata, res_nn = neighbors_mod.compute_neighbors(
        adata,
        n_neighbors=n_neighbors,
        rep_key=INTEGRATED_REP_KEY,
        key_added=None,          # writes to canonical uns["neighbors"]
        metric="euclidean",
        random_state=random_state,
        run_config_path=run_config_path,
    )
    if res_nn.status == "error":
        raise RuntimeError(
            f"coupled_geometry [{method}]: kNN graph failed — {res_nn.reason}"
        )

    conn_key = res_nn.details.get("connectivities_key", "connectivities")
    dist_key = res_nn.details.get("distances_key", "distances")
    nnz = int(adata.obsp[conn_key].nnz) if conn_key in adata.obsp else -1

    logger.info(
        "coupled_geometry [%s]: graph built (nnz=%d) — computing UMAP",
        method, nnz,
    )

    adata, res_umap = neighbors_mod.compute_umap(
        adata,
        neighbors_key="neighbors",
        umap_key="X_umap",
        n_components=n_umap_components,
        min_dist=umap_min_dist,
        spread=umap_spread,
        random_state=random_state,
        run_config_path=run_config_path,
    )
    if res_umap.status == "error":
        raise RuntimeError(
            f"coupled_geometry [{method}]: UMAP failed — {res_umap.reason}"
        )

    log_payload.update({
        "neighbors_status":      "ok",
        "connectivities_key":    conn_key,
        "distances_key":         dist_key,
        "n_neighbors_requested": int(n_neighbors),
        "n_neighbors_actual":    int(res_nn.details.get("n_neighbors_actual",
                                                         n_neighbors)),
        "graph_nnz":             int(nnz),
        "umap_status":           "ok",
    })

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def prepare_integrated_geometry(
    adata: "ad.AnnData",
    integration_method: str,
    *,
    n_pcs: int = 50,
    n_neighbors: int = 20,
    n_umap_components: int = 2,
    umap_min_dist: float = 0.3,
    umap_spread: float = 1.0,
    random_state: int = 0,
    run_config_path: Optional[Union[str, Path]] = None,
) -> "ad.AnnData":
    """
    Prepare canonical geometry slots for the coupled benchmark.

    Called INSIDE run_benchmark, AFTER preprocessing has subsetted adata
    to the lineage task cells. Geometry is therefore always built on the
    correct biological population, not the full atlas.

    Parameters
    ----------
    adata               : AnnData already subsetted to lineage task cells.
    integration_method  : Lowercase method name, e.g. "harmony", "combat".
    n_pcs               : Number of PCs for Route A (ignored for Route B).
    n_neighbors         : Number of neighbours for kNN graph construction.
    n_umap_components   : UMAP output dimensionality.
    umap_min_dist       : UMAP min_dist parameter.
    umap_spread         : UMAP spread parameter.
    random_state        : Seed for PCA (Route A), kNN, and UMAP.
    run_config_path     : Optional path to run_config.json for logging.

    Returns
    -------
    adata with canonical geometry slots populated:
        obsm["X_integrated"]   <- authoritative corrected representation
        obsm["X_pca"]          <- compatibility mirror (same array)
        obsp["connectivities"], obsp["distances"]
        uns["neighbors"]
        obsm["X_umap"]

    Raises
    ------
    ValueError   : unknown integration method name
    RuntimeError : missing required slot, non-finite values, or
                   computation failure in PCA/graph/UMAP steps
    """
    method = str(integration_method).lower().strip()

    known = ROUTE_A_METHODS | ROUTE_B_METHODS
    if method not in known:
        raise ValueError(
            f"coupled_geometry: unknown integration method '{method}'. "
            f"Known methods: {sorted(known)}. "
            "Add it to ROUTE_A_METHODS or ROUTE_B_METHODS as appropriate."
        )

    log_payload: Dict[str, Any] = {
        "integration_method": method,
        "n_cells":            int(adata.n_obs),
        "n_genes":            int(adata.n_vars),
        "n_neighbors":        int(n_neighbors),
        "random_state":       int(random_state),
    }

    # ── Route dispatch ────────────────────────────────────────────────────────
    if method in ROUTE_A_METHODS:
        adata = _route_a(
            adata, method,
            n_pcs=n_pcs,
            random_state=random_state,
            run_config_path=run_config_path,
            log_payload=log_payload,
        )
    else:
        adata = _route_b(adata, method, log_payload=log_payload)

    # ── Shared: graph + UMAP from X_integrated ────────────────────────────────
    adata = _build_graph_and_umap(
        adata, method,
        n_neighbors=n_neighbors,
        n_umap_components=n_umap_components,
        umap_min_dist=umap_min_dist,
        umap_spread=umap_spread,
        random_state=random_state,
        run_config_path=run_config_path,
        log_payload=log_payload,
    )

    if run_config_path is not None:
        merge_json_shallow(
            run_config_path,
            {"coupled_geometry": log_payload},
        )

    logger.info(
        "coupled_geometry [%s]: complete — X_integrated=%s  graph_nnz=%d  X_umap=%s",
        method,
        adata.obsm[INTEGRATED_REP_KEY].shape,
        log_payload.get("graph_nnz", -1),
        adata.obsm["X_umap"].shape,
    )

    return adata


def geometry_summary_from_adata(adata: "ad.AnnData") -> Dict[str, Any]:
    """
    Build a geometry_summary dict compatible with method_runner's expectations
    after coupled geometry preparation.

    Pass this as geometry_summary when skip_geometry=True in run_benchmark,
    ensuring that rep_key, connectivities_key, and umap_key are all resolved
    correctly for root selection and TI execution.
    """
    conn_key = "connectivities"
    dist_key = "distances"

    if "neighbors" in adata.uns:
        conn_key = adata.uns["neighbors"].get("connectivities_key", conn_key)
        dist_key = adata.uns["neighbors"].get("distances_key", dist_key)

    return {
        "pca": {
            "status": "coupled",
            "reason": "geometry from coupled_geometry.prepare_integrated_geometry",
        },
        "neighbors": {"status": "ok"},
        "umap":      {"status": "ok"},
        "geometry_keys": {
            "pca_key":            INTEGRATED_REP_KEY,
            "neighbors_key":      "neighbors",
            "connectivities_key": conn_key,
            "distances_key":      dist_key,
            "umap_key":           "X_umap",
        },
    }


__all__ = [
    "ROUTE_A_METHODS",
    "ROUTE_B_METHODS",
    "INTEGRATED_REP_KEY",
    "LEGACY_PCA_KEY",
    "prepare_integrated_geometry",
    "geometry_summary_from_adata",
]