"""
Integration benchmarking metrics — bounded-runtime scIB adapter.

All 14 metrics are attempted via scib_compat.compute_all_scib_metrics().
The integration_metrics() function is the single public entry point.

Key design decisions
--------------------
1. output_type determines which scIB type_ is used for kBET and LISI:
      'knn'   -> graph-native outputs
      'embed' -> embedding-output methods
      'full'  -> corrected feature-matrix outputs

2. Metrics that compare pre/post integration (PCR, HVG, cell_cycle, trajectory)
   require adata_pre — the unintegrated, preprocessed AnnData. Pass it via
   integration_metrics(..., adata_pre=...).

3. Runtime is bounded with two working subsets:
      metric_subsample_n       default 100_000 cells
      heavy_metric_subsample_n default  50_000 cells
   Small datasets still use all cells. Heavy metrics are not skipped; they are
   computed on the bounded subset.

4. NMI/ARI use an existing cluster_key if present. If it is absent, a single
   Leiden clustering is created on the working subset as a fast fallback.

5. Metrics unavailable for a given output_type return np.nan (not an error).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from thesis_project.Integration.Integration_benchmark.scib_compat import (
    compute_all_scib_metrics,
)

# Valid output types — matches scib's type_ argument
OutputType = Literal["knn", "embed", "full"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def integration_metrics(
    adata_int,
    *,
    # ---- Keys ----
    batch_key: str,
    label_key: str,
    cluster_key: str,
    # ---- Embedding / graph keys ----
    emb_key: Optional[str],
    conn_key: Optional[str] = None,
    dist_key: Optional[str] = None,
    neighbors_uns_key: Optional[str] = None,
    # ---- Method output type ----
    output_type: OutputType = "embed",
    # ---- Comparison with unintegrated data ----
    adata_pre=None,
    # ---- Optional metric toggles ----
    compute_trajectory: bool = False,
    trajectory_adata_pre=None,
    pseudotime_key: str = "dpt_pseudotime",
    # ---- scib parameters ----
    organism: str = "human",
    n_isolated: Optional[int] = None,
    lisi_subsample: Optional[int] = 100,
    verbose: bool = False,
    # ---- Runtime controls ----
    metric_subsample_n: Optional[int] = 100_000,
    heavy_metric_subsample_n: Optional[int] = 50_000,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Compute all 14 scIB integration benchmarking metrics.

    Parameters
    ----------
    adata_int       : AnnData — the integrated output to evaluate
    batch_key       : adata_int.obs column for batch / dataset identity
    label_key       : adata_int.obs column for cell-type annotations
    cluster_key     : adata_int.obs column for Leiden cluster assignments.
                      If missing, a single Leiden clustering is created on the
                      working subset as a fast fallback.
    emb_key         : adata_int.obsm key for the integration embedding.
                      Required for output_type='embed'.
                      Optional for output_type='full'; a temporary PCA is used
                      when needed for embedding-based metrics.
                      For pure knn outputs, keep None.
    conn_key        : adata_int.obsp key for corrected kNN connectivities.
                      Required for output_type='knn'.
    dist_key        : adata_int.obsp key for corrected kNN distances.
    neighbors_uns_key : adata_int.uns key for the neighbors params dict.
    output_type     : One of 'knn' | 'embed' | 'full'.
    adata_pre       : Unintegrated, preprocessed AnnData (same cells, same order).
                      Required for PCR, HVG conservation, cell-cycle
                      conservation, and trajectory conservation.
    compute_trajectory : Whether to compute trajectory conservation.
    trajectory_adata_pre : Optional alternate pre-integration object for
                           trajectory conservation.
    pseudotime_key  : Column in adata_pre.obs containing pseudotime values.
    organism        : 'human' or 'mouse' for cell-cycle conservation.
    n_isolated      : Max batches per label for isolation metrics.
    lisi_subsample  : Percentage (1–100) of the working LISI subset to score.
                      Default 100 means: score the full 50k heavy subset.
    verbose         : Print scib progress messages.
    metric_subsample_n : Max cells for the general metrics subset. Default 100k.
    heavy_metric_subsample_n : Max cells for the heavy metrics subset.
                               Default 50k.
    random_state    : Seed for deterministic metric subsampling.

    Returns
    -------
    dict with scalar float values for all 14 metrics (np.nan if not computable).
    Error messages are stored under <metric>_error or <metric>_note keys.
    """
    _validate_inputs(
        adata_int,
        batch_key=batch_key,
        label_key=label_key,
        emb_key=emb_key,
        conn_key=conn_key,
        output_type=output_type,
    )

    return compute_all_scib_metrics(
        adata_int,
        adata_pre,
        batch_key=batch_key,
        label_key=label_key,
        cluster_key=cluster_key,
        emb_key=emb_key,
        conn_key=conn_key,
        dist_key=dist_key,
        neighbors_uns_key=neighbors_uns_key,
        output_type=output_type,
        organism=organism,
        n_isolated=n_isolated,
        subsample=lisi_subsample,
        compute_trajectory=compute_trajectory,
        trajectory_adata_pre=trajectory_adata_pre,
        pseudotime_key=pseudotime_key,
        verbose=verbose,
        metric_subsample_n=metric_subsample_n,
        heavy_metric_subsample_n=heavy_metric_subsample_n,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    adata_int,
    *,
    batch_key: str,
    label_key: str,
    emb_key: Optional[str],
    conn_key: Optional[str],
    output_type: str,
) -> None:
    """Raise informative errors for clearly wrong inputs before touching scib."""
    valid_types = {"knn", "embed", "full"}
    if output_type not in valid_types:
        raise ValueError(
            f"output_type must be one of {valid_types}, got '{output_type}'"
        )

    missing_obs = [k for k in (batch_key, label_key) if k not in adata_int.obs]
    if missing_obs:
        raise KeyError(f"Keys missing from adata_int.obs: {missing_obs}")

    if output_type == "knn":
        if conn_key is None:
            raise ValueError(
                "output_type='knn' requires conn_key to be set. "
                "Pass the obsp key that holds the corrected connectivities."
            )
        if conn_key not in adata_int.obsp:
            raise KeyError(
                f"conn_key '{conn_key}' not found in adata_int.obsp. "
                f"Available keys: {list(adata_int.obsp.keys())}"
            )

    if output_type == "embed":
        if emb_key is None:
            raise ValueError("output_type='embed' requires emb_key to be set.")
        if emb_key not in adata_int.obsm:
            raise KeyError(
                f"emb_key '{emb_key}' not found in adata_int.obsm. "
                f"Available keys: {list(adata_int.obsm.keys())}"
            )

    if output_type == "full" and emb_key is not None and emb_key not in adata_int.obsm:
        raise KeyError(
            f"emb_key '{emb_key}' not found in adata_int.obsm. "
            f"Available keys: {list(adata_int.obsm.keys())}"
        )


# ---------------------------------------------------------------------------
# Convenience: per-method output_type registry
# ---------------------------------------------------------------------------

METHOD_OUTPUT_TYPES: Dict[str, OutputType] = {
    "bbknn": "knn",
    "harmony": "embed",
    "scvi": "embed",
    "scanvi": "embed",
    "scgen": "embed",
    "liger": "embed",
    "fastmnn": "embed",
    "scanorama": "embed",
    "desc": "embed",
    "seurat_rpca": "embed",
    "seurat": "embed",
    "combat": "full",
    "mnn": "full",
}


def get_output_type(method_name: str) -> OutputType:
    """
    Return the scIB output_type for a given integration method name.
    Falls back to 'embed' if the method is not in the registry.
    """
    return METHOD_OUTPUT_TYPES.get(method_name.lower().replace("-", "_"), "embed")
