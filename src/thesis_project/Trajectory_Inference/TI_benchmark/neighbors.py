"""
Trajectory Inference Benchmarking: Nearest-Neighbor Graph & UMAP (neighbors.py)

Canonical geometry preparation used by method_runner and by stability replicates.

Key compatibility guarantees (publication-grade)
------------------------------------------------
- Neighbors graph defaults to Scanpy canonical keys:
  adata.uns["neighbors"], adata.obsp["connectivities"], adata.obsp["distances"]
  This ensures sc.tl.paga/diffmap/dpt work without special handling.
- UMAP defaults to adata.obsm["X_umap"] and is NEVER deleted.
- If you request a non-default rep_key/umap_key, the result is COPIED there
  but the canonical Scanpy keys are preserved for downstream compatibility.
- All writes to run_config.json are atomic via utils.merge_json_shallow.

Fix history
-----------
2026-03-19 (validation fixes):
  1. Pre-PCA sanity check on adata.X — raises RuntimeError if non-finite;
     warns if values exceed log1p-expected range (max > 50 implies raw counts).
  2. n_neighbors clamped to min(n_neighbors, n_obs - 1) to prevent
     sc.pp.neighbors ValueError in small bootstrap replicates.
  3. metric='euclidean' now logged explicitly in run_config neighbors block.
  4. spread=1.0 now logged explicitly in run_config umap block.
  5. n_epochs logged after sc.tl.umap call (read from adata.uns['umap']['params']).
  6. use_highly_variable exposed as parameter in prepare_geometry() with
     default True for backward compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("neighbors.py requires anndata to be installed.") from e

try:
    import scanpy as sc
except Exception as e:  # pragma: no cover
    raise ImportError("neighbors.py requires scanpy to be installed.") from e

from .utils import OpResult, merge_json_shallow, ok, error

logger = logging.getLogger(__name__)

DEFAULT_N_PCS = 30
DEFAULT_N_NEIGHBORS = 15
DEFAULT_N_UMAP_COMPONENTS = 2
DEFAULT_UMAP_MIN_DIST = 0.3
DEFAULT_UMAP_SPREAD = 1.0
_LOG1P_MAX_EXPECTED = 50.0   # values above this suggest raw counts, not log1p


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_X_for_pca(
    adata: ad.AnnData,
    *,
    n_sample: int = 500,
    max_expected: float = _LOG1P_MAX_EXPECTED,
) -> Dict[str, Any]:
    """
    Sanity-check adata.X before PCA.

    Checks:
      1. X is not None.
      2. A sample of X is fully finite (no NaN / Inf).
      3. Max value is plausibly log1p-normalised (< max_expected).

    Returns a report dict. Raises RuntimeError on hard failures (non-finite).
    Logs a warning on soft failures (value range).
    """
    report: Dict[str, Any] = {"status": "ok", "warnings": [], "n_sample": n_sample}

    if adata.X is None:
        raise RuntimeError("compute_pca: adata.X is None — no expression matrix to decompose.")

    n = int(adata.n_obs)
    idx = slice(None) if n <= n_sample else slice(0, n_sample)
    X_sub = adata.X[idx]

    try:
        X_dense = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub, dtype=float)
    except Exception as e:
        raise RuntimeError(f"compute_pca: could not convert adata.X sample to dense array: {e}") from e

    # Hard check: finiteness
    if not np.isfinite(X_dense).all():
        n_bad = int((~np.isfinite(X_dense)).sum())
        raise RuntimeError(
            f"compute_pca: adata.X contains {n_bad} non-finite value(s) "
            f"(NaN or Inf) in the first {n_sample} rows. "
            "Check preprocessing — normalize_total and log1p must complete "
            "successfully before geometry."
        )

    # Soft check: value range
    x_max = float(X_dense.max())
    x_min = float(X_dense.min())
    x_mean = float(X_dense.mean())
    report["x_max_sample"] = x_max
    report["x_min_sample"] = x_min
    report["x_mean_sample"] = x_mean

    if x_max > max_expected:
        msg = (
            f"compute_pca: adata.X max={x_max:.1f} exceeds {max_expected} "
            f"(sample of {n_sample} rows). This may indicate raw counts rather "
            "than log1p-normalised data. PCA on raw counts produces "
            "highly library-size-dominated components. "
            "Verify preprocessing.log1p=True in run_config."
        )
        logger.warning(msg)
        report["warnings"].append(msg)
        report["status"] = "warn_high_values"

    # Soft check: all-zero matrix
    if x_max == 0.0:
        msg = "compute_pca: adata.X sample is all-zero. Check preprocessing."
        logger.warning(msg)
        report["warnings"].append(msg)
        report["status"] = "warn_all_zero"

    return report


def _neighbors_keys_from_uns(adata: ad.AnnData, neighbors_key: str) -> Tuple[str, str]:
    """
    Return (connectivities_key, distances_key) from adata.uns[neighbors_key],
    with robust fallbacks across scanpy versions.
    """
    if neighbors_key not in adata.uns:
        raise KeyError(f"neighbors_key '{neighbors_key}' not found in adata.uns")
    meta = adata.uns.get(neighbors_key, {})
    ck = meta.get("connectivities_key")
    dk = meta.get("distances_key")

    # Robust defaults across scanpy versions
    if not ck:
        ck = "connectivities" if neighbors_key == "neighbors" else f"{neighbors_key}_connectivities"
    if not dk:
        dk = "distances" if neighbors_key == "neighbors" else f"{neighbors_key}_distances"
    return str(ck), str(dk)


# =============================================================================
# Public API
# =============================================================================

def compute_pca(
    adata: ad.AnnData,
    *,
    n_comps: int = DEFAULT_N_PCS,
    rep_key: str = "X_pca",
    use_highly_variable: bool = True,
    random_state: int = 0,
    run_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[ad.AnnData, OpResult]:
    """
    Compute PCA and store in adata.obsm["X_pca"] (Scanpy canonical key).
    If rep_key != "X_pca", also copy to adata.obsm[rep_key] but keep "X_pca".

    Fixes applied (2026-03-19):
    - Pre-PCA sanity check: raises RuntimeError if adata.X is non-finite;
      warns if values suggest raw counts (max > 50).
    - n_comps clamped to min(n_comps, n_obs-1, n_vars-1) as before.
    - Sanity check report logged to run_config.
    """
    # ── Pre-PCA validation ──────────────────────────────────────────────────
    try:
        x_report = _validate_X_for_pca(adata, n_sample=500, max_expected=_LOG1P_MAX_EXPECTED)
    except RuntimeError as e:
        if run_config_path is not None:
            merge_json_shallow(run_config_path, {"pca": {"status": "error", "error": str(e)}})
        return adata, error(str(e))

    # ── Dimension clamping ──────────────────────────────────────────────────
    hvg_available = "highly_variable" in adata.var.columns
    use_hvg = bool(use_highly_variable and hvg_available)
    n_hvg = int(adata.var["highly_variable"].sum()) if use_hvg else int(adata.n_vars)
    n_genes_for_pca = n_hvg if use_hvg else int(adata.n_vars)

    max_allowed = min(int(n_comps), int(adata.n_obs) - 1, int(n_genes_for_pca) - 1)
    actual_n_comps = max(1, int(max_allowed))

    # ── PCA ────────────────────────────────────────────────────────────────
    sc.pp.pca(
        adata,
        n_comps=int(actual_n_comps),
        use_highly_variable=use_hvg,
        svd_solver="arpack",     # deterministic; randomised SVD varies with thread count
        random_state=int(random_state),
        copy=False,
    )

    if rep_key != "X_pca":
        adata.obsm[rep_key] = adata.obsm["X_pca"].copy()

    details: Dict[str, Any] = {
        "status": "ok",
        "pca_key_canonical": "X_pca",
        "pca_key_requested": rep_key,
        "n_comps_requested": int(n_comps),
        "n_comps_actual": int(actual_n_comps),
        "use_highly_variable": bool(use_hvg),
        "n_hvg": int(n_hvg),
        "n_genes_for_pca": int(n_genes_for_pca),
        "svd_solver": "arpack",
        "random_state": int(random_state),
        "method": "sc.pp.pca",
        "X_validation": x_report,
    }

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"pca": details})

    return adata, ok(details)


def compute_neighbors(
    adata: ad.AnnData,
    *,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    rep_key: str = "X_pca",
    key_added: Optional[str] = None,
    metric: str = "euclidean",
    random_state: int = 0,
    run_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[ad.AnnData, OpResult]:
    """
    Compute kNN graph.

    Canonical behavior:
    - key_added=None => Scanpy defaults:
      uns["neighbors"], obsp["connectivities"], obsp["distances"].

    Fixes applied (2026-03-19):
    - n_neighbors clamped to min(n_neighbors, n_obs - 1) to prevent
      ValueError in small bootstrap replicates.
    - metric exposed as parameter (default 'euclidean') and logged explicitly.
    """
    if rep_key not in adata.obsm:
        return adata, error(
            f"compute_neighbors: obsm key '{rep_key}' not found. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    # ── Clamp n_neighbors to prevent crash in small replicates ──────────────
    max_neighbors = int(adata.n_obs) - 1
    actual_n_neighbors = min(int(n_neighbors), max_neighbors)
    if actual_n_neighbors < n_neighbors:
        logger.warning(
            f"compute_neighbors: n_neighbors clamped from {n_neighbors} to "
            f"{actual_n_neighbors} (n_obs={adata.n_obs})."
        )

    sc.pp.neighbors(
        adata,
        n_neighbors=int(actual_n_neighbors),
        use_rep=rep_key,
        metric=str(metric),
        key_added=key_added,          # None => canonical Scanpy keys
        random_state=int(random_state),
        copy=False,
    )

    neighbors_key = str(key_added) if key_added is not None else "neighbors"
    try:
        conn_key, dist_key = _neighbors_keys_from_uns(adata, neighbors_key)
    except Exception:
        # Fallback (should rarely happen across scanpy versions)
        conn_key = "connectivities" if neighbors_key == "neighbors" else f"{neighbors_key}_connectivities"
        dist_key = "distances" if neighbors_key == "neighbors" else f"{neighbors_key}_distances"

    details: Dict[str, Any] = {
        "status": "ok",
        "neighbors_key": neighbors_key,
        "connectivities_key": conn_key,
        "distances_key": dist_key,
        "n_neighbors_requested": int(n_neighbors),
        "n_neighbors_actual": int(actual_n_neighbors),
        "n_neighbors_clamped": bool(actual_n_neighbors < n_neighbors),
        "rep_key": rep_key,
        "metric": str(metric),               # logged explicitly for reproducibility docs
        "random_state": int(random_state),
        "scanpy_key_added": key_added,
    }

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"neighbors": details})

    return adata, ok(details)


def compute_umap(
    adata: ad.AnnData,
    *,
    neighbors_key: str = "neighbors",
    umap_key: str = "X_umap",
    n_components: int = DEFAULT_N_UMAP_COMPONENTS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    spread: float = DEFAULT_UMAP_SPREAD,
    random_state: int = 0,
    run_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[ad.AnnData, OpResult]:
    """
    Compute UMAP embedding.

    Compatibility guarantee:
    - Scanpy always writes to obsm["X_umap"].
    - If umap_key != "X_umap", we COPY to obsm[umap_key] but KEEP "X_umap".

    Fixes applied (2026-03-19):
    - spread exposed as parameter (default 1.0) and logged explicitly.
    - n_epochs logged from adata.uns['umap']['params'] after call.
    """
    if neighbors_key not in adata.uns:
        return adata, error(
            f"compute_umap: neighbors_key '{neighbors_key}' not found in adata.uns. "
            "Run compute_neighbors first."
        )

    try:
        conn_key, _ = _neighbors_keys_from_uns(adata, neighbors_key)
    except Exception as e:
        return adata, error(f"compute_umap: failed to resolve connectivities key: {e}")

    if conn_key not in adata.obsp:
        return adata, error(
            f"compute_umap: connectivities '{conn_key}' not found in adata.obsp. "
            "Neighbor graph may be missing or corrupted."
        )

    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        n_components=int(n_components),
        min_dist=float(min_dist),
        spread=float(spread),
        random_state=int(random_state),
        copy=False,
    )

    if umap_key != "X_umap":
        if "X_umap" not in adata.obsm:
            return adata, error("compute_umap: scanpy did not produce obsm['X_umap'].")
        adata.obsm[umap_key] = adata.obsm["X_umap"].copy()

    # ── Log n_epochs actually used (UMAP chooses heuristically by n_obs) ───
    n_epochs_actual: Optional[int] = None
    try:
        umap_params = adata.uns.get("umap", {}).get("params", {})
        n_epochs_actual = int(umap_params.get("maxiter") or umap_params.get("n_epochs") or 0) or None
    except Exception:
        pass

    details: Dict[str, Any] = {
        "status": "ok",
        "neighbors_key": neighbors_key,
        "umap_key_canonical": "X_umap",
        "umap_key_requested": umap_key,
        "n_components": int(n_components),
        "min_dist": float(min_dist),
        "spread": float(spread),              # logged explicitly for reproducibility docs
        "min_dist_over_spread": round(float(min_dist) / float(spread), 4),
        "random_state": int(random_state),
        "n_epochs_actual": n_epochs_actual,   # heuristic from UMAP; None if unreadable
    }

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"umap": details})

    return adata, ok(details)


def write_umap_figures(
    adata: ad.AnnData,
    *,
    umap_key: str = "X_umap",
    color_keys: List[str],
    figures_dir: Union[str, Path],
    run_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], OpResult]:
    """
    Write UMAP scatter figures for each color_key using scanpy plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if umap_key not in adata.obsm:
        return [], error(f"write_umap_figures: umap_key '{umap_key}' not in adata.obsm.")

    # Ensure scanpy sees embedding at canonical 'X_umap'
    backup = None
    if umap_key != "X_umap":
        backup = adata.obsm.get("X_umap")
        adata.obsm["X_umap"] = adata.obsm[umap_key]

    present = [k for k in color_keys if k in adata.obs.columns or k in adata.var_names]
    missing = [k for k in color_keys if k not in present]

    written: List[str] = []
    errors: List[str] = []

    for key in present:
        try:
            fig = sc.pl.umap(adata, color=key, show=False, return_fig=True)
            out_path = fig_dir / f"umap_{key}.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(str(out_path))
        except Exception as e:
            errors.append(f"{key}: {e}")

    # Restore canonical key state
    if umap_key != "X_umap":
        if backup is None:
            adata.obsm.pop("X_umap", None)
        else:
            adata.obsm["X_umap"] = backup

    details: Dict[str, Any] = {
        "umap_key": umap_key,
        "written": written,
        "n_written": len(written),
        "missing_keys": missing,
        "errors": errors,
    }

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"umap_figures": details})

    if errors:
        return written, error(f"{len(errors)} figure(s) failed: {errors[:5]}", details)
    return written, ok(details)


def prepare_geometry(
    adata: ad.AnnData,
    *,
    n_pcs: int = DEFAULT_N_PCS,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    rep_key: str = "X_pca",
    neighbors_key_added: Optional[str] = None,
    umap_key: str = "X_umap",
    n_umap_components: int = DEFAULT_N_UMAP_COMPONENTS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    umap_spread: float = DEFAULT_UMAP_SPREAD,
    use_highly_variable: bool = True,
    metric: str = "euclidean",
    color_keys: Optional[List[str]] = None,
    figures_dir: Optional[Union[str, Path]] = None,
    random_state: int = 0,
    run_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    """
    Full geometry preparation:
    1) PCA (with pre-run adata.X sanity check)
    2) neighbors (kNN clamped, metric logged)
    3) UMAP (spread logged, n_epochs logged)
    4) optional figures

    Returns (adata, summary_dict) with explicit geometry_keys.

    Parameters added (2026-03-19):
    - umap_spread: controls UMAP point spread (default 1.0, now exposed + logged).
    - use_highly_variable: passed through to compute_pca (default True).
    - metric: kNN metric passed through to compute_neighbors (default 'euclidean').
    """
    summary: Dict[str, Any] = {}

    # 1) PCA
    adata, res_pca = compute_pca(
        adata,
        n_comps=n_pcs,
        rep_key=rep_key,
        use_highly_variable=use_highly_variable,
        random_state=int(random_state),
        run_config_path=run_config_path,
    )
    summary["pca"] = res_pca.to_dict()
    if res_pca.status == "error":
        return adata, summary

    # 2) kNN graph
    adata, res_nn = compute_neighbors(
        adata,
        n_neighbors=n_neighbors,
        rep_key=rep_key,
        key_added=neighbors_key_added,
        metric=metric,
        random_state=int(random_state),
        run_config_path=run_config_path,
    )
    summary["neighbors"] = res_nn.to_dict()
    if res_nn.status == "error":
        return adata, summary

    neighbors_key = res_nn.details.get("neighbors_key", "neighbors")
    conn_key = res_nn.details.get("connectivities_key", "connectivities")
    dist_key = res_nn.details.get("distances_key", "distances")

    # 3) UMAP
    adata, res_umap = compute_umap(
        adata,
        neighbors_key=str(neighbors_key),
        umap_key=umap_key,
        n_components=n_umap_components,
        min_dist=umap_min_dist,
        spread=umap_spread,
        random_state=int(random_state),
        run_config_path=run_config_path,
    )
    summary["umap"] = res_umap.to_dict()

    # 4) Optional figures
    if figures_dir is not None and color_keys:
        _, res_figs = write_umap_figures(
            adata,
            umap_key=umap_key,
            color_keys=color_keys,
            figures_dir=figures_dir,
            run_config_path=run_config_path,
        )
        summary["figures"] = res_figs.to_dict()

    summary["geometry_keys"] = {
        "pca_key": rep_key,
        "neighbors_key": str(neighbors_key),
        "connectivities_key": str(conn_key),
        "distances_key": str(dist_key),
        "umap_key": umap_key,
    }

    return adata, summary


__all__ = [
    "DEFAULT_N_PCS",
    "DEFAULT_N_NEIGHBORS",
    "DEFAULT_N_UMAP_COMPONENTS",
    "DEFAULT_UMAP_MIN_DIST",
    "DEFAULT_UMAP_SPREAD",
    "compute_pca",
    "compute_neighbors",
    "compute_umap",
    "write_umap_figures",
    "prepare_geometry",
]