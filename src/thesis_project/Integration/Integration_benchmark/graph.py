"""
graph.py
--------
Graph construction utilities for single-cell RNA-seq integration benchmarking.

These helpers provide a safe, namespaced layer over Scanpy so multiple method
runs can coexist in a single AnnData object without colliding in ``.uns``,
``.obsm`` or ``.obsp``.
"""

from __future__ import annotations

from typing import Optional, Tuple
import copy
import warnings

import numpy as np
import scanpy as sc

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional dependency guard
    sp = None


def _require_nonempty_key(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def _validate_positive_float(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    if allow_zero:
        ok = value >= 0.0
        cond = "non-negative"
    else:
        ok = value > 0.0
        cond = "positive"
    if not ok:
        raise ValueError(f"{name} must be {cond}, got {value!r}.")
    return value


def _require_obsm_rep(ad, use_rep: str) -> None:
    if use_rep not in ad.obsm:
        raise KeyError(f"use_rep='{use_rep}' not found in ad.obsm.")

    Z = ad.obsm[use_rep]
    shape = getattr(Z, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"ad.obsm['{use_rep}'] must be 2D, got shape={shape!r}.")
    if shape[0] != ad.n_obs:
        raise ValueError(
            f"ad.obsm['{use_rep}'] has {shape[0]} rows but ad.n_obs={ad.n_obs}."
        )

    if sp is not None and sp.issparse(Z):
        if Z.data.size and not np.isfinite(Z.data).all():
            raise ValueError(f"ad.obsm['{use_rep}'] contains non-finite values.")
    else:
        arr = np.asarray(Z)
        if not np.isfinite(arr).all():
            raise ValueError(f"ad.obsm['{use_rep}'] contains non-finite values.")


def _validate_neighbors_key(ad, neighbors_key: str) -> dict:
    _require_nonempty_key("neighbors_key", neighbors_key)

    if neighbors_key not in ad.uns:
        raise KeyError(f"neighbors_key '{neighbors_key}' not found in ad.uns.")
    meta = ad.uns[neighbors_key]
    if not isinstance(meta, dict):
        raise TypeError(f"ad.uns['{neighbors_key}'] must be a dict, got {type(meta).__name__}.")

    conn_key = meta.get("connectivities_key")
    dist_key = meta.get("distances_key")
    if not conn_key or not dist_key:
        raise KeyError(
            f"ad.uns['{neighbors_key}'] must contain 'connectivities_key' and 'distances_key'."
        )
    if conn_key not in ad.obsp:
        raise KeyError(
            f"Connectivity matrix '{conn_key}' referenced by '{neighbors_key}' not found in ad.obsp."
        )
    if dist_key not in ad.obsp:
        raise KeyError(
            f"Distance matrix '{dist_key}' referenced by '{neighbors_key}' not found in ad.obsp."
        )
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# UMAP coordinate normalization
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_umap_coords(ad, umap_key: str) -> None:
    """
    Rescale UMAP coordinates to [-10, 10] on both axes in-place.

    This is a purely cosmetic transformation applied after embedding
    computation. It makes all method UMAPs occupy the same visual coordinate
    space so thesis panel figures are directly comparable without axis
    cropping or empty padding.

    Important: this step is applied after build_umap returns and before saving
    the h5ad. It does not affect any metric computation, neighbor graphs, or
    Leiden clusters — all of which are computed from the original embedding
    before normalization.
    """
    coords = np.asarray(ad.obsm[umap_key], dtype=np.float32).copy()
    for dim in range(coords.shape[1]):
        col = coords[:, dim]
        col_min, col_max = col.min(), col.max()
        span = col_max - col_min
        if span > 0:
            coords[:, dim] = 20.0 * (col - col_min) / span - 10.0
        # if span == 0 (degenerate axis), leave as-is rather than produce NaN
    ad.obsm[umap_key] = coords


# ──────────────────────────────────────────────────────────────────────────────
# Core graph-building primitives
# ──────────────────────────────────────────────────────────────────────────────

def build_neighbors(
    ad,
    use_rep: str,
    n_neighbors: int,
    key_added: str,
    random_state: int = 0,
) -> str:
    """
    Compute a k-nearest-neighbour graph and store it under a namespaced key.
    """
    _require_nonempty_key("use_rep", use_rep)
    _require_nonempty_key("key_added", key_added)
    _validate_positive_int("n_neighbors", n_neighbors)
    _require_obsm_rep(ad, use_rep)

    sc.pp.neighbors(
        ad,
        use_rep=use_rep,
        n_neighbors=int(n_neighbors),
        key_added=key_added,
        random_state=int(random_state),
    )

    # New post-check: fail early if Scanpy did not create the expected
    # namespaced graph metadata and matrices.
    meta = _validate_neighbors_key(ad, key_added)
    _ = meta

    return key_added


def build_umap(
    ad,
    neighbors_key: str,
    key_umap: str,
    min_dist: float,
    spread: float,
    random_state: int = 0,
) -> str:
    """
    Compute a UMAP embedding and store it under a namespaced obsm key.

    Scanpy always writes the embedding to ``ad.obsm['X_umap']`` and stores run
    parameters in ``ad.uns['umap']``. This function moves those outputs into
    per-run namespaced keys so repeated calls do not overwrite each other.

    After storing the embedding, coordinates are rescaled to [-10, 10] on both
    axes via _normalize_umap_coords. This is cosmetic only — it makes all
    method UMAPs visually comparable in panel figures without affecting any
    metric, graph, or clustering result.
    """
    _validate_neighbors_key(ad, neighbors_key)
    _require_nonempty_key("key_umap", key_umap)
    _validate_positive_float("min_dist", min_dist, allow_zero=True)
    _validate_positive_float("spread", spread)

    sc.tl.umap(
        ad,
        neighbors_key=neighbors_key,
        min_dist=float(min_dist),
        spread=float(spread),
        random_state=int(random_state),
        init_pos="random",
    )

    if "X_umap" not in ad.obsm:
        raise RuntimeError(
            "sc.tl.umap completed but did not write ad.obsm['X_umap'] as expected."
        )

    if key_umap != "X_umap":
        ad.obsm[key_umap] = ad.obsm["X_umap"].copy()
        del ad.obsm["X_umap"]

    # Namespace the UMAP run metadata so successive method runs do not
    # overwrite each other's parameter logs.
    umap_meta_key = f"umap_{key_umap}"
    if "umap" in ad.uns:
        ad.uns[umap_meta_key] = copy.deepcopy(ad.uns.pop("umap"))

    # Normalize coordinates to [-10, 10] for consistent visual comparison
    # across all 11 methods and the unintegrated reference.
    _normalize_umap_coords(ad, key_umap)

    return key_umap


def build_leiden(
    ad,
    neighbors_key: str,
    key_leiden: str,
    resolution: float,
    random_state: int = 0,
) -> str:
    """Run Leiden community detection and store labels under a namespaced key."""
    _validate_neighbors_key(ad, neighbors_key)
    _require_nonempty_key("key_leiden", key_leiden)
    _validate_positive_float("resolution", resolution)

    sc.tl.leiden(
        ad,
        neighbors_key=neighbors_key,
        resolution=float(resolution),
        key_added=key_leiden,
        random_state=int(random_state),
    )
    return key_leiden


# ──────────────────────────────────────────────────────────────────────────────
# BBKNN-specific namespace helper
# ──────────────────────────────────────────────────────────────────────────────

def namespace_existing_graph(
    ad,
    run_tag: str,
    n_neighbors: Optional[int] = None,
    delete_default_slots: bool = True,
) -> Tuple[str, str, str]:
    """
    Move BBKNN's default graph slots to method-specific namespaced keys.

    BBKNN writes its output to the default Scanpy slots:
        ad.obsp['connectivities'], ad.obsp['distances'], ad.uns['neighbors']

    This function copies those matrices and metadata to namespaced keys derived
    from ``run_tag`` so multiple integration methods can coexist in one AnnData
    object without overwriting each other.
    """
    _require_nonempty_key("run_tag", run_tag)
    if n_neighbors is not None:
        _validate_positive_int("n_neighbors", n_neighbors)

    conn_key = f"{run_tag}_connectivities"
    dist_key = f"{run_tag}_distances"
    neighbors_key = f"neighbors_{run_tag}"

    missing = [slot for slot in ("connectivities", "distances") if slot not in ad.obsp]
    if missing:
        raise ValueError(
            f"Expected ad.obsp {missing} after BBKNN graph construction. "
            "Ensure bbknn has been called before namespace_existing_graph()."
        )

    ad.obsp[conn_key] = ad.obsp["connectivities"].copy()
    ad.obsp[dist_key] = ad.obsp["distances"].copy()

    if "neighbors" in ad.uns and isinstance(ad.uns["neighbors"], dict):
        neighbors_meta = copy.deepcopy(ad.uns["neighbors"])
    else:
        warnings.warn(
            f"ad.uns['neighbors'] not found after BBKNN for run '{run_tag}'. "
            "Creating minimal metadata. Some downstream tools may fail.",
            RuntimeWarning,
            stacklevel=2,
        )
        neighbors_meta = {"params": {}}

    if "params" not in neighbors_meta or not isinstance(neighbors_meta["params"], dict):
        neighbors_meta["params"] = {}

    if n_neighbors is not None:
        neighbors_meta["params"]["n_neighbors"] = int(n_neighbors)
    elif "n_neighbors" not in neighbors_meta.get("params", {}):
        warnings.warn(
            f"n_neighbors not available in uns['neighbors']['params'] for run '{run_tag}' and "
            "was not passed explicitly. Tools such as sc.tl.diffmap that read this field may fail. "
            "Pass n_neighbors= to namespace_existing_graph() when such tools are used.",
            RuntimeWarning,
            stacklevel=2,
        )

    neighbors_meta["connectivities_key"] = conn_key
    neighbors_meta["distances_key"] = dist_key
    ad.uns[neighbors_key] = neighbors_meta

    if delete_default_slots:
        del ad.obsp["connectivities"]
        del ad.obsp["distances"]
        if "neighbors" in ad.uns:
            del ad.uns["neighbors"]

    return neighbors_key, conn_key, dist_key