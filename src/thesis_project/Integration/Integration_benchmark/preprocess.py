from __future__ import annotations

from typing import Optional

import numpy as np
import scanpy as sc

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - optional dependency guard
    sp = None


def _validate_clip_strategy(strategy: str) -> str:
    valid = {"absolute", "quantile"}
    if strategy not in valid:
        raise ValueError(f"Unsupported clip strategy '{strategy}'. Expected one of {sorted(valid)}.")
    return strategy


def _validate_quantile(q: float) -> float:
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q!r}.")
    return float(q)


def _resolve_clip_value(*, values: np.ndarray, max_value: Optional[float], strategy: str, q: float) -> Optional[float]:
    """Return the upper clipping bound, or None when no clipping is required."""
    strategy = _validate_clip_strategy(strategy)
    q = _validate_quantile(q)

    if strategy == "absolute":
        if max_value is None:
            return None
        vmax = float(max_value)
    else:
        if values.size == 0:
            return None
        vmax = float(np.quantile(values, q))

    if vmax < 0:
        raise ValueError(f"Clipping bound must be non-negative, got {vmax!r}.")
    if vmax == 0:
        return None
    return vmax


def clip_nonzero_inplace(
    X,
    max_value: Optional[float] = None,
    strategy: str = "absolute",
    q: float = 0.999,
) -> None:
    """
    Clip only non-zero entries of a matrix in place.

    This helper is sparse-safe and keeps the clipping semantics consistent across
    sparse and dense matrices: when ``strategy='quantile'``, the quantile is
    computed over non-zero values only.
    """
    strategy = _validate_clip_strategy(strategy)
    q = _validate_quantile(q)

    if strategy == "absolute" and max_value is None:
        return

    if sp is not None and sp.issparse(X):
        data = X.data
        if data.size == 0:
            return
        vmax = _resolve_clip_value(values=np.asarray(data), max_value=max_value, strategy=strategy, q=q)
        if vmax is None:
            return
        data[data > vmax] = vmax
        return

    arr = np.asarray(X)
    if arr.size == 0:
        return

    nonzero = arr[arr != 0]
    vmax = _resolve_clip_value(values=np.asarray(nonzero), max_value=max_value, strategy=strategy, q=q)
    if vmax is None:
        return

    np.minimum(arr, vmax, out=arr)

    # np.asarray(X) returns a view for ndarray inputs; if it produced a copy for
    # another dense array-like object, write the result back explicitly.
    if arr is not X:
        try:
            X[...] = arr
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(
                "clip_nonzero_inplace() could not write clipped values back to the input object. "
                "Pass a writable numpy array or sparse matrix."
            ) from exc



def run_pca(
    ad,
    n_pcs: int = 50,
    scale: bool = False,
    scale_max_value: float = 10.0,
    solver: str = "arpack",
):
    """Shared PCA preparation step for methods that consume ``X_pca`` as input."""
    if int(n_pcs) <= 0:
        raise ValueError(f"n_pcs must be positive, got {n_pcs!r}.")
    if scale and float(scale_max_value) <= 0:
        raise ValueError(f"scale_max_value must be positive when scaling is enabled, got {scale_max_value!r}.")

    if scale:
        sc.pp.scale(ad, max_value=float(scale_max_value))
    sc.pp.pca(ad, n_comps=int(n_pcs), svd_solver=solver)
    return ad



def run_pca_sparse_safe(
    ad,
    n_pcs: int = 50,
    clip_nonzero_max: Optional[float] = 10.0,
    clip_strategy: str = "absolute",
    clip_quantile: float = 0.999,
    solver: str = "arpack",
):
    """Sparse-safe PCA preparation used by workflows that clip extreme values before PCA."""
    if int(n_pcs) <= 0:
        raise ValueError(f"n_pcs must be positive, got {n_pcs!r}.")

    if clip_nonzero_max is not None or clip_strategy == "quantile":
        clip_nonzero_inplace(
            ad.X,
            max_value=clip_nonzero_max,
            strategy=clip_strategy,
            q=clip_quantile,
        )
    sc.pp.pca(ad, n_comps=int(n_pcs), svd_solver=solver)
    return ad
