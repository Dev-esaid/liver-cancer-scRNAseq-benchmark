from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple


def subset_and_cast_obs(
    adata,
    batch_key: str,
    label_key: str,
    exclude: Optional[Iterable[str]] = None,
) -> Tuple[Any, Tuple[str, ...]]:
    """
    Subset out excluded batches and coerce the key obs columns to string.

    Returns a copied AnnData object so downstream integrations can mutate it
    safely without touching the caller's object.
    """
    if batch_key not in adata.obs:
        raise ValueError(f"Missing obs column '{batch_key}'.")
    if label_key not in adata.obs:
        raise ValueError(f"Missing obs column '{label_key}'.")

    exclude_tuple: Tuple[str, ...] = tuple(exclude or ())

    mask = ~adata.obs[batch_key].isin(exclude_tuple)
    ad = adata[mask].copy()

    ad.obs[batch_key] = ad.obs[batch_key].astype(str)
    ad.obs[label_key] = ad.obs[label_key].astype(str)
    return ad, exclude_tuple
