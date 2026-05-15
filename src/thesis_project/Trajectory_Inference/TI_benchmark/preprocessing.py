"""
Trajectory Inference Benchmarking: Preprocessing (preprocessing.py)

Gene filtering, normalization, and log-transform pipeline for TI benchmarking.

Contract guarantees
-------------------
- preprocess_adata() provides correct type overloads for return_report True/False.
- run_config.json is updated via atomic merge_json_shallow when run_config_path is provided.
- Bootstrap replicate preprocessing should NOT write to run_config.json (caller responsibility).
- Cell-type subsetting + label replacement are handled here (include/exclude/replace_labels_json),
  because method_runner forwards those arguments to preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("preprocessing.py requires anndata to be installed.") from e

try:
    import scanpy as sc
except Exception as e:  # pragma: no cover
    raise ImportError("preprocessing.py requires scanpy to be installed.") from e

from .utils import is_sparse, merge_json_shallow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_variance(X: Any) -> np.ndarray:
    """Numerically stable per-gene variance (densify sparse to avoid cancellation)."""
    if is_sparse(X):
        Xd = np.asarray(X.toarray(), dtype=float)
    else:
        Xd = np.asarray(X, dtype=float)
    return Xd.var(axis=0, ddof=0)


def _parse_replace_labels_json(s: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Parse replace_labels_json into a dict[str,str].
    Returns None if s is None/empty.
    Raises ValueError on invalid JSON or invalid mapping.
    """
    if s is None:
        return None
    s2 = str(s).strip()
    if not s2:
        return None

    try:
        obj = json.loads(s2)
    except Exception as e:
        raise ValueError(f"replace_labels_json is not valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("replace_labels_json must decode to a JSON object (dict).")

    out: Dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("replace_labels_json must map string -> string only.")
        out[k] = v
    return out if out else None


def apply_label_replacements(
    adata: "ad.AnnData",
    *,
    key: str,
    mapping: Dict[str, str],
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    """
    Replace values in adata.obs[key] according to mapping.
    Works for string/category columns; preserves as category if possible.
    """
    if key not in adata.obs.columns:
        raise KeyError(f"apply_label_replacements: obs column not found: '{key}'")

    s = adata.obs[key]
    was_category = pd.api.types.is_categorical_dtype(s)

    s_str = s.astype("string")
    before_counts = s_str.value_counts(dropna=False).to_dict()

    # replace operates on strings (NA-safe)
    s_repl = s_str.replace(mapping)

    after_counts = s_repl.value_counts(dropna=False).to_dict()
    n_changed = int((s_str != s_repl).sum())

    # restore dtype
    if was_category:
        adata.obs[key] = pd.Categorical(s_repl)
    else:
        adata.obs[key] = s_repl.astype("string")

    return adata, {
        "step": "replace_labels",
        "key": str(key),
        "n_changed": n_changed,
        "mapping": dict(mapping),
        "counts_before_top": dict(list(before_counts.items())[:20]),
        "counts_after_top": dict(list(after_counts.items())[:20]),
    }


def subset_obs_include_exclude(
    adata: "ad.AnnData",
    *,
    include_key: Optional[str],
    include_values: Optional[List[str]],
    exclude_key: Optional[str],
    exclude_values: Optional[List[str]],
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    """
    Subset cells by include/exclude filters.

    Semantics
    ---------
    - If include_key and include_values are provided:
        keep only cells where obs[include_key] is in include_values.
    - If exclude_key and exclude_values are provided:
        drop cells where obs[exclude_key] is in exclude_values.
    - Both can be active; include is applied first, then exclude.

    Notes
    -----
    - Values are compared as strings (robust to categories).
    - Returns a copied AnnData (important to avoid view pitfalls).
    """
    n0 = int(adata.n_obs)
    report: Dict[str, Any] = {
        "step": "subset_cells",
        "n_cells_before": n0,
        "include": None,
        "exclude": None,
    }

    mask = np.ones(n0, dtype=bool)

    if include_key is not None and include_values:
        if include_key not in adata.obs.columns:
            raise KeyError(f"include_key not found in adata.obs: '{include_key}'")
        vals = set(str(v) for v in include_values)
        s = adata.obs[include_key].astype("string")
        m_inc = s.isin(list(vals)).to_numpy()
        mask &= m_inc
        report["include"] = {
            "key": str(include_key),
            "n_values": int(len(vals)),
            "values": list(vals),
            "n_kept_after_include": int(mask.sum()),
        }

    if exclude_key is not None and exclude_values:
        if exclude_key not in adata.obs.columns:
            raise KeyError(f"exclude_key not found in adata.obs: '{exclude_key}'")
        vals = set(str(v) for v in exclude_values)
        s = adata.obs[exclude_key].astype("string")
        m_exc = s.isin(list(vals)).to_numpy()
        mask &= ~m_exc
        report["exclude"] = {
            "key": str(exclude_key),
            "n_values": int(len(vals)),
            "values": list(vals),
            "n_kept_after_exclude": int(mask.sum()),
        }

    n1 = int(mask.sum())
    report["n_cells_after"] = n1
    report["n_cells_removed"] = int(n0 - n1)

    # If no filtering requested, keep as-is (but still report)
    if n1 == n0:
        return adata, report

    # Copy to avoid views and downstream surprises
    return adata[mask].copy(), report


# ---------------------------------------------------------------------------
# Gene-level preprocessing steps
# ---------------------------------------------------------------------------

def filter_min_cells(
    adata: "ad.AnnData",
    *,
    min_cells: int,
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    n_before = int(adata.n_vars)
    sc.pp.filter_genes(adata, min_cells=int(min_cells), inplace=True)
    n_after = int(adata.n_vars)
    return adata, {
        "step": "filter_min_cells",
        "min_cells": int(min_cells),
        "n_genes_before": n_before,
        "n_genes_after": n_after,
        "n_genes_removed": int(n_before - n_after),
    }


def filter_min_counts(
    adata: "ad.AnnData",
    *,
    min_counts: int,
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    n_before = int(adata.n_vars)
    sc.pp.filter_genes(adata, min_counts=int(min_counts), inplace=True)
    n_after = int(adata.n_vars)
    return adata, {
        "step": "filter_min_counts",
        "min_counts": int(min_counts),
        "n_genes_before": n_before,
        "n_genes_after": n_after,
        "n_genes_removed": int(n_before - n_after),
    }


def select_highly_variable_genes(
    adata: "ad.AnnData",
    *,
    n_top_genes: int,
    flavor: str = "seurat",
    batch_key: Optional[str] = None,
    subset: bool = False,
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    n_before = int(adata.n_vars)
    used_flavor = str(flavor)

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor=str(flavor),
            batch_key=batch_key,
            inplace=True,
        )
    except Exception as e:
        warnings.warn(
            f"highly_variable_genes(flavor='{flavor}') failed: {e}. "
            f"Falling back to variance selection.",
            stacklevel=2,
        )
        var = _safe_variance(adata.X)
        top_idx = np.argsort(var)[::-1][: int(n_top_genes)]
        mask = np.zeros(adata.n_vars, dtype=bool)
        mask[top_idx] = True
        adata.var["highly_variable"] = mask
        used_flavor = "variance_fallback"

    if "highly_variable" not in adata.var.columns:
        # Very defensive: Scanpy should always create it, but avoid KeyError.
        adata.var["highly_variable"] = False

    n_hvg = int(pd.Series(adata.var["highly_variable"].values).sum())
    if subset:
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata, {
        "step": "select_highly_variable_genes",
        "n_top_genes": int(n_top_genes),
        "flavor": used_flavor,
        "batch_key": batch_key,
        "subset": bool(subset),
        "n_genes_before": n_before,
        "n_hvg_selected": int(n_hvg),
        "n_genes_after": int(adata.n_vars),
    }


def normalize_total(
    adata: "ad.AnnData",
    *,
    target_sum: float = 1e4,
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    sc.pp.normalize_total(adata, target_sum=float(target_sum), inplace=True)
    return adata, {"step": "normalize_total", "target_sum": float(target_sum)}


def log1p_transform(adata: "ad.AnnData") -> Tuple["ad.AnnData", Dict[str, Any]]:
    sc.pp.log1p(adata)
    return adata, {"step": "log1p_transform"}


def scale_to_unit_variance(
    adata: "ad.AnnData",
    *,
    max_value: Optional[float] = 10.0,
    zero_center: bool = True,
) -> Tuple["ad.AnnData", Dict[str, Any]]:
    sc.pp.scale(adata, zero_center=bool(zero_center), max_value=max_value, copy=False)
    return adata, {
        "step": "scale_to_unit_variance",
        "zero_center": bool(zero_center),
        "max_value": max_value,
    }


# ---------------------------------------------------------------------------
# Main pipeline (typed overloads for type-safe return_report flag)
# ---------------------------------------------------------------------------

@overload
def preprocess_adata(
    adata: "ad.AnnData",
    *,
    # --- cell subsetting / label cleaning ---
    include_key: Optional[str] = None,
    include_values: Optional[List[str]] = None,
    exclude_key: Optional[str] = None,
    exclude_values: Optional[List[str]] = None,
    replace_labels_json: Optional[str] = None,
    replace_labels_key: Optional[str] = None,
    # --- preprocessing ---
    min_cells: int = 3,
    min_counts: int = 1,
    n_top_genes: int = 3000,
    normalize: bool = True,
    log1p: bool = True,
    scale: bool = False,
    target_sum: float = 1e4,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
    batch_key: Optional[str] = None,
    run_config_path: Optional[Union[str, Path]] = None,
    tables_dir: Optional[Union[str, Path]] = None,
    return_report: Literal[False] = False,
) -> "ad.AnnData": ...


@overload
def preprocess_adata(
    adata: "ad.AnnData",
    *,
    # --- cell subsetting / label cleaning ---
    include_key: Optional[str] = None,
    include_values: Optional[List[str]] = None,
    exclude_key: Optional[str] = None,
    exclude_values: Optional[List[str]] = None,
    replace_labels_json: Optional[str] = None,
    replace_labels_key: Optional[str] = None,
    # --- preprocessing ---
    min_cells: int = 3,
    min_counts: int = 1,
    n_top_genes: int = 3000,
    normalize: bool = True,
    log1p: bool = True,
    scale: bool = False,
    target_sum: float = 1e4,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
    batch_key: Optional[str] = None,
    run_config_path: Optional[Union[str, Path]] = None,
    tables_dir: Optional[Union[str, Path]] = None,
    return_report: Literal[True] = True,
) -> Tuple["ad.AnnData", Dict[str, Any]]: ...


def preprocess_adata(
    adata: "ad.AnnData",
    *,
    # --- cell subsetting / label cleaning ---
    include_key: Optional[str] = None,
    include_values: Optional[List[str]] = None,
    exclude_key: Optional[str] = None,
    exclude_values: Optional[List[str]] = None,
    replace_labels_json: Optional[str] = None,
    replace_labels_key: Optional[str] = None,
    # --- preprocessing ---
    min_cells: int = 3,
    min_counts: int = 1,
    n_top_genes: int = 3000,
    normalize: bool = True,
    log1p: bool = True,
    scale: bool = False,
    target_sum: float = 1e4,
    hvg_flavor: str = "seurat",
    hvg_subset: bool = False,
    batch_key: Optional[str] = None,
    run_config_path: Optional[Union[str, Path]] = None,
    tables_dir: Optional[Union[str, Path]] = None,
    return_report: bool = False,
) -> Union["ad.AnnData", Tuple["ad.AnnData", Dict[str, Any]]]:
    """
    Standard preprocessing pipeline for TI benchmarking.

    Order
    -----
    (A) Optional label replacement (replace_labels_json) on replace_labels_key
        (defaults to include_key if set else exclude_key if set else None)
    (B) Optional cell subsetting (include/exclude)
    (C) Gene filtering + normalize + log1p + HVG + optional scaling

    Notes
    -----
    - If run_config_path is provided, merges preprocessing report atomically into
      run_config.json under the "preprocessing" key.
    - For bootstrap replicates, callers MUST pass run_config_path=None to avoid
      overwriting base run's preprocessing log.
    """
    n_cells_start = int(adata.n_obs)
    n_genes_start = int(adata.n_vars)
    steps: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 0) Label replacements (obs)
    # ------------------------------------------------------------------
    mapping = _parse_replace_labels_json(replace_labels_json)

    # Choose which key to apply replacement on:
    # - explicit replace_labels_key wins
    # - else prefer include_key, else exclude_key
    repl_key = replace_labels_key or include_key or exclude_key

    if mapping is not None:
        if repl_key is None:
            warnings.warn(
                "replace_labels_json provided but no replace_labels_key/include_key/exclude_key "
                "was provided; skipping label replacement.",
                stacklevel=2,
            )
        else:
            adata, rep = apply_label_replacements(adata, key=str(repl_key), mapping=mapping)
            steps.append(rep)

    # ------------------------------------------------------------------
    # 1) Cell subsetting
    # ------------------------------------------------------------------
    if (include_key is not None and include_values) or (exclude_key is not None and exclude_values):
        adata, rep = subset_obs_include_exclude(
            adata,
            include_key=include_key,
            include_values=include_values,
            exclude_key=exclude_key,
            exclude_values=exclude_values,
        )
        steps.append(rep)

    # ------------------------------------------------------------------
    # 2) Gene filtering / preprocessing
    # ------------------------------------------------------------------
    if min_cells > 0:
        adata, rep = filter_min_cells(adata, min_cells=int(min_cells))
        steps.append(rep)

    if min_counts > 0:
        adata, rep = filter_min_counts(adata, min_counts=int(min_counts))
        steps.append(rep)

    if normalize:
        adata, rep = normalize_total(adata, target_sum=float(target_sum))
        steps.append(rep)

    if log1p:
        adata, rep = log1p_transform(adata)
        steps.append(rep)

    if n_top_genes > 0:
        adata, rep = select_highly_variable_genes(
            adata,
            n_top_genes=int(n_top_genes),
            flavor=str(hvg_flavor),
            batch_key=batch_key,
            subset=bool(hvg_subset),
        )
        steps.append(rep)

    if scale:
        adata, rep = scale_to_unit_variance(adata)
        steps.append(rep)

    report: Dict[str, Any] = {
        "n_cells_input": n_cells_start,
        "n_genes_input": n_genes_start,
        "n_cells_output": int(adata.n_obs),
        "n_genes_output": int(adata.n_vars),
        "steps": steps,
        "params": {
            # cell subsetting / labels
            "include_key": include_key,
            "include_values": include_values,
            "exclude_key": exclude_key,
            "exclude_values": exclude_values,
            "replace_labels_json": replace_labels_json,
            "replace_labels_key": repl_key,
            # preprocessing
            "min_cells": int(min_cells),
            "min_counts": int(min_counts),
            "n_top_genes": int(n_top_genes),
            "normalize": bool(normalize),
            "log1p": bool(log1p),
            "scale": bool(scale),
            "target_sum": float(target_sum),
            "hvg_flavor": str(hvg_flavor),
            "hvg_subset": bool(hvg_subset),
            "batch_key": batch_key,
        },
    }

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"preprocessing": report})

    if tables_dir is not None:
        td = Path(tables_dir)
        td.mkdir(parents=True, exist_ok=True)
        from .utils import write_json as _write_json  # deferred import — avoid top-level cycle
        _write_json(td / "preprocessing_report.json", report, indent=2, atomic=True)

    if return_report:
        return adata, report
    return adata


def get_preprocessing_checksum(adata: "ad.AnnData") -> str:
    genes_hash = hashlib.md5(",".join(sorted(adata.var_names.tolist())).encode()).hexdigest()
    cells_hash = hashlib.md5(",".join(sorted(adata.obs_names.tolist())).encode()).hexdigest()
    return f"genes:{genes_hash[:8]}_cells:{cells_hash[:8]}"


__all__ = [
    "apply_label_replacements",
    "subset_obs_include_exclude",
    "filter_min_cells",
    "filter_min_counts",
    "select_highly_variable_genes",
    "normalize_total",
    "log1p_transform",
    "scale_to_unit_variance",
    "preprocess_adata",
    "get_preprocessing_checksum",
]