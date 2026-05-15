"""
Trajectory Inference Benchmarking: Root Cell Selection (root_selection.py)

Update in this pass
-------------------
- select_root now supports an explicit cli_group_key override so that runs
  without priors (or with group_key override) can still use centroid rooting.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("root_selection.py requires anndata to be installed.") from e

from .shared_types import TaskPriors
from .utils import merge_json_shallow


def select_root_by_cell_id(
    adata: "ad.AnnData",
    root_cell_id: str,
) -> Tuple[str, Dict[str, Any]]:
    cid = str(root_cell_id)
    if cid not in adata.obs_names:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "error",
            "strategy": "explicit_cell_id",
            "reason": f"root_cell_id '{cid}' not found; falling back to first cell",
            "fallback": fb,
        }
    return cid, {"status": "ok", "strategy": "explicit_cell_id", "root_cell_id": cid}


def select_root_by_group(
    adata: "ad.AnnData",
    root_group: str,
    *,
    group_key: str,
    rep_key: str = "X_pca",
    k: int = 1,
    random_state: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    if group_key not in adata.obs.columns:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "error",
            "strategy": "group_centroid",
            "reason": f"group_key '{group_key}' missing",
            "fallback": fb,
        }

    if rep_key not in adata.obsm:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "error",
            "strategy": "group_centroid",
            "reason": f"rep_key '{rep_key}' missing",
            "fallback": fb,
        }

    group_mask = adata.obs[group_key].astype(str) == str(root_group)
    group_idx = np.where(group_mask.values)[0]
    n_group = int(group_idx.size)

    if n_group == 0:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "error",
            "strategy": "group_centroid",
            "reason": f"root_group '{root_group}' not found in adata.obs['{group_key}']",
            "fallback": fb,
        }

    X = np.asarray(adata.obsm[rep_key], dtype=float)
    Xg = X[group_idx]
    centroid = Xg.mean(axis=0)

    dists = np.linalg.norm(Xg - centroid[None, :], axis=1)
    order = np.argsort(dists, kind="mergesort")  # deterministic tie-breaking

    k_req = int(k)
    if k_req < 1:
        k_req = 1
    k_clamped = min(k_req, n_group)
    if k_req != k_clamped:
        warnings.warn(
            f"select_root_by_group: requested k={k_req} but group size is {n_group}; using k={k_clamped}.",
            stacklevel=2,
        )

    chosen_local = int(order[k_clamped - 1])
    chosen_idx = int(group_idx[chosen_local])
    root_cell = str(adata.obs_names[chosen_idx])

    return root_cell, {
        "status": "ok",
        "strategy": "group_centroid",
        "root_group": str(root_group),
        "group_key": group_key,
        "rep_key": rep_key,
        "n_group_cells": int(n_group),
        "k": int(k_clamped),
        "dist_to_centroid": float(dists[chosen_local]),
        "root_cell_id": root_cell,
        "random_state": int(random_state),
    }


def select_root_by_early_markers(
    adata: "ad.AnnData",
    priors: TaskPriors,
    *,
    layer: Optional[str] = None,
    bottom_quantile: float = 0.1,
) -> Tuple[str, Dict[str, Any]]:
    from .metrics import gene_set_score  # local import to avoid circular deps

    if priors.root is None:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "skipped",
            "strategy": "early_markers",
            "reason": "no root priors",
            "fallback": fb,
        }

    early_names = list(priors.root.early_marker_programs)
    if not early_names:
        early_names = [p.name for p in priors.marker_programs if p.expected_direction == "negative"]
    if not early_names:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "skipped",
            "strategy": "early_markers",
            "reason": "no early marker programs",
            "fallback": fb,
        }

    all_genes = []
    for name in early_names:
        for p in priors.marker_programs:
            if p.name == name:
                all_genes.extend(list(p.genes))
    all_genes = list(dict.fromkeys(all_genes))  # stable unique

    if not all_genes:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "skipped",
            "strategy": "early_markers",
            "reason": "no genes in early programs",
            "fallback": fb,
        }

    try:
        score, rep = gene_set_score(adata, all_genes, layer=layer, zscore_per_gene=True, allow_partial=True)
    except Exception as e:
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "error",
            "strategy": "early_markers",
            "reason": f"gene_set_score failed: {e}",
            "fallback": fb,
        }

    sv = pd.to_numeric(score, errors="coerce").astype(float).values
    finite = np.isfinite(sv)
    if not finite.any():
        fb = str(adata.obs_names[0])
        return fb, {
            "status": "error",
            "strategy": "early_markers",
            "reason": "no finite marker scores",
            "fallback": fb,
        }

    thr = float(np.nanquantile(sv[finite], float(bottom_quantile)))
    tail_idx = np.where(sv <= thr)[0]
    chosen_idx = int(tail_idx[np.nanargmin(sv[tail_idx])]) if tail_idx.size > 0 else int(np.nanargmin(sv))
    root_cell = str(adata.obs_names[chosen_idx])

    return root_cell, {
        "status": "ok",
        "strategy": "early_markers",
        "early_marker_programs": early_names,
        "n_genes_used": int(rep["n_present"]),
        "bottom_quantile": float(bottom_quantile),
        "score_threshold": float(thr),
        "root_cell_id": root_cell,
    }


def select_root(
    adata: "ad.AnnData",
    priors: TaskPriors,
    *,
    cli_group_key: Optional[str] = None,
    cli_root_group: Optional[str] = None,
    cli_root_cell_id: Optional[str] = None,
    rep_key: str = "X_pca",
    layer: Optional[str] = None,
    k_nearest_centroid: int = 1,
    random_state: int = 0,
    run_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, Dict[str, Any]]:
    summary: Dict[str, Any] = {
        "n_cells": int(adata.n_obs),
        "rep_key": rep_key,
        "random_state": int(random_state),
    }

    # 1) CLI root_cell_id
    if cli_root_cell_id is not None:
        root, sel = select_root_by_cell_id(adata, cli_root_cell_id)
        summary.update({"selection": sel, "source": "cli_root_cell_id"})
        if run_config_path is not None:
            merge_json_shallow(run_config_path, {"root_selection": summary})
        return root, summary

    # 2) priors root_cell_id
    if priors.root_cell_id is not None:
        root, sel = select_root_by_cell_id(adata, priors.root_cell_id)
        summary.update({"selection": sel, "source": "priors_root_cell_id"})
        if run_config_path is not None:
            merge_json_shallow(run_config_path, {"root_selection": summary})
        return root, summary

    effective_root_group = cli_root_group or priors.root_group
    effective_group_key = cli_group_key or priors.root_group_key or priors.group_key

    # 3/4) root_group centroid strategy
    if effective_root_group is not None:
        if effective_group_key is None:
            warnings.warn(
                "root_group set but no group_key/root_group_key/cli_group_key available; skipping centroid strategy",
                stacklevel=2,
            )
        else:
            k = (
                int(priors.root.k_nearest_centroid)
                if (priors.root and priors.root.k_nearest_centroid is not None)
                else int(k_nearest_centroid)
            )
            root, sel = select_root_by_group(
                adata,
                effective_root_group,
                group_key=effective_group_key,
                rep_key=rep_key,
                k=k,
                random_state=int(random_state),
            )
            summary.update({
                "selection": sel,
                "source": "cli_root_group" if cli_root_group else "priors_root_group",
                "effective_group_key": effective_group_key,
            })
            if run_config_path is not None:
                merge_json_shallow(run_config_path, {"root_selection": summary})
            return root, summary

    # 5) early markers (always returns a fallback cell_id)
    root, sel = select_root_by_early_markers(adata, priors, layer=layer)
    summary.update({"selection": sel, "source": "early_markers"})
    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"root_selection": summary})
    return root, summary


__all__ = [
    "select_root_by_cell_id",
    "select_root_by_group",
    "select_root_by_early_markers",
    "select_root",
]