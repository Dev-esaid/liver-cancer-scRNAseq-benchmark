#!/usr/bin/env python3
"""
VIA adapter hotfix for TI_benchmark.

This is a drop-in replacement for the user-provided via.py with three changes:
1) SciPy-compatible + symmetry-preserving patch of pyVIA.utils_via.get_sparse_from_igraph
2) Safe handling of zero-outdegree rows in VIA's Markov/RW2 helpers
3) Do NOT pass user group labels as VIA.labels by default; let VIA/PARC cluster internally

Notes:
- Keep true_label for annotation / root reporting.
- Preserve existing group-level benchmark outputs by aggregating the cell graph back to group_key.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("via.py requires anndata to be installed.") from e

try:
    import scipy.sparse as sp
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
except Exception as e:  # pragma: no cover
    raise ImportError("via.py requires scipy to be installed.") from e

# --- Robust import across pyVIA releases ---
try:
    import pyVIA.core as via_core  # type: ignore
except Exception:
    try:
        import VIA.core as via_core  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "via.py requires pyVIA (or VIA). Install e.g.: pip install pyVIA"
        ) from e

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput
from TI_benchmark.method_runner import (
    RunSpec,
    run_benchmark,
    build_arg_parser,
    args_to_run_spec,
)

logger = logging.getLogger(__name__)
METHOD_NAME = "via"


# =============================================================================
# Upstream hotfixes for pyVIA 0.2.x
# =============================================================================

def _ensure_nonzero_rows(A: Any) -> np.ndarray:
    """Make a dense nonnegative matrix with at least one outgoing edge per row.

    pyVIA's directed cluster graph can contain rows that sum to zero after edge
    biasing. In that case, making the node absorbing with a self-loop is the
    safest behavior and matches the intent of terminal/sink states.
    """
    M = np.asarray(A, dtype=float).copy()
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square 2D array; got shape={M.shape}")
    M[~np.isfinite(M)] = 0.0
    M[M < 0] = 0.0
    rs = M.sum(axis=1)
    zero = np.where(rs <= 0)[0]
    if zero.size:
        M[zero, zero] = 1.0
    return M


def _patch_pyvia() -> None:
    try:
        try:
            import pyVIA.utils_via as _via_utils  # type: ignore
        except Exception:
            import VIA.utils_via as _via_utils  # type: ignore

        from scipy.sparse import csr_matrix as _csr_matrix

        # ------------------------------------------------------------------
        # 1) Fix get_sparse_from_igraph for modern SciPy, while preserving the
        #    original semantics for undirected graphs (mirror edges both ways).
        # ------------------------------------------------------------------
        def _safe_get_sparse_from_igraph(graph, weight_attr: str | None = None):
            n = int(graph.vcount())
            edges = list(graph.get_edgelist())
            if len(edges) == 0:
                logger.warning(
                    "pyVIA get_sparse_from_igraph: graph has 0 edges; returning empty %dx%d sparse matrix",
                    n, n,
                )
                return _csr_matrix((n, n), dtype=float)

            try:
                weights = list(graph.es[weight_attr]) if weight_attr else [1.0] * len(edges)
            except Exception:
                weights = [1.0] * len(edges)

            if not graph.is_directed():
                rev_edges = [(v, u) for (u, v) in edges]
                edges = edges + rev_edges
                weights = list(weights) + list(weights)

            rows, cols = zip(*edges)
            rows = np.asarray(rows, dtype=int)
            cols = np.asarray(cols, dtype=int)
            weights = np.asarray(weights, dtype=float)
            return _csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=float)

        _via_utils.get_sparse_from_igraph = _safe_get_sparse_from_igraph
        if hasattr(via_core, "get_sparse_from_igraph"):
            via_core.get_sparse_from_igraph = _safe_get_sparse_from_igraph

        # ------------------------------------------------------------------
        # 2) Fix RW2 helper by ensuring zero-outdegree rows become absorbing.
        # ------------------------------------------------------------------
        if hasattr(via_core, "_rw2_walks"):
            _orig_rw2_walks = via_core._rw2_walks

            def _safe_rw2_walks(A, root, memory=0.8, weighted=True, implicit_ids=False,
                                num_walks=100, p_memory=1.0, x_lazy=0.95, alpha_teleport=0.99):
                A_safe = _ensure_nonzero_rows(A)
                return _orig_rw2_walks(
                    A_safe,
                    root=root,
                    memory=memory,
                    weighted=weighted,
                    implicit_ids=implicit_ids,
                    num_walks=num_walks,
                    p_memory=p_memory,
                    x_lazy=x_lazy,
                    alpha_teleport=alpha_teleport,
                )

            via_core._rw2_walks = _safe_rw2_walks

        # ------------------------------------------------------------------
        # 3) Replace _simulate_markov with a robust single-process version.
        #    In pyVIA 0.2.4 the original implementation hard-codes two worker
        #    processes and crashes if any worker dies; moreover its result is
        #    overwritten later by RW2 hitting times in run_subVIA.
        # ------------------------------------------------------------------
        def _safe_simulate_markov(self, A, root):
            P = _ensure_nonzero_rows(A)
            n_states = P.shape[0]
            P = P / P.sum(axis=1, keepdims=True)
            x_lazy = float(getattr(self, "x_lazy", 0.99))
            alpha = float(getattr(self, "alpha_teleport", 0.99))
            P = x_lazy * P + (1.0 - x_lazy) * np.eye(n_states)
            P = alpha * P + ((1.0 - alpha) * (1.0 / n_states) * np.ones((n_states, n_states)))
            P = P / P.sum(axis=1, keepdims=True)

            num_sim = int(getattr(self, "num_mcmc_simulations", 1300))
            n_steps = int(2 * n_states)
            rng = np.random.RandomState(int(getattr(self, "random_seed", 0)))

            hitting_array = np.full((n_states, num_sim), n_steps + 1.0, dtype=float)
            for sim_i in range(num_sim):
                cur = int(root)
                first_hit = np.full(n_states, -1, dtype=int)
                first_hit[cur] = 0
                dist_list: list[float] = []
                for step in range(n_steps):
                    nxt = int(rng.choice(np.arange(n_states), p=P[cur]))
                    dist = float(P[cur, nxt])
                    dist_list.append(1.0 / (1.0 + math.exp(dist - 1.0)))
                    cur = nxt
                    if first_hit[cur] < 0:
                        first_hit[cur] = step + 1
                csum = np.concatenate([[0.0], np.cumsum(np.asarray(dist_list, dtype=float))])
                reached = np.where(first_hit >= 0)[0]
                hitting_array[reached, sim_i] = csum[first_hit[reached]]

            out = np.zeros(n_states, dtype=float)
            for i in range(n_states):
                row = hitting_array[i, :]
                ok = row != (n_steps + 1)
                if np.any(ok):
                    perc = np.percentile(row[ok], 15) + 0.001
                    out[i] = float(np.mean(row[ok & (row <= perc)]))
                else:
                    out[i] = float(n_steps + 1)
            return out

        via_core.VIA._simulate_markov = _safe_simulate_markov

        logger.info("Applied pyVIA hotfixes: symmetric get_sparse_from_igraph, safe RW2, safe _simulate_markov")

    except Exception as e:
        logger.warning("Could not apply pyVIA hotfixes: %s", e)


_patch_pyvia()


# =============================================================================
# CLI args
# =============================================================================

def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("VIA options")

    g.add_argument(
        "--rep-key", "--rep_key",
        dest="rep_key",
        default="X_pca",
        help="adata.obsm key used as input to VIA (default: X_pca).",
    )
    g.add_argument(
        "--rep-dims", "--rep_dims",
        dest="rep_dims",
        type=int,
        default=None,
        help="Use only first N dims from representation (e.g. 20). Default: use all dims.",
    )

    g.add_argument(
        "--via-knn",
        dest="via_knn",
        type=int,
        default=30,
        help="VIA knn parameter (default: 30).",
    )
    g.add_argument(
        "--via-distance",
        dest="via_distance",
        choices=("l2", "cosine", "ip"),
        default="l2",
        help="Distance metric for VIA KNN graph construction (default: l2).",
    )
    g.add_argument(
        "--via-cluster-graph-pruning",
        dest="via_cluster_graph_pruning",
        type=float,
        default=0.15,
        help="cluster_graph_pruning in VIA (default: 0.15).",
    )
    g.add_argument(
        "--via-edgepruning-clustering-resolution",
        dest="via_edgepruning_clustering_resolution",
        type=float,
        default=0.15,
        help="edgepruning_clustering_resolution in VIA (default: 0.15).",
    )
    g.add_argument(
        "--via-preserve-disconnected",
        dest="via_preserve_disconnected",
        action="store_true",
        help="If set, preserve_disconnected=True.",
    )
    g.add_argument(
        "--via-num-threads",
        dest="via_num_threads",
        type=int,
        default=1,
        help="num_threads for VIA KNN construction (default: 1).",
    )
    g.add_argument(
        "--via-use-user-labels",
        dest="via_use_user_labels",
        action="store_true",
        help="Use obs[group_key] integer codes as VIA.labels. Default: False (recommended).",
    )
    g.add_argument(
        "--group-graph-weight",
        dest="group_graph_weight",
        choices=("unit", "distance", "similarity"),
        default="similarity",
        help=(
            "How to set edge weights in exported group-level graph. "
            "unit=1; distance=MST distance; similarity=topology_matrix similarity."
        ),
    )


def build_legacy_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("-i", "--input", required=True, help="Input .h5ad")
    p.add_argument("-o", "--output", required=True, help="Output run directory")
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--root-cell-id", "--root_cell_id", dest="root_cell_id", default=None)
    p.add_argument("--group_key", required=True, help="obs column for clusters/groups")
    p.add_argument("--root_group", required=True, help="root cluster label (string)")
    p.add_argument("--priors_path", default=None)
    p.add_argument("--priors_root", default=None)
    p.add_argument("--expression_layer", default=None)
    p.add_argument("--batch_key", default=None)
    p.add_argument("--n_neighbors", type=int, default=20)
    p.add_argument("--n_pcs", type=int, default=30)
    p.add_argument("--n_bootstraps", type=int, default=20)
    p.add_argument("--bootstrap_frac", type=float, default=0.8)
    p.add_argument("--bootstrap_seed", type=int, default=0)
    p.add_argument("--bootstrap_stratify_by", default=None)
    p.add_argument("--skip_stability", action="store_true")
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--min_counts", type=int, default=1)
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--hvg_flavor", default="seurat")
    p.add_argument("--hvg_subset", action="store_true")
    p.add_argument("--target_sum", type=float, default=1e4)
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--scale", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rep-key", "--rep_key", dest="rep_key", default="X_pca")
    p.add_argument("--rep-dims", "--rep_dims", dest="rep_dims", type=int, default=None)
    p.add_argument("--via-knn", dest="via_knn", type=int, default=30)
    p.add_argument("--via-distance", dest="via_distance", choices=("l2", "cosine", "ip"), default="l2")
    p.add_argument("--via-cluster-graph-pruning", dest="via_cluster_graph_pruning", type=float, default=0.15)
    p.add_argument("--via-edgepruning-clustering-resolution", dest="via_edgepruning_clustering_resolution", type=float, default=0.15)
    p.add_argument("--via-preserve-disconnected", dest="via_preserve_disconnected", action="store_true")
    p.add_argument("--via-num-threads", dest="via_num_threads", type=int, default=1)
    p.add_argument("--via-use-user-labels", dest="via_use_user_labels", action="store_true")
    p.add_argument("--group-graph-weight", dest="group_graph_weight", choices=("unit", "distance", "similarity"), default="similarity")
    p.add_argument("--preprocess", default=None)
    p.add_argument("--include_key", default=None)
    p.add_argument("--include_values", default=None)
    p.add_argument("--exclude_key", default=None)
    p.add_argument("--exclude_values", default=None)
    p.add_argument("--replace_labels_json", default=None)
    return p


# =============================================================================
# Helpers
# =============================================================================

def _as_dense_float_matrix(x: Any) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    X = np.asarray(x)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix; got shape={X.shape}")
    X = X.astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        bad = int(np.sum(~np.isfinite(X)))
        raise ValueError(f"Input representation contains {bad} non-finite values.")
    return X


def _normalize_01(pt: pd.Series) -> pd.Series:
    v = pt.values.astype(float, copy=False)
    mn = float(np.nanmin(v))
    mx = float(np.nanmax(v))
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise RuntimeError("Pseudotime contains non-finite values.")
    if mx <= mn:
        return pt * 0.0
    return (pt - mn) / (mx - mn)


def _safe_filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _group_similarity_from_cell_graph(G: Any, group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    if not sp.issparse(G):
        G = csr_matrix(G)
    coo = G.tocoo()
    if coo.nnz == 0:
        S = np.zeros((n_groups, n_groups), dtype=float)
        np.fill_diagonal(S, 1.0)
        return S
    gi = group_codes[coo.row]
    gj = group_codes[coo.col]
    w = np.asarray(coo.data, dtype=float)
    m = np.isfinite(w) & (gi >= 0) & (gj >= 0)
    gi = gi[m].astype(int, copy=False)
    gj = gj[m].astype(int, copy=False)
    w = w[m]
    sum_w = np.zeros((n_groups, n_groups), dtype=float)
    cnt = np.zeros((n_groups, n_groups), dtype=float)
    np.add.at(sum_w, (gi, gj), w)
    np.add.at(cnt, (gi, gj), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_w = sum_w / cnt
    mean_w[cnt == 0] = np.nan
    mean_w = 0.5 * (mean_w + mean_w.T)
    off = mean_w.copy()
    np.fill_diagonal(off, np.nan)
    finite_off = off[np.isfinite(off)]
    if finite_off.size == 0:
        S = np.zeros((n_groups, n_groups), dtype=float)
        np.fill_diagonal(S, 1.0)
        return S
    wmin = float(np.nanmin(finite_off))
    wmax = float(np.nanmax(finite_off))
    if wmin >= 0.0 and wmax <= 1.5:
        sim = np.where(np.isfinite(mean_w), mean_w, 0.0)
    else:
        sim = np.where(np.isfinite(mean_w), 1.0 / (1.0 + mean_w), 0.0)
    np.fill_diagonal(sim, 1.0)
    off2 = sim.copy()
    np.fill_diagonal(off2, np.nan)
    mx = float(np.nanmax(off2)) if np.isfinite(off2).any() else 0.0
    if mx > 0:
        sim = sim / mx
        np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def _mst_edge_list_over_groups(sim: np.ndarray, group_labels: pd.Index, group_medians: np.ndarray, weight_mode: str) -> pd.DataFrame:
    n = int(sim.shape[0])
    if n < 2:
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])
    eps = 1e-12
    dist = 1.0 / (sim + eps)
    dist = dist.astype(float)
    np.fill_diagonal(dist, 0.0)
    mst = minimum_spanning_tree(csr_matrix(dist)).tocoo()
    rows = []
    for i, j, d in zip(mst.row, mst.col, mst.data):
        i = int(i)
        j = int(j)
        d = float(d)
        src, tgt = i, j
        if group_medians[src] > group_medians[tgt]:
            src, tgt = tgt, src
        if weight_mode == "unit":
            w = 1.0
        elif weight_mode == "distance":
            w = d
        elif weight_mode == "similarity":
            w = float(sim[i, j])
        else:
            raise ValueError(f"Unknown weight_mode='{weight_mode}'.")
        rows.append((str(group_labels[src]), str(group_labels[tgt]), float(w), True))
    return pd.DataFrame(rows, columns=["source", "target", "weight", "directed"]).reset_index(drop=True)


def _raise_with_traceback(prefix: str, exc: Exception) -> None:
    tb = traceback.format_exc()
    logger.error("%s\n%s", prefix, tb)
    raise RuntimeError(f"{prefix}: {type(exc).__name__}: {exc}\n\n{tb}") from exc


# =============================================================================
# TI runner
# =============================================================================

def ti_runner(
    adata: "ad.AnnData",
    root_cell_id: str,
    seed: int,
    *,
    bootstrap_index: Optional[int] = None,
    rep_key: str = "X_pca",
    rep_dims: Optional[int] = None,
    group_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    connectivities_key: str = "connectivities",
    via_knn: int = 30,
    via_distance: str = "l2",
    via_cluster_graph_pruning: float = 0.15,
    via_edgepruning_clustering_resolution: float = 0.15,
    via_preserve_disconnected: bool = True,
    via_num_threads: int = 1,
    via_use_user_labels: bool = False,
    group_graph_weight: str = "similarity",
    **kwargs: Any,
) -> TIOutput:
    np.random.seed(int(seed))
    random.seed(int(seed))

    logger.info(
        "VIA ti_runner start | seed=%s | bootstrap_index=%s | rep_key=%s | rep_dims=%s | via_use_user_labels=%s",
        int(seed), bootstrap_index, str(rep_key), rep_dims, bool(via_use_user_labels),
    )

    rk = str(rep_key)
    if rk in adata.obsm:
        X = adata.obsm[rk]
    elif rk == "X":
        X = adata.X
    else:
        raise ValueError(
            f"VIA requires rep_key '{rk}' in adata.obsm (or rep_key='X'). "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    X = _as_dense_float_matrix(X)
    if rep_dims is not None:
        d = int(rep_dims)
        if d <= 0:
            raise ValueError("rep_dims must be a positive int.")
        if d > X.shape[1]:
            raise ValueError(f"rep_dims={d} exceeds X.shape[1]={X.shape[1]}.")
        X = X[:, :d]

    gk = group_key or cluster_key
    if gk is None or gk not in adata.obs.columns:
        raise ValueError("via.py requires group_key/cluster_key present in adata.obs.")
    adata.obs[gk] = adata.obs[gk].astype("string").astype("category")
    cats = pd.Index(adata.obs[gk].cat.categories).astype(str)
    if len(cats) < 2:
        raise ValueError(f"Need at least 2 groups in obs['{gk}']; got {len(cats)}.")
    group_codes = adata.obs[gk].cat.codes.to_numpy()
    if np.any(group_codes < 0):
        raise ValueError(f"obs['{gk}'] contains missing values; cannot build VIA labels.")
    group_codes = group_codes.astype(int, copy=False)
    n_groups = int(len(cats))

    obs_ids = adata.obs_names.astype(str).to_numpy()
    m = np.where(obs_ids == str(root_cell_id))[0]
    if m.size == 0:
        raise ValueError(f"root_cell_id '{root_cell_id}' not found in adata.obs_names.")
    root_idx = int(m[0])

    true_label = adata.obs[gk].astype(str).tolist()
    labels = group_codes if bool(via_use_user_labels) else None
    if labels is not None and n_groups < 10:
        logger.warning(
            "Using %d coarse user labels as VIA.labels. This is supported, but may make the VIA cluster graph numerically fragile. Recommended default is via_use_user_labels=False.",
            n_groups,
        )
    ncomps = X.shape[1]

    init_kwargs: Dict[str, Any] = dict(
        true_label=true_label,
        labels=labels,
        knn=int(via_knn),
        ncomps=int(ncomps),
        distance=str(via_distance),
        root_user=[root_idx],
        dataset="",
        random_seed=int(seed),
        num_threads=int(via_num_threads),
        cluster_graph_pruning=float(via_cluster_graph_pruning),
        edgepruning_clustering_resolution=float(via_edgepruning_clustering_resolution),
    )
    if bool(via_preserve_disconnected):
        init_kwargs["preserve_disconnected"] = True
    init_kwargs = _safe_filter_kwargs(via_core.VIA, init_kwargs)

    try:
        via_obj = via_core.VIA(X, **init_kwargs)
    except Exception as e:
        _raise_with_traceback("VIA construction failed", e)

    try:
        if hasattr(via_obj, "run_VIA") and callable(getattr(via_obj, "run_VIA")):
            via_obj.run_VIA()
    except Exception as e:
        _raise_with_traceback("VIA run_VIA() failed", e)

    try:
        pt_raw = None
        for attr in ("single_cell_pt_markov", "single_cell_pt", "pseudotime"):
            if hasattr(via_obj, attr):
                pt_raw = getattr(via_obj, attr)
                break
        if pt_raw is None:
            raise RuntimeError("Could not find VIA pseudotime attribute (single_cell_pt_markov / single_cell_pt / pseudotime).")
        pt_arr = np.asarray(pt_raw, dtype=float).reshape(-1)
        if pt_arr.shape[0] != adata.n_obs:
            raise RuntimeError(f"VIA pseudotime length mismatch: got {pt_arr.shape[0]} vs n_cells={adata.n_obs}")
        if not np.isfinite(pt_arr).all():
            bad = int(np.sum(~np.isfinite(pt_arr)))
            bad_idx = np.where(~np.isfinite(pt_arr))[0][:20].tolist()
            raise RuntimeError(f"VIA pseudotime contains {bad} non-finite values. First bad indices: {bad_idx}")
        pt = pd.Series(pt_arr, index=adata.obs_names, name="pseudotime")
        pt = _normalize_01(pt)
    except Exception as e:
        _raise_with_traceback("VIA pseudotime extraction failed", e)

    try:
        G_cell = getattr(via_obj, "csr_full_graph", None)
        if G_cell is None:
            G_cell = adata.obsp.get(str(connectivities_key))
        if G_cell is None:
            raise RuntimeError("Could not access VIA csr_full_graph and no scanpy connectivities available for topology.")
        sim = _group_similarity_from_cell_graph(G_cell, group_codes=group_codes, n_groups=n_groups)
        topology_matrix = pd.DataFrame(sim, index=cats, columns=cats)
        group_medians = np.zeros(n_groups, dtype=float)
        ptv = pt.values
        for gi in range(n_groups):
            group_medians[gi] = float(np.median(ptv[group_codes == gi]))
        edge_list = _mst_edge_list_over_groups(
            sim=sim,
            group_labels=cats,
            group_medians=group_medians,
            weight_mode=str(group_graph_weight),
        )
    except Exception as e:
        _raise_with_traceback("VIA topology construction failed", e)

    extras = {
        "via_knn": int(via_knn),
        "via_distance": str(via_distance),
        "via_cluster_graph_pruning": float(via_cluster_graph_pruning),
        "via_edgepruning_clustering_resolution": float(via_edgepruning_clustering_resolution),
        "via_preserve_disconnected": bool(via_preserve_disconnected),
        "via_num_threads": int(via_num_threads),
        "via_use_user_labels": bool(via_use_user_labels),
        "n_groups": int(n_groups),
        "rep_key_used": str(rep_key),
        "rep_dims_used": int(rep_dims) if rep_dims is not None else None,
        "internal_via_n_labels": int(len(set(map(int, np.asarray(via_obj.labels).tolist())))) if getattr(via_obj, "labels", None) is not None else None,
    }

    return TIOutput(
        pseudotime=pt,
        topology_matrix=topology_matrix,
        edge_list=edge_list,
        method_name=METHOD_NAME,
        extras=extras,
    )


# =============================================================================
# main
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    argv = sys.argv[1:]
    legacy_mode = any(a in argv for a in ("-i", "--input", "-o", "--output"))
    if legacy_mode:
        p = build_legacy_parser()
        a = p.parse_args()
        spec = RunSpec(
            method_name=METHOD_NAME,
            dataset_name=str(a.dataset),
            task_name=str(a.task),
            adata_path=str(a.input),
            run_dir=str(a.output),
            priors_path=a.priors_path,
            priors_root=a.priors_root,
            root_group=str(a.root_group),
            root_cell_id=str(a.root_cell_id) if a.root_cell_id is not None else None,
            expression_layer=a.expression_layer,
            batch_key=a.batch_key,
            n_pcs=int(a.n_pcs),
            n_neighbors=int(a.n_neighbors),
            n_bootstrap=int(a.n_bootstraps),
            bootstrap_frac=float(a.bootstrap_frac),
            bootstrap_seed=int(a.bootstrap_seed),
            bootstrap_stratify_by=a.bootstrap_stratify_by,
            skip_stability=bool(a.skip_stability),
            random_state=int(a.seed),
            min_cells=int(a.min_cells),
            min_counts=int(a.min_counts),
            n_top_genes=int(a.n_top_genes),
            hvg_flavor=str(a.hvg_flavor),
            hvg_subset=bool(a.hvg_subset),
            target_sum=float(a.target_sum),
            normalize=not bool(a.no_normalize),
            log1p=not bool(a.no_log1p),
            scale=bool(a.scale),
            color_keys=[str(a.group_key)],
        )
        runner_extra_kwargs: Dict[str, Any] = dict(
            group_key=str(a.group_key),
            cluster_key=str(a.group_key),
            rep_key=str(a.rep_key),
            rep_dims=int(a.rep_dims) if a.rep_dims is not None else None,
            via_knn=int(a.via_knn),
            via_distance=str(a.via_distance),
            via_cluster_graph_pruning=float(a.via_cluster_graph_pruning),
            via_edgepruning_clustering_resolution=float(a.via_edgepruning_clustering_resolution),
            via_preserve_disconnected=bool(a.via_preserve_disconnected),
            via_num_threads=int(a.via_num_threads),
            via_use_user_labels=bool(a.via_use_user_labels),
            group_graph_weight=str(a.group_graph_weight),
        )
        run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)
        return

    p = build_arg_parser()
    add_method_args(p)
    argv2 = list(argv)
    if "--method" not in argv2:
        argv2 = ["--method", METHOD_NAME] + argv2
    a = p.parse_args(argv2)
    spec = args_to_run_spec(a)
    spec.method_name = METHOD_NAME
    runner_extra_kwargs = dict(
        rep_key=str(a.rep_key),
        rep_dims=int(a.rep_dims) if a.rep_dims is not None else None,
        via_knn=int(a.via_knn),
        via_distance=str(a.via_distance),
        via_cluster_graph_pruning=float(a.via_cluster_graph_pruning),
        via_edgepruning_clustering_resolution=float(a.via_edgepruning_clustering_resolution),
        via_preserve_disconnected=bool(a.via_preserve_disconnected),
        via_num_threads=int(a.via_num_threads),
        via_use_user_labels=bool(a.via_use_user_labels),
        group_graph_weight=str(a.group_graph_weight),
    )
    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()
