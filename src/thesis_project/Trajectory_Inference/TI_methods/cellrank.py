"""
cellrank.py
===========
\
Trajectory inference adapter using CellRank *without RNA velocity*.

This implementation uses CellRank's CytoTRACEKernel to compute a CytoTRACE-derived pseudotime
and a biased transition matrix based on the kNN graph.

Outputs (returned to the benchmark runner):
- pseudotime: pandas.Series (index=cell ids) in [0, 1]
- edge_list:  pandas.DataFrame with columns [source, target, weight, directed]
- topology_matrix: pandas.DataFrame (groups x groups) adjacency weights
- branch_labels: pandas.Series (index=cell ids) - by default equals group labels
- extras: dict with CellRank parameters actually used

The method runner (TI_benchmark.method_runner) is expected to take care of preprocessing
(HVG selection, scaling), geometry (PCA, neighbors), root selection, bootstrapping, I/O, and plotting.

CellRank API notes (official docs):
- CytoTRACEKernel.compute_cytotrace() writes `ct_score`, `ct_pseudotime`, etc. into `adata.obs`.
- CytoTRACEKernel.compute_transition_matrix() accepts:
  threshold_scheme, frac_to_keep, b, nu, check_irreducibility, n_jobs, backend, show_progress_bar, ...

Fix history
-----------
2026-03-19 (cytotrace_layer default — Fix 1):
  Changed default cytotrace_layer from 'X' to 'counts' in both ti_runner()
  and add_method_args() to match the bash script and prevent silent fallback
  to log1p-normalised HVG data if the CLI argument is ever omitted.

  Rationale: CytoTRACE requires raw integer counts to compute gene-detection
  diversity. The previous default 'X' pointed to the log1p-normalised,
  HVG-subsetted expression matrix, which collapses the diversity signal and
  produces degenerate ct_pseudotime ≈ 0 for all cells (dark purple UMAP).
  Using layer='counts' is consistent with the original CytoTRACE algorithm
  (Gulati et al. 2020) and differs from the CellRank tutorial which uses
  layer='Ms' (kNN-imputed). The deviation is documented: raw counts provide
  a binary detected/undetected signal appropriate for the gene-count diversity
  metric, and avoid introducing a scVelo imputation dependency.

Reference:
- Official CellRank repo: https://github.com/scverse/cellrank
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# IMPORTANT: make TI_benchmark importable (same pattern as via.py)
# ---------------------------------------------------------------------
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # .../TI_methods
TI_ROOT = THIS_DIR.parent                          # .../Trajectory_Inference
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.method_runner import (  # noqa: E402
    run_benchmark,
    build_arg_parser,
    args_to_run_spec,
)
from TI_benchmark.shared_types import TIOutput  # noqa: E402

# ---------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------
from typing import Any, Dict, Optional, Tuple
import argparse
import inspect
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import anndata as ad
import scanpy as sc  # noqa: F401 (runner computes neighbors; import keeps env consistent)

logger = logging.getLogger(__name__)
METHOD_NAME = "cellrank"


# ---------------------------------------------------------------------
# Global fix: allow writing nullable strings BEFORE run_benchmark() exports
# ---------------------------------------------------------------------
try:
    # anndata>=0.11
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass


# ---------------------------------------------------------------------
# Safe CellRank import (avoid shadowing by this file name "cellrank.py")
# ---------------------------------------------------------------------
def _import_cellrank_package() -> Any:
    """
    Import the *installed* CellRank package, not this adapter file (cellrank.py).

    Why needed:
      this file is named `cellrank.py`, which can shadow the real `cellrank` package.
    """
    import importlib

    this_file = Path(__file__).resolve()
    this_dir = this_file.parent.resolve()

    # If something already imported "cellrank" and it's actually this file, remove it.
    mod = sys.modules.get("cellrank")
    if mod is not None:
        mod_file = getattr(mod, "__file__", None)
        if mod_file is not None:
            try:
                if Path(mod_file).resolve() == this_file:
                    del sys.modules["cellrank"]
            except Exception:
                pass

    # Temporarily remove THIS_DIR from sys.path so `import cellrank` can't resolve to ./cellrank.py
    path_backup = list(sys.path)
    try:
        filtered: list[str] = []
        for p in sys.path:
            try:
                if Path(p).resolve() == this_dir:
                    continue
            except Exception:
                # keep weird/non-path entries as-is
                pass
            filtered.append(p)
        sys.path = filtered

        cr = importlib.import_module("cellrank")
    finally:
        sys.path = path_backup

    cr_file = getattr(cr, "__file__", None)
    if cr_file is not None:
        try:
            if Path(cr_file).resolve() == this_file:
                raise ImportError(
                    "Failed to import the real CellRank package because this adapter "
                    f"({this_file}) is shadowing it. Imported module file: {cr_file}"
                )
        except Exception:
            pass

    return cr


def _load_cytotrace_kernel_class() -> Tuple[Any, Any]:
    """
    Return (cellrank_module, CytoTRACEKernel_class) across CellRank releases.
    """
    import importlib

    cr = _import_cellrank_package()

    # Preferred: official public module path
    candidates = [
        "cellrank.kernels",
        # fallback (older/internal layouts)
        "cellrank.tl.kernels",
    ]

    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, "CytoTRACEKernel", None)
            if cls is not None:
                return cr, cls
        except Exception:
            continue

    # Last resort: maybe `cr.kernels` exists but isn't imported into __init__
    kernels = getattr(cr, "kernels", None)
    if kernels is not None and hasattr(kernels, "CytoTRACEKernel"):
        return cr, getattr(kernels, "CytoTRACEKernel")

    raise ImportError(
        "Could not locate CytoTRACEKernel in your CellRank installation. "
        "Make sure you installed scverse CellRank (pip/conda) and that it's importable."
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _set_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def _to_dense_1d(x: Any) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr).reshape(-1)


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.all(finite):
        raise RuntimeError(
            f"Pseudotime contains {np.size(x) - int(np.sum(finite))} non-finite values."
        )
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def _build_membership_matrix(group_codes: np.ndarray, n_groups: int) -> sp.csr_matrix:
    if np.any(group_codes < 0):
        raise ValueError("group_codes contains -1 (missing category). Cannot aggregate.")
    n_cells = int(group_codes.shape[0])
    data = np.ones(n_cells, dtype=float)
    row = np.arange(n_cells, dtype=int)
    col = group_codes.astype(int, copy=False)
    return sp.csr_matrix((data, (row, col)), shape=(n_cells, n_groups))


def _aggregate_group_transitions(
    T: sp.spmatrix,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    if not sp.issparse(T):
        T = sp.csr_matrix(T)
    T = T.tocsr()

    M = _build_membership_matrix(group_codes, n_groups)  # n_cells x n_groups
    group_sizes = np.asarray(M.sum(axis=0)).reshape(-1)  # n_groups
    if np.any(group_sizes <= 0):
        raise RuntimeError(f"Found empty groups (sizes={group_sizes}).")

    G_sum = (M.T @ T) @ M  # n_groups x n_groups
    G_sum = G_sum.tocsr()

    inv_sizes = 1.0 / group_sizes
    P = G_sum.multiply(inv_sizes[:, None]).toarray().astype(float)

    P[~np.isfinite(P)] = 0.0
    P[P < 0] = 0.0
    P[P > 1] = 1.0
    return P


def _symmetrize(mat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return 0.5 * (mat + mat.T)
    if mode == "max":
        return np.maximum(mat, mat.T)
    if mode == "min":
        return np.minimum(mat, mat.T)
    raise ValueError(f"Unknown symmetrize mode: {mode!r}")


def _group_median_pseudotime(
    pseudotime: np.ndarray, group_codes: np.ndarray, n_groups: int
) -> np.ndarray:
    med = np.zeros(n_groups, dtype=float)
    for g in range(n_groups):
        vals = pseudotime[group_codes == g]
        med[g] = float(np.median(vals)) if vals.size else math.nan
    if not np.all(np.isfinite(med)):
        raise RuntimeError("Non-finite group pseudotime medians encountered.")
    return med


def _topology_from_similarity_mst(
    groups: list[str],
    sim: np.ndarray,
    group_pt: np.ndarray,
    directed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(groups)
    if sim.shape != (n, n):
        raise ValueError(f"Expected sim matrix shape {(n, n)}, found {sim.shape}")

    sim = sim.copy().astype(float)
    sim[~np.isfinite(sim)] = 0.0
    np.fill_diagonal(sim, 0.0)

    # similarity -> distance
    dist = 1.0 - sim
    dist[dist < 0] = 0.0
    np.fill_diagonal(dist, 0.0)

    mst = minimum_spanning_tree(csr_matrix(dist)).tocoo()

    edges = []
    adj = np.zeros((n, n), dtype=float)

    for i, j, d in zip(mst.row, mst.col, mst.data):
        i = int(i)
        j = int(j)
        w = 1.0 - float(d)

        if directed:
            src, tgt = (i, j) if group_pt[i] <= group_pt[j] else (j, i)
            edges.append((groups[src], groups[tgt], float(w), True))
            adj[src, tgt] = float(w)
        else:
            edges.append((groups[i], groups[j], float(w), False))
            edges.append((groups[j], groups[i], float(w), False))
            adj[i, j] = float(w)
            adj[j, i] = float(w)

    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight", "directed"])
    adj_df = pd.DataFrame(adj, index=groups, columns=groups)
    return edges_df, adj_df


def _safe_filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to those accepted by callable_obj (helps across CellRank releases).
    """
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


# ---------------------------------------------------------------------
# TI runner
# ---------------------------------------------------------------------
def ti_runner(
    adata: ad.AnnData,
    root_cell_id: str,
    seed: int,
    *,
    bootstrap_index: Optional[int] = None,
    group_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    cellrank_kernel: str = "cytotrace",
    cytotrace_layer: str = "counts",       # Fix 1: was "X" — raw counts required for CytoTRACE
    cytotrace_use_raw: bool = False,
    cytotrace_n_genes: int = 200,
    cytotrace_aggregation: str = "mean",
    threshold_scheme: str = "hard",
    frac_to_keep: float = 0.3,
    b: float = 10.0,
    nu: float = 0.5,
    n_jobs: int = 1,
    orient_to_root: bool = True,
    graph_mode: str = "mst",
    symmetrize_mode: str = "mean",
    graph_directed: bool = True,
    **_: Any,
) -> TIOutput:
    """
    Run CellRank (no velocity) on the provided AnnData.

    Note on cytotrace_layer default:
      Defaults to 'counts' (raw integer counts) rather than the CellRank
      tutorial default 'Ms' (kNN-imputed). Raw counts are required for the
      gene-detection diversity score and are consistent with Gulati et al. 2020.
      CytoTRACE must receive the full gene panel (all ~20k genes) — the HVG-
      subsetted working adata collapses the diversity signal. Pass adata_full
      (adata_metrics in method_runner) to this function for CellRank.
    """
    # Ensure this is enabled inside the run too (harmless if already True)
    try:
        ad.settings.allow_write_nullable_strings = True
    except Exception:
        pass

    _set_seeds(seed)

    # Determine group key
    gk = group_key or cluster_key
    if gk is None:
        for cand in ("cluster_annotation", "clusters", "cluster", "louvain", "leiden"):
            if cand in adata.obs:
                gk = cand
                break
    if gk is None:
        raise ValueError("cellrank.ti_runner requires `group_key` (cluster labels).")
    if gk not in adata.obs:
        raise KeyError(f"`group_key={gk}` not found in adata.obs columns.")

    obs_names_str = adata.obs_names.astype(str)
    if str(root_cell_id) not in set(obs_names_str):
        raise KeyError(f"Root cell id {root_cell_id!r} not found in adata.obs_names.")

    kernel_type = str(cellrank_kernel).lower()
    if kernel_type != "cytotrace":
        raise NotImplementedError(
            "This adapter supports only `--cellrank-kernel cytotrace` (no velocity)."
        )

    # Import CellRank safely
    cr, CytoTRACEKernel = _load_cytotrace_kernel_class()
    cr_version = getattr(cr, "__version__", None)

    # Choose layer safely — raise rather than silently fall back to 'X'
    # to prevent degenerate CytoTRACE scores on log1p-normalised data.
    layer = str(cytotrace_layer)
    if layer != "X" and layer not in adata.layers:
        available = list(adata.layers.keys())
        raise RuntimeError(
            f"CytoTRACE layer '{layer}' not found in adata.layers. "
            f"Available layers: {available}. "
            "CytoTRACE requires raw integer counts (layer='counts'). "
            "Ensure method_runner passes adata_full (full gene panel) to CellRank, "
            "not the HVG-subsetted working adata."
        )

    # If use_raw requested but raw missing and layer != X, temporarily set raw
    raw_backup = adata.raw
    raw_set = False
    if bool(cytotrace_use_raw) and raw_backup is None and layer != "X":
        tmp = ad.AnnData(X=adata.layers[layer], obs=adata.obs.copy(), var=adata.var.copy())
        adata.raw = tmp
        raw_set = True

    try:
        ctk = CytoTRACEKernel(adata)

        # compute_cytotrace kwargs (official: layer, aggregation, use_raw, n_genes)
        # Note: official CellRank tutorial default is layer='Ms' (kNN-imputed).
        # We use layer='counts' (raw integers) per Gulati et al. 2020 original CytoTRACE.
        cyt_kwargs = dict(
            layer=layer,
            aggregation=str(cytotrace_aggregation),
            use_raw=bool(cytotrace_use_raw),
            n_genes=int(cytotrace_n_genes),
        )
        cyt_kwargs = _safe_filter_kwargs(ctk.compute_cytotrace, cyt_kwargs)
        ctk.compute_cytotrace(**cyt_kwargs)

        # compute_transition_matrix — all parameters at official defaults
        tm_kwargs = dict(
            threshold_scheme=str(threshold_scheme),
            frac_to_keep=float(frac_to_keep),
            b=float(b),
            nu=float(nu),
            n_jobs=int(n_jobs),
        )
        tm_kwargs = _safe_filter_kwargs(ctk.compute_transition_matrix, tm_kwargs)
        ctk.compute_transition_matrix(**tm_kwargs)

    finally:
        if raw_set:
            adata.raw = raw_backup

    if "ct_pseudotime" not in adata.obs:
        raise RuntimeError("Expected CellRank to write `adata.obs['ct_pseudotime']`.")

    pt = _to_dense_1d(adata.obs["ct_pseudotime"]).astype(float)
    pt = _minmax01(pt)

    if bool(orient_to_root):
        root_idx = int(np.where(obs_names_str == str(root_cell_id))[0][0])
        if pt[root_idx] > 0.5:
            pt = 1.0 - pt

    pseudotime = pd.Series(pt, index=adata.obs_names, name="pseudotime")

    groups_cat = adata.obs[gk].astype("category")
    groups = [str(x) for x in groups_cat.cat.categories.tolist()]
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups in obs['{gk}']; got {len(groups)}.")
    group_codes = groups_cat.cat.codes.to_numpy()
    n_groups = int(len(groups))

    T = getattr(ctk, "transition_matrix", None)
    if T is None:
        raise RuntimeError("CellRank kernel did not create `transition_matrix`.")

    # group transitions
    P = _aggregate_group_transitions(T, group_codes, n_groups)
    group_pt = _group_median_pseudotime(pt, group_codes, n_groups)

    if str(graph_mode) != "mst":
        raise NotImplementedError("Only `graph_mode='mst'` is implemented.")

    sim = _symmetrize(P, mode=str(symmetrize_mode))
    edge_list, topology_matrix = _topology_from_similarity_mst(
        groups=groups, sim=sim, group_pt=group_pt, directed=bool(graph_directed)
    )

    branch_labels = pd.Series(
        groups_cat.astype(str).to_numpy(), index=adata.obs_names, name="branch"
    )

    extras = dict(
        cellrank_version=cr_version,
        cellrank_kernel=kernel_type,
        cytotrace_layer_used=layer,
        cytotrace_use_raw=bool(cytotrace_use_raw),
        cytotrace_n_genes=int(cytotrace_n_genes),
        cytotrace_aggregation=str(cytotrace_aggregation),
        threshold_scheme=str(threshold_scheme),
        frac_to_keep=float(frac_to_keep),
        b=float(b),
        nu=float(nu),
        n_jobs=int(n_jobs),
        orient_to_root=bool(orient_to_root),
        graph_mode=str(graph_mode),
        symmetrize_mode=str(symmetrize_mode),
        graph_directed=bool(graph_directed),
        bootstrap_index=int(bootstrap_index) if bootstrap_index is not None else None,
        group_key=str(gk),
    )

    return TIOutput(
        pseudotime=pseudotime,
        topology_matrix=topology_matrix,
        edge_list=edge_list,
        branch_labels=branch_labels,
        method_name=METHOD_NAME,
        extras=extras,
    )


# ---------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------
def add_method_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("CellRank (no-velocity) parameters")

    g.add_argument(
        "--cellrank-kernel",
        default="cytotrace",
        choices=["cytotrace"],
        help="Which CellRank kernel to use (this adapter supports only 'cytotrace' in no-velocity mode).",
    )

    g.add_argument(
        "--cellrank-cytotrace-layer",
        default="counts",      # Fix 1: was "X" — must match ti_runner default and bash script
        help=(
            "Layer for CytoTRACE score computation: key in adata.layers or 'X' for adata.X. "
            "Defaults to 'counts' (raw integers) per Gulati et al. 2020. "
            "Note: CellRank tutorial uses 'Ms' (kNN-imputed); we use raw counts to avoid "
            "a scVelo dependency and to preserve binary gene-detection signal."
        ),
    )
    g.add_argument(
        "--cellrank-cytotrace-n-genes",
        type=int,
        default=200,
        help="Number of top positively correlated genes for CytoTRACE score.",
    )
    g.add_argument(
        "--cellrank-cytotrace-aggregation",
        default="mean",
        choices=["mean", "median", "hmean", "gmean"],
        help="Aggregation of top-correlating genes (CytoTRACEKernel).",
    )
    g.add_argument(
        "--cellrank-cytotrace-use-raw",
        action="store_true",
        help="Use adata.raw for CytoTRACE computation.",
    )

    g.add_argument(
        "--cellrank-threshold-scheme",
        default="hard",
        choices=["hard", "soft"],
        help="Threshold scheme for transition matrix biasing.",
    )
    g.add_argument(
        "--cellrank-frac-to-keep",
        type=float,
        default=0.3,
        help="Fraction of neighbors to keep when thresholding transitions.",
    )
    g.add_argument(
        "--cellrank-b",
        type=float,
        default=10.0,
        help="Sigmoid steepness parameter 'b' for soft thresholding.",
    )
    g.add_argument(
        "--cellrank-nu",
        type=float,
        default=0.5,
        help="Bias strength 'nu' in [0,1].",
    )
    g.add_argument(
        "--cellrank-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs (n_jobs) for some CellRank computations.",
    )

    orient = g.add_mutually_exclusive_group()
    orient.add_argument(
        "--cellrank-orient-to-root",
        dest="cellrank_orient_to_root",
        action="store_true",
        help="Orient pseudotime so the selected root cell is near 0 (may invert).",
    )
    orient.add_argument(
        "--cellrank-no-orient-to-root",
        dest="cellrank_orient_to_root",
        action="store_false",
        help="Do not auto-orient pseudotime to the root cell.",
    )
    parser.set_defaults(cellrank_orient_to_root=True)

    gg = parser.add_argument_group("CellRank topology graph construction (group-level)")
    gg.add_argument(
        "--cellrank-graph-mode",
        default="mst",
        choices=["mst"],
        help="How to derive a sparse group topology from group transition probabilities.",
    )
    gg.add_argument(
        "--cellrank-symmetrize",
        default="mean",
        choices=["mean", "max", "min"],
        help="How to symmetrize group transition matrix before building MST.",
    )
    gg.add_argument(
        "--cellrank-graph-directed",
        action="store_true",
        help="Direct MST edges by increasing group median pseudotime.",
    )
    gg.add_argument(
        "--cellrank-graph-undirected",
        dest="cellrank_graph_directed",
        action="store_false",
        help="Export MST edges as undirected (store weights symmetrically).",
    )
    parser.set_defaults(cellrank_graph_directed=True)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        ad.settings.allow_write_nullable_strings = True
    except Exception:
        pass

    parser = build_arg_parser()
    add_method_args(parser)

    argv2 = list(sys.argv[1:] if argv is None else argv)
    if "--method" not in argv2:
        argv2 = ["--method", METHOD_NAME] + argv2
    args = parser.parse_args(argv2)

    spec = args_to_run_spec(args)
    spec.method_name = METHOD_NAME

    runner_extra_kwargs = dict(
        cellrank_kernel=args.cellrank_kernel,
        cytotrace_layer=args.cellrank_cytotrace_layer,
        cytotrace_use_raw=bool(args.cellrank_cytotrace_use_raw),
        cytotrace_n_genes=int(args.cellrank_cytotrace_n_genes),
        cytotrace_aggregation=str(args.cellrank_cytotrace_aggregation),
        threshold_scheme=str(args.cellrank_threshold_scheme),
        frac_to_keep=float(args.cellrank_frac_to_keep),
        b=float(args.cellrank_b),
        nu=float(args.cellrank_nu),
        n_jobs=int(args.cellrank_n_jobs),
        orient_to_root=bool(args.cellrank_orient_to_root),
        graph_mode=str(args.cellrank_graph_mode),
        symmetrize_mode=str(args.cellrank_symmetrize),
        graph_directed=bool(args.cellrank_graph_directed),
    )

    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())