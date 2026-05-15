# TI_benchmark/TI_methods/paga.py
"""
PAGA (+ Diffusion Pseudotime) adapter.

ti_runner(adata, root_cell_id, seed, *, bootstrap_index=None, **kwargs) -> TIOutput

Key compatibility
-----------------
- Uses neighbors_key for PAGA, DiffMap, and DPT; supports both `neighbor_key`
  and `neighbors_key` kwarg names (framework passes `neighbors_key`).
- Requires group_key/cluster_key in adata.obs.
- Robust to Scanpy returning dense or sparse PAGA connectivities.
- Guards against fragile / duplicated group categories after label cleaning.

Fix history
-----------
2026-03-18 (connectivities_tree fix):
  edge_list now uses adata.uns["paga"]["connectivities_tree"] for the
  TOPOLOGY GRAPH instead of thresholding connectivities at paga_threshold.

  Rationale:
  connectivities_tree is PAGA's own internally computed spanning tree —
  the maximally parsimonious tree derived from the full confidence matrix.
  Using connectivities + threshold=N for the edge_list was:
    (a) Arbitrary — the threshold is a display parameter, not a principled
        topology decision.
    (b) Over-connected — at threshold=0.2 it produced loopy graphs with
        2-3x more edges than expected for a tree, inconsistent with all
        other benchmarked methods which produce spanning trees.
    (c) Unfair for cross-method comparison — all other methods produce
        sparse trees; thresholded connectivities gives PAGA more edges.

  connectivities_tree is always a spanning tree (n_groups-1 edges for a
  connected graph), making PAGA directly comparable to Slingshot (MST),
  TSCAN (MST), VIA (MST), SCORPIUS (linear chain), and ElPiGraph (tree).

  Edge weights in the exported edge_list are taken from the continuous
  connectivities matrix (not the binary tree matrix), preserving graded
  confidence information in the weight column.

  paga_threshold is retained but its role is now restricted to:
    - topology_matrix heatmap display (which edges to highlight)
    - Documentation of the full confidence landscape
  It no longer controls which edges appear in edge_list.

2026-03-18 (duplicate import fix):
  Removed the redundant try/except import anndata block. The unconditional
  import at the top of the file is sufficient.

2026-03-18 (threshold default):
  paga_threshold default raised from 0.2 → 0.5 to match Wolf et al. 2019
  main-text figures. This only affects topology_matrix display since
  edge_list now uses connectivities_tree.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import anndata as ad

ad.settings.allow_write_nullable_strings = True

try:
    import scanpy as sc
except Exception as e:  # pragma: no cover
    raise ImportError("paga.py requires scanpy to be installed.") from e

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT  = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput
from TI_benchmark.method_runner import RunSpec, run_benchmark, build_arg_parser, args_to_run_spec

logger = logging.getLogger(__name__)
METHOD_NAME = "paga"


# =============================================================================
# CLI
# =============================================================================

def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("PAGA options")
    g.add_argument(
        "--paga-threshold", "--paga_threshold",
        dest="paga_threshold",
        type=float, default=0.5,
        help=(
            "Threshold for PAGA connectivity display in topology_matrix heatmap "
            "(default: 0.5, matching Wolf et al. 2019). "
            "Does NOT affect edge_list, which uses connectivities_tree."
        ),
    )
    g.add_argument(
        "--diffmap-n-comps", "--diffmap_n_comps",
        dest="diffmap_n_comps",
        type=int, default=15,
        help="n_comps passed to sc.tl.diffmap (default: 15).",
    )
    g.add_argument(
        "--dpt-n-dcs", "--dpt_n_dcs",
        dest="dpt_n_dcs",
        type=int, default=10,
        help="n_dcs passed to sc.tl.dpt, clamped to actual diffmap dims (default: 10).",
    )


def build_legacy_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)

    p.add_argument("-i", "--input",  required=True, help="Input .h5ad")
    p.add_argument("-o", "--output", required=True, help="Output run directory")

    p.add_argument("--dataset", required=True)
    p.add_argument("--task",    required=True)

    p.add_argument("--root-cell-id", "--root_cell_id", dest="root_cell_id", default=None)
    p.add_argument("--group_key",   required=True)
    p.add_argument("--root_group",  required=True)

    p.add_argument("--priors_path",  default=None)
    p.add_argument("--priors_root",  default=None)
    p.add_argument("--expression_layer", default=None)
    p.add_argument("--batch_key",    default=None)
    p.add_argument("--n_neighbors",  type=int,   default=15)
    p.add_argument("--n_pcs",        type=int,   default=30)

    p.add_argument("--n_bootstraps",            type=int,   default=20)
    p.add_argument("--bootstrap_frac",          type=float, default=0.8)
    p.add_argument("--bootstrap_seed",          type=int,   default=42)
    p.add_argument("--bootstrap_stratify_by",   default=None)
    p.add_argument("--skip_stability",          action="store_true")

    p.add_argument("--min_cells",   type=int,   default=3)
    p.add_argument("--min_counts",  type=int,   default=1)
    p.add_argument("--n_top_genes", type=int,   default=3000)
    p.add_argument("--hvg_flavor",  default="seurat")
    p.add_argument("--hvg_subset",  action="store_true")
    p.add_argument("--target_sum",  type=float, default=1e4)
    p.add_argument("--no_normalize",action="store_true")
    p.add_argument("--no_log1p",    action="store_true")
    p.add_argument("--scale",       action="store_true")
    p.add_argument("--seed",        type=int,   default=0)

    p.add_argument("--paga-threshold",  "--paga_threshold",
                   dest="paga_threshold",  type=float, default=0.5)
    p.add_argument("--diffmap-n-comps", "--diffmap_n_comps",
                   dest="diffmap_n_comps", type=int,   default=15)
    p.add_argument("--dpt-n-dcs",       "--dpt_n_dcs",
                   dest="dpt_n_dcs",      type=int,   default=10)

    # Accept-and-ignore legacy extras safely
    p.add_argument("--preprocess",       default=None)
    p.add_argument("--include_key",      default=None)
    p.add_argument("--include_values",   default=None)
    p.add_argument("--exclude_key",      default=None)
    p.add_argument("--exclude_values",   default=None)
    p.add_argument("--replace_labels_json", default=None)

    return p


# =============================================================================
# Edge extraction helpers
# =============================================================================

def _edges_from_connectivities_tree(
    tree_mat: np.ndarray,
    conn_mat: np.ndarray,
    cats: pd.Index,
) -> pd.DataFrame:
    """
    Build edge_list from PAGA's internally computed spanning tree.

    Uses connectivities_tree for STRUCTURE (which cluster pairs are connected)
    and connectivities for WEIGHTS (continuous confidence values in [0,1]).

    Parameters
    ----------
    tree_mat : Binary adjacency matrix from adata.uns["paga"]["connectivities_tree"].
               Shape (n_groups, n_groups). Entries are 0 or 1.
    conn_mat : Continuous confidence matrix from adata.uns["paga"]["connectivities"].
               Shape (n_groups, n_groups). Entries in [0, 1].
    cats     : Cluster category labels (ordered, matching matrix rows/cols).

    Returns
    -------
    pd.DataFrame with columns [source, target, weight, directed].
    """
    if tree_mat.shape != conn_mat.shape:
        raise ValueError(
            f"connectivities_tree shape {tree_mat.shape} != "
            f"connectivities shape {conn_mat.shape}"
        )

    iu = np.triu_indices(tree_mat.shape[0], k=1)   # upper triangle, no diagonal
    tree_mask = tree_mat[iu] > 0                    # where tree has an edge

    if not np.any(tree_mask):
        logger.warning(
            "connectivities_tree has no edges in the upper triangle. "
            "Falling back to empty edge_list. "
            "This may indicate a disconnected PAGA graph."
        )
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    src = cats.to_numpy()[iu[0][tree_mask]]
    tgt = cats.to_numpy()[iu[1][tree_mask]]
    ww  = conn_mat[iu][tree_mask].astype(float)     # continuous weights

    df = pd.DataFrame({
        "source":   src,
        "target":   tgt,
        "weight":   ww,
        "directed": False,
    })
    df = df.sort_values("weight", ascending=False, kind="mergesort")
    return df[["source", "target", "weight", "directed"]].reset_index(drop=True)


def _paga_edges_from_connectivities(
    conn_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Fallback: extract edges from connectivities matrix using a threshold.

    Used ONLY for topology_matrix display or if connectivities_tree is
    unavailable. NOT used for the primary edge_list in ti_runner().
    """
    thr = float(threshold)
    cats = pd.Index(conn_df.index).astype(str)
    mat  = conn_df.to_numpy(dtype=float, copy=False)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"connectivities matrix must be square; got shape={mat.shape}")

    iu   = np.triu_indices(mat.shape[0], k=1)
    w    = mat[iu]
    mask = np.isfinite(w) & (w >= thr)

    if not np.any(mask):
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    src = cats.to_numpy()[iu[0][mask]]
    tgt = cats.to_numpy()[iu[1][mask]]
    ww  = w[mask].astype(float, copy=False)

    df = pd.DataFrame({"source": src, "target": tgt, "weight": ww, "directed": False})
    df = df.sort_values("weight", ascending=False, kind="mergesort").reset_index(drop=True)
    return df[["source", "target", "weight", "directed"]]


# =============================================================================
# TI runner
# =============================================================================

def ti_runner(
    adata: "ad.AnnData",
    root_cell_id: str,
    seed: int,
    *,
    bootstrap_index: Optional[int] = None,
    neighbor_key:   Optional[str]  = None,
    neighbors_key:  Optional[str]  = None,
    group_key:      Optional[str]  = None,
    cluster_key:    Optional[str]  = None,
    paga_threshold: float = 0.5,
    diffmap_n_comps: int  = 15,
    dpt_n_dcs:      int   = 10,
    **kwargs: Any,
) -> TIOutput:
    # ── Neighbor graph validation ────────────────────────────────────────────
    nk = str(neighbor_key or neighbors_key or "neighbors")
    if nk not in adata.uns:
        raise RuntimeError(
            f"PAGA requires a pre-computed neighbor graph under adata.uns['{nk}']. "
            f"Available adata.uns keys: {list(adata.uns.keys())[:20]}"
        )

    # ── Group key validation ─────────────────────────────────────────────────
    gk = group_key or cluster_key
    if gk is None or gk not in adata.obs.columns:
        raise ValueError(
            "PAGA requires group_key/cluster_key present in adata.obs (clusters/groups)."
        )

    adata.obs[gk] = adata.obs[gk].astype("string").astype("category")
    cats = pd.Index(adata.obs[gk].cat.categories).astype(str)
    if cats.has_duplicates:
        raise ValueError(
            f"PAGA groups are not unique after label processing in obs['{gk}']. "
            "This can happen after label replacement/merging."
        )
    if len(cats) < 2:
        raise ValueError(
            f"PAGA requires at least 2 groups in obs['{gk}']; got {len(cats)}."
        )

    # ── Run PAGA ─────────────────────────────────────────────────────────────
    # Parameters: groups, neighbors_key at user-defined values.
    # All other sc.tl.paga() parameters at official defaults:
    #   use_rna_velocity=False, model='v1.2'
    sc.tl.paga(adata, groups=gk, neighbors_key=nk)

    # ── Extract connectivities matrix ────────────────────────────────────────
    conn = adata.uns["paga"]["connectivities"]
    mat  = conn.toarray() if hasattr(conn, "toarray") else np.asarray(conn)
    if mat.shape[0] != len(cats) or mat.shape[1] != len(cats):
        raise RuntimeError(
            f"Unexpected PAGA connectivities shape {mat.shape} "
            f"vs n_groups={len(cats)} (group_key='{gk}')."
        )

    conn_df = pd.DataFrame(mat, index=cats, columns=cats)

    # ── Build edge_list from connectivities_TREE (Fix 2026-03-18) ───────────
    # connectivities_tree is PAGA's own spanning tree — the maximally
    # parsimonious topology derived internally by PAGA.  This is preferable
    # to thresholding connectivities because:
    #   1. No arbitrary threshold — PAGA determines the tree structure.
    #   2. Always produces a sparse tree (n_groups-1 edges), comparable
    #      to all other benchmarked methods (Slingshot, TSCAN, VIA etc).
    #   3. Eliminates over-connectivity observed at threshold=0.2.
    # Edge weights are taken from the continuous connectivities matrix
    # (not the binary tree matrix), preserving graded confidence values.
    paga_out = adata.uns["paga"]
    if "connectivities_tree" in paga_out:
        raw_tree = paga_out["connectivities_tree"]
        tree_mat = (
            raw_tree.toarray() if hasattr(raw_tree, "toarray") else np.asarray(raw_tree)
        )
        logger.info(
            "Using connectivities_tree for edge_list: "
            "%d non-zero entries in upper triangle.",
            int(np.sum(np.triu(tree_mat, k=1) > 0)),
        )
        edges = _edges_from_connectivities_tree(tree_mat, mat, cats)
    else:
        # Graceful fallback for scanpy versions that do not emit connectivities_tree
        logger.warning(
            "connectivities_tree not found in adata.uns['paga'] "
            "(scanpy version may be older). "
            "Falling back to threshold-based edge extraction (threshold=%.2f).",
            float(paga_threshold),
        )
        edges = _paga_edges_from_connectivities(conn_df, threshold=float(paga_threshold))

    logger.info(
        "PAGA edge_list: %d edges from %d groups.",
        len(edges), len(cats),
    )

    # ── Root cell for DPT ────────────────────────────────────────────────────
    obs_ids = adata.obs_names.astype(str)
    matches = np.where(obs_ids == str(root_cell_id))[0]
    if matches.size == 0:
        raise ValueError(
            f"root_cell_id '{root_cell_id}' not found in adata.obs_names."
        )
    adata.uns["iroot"] = int(matches[0])

    # ── Diffusion map ────────────────────────────────────────────────────────
    # Parameters at official defaults:
    #   n_comps=15, neighbors_key=nk, random_state=0
    sc.tl.diffmap(adata, n_comps=int(diffmap_n_comps), neighbors_key=nk)

    # ── DPT pseudotime ───────────────────────────────────────────────────────
    # n_dcs clamped to actual diffmap dims to prevent IndexError.
    # Other parameters at official defaults:
    #   n_branchings=0, allow_kendall_tau_shift=True
    n_dcs_eff = int(min(int(dpt_n_dcs), int(adata.obsm["X_diffmap"].shape[1])))
    sc.tl.dpt(adata, n_dcs=n_dcs_eff, neighbors_key=nk)

    pt = adata.obs["dpt_pseudotime"].copy()
    pt.index = adata.obs_names
    pt = pd.to_numeric(pt, errors="coerce").astype(float)

    if not np.isfinite(pt.values).all():
        bad = int(np.sum(~np.isfinite(pt.values)))
        raise RuntimeError(
            f"DPT pseudotime contains {bad} non-finite values (NaN/inf)."
        )

    return TIOutput(
        pseudotime=pt,
        topology_matrix=conn_df,   # full continuous connectivities (for heatmap)
        edge_list=edges,           # PAGA spanning tree (connectivities_tree)
        method_name=METHOD_NAME,
    )


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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
            root_cell_id=(
                str(a.root_cell_id) if a.root_cell_id is not None else None
            ),
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

        runner_extra_kwargs: Dict[str, Any] = {
            "group_key":       str(a.group_key),
            "cluster_key":     str(a.group_key),
            "paga_threshold":  float(a.paga_threshold),
            "diffmap_n_comps": int(a.diffmap_n_comps),
            "dpt_n_dcs":       int(a.dpt_n_dcs),
        }
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

    runner_extra_kwargs = {
        "paga_threshold":  float(a.paga_threshold),
        "diffmap_n_comps": int(a.diffmap_n_comps),
        "dpt_n_dcs":       int(a.dpt_n_dcs),
    }
    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()