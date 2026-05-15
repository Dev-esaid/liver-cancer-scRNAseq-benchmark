# TI_benchmark/TI_methods/elpigraph.py
"""
ElPiGraph (elpigraph-python) adapter.

Fixes vs previous draft
-----------------------
1) Avoids NAME COLLISION:
   This adapter file is named `elpigraph.py`. If you do `import elpigraph` at top-level,
   Python may import *this adapter file* as the `elpigraph` module, causing:
       AttributeError: module 'elpigraph' has no attribute 'computeElasticPrincipalTree'
   We therefore import the library via a guarded importer that temporarily removes this
   directory from sys.path.

2) Uses the OFFICIAL elpigraph-python API:
   - computeElasticPrincipalTree exists in elpigraph-python and returns a list of PG dicts. :contentReference[oaicite:2]{index=2}
   - The function signature does NOT accept drawAccuracyComplexity/drawEnergy kwargs
     (they are commented out in source). :contentReference[oaicite:3]{index=3}
   - utils.getProjection(...) and utils.getPseudotime(..., project=...) are the intended helpers. :contentReference[oaicite:4]{index=4}

Outputs (TIOutput)
------------------
- pseudotime: pd.Series indexed by adata.obs_names (normalized to [0, 1])
- edge_list: pd.DataFrame with columns [source, target, weight, directed] over groups
- topology_matrix: pd.DataFrame (group x group) similarity derived from tree distances
- extras: small JSON-safe dict (version/params/root_node, etc.)

Dependencies
------------
- elpigraph-python (import name: `elpigraph`)
  Install: pip install elpigraph-python
- scipy
"""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
    ad.settings.allow_write_nullable_strings = True
except Exception as e:  # pragma: no cover
    raise ImportError("elpigraph.py requires anndata to be installed.") from e

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
except Exception as e:  # pragma: no cover
    raise ImportError("elpigraph.py requires scipy to be installed.") from e

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput
from TI_benchmark.method_runner import RunSpec, run_benchmark, build_arg_parser, args_to_run_spec

logger = logging.getLogger(__name__)
METHOD_NAME = "elpigraph"


# -----------------------------------------------------------------------------
# Robust import of elpigraph-python (avoids local-file shadowing)
# -----------------------------------------------------------------------------
def _import_elpigraph_lib() -> Any:
    """
    Import the *installed* elpigraph-python package as a module object.

    Critical: this adapter file is named elpigraph.py, so a plain `import elpigraph`
    can self-import this file instead of the library. We temporarily remove THIS_DIR
    (and '' if present) from sys.path to ensure we import the external package.
    """
    this_dir = str(THIS_DIR)
    orig_path = list(sys.path)

    # Remove adapter directory & empty entry (both can cause self-shadowing)
    cleaned = [p for p in sys.path if p not in ("", this_dir)]
    sys.path = cleaned
    try:
        mod = importlib.import_module("elpigraph")
    except Exception as e:
        raise ImportError(
            "Failed to import the elpigraph-python library.\n"
            "Install it with: pip install elpigraph-python\n"
            "If you *did* install it, check for shadowing by running:\n"
            "  python -c \"import elpigraph; print(elpigraph.__file__)\""
        ) from e
    finally:
        sys.path = orig_path

    # Validate we imported the right module
    mod_file = getattr(mod, "__file__", None)
    if mod_file is not None:
        try:
            if Path(mod_file).resolve() == Path(__file__).resolve():
                raise ImportError(
                    "Import shadowing detected: `import elpigraph` resolved to this adapter file.\n"
                    "The guarded importer should prevent this; if it still happens, you likely have\n"
                    "another local elpigraph.py earlier on sys.path."
                )
        except Exception:
            # If Path resolve fails for any reason, ignore.
            pass

    if not hasattr(mod, "computeElasticPrincipalTree"):
        raise ImportError(
            "Imported module 'elpigraph' does not expose computeElasticPrincipalTree.\n"
            f"Imported from: {mod_file}\n"
            "This usually means:\n"
            "  - You installed a different package named 'elpigraph' (not elpigraph-python), OR\n"
            "  - There is still a name collision on sys.path.\n"
            "Expected API (elpigraph-python) includes computeElasticPrincipalTree. "
        )

    if not hasattr(mod, "utils") or not hasattr(mod.utils, "getPseudotime"):
        raise ImportError(
            "elpigraph-python utils helpers not found (expected mod.utils.getPseudotime). "
            f"Imported from: {mod_file}"
        )

    return mod


# -----------------------------------------------------------------------------
# CLI args
# -----------------------------------------------------------------------------
def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("ElPiGraph options")

    g.add_argument(
        "--rep-key", "--rep_key",
        dest="rep_key",
        default="X_pca",
        help="adata.obsm key used to fit ElPiGraph (default: X_pca).",
    )
    g.add_argument(
        "--rep-dims", "--rep_dims",
        dest="rep_dims",
        type=int,
        default=None,
        help="Optional: use only the first N dims from the representation (e.g. 20).",
    )

    g.add_argument(
        "--elpigraph-num-nodes", "--elpigraph_num_nodes",
        dest="elpigraph_num_nodes",
        type=int,
        default=50,
        help="NumNodes for computeElasticPrincipalTree.",
    )
    g.add_argument(
        "--elpigraph-lambda", "--elpigraph_lambda",
        dest="elpigraph_lambda",
        type=float,
        default=0.01,
        help="Lambda (edge elasticity) for computeElasticPrincipalTree.",
    )
    g.add_argument(
        "--elpigraph-mu", "--elpigraph_mu",
        dest="elpigraph_mu",
        type=float,
        default=0.1,
        help="Mu (star elasticity) for computeElasticPrincipalTree.",
    )
    g.add_argument(
        "--elpigraph-trimming-radius", "--elpigraph_trimming_radius",
        dest="elpigraph_trimming_radius",
        type=float,
        default=float("inf"),
        help="TrimmingRadius for computeElasticPrincipalTree.",
    )
    g.add_argument(
        "--elpigraph-max-iter", "--elpigraph_max_iter",
        dest="elpigraph_max_iter",
        type=int,
        default=10,
        help="MaxNumberOfIterations for computeElasticPrincipalTree.",
    )
    g.add_argument(
        "--elpigraph-eps",
        dest="elpigraph_eps",
        type=float,
        default=0.01,
        help="eps stopping criterion for computeElasticPrincipalTree.",
    )
    g.add_argument(
        "--elpigraph-n-reps", "--elpigraph_n_reps",
        dest="elpigraph_n_reps",
        type=int,
        default=1,
        help="nReps for computeElasticPrincipalTree (keep 1 for speed/determinism).",
    )
    g.add_argument(
        "--elpigraph-do-pca", "--elpigraph_do_pca",
        dest="elpigraph_do_pca",
        action="store_true",
        help="If set, ElPiGraph will run PCA internally (usually FALSE when using X_pca).",
    )
    g.add_argument(
        "--elpigraph-center-data", "--elpigraph_center_data",
        dest="elpigraph_center_data",
        action="store_true",
        help="If set, ElPiGraph will center data internally.",
    )

    g.add_argument(
        "--group-graph-weight",
        dest="group_graph_weight",
        choices=("unit", "distance", "similarity"),
        default="unit",
        help=(
            "How to set edge weights in the exported *group-level* topology graph. "
            "unit=1 for every edge; distance=tree distance; similarity=1/(1+distance)."
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
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--n_pcs", type=int, default=30)

    p.add_argument("--n_bootstraps", type=int, default=20)
    p.add_argument("--bootstrap_frac", type=float, default=0.8)
    p.add_argument("--bootstrap_seed", type=int, default=42)
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

    # ElPiGraph options
    p.add_argument("--rep-key", "--rep_key", dest="rep_key", default="X_pca")
    p.add_argument("--rep-dims", "--rep_dims", dest="rep_dims", type=int, default=None)

    p.add_argument("--elpigraph-num-nodes", "--elpigraph_num_nodes", dest="elpigraph_num_nodes", type=int, default=50)
    p.add_argument("--elpigraph-lambda", "--elpigraph_lambda", dest="elpigraph_lambda", type=float, default=0.01)
    p.add_argument("--elpigraph-mu", "--elpigraph_mu", dest="elpigraph_mu", type=float, default=0.1)
    p.add_argument("--elpigraph-trimming-radius", "--elpigraph_trimming_radius", dest="elpigraph_trimming_radius",
                   type=float, default=float("inf"))
    p.add_argument("--elpigraph-max-iter", "--elpigraph_max_iter", dest="elpigraph_max_iter", type=int, default=10)
    p.add_argument("--elpigraph-eps", dest="elpigraph_eps", type=float, default=0.01)
    p.add_argument("--elpigraph-n-reps", "--elpigraph_n_reps", dest="elpigraph_n_reps", type=int, default=1)
    p.add_argument("--elpigraph-do-pca", "--elpigraph_do_pca", dest="elpigraph_do_pca", action="store_true")
    p.add_argument("--elpigraph-center-data", "--elpigraph_center_data", dest="elpigraph_center_data", action="store_true")

    p.add_argument(
        "--group-graph-weight",
        dest="group_graph_weight",
        choices=("unit", "distance", "similarity"),
        default="unit",
    )

    # Accept-and-ignore legacy extras safely
    p.add_argument("--preprocess", default=None)
    p.add_argument("--include_key", default=None)
    p.add_argument("--include_values", default=None)
    p.add_argument("--exclude_key", default=None)
    p.add_argument("--exclude_values", default=None)
    p.add_argument("--replace_labels_json", default=None)

    return p


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_float_matrix(x: Any) -> np.ndarray:
    X = np.asarray(x)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={X.shape}")
    return X.astype(float, copy=False)


def _principal_edges_and_lengths(PG: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    if "NodePositions" not in PG:
        raise RuntimeError("ElPiGraph PG dict missing 'NodePositions'.")
    if "Edges" not in PG:
        raise RuntimeError("ElPiGraph PG dict missing 'Edges'.")

    node_pos = np.asarray(PG["NodePositions"])
    if node_pos.ndim != 2:
        raise ValueError(f"NodePositions must be 2D; got shape={node_pos.shape}")
    n_nodes = int(node_pos.shape[0])

    Ed = PG["Edges"]
    edges = np.asarray(Ed[0] if isinstance(Ed, (list, tuple)) else Ed, dtype=int)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Unexpected edges shape {edges.shape}; expected (n_edges, 2).")

    # Prefer projection edge lengths if present; else compute Euclidean
    edge_len = None
    try:
        proj = PG.get("projection", {})
        cand = proj.get("edge_len", None)
        if cand is not None:
            cand = np.asarray(cand, dtype=float)
            if cand.shape[0] == edges.shape[0]:
                edge_len = cand
    except Exception:
        edge_len = None

    if edge_len is None:
        u = edges[:, 0]
        v = edges[:, 1]
        edge_len = np.linalg.norm(node_pos[u] - node_pos[v], axis=1)

    edge_len = np.asarray(edge_len, dtype=float)
    if not np.isfinite(edge_len).all():
        raise RuntimeError("Non-finite edge lengths encountered in ElPiGraph output.")
    edge_len = np.maximum(edge_len, 0.0)

    return edges, edge_len, n_nodes


def _group_rep_nodes(
    group_codes: np.ndarray,
    node_ids: np.ndarray,
    n_groups: int,
    n_nodes: int,
) -> np.ndarray:
    rep = np.empty(n_groups, dtype=int)
    for gi in range(n_groups):
        mask = group_codes == gi
        if not np.any(mask):
            raise ValueError(f"Group code {gi} has no cells; cannot create group topology.")
        counts = np.bincount(node_ids[mask], minlength=n_nodes)
        rep[gi] = int(np.argmax(counts))
    return rep


def _normalize_01(x: pd.Series) -> pd.Series:
    mn = float(np.nanmin(x.values))
    mx = float(np.nanmax(x.values))
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise RuntimeError("Pseudotime contains non-finite values.")
    if mx <= mn:
        return x * 0.0
    return (x - mn) / (mx - mn)


def _mst_edges_over_groups(
    dist_groups: np.ndarray,
    group_labels: pd.Index,
    group_medians: np.ndarray,
    weight_mode: str,
) -> pd.DataFrame:
    n = int(dist_groups.shape[0])
    if dist_groups.shape != (n, n):
        raise ValueError("dist_groups must be square.")
    if n < 2:
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    D = np.asarray(dist_groups, dtype=float)
    np.fill_diagonal(D, 0.0)

    mst = minimum_spanning_tree(csr_matrix(D)).tocoo()

    rows = []
    for i, j, w in zip(mst.row, mst.col, mst.data):
        i = int(i)
        j = int(j)
        w = float(w)

        # direct along increasing median pseudotime
        src = i
        tgt = j
        if group_medians[src] > group_medians[tgt]:
            src, tgt = tgt, src

        if weight_mode == "unit":
            ww = 1.0
        elif weight_mode == "distance":
            ww = w
        elif weight_mode == "similarity":
            ww = 1.0 / (1.0 + w) if np.isfinite(w) else 0.0
        else:
            raise ValueError(f"Unknown weight_mode='{weight_mode}'.")

        rows.append((str(group_labels[src]), str(group_labels[tgt]), float(ww), True))

    df = pd.DataFrame(rows, columns=["source", "target", "weight", "directed"])
    df = df.sort_values("weight", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
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
    elpigraph_num_nodes: int = 50,
    elpigraph_lambda: float = 0.01,
    elpigraph_mu: float = 0.1,
    elpigraph_trimming_radius: float = float("inf"),
    elpigraph_max_iter: int = 10,
    elpigraph_eps: float = 0.01,
    elpigraph_n_reps: int = 1,
    elpigraph_do_pca: bool = False,
    elpigraph_center_data: bool = False,
    group_graph_weight: str = "unit",
    **kwargs: Any,
) -> TIOutput:
    # Reproducibility
    np.random.seed(int(seed))
    random.seed(int(seed))

    epg = _import_elpigraph_lib()

    rk = str(rep_key)
    if rk not in adata.obsm:
        raise ValueError(
            f"ElPiGraph requires adata.obsm['{rk}'] as input representation. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )

    X = _as_float_matrix(adata.obsm[rk])
    if rep_dims is not None:
        d = int(rep_dims)
        if d <= 0:
            raise ValueError("rep_dims must be a positive int.")
        if d > X.shape[1]:
            raise ValueError(f"rep_dims={d} exceeds X.shape[1]={X.shape[1]}.")
        X = X[:, :d]

    # Root cell index
    obs_ids = adata.obs_names.astype(str).to_numpy()
    matches = np.where(obs_ids == str(root_cell_id))[0]
    if matches.size == 0:
        raise ValueError(f"root_cell_id '{root_cell_id}' not found in adata.obs_names.")
    root_idx = int(matches[0])

    logger.info(
        "Fitting ElPiGraph principal tree: NumNodes=%s Lambda=%s Mu=%s nReps=%s",
        elpigraph_num_nodes, elpigraph_lambda, elpigraph_mu, elpigraph_n_reps
    )

    # IMPORTANT: only pass kwargs supported by elpigraph-python signature. :contentReference[oaicite:5]{index=5}
    PG_list = epg.computeElasticPrincipalTree(
        X,
        NumNodes=int(elpigraph_num_nodes),
        Lambda=float(elpigraph_lambda),
        Mu=float(elpigraph_mu),
        TrimmingRadius=float(elpigraph_trimming_radius),
        MaxNumberOfIterations=int(elpigraph_max_iter),
        eps=float(elpigraph_eps),
        nReps=int(elpigraph_n_reps),
        Do_PCA=bool(elpigraph_do_pca),
        CenterData=bool(elpigraph_center_data),
        verbose=False,
        ShowTimer=False,
        n_cores=1,
    )

    PG = PG_list[-1] if isinstance(PG_list, (list, tuple)) else PG_list
    if not isinstance(PG, dict):
        raise RuntimeError(f"Unexpected ElPiGraph return type: {type(PG)}")

    # Projection + pseudotime (official helpers) :contentReference[oaicite:6]{index=6}
    epg.utils.getProjection(X, PG)

    if "projection" not in PG or "node_id" not in PG["projection"]:
        raise RuntimeError("ElPiGraph projection did not populate PG['projection']['node_id'].")

    node_ids = np.asarray(PG["projection"]["node_id"], dtype=int)
    if node_ids.shape[0] != X.shape[0]:
        raise RuntimeError("Projection node_id length mismatch with n_cells.")

    root_node = int(node_ids[root_idx])

    epg.utils.getPseudotime(X, PG, source=root_node, project=False)

    if "pseudotime" not in PG:
        raise RuntimeError("ElPiGraph did not populate PG['pseudotime'] after getPseudotime().")

    pt = pd.Series(np.asarray(PG["pseudotime"], dtype=float), index=adata.obs_names, name="pseudotime")
    pt = pd.to_numeric(pt, errors="coerce").astype(float)
    if not np.isfinite(pt.values).all():
        bad = int(np.sum(~np.isfinite(pt.values)))
        raise RuntimeError(f"ElPiGraph pseudotime contains {bad} non-finite values (NaN/inf).")

    pt = _normalize_01(pt)

    # Build group-level topology (nodes = clusters)
    gk = group_key or cluster_key
    if gk is None or gk not in adata.obs.columns:
        raise ValueError("ElPiGraph adapter requires group_key/cluster_key present in adata.obs.")

    adata.obs[gk] = adata.obs[gk].astype("string").astype("category")
    cats = pd.Index(adata.obs[gk].cat.categories).astype(str)
    if cats.has_duplicates:
        raise ValueError(
            f"Groups in obs['{gk}'] are not unique after label processing. "
            "This can happen after label replacement/merging."
        )
    if len(cats) < 2:
        raise ValueError(f"Need at least 2 groups in obs['{gk}']; got {len(cats)}.")

    group_codes = adata.obs[gk].cat.codes.to_numpy()
    n_groups = int(len(cats))

    # Principal graph distances (between principal nodes)
    edges, edge_len, n_nodes = _principal_edges_and_lengths(PG)

    r = np.concatenate([edges[:, 0], edges[:, 1]])
    c = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_len, edge_len])

    A = csr_matrix((data, (r, c)), shape=(n_nodes, n_nodes))

    dist_nodes = shortest_path(A, directed=False, unweighted=False)
    if not np.isfinite(dist_nodes).all():
        raise RuntimeError("Non-finite distances in principal tree shortest_path result.")

    # Representative principal node per group
    rep_nodes = _group_rep_nodes(group_codes, node_ids, n_groups=n_groups, n_nodes=n_nodes)

    # Group distance matrix
    dist_groups = dist_nodes[np.ix_(rep_nodes, rep_nodes)].astype(float)

    # Group pseudotime medians (for edge direction)
    group_medians = np.zeros(n_groups, dtype=float)
    pt_vals = pt.values
    for gi in range(n_groups):
        group_medians[gi] = float(np.median(pt_vals[group_codes == gi]))

    edge_list = _mst_edges_over_groups(
        dist_groups=dist_groups,
        group_labels=cats,
        group_medians=group_medians,
        weight_mode=str(group_graph_weight),
    )

    topo_sim = 1.0 / (1.0 + dist_groups)
    np.fill_diagonal(topo_sim, 1.0)
    topology_matrix = pd.DataFrame(topo_sim, index=cats, columns=cats)

    extras = {
        "library_module_file": getattr(epg, "__file__", None),
        "library_version": getattr(epg, "__version__", None),
        "rep_key": rk,
        "rep_dims": int(rep_dims) if rep_dims is not None else None,
        "root_node": int(root_node),
        "n_principal_nodes": int(n_nodes),
        "elpigraph_num_nodes": int(elpigraph_num_nodes),
        "elpigraph_lambda": float(elpigraph_lambda),
        "elpigraph_mu": float(elpigraph_mu),
        "elpigraph_trimming_radius": float(elpigraph_trimming_radius),
        "elpigraph_max_iter": int(elpigraph_max_iter),
        "elpigraph_eps": float(elpigraph_eps),
        "elpigraph_n_reps": int(elpigraph_n_reps),
        "elpigraph_do_pca": bool(elpigraph_do_pca),
        "elpigraph_center_data": bool(elpigraph_center_data),
        "group_graph_weight": str(group_graph_weight),
    }

    return TIOutput(
        pseudotime=pt,
        topology_matrix=topology_matrix,
        edge_list=edge_list,
        method_name=METHOD_NAME,
        extras=extras,
    )


# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
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

        runner_extra_kwargs: Dict[str, Any] = {
            "group_key": str(a.group_key),
            "cluster_key": str(a.group_key),
            "rep_key": str(a.rep_key),
            "rep_dims": int(a.rep_dims) if a.rep_dims is not None else None,
            "elpigraph_num_nodes": int(a.elpigraph_num_nodes),
            "elpigraph_lambda": float(a.elpigraph_lambda),
            "elpigraph_mu": float(a.elpigraph_mu),
            "elpigraph_trimming_radius": float(a.elpigraph_trimming_radius),
            "elpigraph_max_iter": int(a.elpigraph_max_iter),
            "elpigraph_eps": float(a.elpigraph_eps),
            "elpigraph_n_reps": int(a.elpigraph_n_reps),
            "elpigraph_do_pca": bool(a.elpigraph_do_pca),
            "elpigraph_center_data": bool(a.elpigraph_center_data),
            "group_graph_weight": str(a.group_graph_weight),
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
        "rep_key": str(a.rep_key),
        "rep_dims": int(a.rep_dims) if a.rep_dims is not None else None,
        "elpigraph_num_nodes": int(a.elpigraph_num_nodes),
        "elpigraph_lambda": float(a.elpigraph_lambda),
        "elpigraph_mu": float(a.elpigraph_mu),
        "elpigraph_trimming_radius": float(a.elpigraph_trimming_radius),
        "elpigraph_max_iter": int(a.elpigraph_max_iter),
        "elpigraph_eps": float(a.elpigraph_eps),
        "elpigraph_n_reps": int(a.elpigraph_n_reps),
        "elpigraph_do_pca": bool(a.elpigraph_do_pca),
        "elpigraph_center_data": bool(a.elpigraph_center_data),
        "group_graph_weight": str(a.group_graph_weight),
    }
    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()