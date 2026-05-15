#!/usr/bin/env python3
"""
Monocle3 adapter (R) for TI benchmarking (TIOutput contract).

Fixes (2026-02-28):
1) AnnData export nullable strings (Option A):
   - anndata.settings.allow_write_nullable_strings = True
   This must run early (before run_benchmark exports .h5ad artifacts).

2) Monocle3 input must be non-negative counts:
   - Export adata.layers['counts'] by default (or configurable layer),
     NOT scaled adata.X (which can be negative and breaks monocle3 size factors).

3) Robust pseudotime contract:
   - If monocle3 returns NA pseudotime for some cells, impute deterministically
     using nearest-neighbor in UMAP.
   - Optionally normalize to [0, 1].

4) Enforce edge_list schema + build topology_matrix for plotting/stability parity.

2026-04-06:
- Compatibility with structured traceback logging in method_runner.py:
  raise clear, specific exceptions and preserve R log path in failures.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.io import mmwrite
from scipy.spatial import cKDTree

try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput
from TI_benchmark.method_runner import run_benchmark, build_arg_parser, args_to_run_spec

logger = logging.getLogger(__name__)
METHOD_NAME = "monocle3"


# ---------------------------------------------------------------------
# CLI (method-specific)
# ---------------------------------------------------------------------
def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Monocle3 options")
    g.add_argument("--rscript", default="Rscript", help="Path to Rscript binary")
    g.add_argument(
        "--monocle3-use-partition",
        action="store_true",
        default=False,
        help="Use Monocle3 partitions when learning graph / ordering (passed to R).",
    )
    g.add_argument(
        "--monocle3-umap-key",
        default=None,
        help="Override UMAP key in adata.obsm (default: framework umap_key, else 'X_umap').",
    )
    g.add_argument(
        "--monocle3-prefer-layer",
        default="counts",
        help="Expression layer to export to Monocle3 (default: counts).",
    )
    g.add_argument(
        "--pt-normalize-01",
        action="store_true",
        help="Normalize pseudotime to [0,1] after (possible) imputation.",
    )


# ---------------------------------------------------------------------
# R runner (self-contained)
# ---------------------------------------------------------------------
def sh_quote(s: str) -> str:
    if any(c in s for c in (" ", "\t", "\n", '"', "'", "(", ")", "&", "|", ";")):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _run_rscript(
    *,
    rscript: str,
    script_path: Path,
    args: Dict[str, Any],
    log_path: Path,
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    cmd = [str(rscript), str(script_path)]
    for k, v in args.items():
        if v is None:
            continue
        cmd += [f"--{k}", str(v)]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running Rscript: %s", " ".join([sh_quote(x) for x in cmd]))
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("# Command:\n")
        fh.write(" ".join([sh_quote(x) for x in cmd]) + "\n\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)

    if p.returncode != 0:
        raise RuntimeError(f"Monocle3 R step failed (exit={p.returncode}). See log: {log_path}")


# ---------------------------------------------------------------------
# Helpers: expression export
# ---------------------------------------------------------------------
def _clip_sparse_negatives_to_zero(X: sparse.spmatrix) -> sparse.csr_matrix:
    X = X.tocsr(copy=True)
    if X.data.size == 0:
        return X
    neg = X.data < 0
    if np.any(neg):
        nneg = int(np.sum(neg))
        logger.warning("Monocle3: found %d negative stored entries; clipping to 0.", nneg)
        X.data[neg] = 0.0
        X.eliminate_zeros()
    return X


def _select_expression_matrix(adata: ad.AnnData, *, prefer_layer: str) -> sparse.csr_matrix:
    """
    Monocle3 expects non-negative counts-like data.
    Prefer adata.layers[prefer_layer] (default: 'counts'), else fall back to adata.X.
    Any negative values are clipped to 0 to prevent size factor NaNs.
    """
    if prefer_layer and prefer_layer in adata.layers:
        X = adata.layers[prefer_layer]
        logger.info("Monocle3: using layer=%r for expression export.", prefer_layer)
    else:
        X = adata.X
        logger.warning(
            "Monocle3: layer=%r not present; falling back to adata.X. "
            "If X is scaled/negative, monocle3 may fail.",
            prefer_layer,
        )

    if sparse.issparse(X):
        return _clip_sparse_negatives_to_zero(X).tocsr()

    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise RuntimeError("Monocle3 expression matrix is empty.")
    if np.nanmin(X) < 0:
        logger.warning("Monocle3: dense expression has negatives; clipping to 0.")
        X = np.maximum(X, 0.0)
    return sparse.csr_matrix(X)


def _export_mtx_bundle(adata: ad.AnnData, out_dir: Path, *, prefer_layer: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    X = _select_expression_matrix(adata, prefer_layer=prefer_layer)
    expr = X.T.tocsr()  # genes × cells

    expr_mtx = out_dir / "expr.mtx"
    mmwrite(str(expr_mtx), expr)

    genes_csv = out_dir / "genes.csv"
    cells_csv = out_dir / "cells.csv"
    pd.Series(adata.var_names.astype(str)).to_csv(genes_csv, index=False, header=False)
    pd.Series(adata.obs_names.astype(str)).to_csv(cells_csv, index=False, header=False)

    return {"expr_mtx": expr_mtx, "genes_csv": genes_csv, "cells_csv": cells_csv}


# ---------------------------------------------------------------------
# Helpers: pseudotime sanity
# ---------------------------------------------------------------------
def _normalize_01(pt: pd.Series) -> pd.Series:
    v = pd.to_numeric(pt, errors="coerce").astype(float).to_numpy()
    m = np.isfinite(v)
    if m.sum() == 0:
        return pt
    lo = float(v[m].min())
    hi = float(v[m].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pt * 0.0
    return (pt - lo) / (hi - lo)


def _impute_missing_pseudotime(pt: pd.Series, coords: np.ndarray) -> Tuple[pd.Series, int]:
    v = pd.to_numeric(pt, errors="coerce").astype(float).to_numpy()
    finite = np.isfinite(v)
    if finite.sum() == 0:
        raise RuntimeError("Monocle3 pseudotime has no finite values (cannot impute).")

    missing = ~finite
    if missing.sum() == 0:
        return pt, 0

    tree = cKDTree(coords[finite, :])
    _, nn = tree.query(coords[missing, :], k=1)
    v[missing] = v[finite][nn]
    out = pd.Series(v, index=pt.index, name=pt.name)
    return out, int(missing.sum())


# ---------------------------------------------------------------------
# Helpers: edges + topology
# ---------------------------------------------------------------------
def _ensure_edge_list_schema(edge_list: Optional[pd.DataFrame]) -> pd.DataFrame:
    if edge_list is None or edge_list.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    df = edge_list.copy()

    if "from" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"from": "source"})
    if "to" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"to": "target"})

    if not {"source", "target"}.issubset(df.columns):
        logger.warning("edges.csv missing required columns. Found=%s", list(df.columns))
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "directed" not in df.columns:
        df["directed"] = False

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)

    if df["directed"].dtype != bool:
        df["directed"] = df["directed"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"])

    df = df[(df["source"] != df["target"]) & (df["source"].ne("nan")) & (df["target"].ne("nan"))]
    df = df.reset_index(drop=True)
    return df


def _build_topology_matrix(edge_list: pd.DataFrame) -> Optional[pd.DataFrame]:
    if edge_list is None or edge_list.empty:
        return None

    nodes = sorted(set(edge_list["source"]) | set(edge_list["target"]))
    if not nodes:
        return None

    idx = {n: i for i, n in enumerate(nodes)}
    mat = np.zeros((len(nodes), len(nodes)), dtype=float)

    for s, t, w in zip(edge_list["source"], edge_list["target"], edge_list["weight"]):
        if s == t:
            continue
        i, j = idx.get(s), idx.get(t)
        if i is None or j is None:
            continue
        ww = float(w) if np.isfinite(w) else 1.0
        mat[i, j] = max(mat[i, j], ww)
        mat[j, i] = max(mat[j, i], ww)

    return pd.DataFrame(mat, index=nodes, columns=nodes)


# ---------------------------------------------------------------------
# Core TI runner
# ---------------------------------------------------------------------
def ti_runner(
    adata: ad.AnnData,
    root_cell_id: str,
    seed: int,
    *,
    bootstrap_index: Optional[int] = None,
    group_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    umap_key: Optional[str] = None,
    run_dir: Optional[str] = None,
    rscript: str = "Rscript",
    monocle3_use_partition: bool = False,
    monocle3_umap_key: Optional[str] = None,
    monocle3_prefer_layer: str = "counts",
    pt_normalize_01: bool = False,
    **kwargs: Any,
) -> TIOutput:
    gk = group_key or cluster_key
    if gk is None or gk not in adata.obs.columns:
        raise ValueError("Monocle3 requires group_key/cluster_key in adata.obs.")

    uk = monocle3_umap_key or umap_key or "X_umap"
    if uk not in adata.obsm:
        raise RuntimeError(
            f"Monocle3 requires a 2-D UMAP embedding at adata.obsm['{uk}']. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )

    umap_arr = np.asarray(adata.obsm[uk])
    if umap_arr.ndim != 2 or umap_arr.shape[1] < 2:
        raise RuntimeError(f"adata.obsm['{uk}'] must be (n_cells, >=2). Got {umap_arr.shape}")

    coords = np.asarray(umap_arr[:, :2], dtype=float)
    if not np.isfinite(coords).all():
        raise RuntimeError(f"UMAP embedding adata.obsm['{uk}'] contains non-finite values.")

    base = Path(run_dir) if run_dir is not None else Path.cwd()
    tag = "main" if bootstrap_index is None else f"bootstrap_{int(bootstrap_index)}"
    tmp_dir = base / "logs" / "tmp" / "monocle3" / tag
    tmp_dir.mkdir(parents=True, exist_ok=True)

    bundle = _export_mtx_bundle(adata, tmp_dir / "bundle", prefer_layer=str(monocle3_prefer_layer))

    meta = pd.DataFrame({"cell_id": adata.obs_names.astype(str), gk: adata.obs[gk].astype(str).values})
    meta_path = tmp_dir / "meta.csv"
    meta.to_csv(meta_path, index=False)

    umap_df = pd.DataFrame(coords, index=adata.obs_names.astype(str), columns=["UMAP1", "UMAP2"])
    umap_path = tmp_dir / "umap.csv"
    umap_df.to_csv(umap_path)

    root_id = str(root_cell_id)
    obs_ids = adata.obs_names.astype(str)
    if root_id not in set(obs_ids):
        logger.warning("root_cell_id=%r not found in this adata; falling back to first cell.", root_id)
        root_id = str(obs_ids[0])

    r_out = tmp_dir / "r_out"
    r_out.mkdir(parents=True, exist_ok=True)
    r_script = THIS_DIR / "R" / "monocle3_run.R"
    log_path = tmp_dir / "monocle3_R.log"

    use_partition_str = "TRUE" if bool(monocle3_use_partition) else "FALSE"

    _run_rscript(
        rscript=str(rscript),
        script_path=r_script,
        args={
            "expr_mtx": bundle["expr_mtx"],
            "genes_csv": bundle["genes_csv"],
            "cells_csv": bundle["cells_csv"],
            "meta_csv": meta_path,
            "umap_csv": umap_path,
            "group_key": gk,
            "root_cell_id": root_id,
            "use_partition": use_partition_str,
            "seed": int(seed),
            "out_dir": r_out,
        },
        log_path=log_path,
    )

    # Read pseudotime
    pt_path = r_out / "pseudotime.csv"
    if not pt_path.exists():
        raise RuntimeError(f"R completed but pseudotime.csv not found at: {pt_path}")

    pt_df = pd.read_csv(pt_path)
    if not {"cell_id", "pseudotime"}.issubset(pt_df.columns):
        raise RuntimeError(f"pseudotime.csv missing required columns. Found: {list(pt_df.columns)}")

    pt = pd.Series(
        pd.to_numeric(pt_df["pseudotime"], errors="coerce").astype(float).to_numpy(),
        index=pt_df["cell_id"].astype(str).to_numpy(),
        name="pseudotime",
    )

    pt = pt.reindex(adata.obs_names.astype(str))
    pt.index = adata.obs_names

    n_finite = int(np.isfinite(pt.to_numpy()).sum())
    if n_finite == 0:
        raise RuntimeError(f"Monocle3 produced no finite pseudotime values. See R log: {log_path}")

    if n_finite < adata.n_obs:
        pt, n_imputed = _impute_missing_pseudotime(pt, coords)
        logger.warning("Monocle3: imputed %d missing pseudotime values via NN in UMAP.", n_imputed)

    if pt_normalize_01:
        pt = _normalize_01(pt)
        pt.name = "pseudotime"

    if not np.isfinite(pt.values).all():
        raise RuntimeError("Monocle3 pseudotime contains non-finite values after imputation/normalization.")
    if int(pd.Series(pt.values).nunique()) < 2:
        raise RuntimeError("Monocle3 pseudotime is constant after imputation/normalization.")

    edges_path = r_out / "edges.csv"
    edge_list = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()
    edge_list = _ensure_edge_list_schema(edge_list)
    topology_matrix = _build_topology_matrix(edge_list)

    return TIOutput(
        pseudotime=pt,
        edge_list=edge_list,
        topology_matrix=topology_matrix,
        method_name=METHOD_NAME,
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    p = build_arg_parser()
    add_method_args(p)
    a = p.parse_args()

    spec = args_to_run_spec(a)
    spec.method_name = METHOD_NAME

    runner_extra_kwargs = {
        "rscript": str(a.rscript),
        "monocle3_use_partition": bool(a.monocle3_use_partition),
        "monocle3_umap_key": a.monocle3_umap_key,
        "monocle3_prefer_layer": str(a.monocle3_prefer_layer),
        "pt_normalize_01": bool(getattr(a, "pt_normalize_01", False)),
    }
    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()