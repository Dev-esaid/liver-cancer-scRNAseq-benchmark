#!/usr/bin/env python3
"""
Slingshot adapter (R) for TI benchmarking (TIOutput contract).

Fixes applied (2026-02-28):
1) AnnData export nullable-string issue (Option A):
   set anndata.settings.allow_write_nullable_strings = True early in process.
2) Robust pseudotime: R now uses slingPseudotime(..., na=False) + fallbacks, so
   pseudotime is finite for all cells.
3) Robust MST parsing: slingMST may be igraph or adjacency matrix (handled in R);
   Python still validates edge schema + builds topology_matrix.
4) Optional pseudotime normalization to [0, 1].
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

# -----------------------------
# Option A: Fix AnnData export
# -----------------------------
try:
    # This is exactly what the runtime error recommends.
    ad.settings.allow_write_nullable_strings = True
except Exception:
    # Keep going; benchmark can still run, but exports may fail if obs/var have StringArray dtypes.
    pass

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput
from TI_benchmark.method_runner import run_benchmark, build_arg_parser, args_to_run_spec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CLI (method-specific)
# ---------------------------------------------------------------------
def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Slingshot options")
    g.add_argument("--rscript", default="Rscript", help="Path to Rscript binary")
    g.add_argument("--slingshot-n-dims", type=int, default=20, help="Number of embedding dims passed to Slingshot")
    g.add_argument(
        "--pt-normalize-01",
        action="store_true",
        help="Normalize pseudotime to [0, 1] (recommended for consistent plotting/metrics).",
    )


# ---------------------------------------------------------------------
# R runner
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
    """
    Run an Rscript with --key value args.
    Writes combined stdout/stderr to log_path.
    Raises RuntimeError on non-zero exit.
    """
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

        p = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if p.returncode != 0:
        raise RuntimeError(
            f"Slingshot R step failed (exit={p.returncode}). "
            f"See log: {log_path}"
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _normalize_01(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce").astype(float)
    m = np.isfinite(vals.values)
    if m.sum() == 0:
        return vals
    lo = float(np.nanmin(vals.values[m]))
    hi = float(np.nanmax(vals.values[m]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals
    return (vals - lo) / (hi - lo)


def _validate_pseudotime(pt: pd.Series) -> Tuple[bool, str]:
    if pt is None or len(pt) == 0:
        return False, "pseudotime is empty"
    vals = pd.to_numeric(pt, errors="coerce").astype(float).values
    n_finite = int(np.isfinite(vals).sum())
    if n_finite == 0:
        return False, "pseudotime has no finite values"
    if n_finite != len(vals):
        return False, "pseudotime contains non-finite values"
    return True, "ok"


def _ensure_edge_list_schema(edge_list: Optional[pd.DataFrame]) -> pd.DataFrame:
    if edge_list is None or edge_list.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    df = edge_list.copy()
    if "from" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"from": "source"})
    if "to" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"to": "target"})

    required = {"source", "target"}
    if not required.issubset(df.columns):
        logger.warning("edges.csv missing required columns %s. Found=%s", required, list(df.columns))
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
    return df.reset_index(drop=True)


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
        i = idx.get(s)
        j = idx.get(t)
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
    rep_key: str = "X_pca",
    group_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    run_dir: Optional[str] = None,
    rscript: str = "Rscript",
    slingshot_n_dims: int = 30,
    pt_normalize_01: bool = False,
    **kwargs: Any,
) -> TIOutput:
    gk = group_key or cluster_key
    if gk is None or gk not in adata.obs.columns:
        raise ValueError("Slingshot requires group_key/cluster_key in adata.obs.")
    if rep_key not in adata.obsm:
        raise KeyError(f"rep_key '{rep_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    root_cell_id = str(root_cell_id)
    if root_cell_id not in adata.obs_names.astype(str):
        raise ValueError(f"root_cell_id '{root_cell_id}' not found in adata.obs_names.")

    # Root cluster label
    root_cluster = str(adata.obs.loc[root_cell_id, gk])

    base = Path(run_dir) if run_dir is not None else Path.cwd()
    tag = "main" if bootstrap_index is None else f"bootstrap_{int(bootstrap_index)}"
    tmp_dir = base / "logs" / "tmp" / "slingshot" / tag
    tmp_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(adata.obsm[rep_key])
    if X.ndim != 2 or X.shape[0] != adata.n_obs:
        raise RuntimeError(f"Invalid embedding at obsm['{rep_key}']: shape={getattr(X, 'shape', None)}")

    n_dims = int(X.shape[1])
    if n_dims < 2:
        raise RuntimeError(f"Need at least 2 dims for slingshot; got n_dims={n_dims} from embedding shape={X.shape}")

    # Write inputs for R
    emb = pd.DataFrame(X[:, :n_dims], index=adata.obs_names.astype(str))
    emb_path = tmp_dir / "embedding.csv"
    emb.to_csv(emb_path)

    meta = pd.DataFrame({"cell_id": adata.obs_names.astype(str), gk: adata.obs[gk].astype(str).values})
    meta_path = tmp_dir / "meta.csv"
    meta.to_csv(meta_path, index=False)

    r_out = tmp_dir / "r_out"
    r_out.mkdir(parents=True, exist_ok=True)
    r_script = THIS_DIR / "R" / "slingshot_run.R"
    log_path = tmp_dir / "slingshot_R.log"

    _run_rscript(
        rscript=str(rscript),
        script_path=r_script,
        args={
            "embedding": emb_path,
            "meta": meta_path,
            "cluster_key": gk,
            "root_cluster": root_cluster,
            "seed": int(seed),
            "out_dir": r_out,
        },
        log_path=log_path,
    )

    # Load pseudotime
    pt_path = r_out / "pseudotime.csv"
    if not pt_path.exists():
        raise RuntimeError(f"R completed but pseudotime.csv not found at: {pt_path}")

    pt_df = pd.read_csv(pt_path)
    if not {"cell_id", "pseudotime"}.issubset(pt_df.columns):
        raise RuntimeError(f"pseudotime.csv missing required columns. Found: {list(pt_df.columns)}")

    pt = pd.Series(
        pd.to_numeric(pt_df["pseudotime"], errors="coerce").astype(float).values,
        index=pt_df["cell_id"].astype(str).values,
        name="pseudotime",
    )

    # Align exactly to adata obs_names
    pt = pt.reindex(adata.obs_names.astype(str))
    pt.index = adata.obs_names

    if pt_normalize_01:
        pt = _normalize_01(pt)
        pt.name = "pseudotime"

    ok, reason = _validate_pseudotime(pt)
    if not ok:
        # Attempt one salvage: recompute consensus from per-lineage file if present
        lin_path = r_out / "pseudotime_lineages.csv"
        if lin_path.exists():
            lin = pd.read_csv(lin_path)
            if "cell_id" in lin.columns:
                lin = lin.set_index(lin["cell_id"].astype(str))
                lineage_cols = [c for c in lin.columns if c != "cell_id"]
                if lineage_cols:
                    mat = lin[lineage_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    row_min = np.nanmin(mat, axis=1)
                    pt2 = pd.Series(row_min, index=lin.index, name="pseudotime").reindex(adata.obs_names.astype(str))
                    pt2.index = adata.obs_names
                    ok2, reason2 = _validate_pseudotime(pt2)
                    if ok2:
                        pt = pt2
                        reason = "ok (salvaged from pseudotime_lineages.csv)"

        ok_final, reason_final = _validate_pseudotime(pt)
        if not ok_final:
            raise RuntimeError(reason_final)

    # Load edges
    edges_path = r_out / "edges.csv"
    edge_list = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()
    edge_list = _ensure_edge_list_schema(edge_list)

    topology_matrix = _build_topology_matrix(edge_list)

    return TIOutput(
        pseudotime=pt,
        edge_list=edge_list,
        topology_matrix=topology_matrix,
        method_name="slingshot",
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
    spec.method_name = "slingshot"

    runner_extra_kwargs = {
        "rscript": str(a.rscript),
        "slingshot_n_dims": int(a.slingshot_n_dims),
        "pt_normalize_01": bool(getattr(a, "pt_normalize_01", False)),
    }
    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()