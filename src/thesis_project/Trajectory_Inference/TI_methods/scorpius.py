#!/usr/bin/env python3
"""
SCORPIUS adapter (R) for TI benchmarking (TIOutput contract).

Fix history
-----------
2026-03-11:
  - Removed erroneous top-level call `adata.write_h5ad(...)` which caused:
      NameError: name 'adata' is not defined
    because `adata` only exists inside ti_runner().

2026-03-17 (root-orientation fix):
  - Added post-hoc pseudotime orientation step.
    SCORPIUS::infer_trajectory() fits a principal curve via a TSP solver
    and assigns arc-length pseudotime in [0, 1] with no concept of a
    biological root. The resulting orientation is arbitrary and is
    equally likely to be root→terminal or terminal→root.

    Fix: after reading pseudotime from the R output, check whether the
    designated root cell sits at pseudotime > 0.5. If so, flip the
    entire pseudotime vector: pt_flipped = pt.max() - pt
    (equivalent to 1 - pt for SCORPIUS output which is already in [0,1]).

  - Added edge orientation correction.
    The R script builds group edges by sorting cluster medians in
    ascending pseudotime order. If the pseudotime was inverted when
    edges.csv was written, the source/target pairs are also inverted.
    When the Python adapter flips pseudotime it also swaps source↔target
    in the edge list to keep topology consistent with the corrected
    pseudotime direction.

2026-04-06:
  - Compatibility with structured traceback logging in method_runner.py:
    raise clear, specific exceptions and preserve R log path in failures.
  - Fixed scorpius_n_dims handling: exported embedding now correctly uses
    min(scorpius_n_dims, X.shape[1]) instead of always passing all dims.

Key features:
- Automatically exports meta.csv from adata.obs[group_key] (no external meta_csv needed).
- Passes --meta_csv and --group_key to scorpius_run.R so it can build cluster-to-cluster edges.
- Efficient topology by default:
    edge_mode=group when group_key exists, else waypoints.
- Post-hoc pseudotime orientation anchored to root_cell_id.

Tmp artifacts:
  <run_dir>/logs/tmp/scorpius/{main|bootstrap_<k>}/...

Requires:
- embedding in adata.obsm[rep_key] (default: X_pca)
- R script: THIS_DIR/R/scorpius_run.R which writes:
    out_dir/pseudotime.csv  (cell_id, pseudotime)
    out_dir/edges.csv       (source, target, weight, directed)
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

# Best-effort setting; don't crash if unsupported in your anndata version.
try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput  # noqa: E402
from TI_benchmark.method_runner import run_benchmark, build_arg_parser, args_to_run_spec  # noqa: E402

logger = logging.getLogger(__name__)
METHOD_NAME = "scorpius"


# ---------------------------------------------------------------------
# CLI (method-specific)
# ---------------------------------------------------------------------
def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("SCORPIUS options")
    g.add_argument("--rscript", default="Rscript", help="Path to Rscript binary")
    g.add_argument(
        "--scorpius-n-dims",
        type=int,
        default=20,
        help="Number of embedding dims passed to SCORPIUS",
    )

    # Topology controls (efficient by default)
    g.add_argument(
        "--scorpius-edge-mode",
        choices=["auto", "group", "waypoints", "cells"],
        default="auto",
        help="Topology construction mode. auto=group if group_key is available, else waypoints.",
    )
    g.add_argument(
        "--scorpius-n-waypoints",
        type=int,
        default=50,
        help="Number of waypoints if edge_mode=waypoints (keeps edges small).",
    )
    g.add_argument(
        "--scorpius-weight-mode",
        choices=["unit", "pseudotime", "pseudotime_scaled", "euclidean"],
        default="unit",
        help="Edge weight mode written by the R script.",
    )
    g.add_argument(
        "--scorpius-directed",
        type=int,
        choices=[0, 1],
        default=0,
        help="Write directed edges? 0=undirected, 1=directed.",
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
            f"SCORPIUS R step failed (exit={p.returncode}). See log: {log_path}"
        )


# ---------------------------------------------------------------------
# Edge schema helper
# ---------------------------------------------------------------------
def _ensure_edge_list_schema(edge_list: Optional[pd.DataFrame]) -> pd.DataFrame:
    if edge_list is None or edge_list.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    df = edge_list.copy()

    if "from" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"from": "source"})
    if "to" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"to": "target"})

    if "source" not in df.columns:
        df["source"] = ""
    if "target" not in df.columns:
        df["target"] = ""

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

    return df[["source", "target", "weight", "directed"]]


# ---------------------------------------------------------------------
# Pseudotime orientation helper
# ---------------------------------------------------------------------
def _orient_pseudotime_to_root(
    pt: pd.Series,
    root_cell_id: str,
    edge_list: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame], bool]:
    """
    Ensure the designated root cell has the minimum pseudotime.
    """
    root_cell_id_str = str(root_cell_id)

    if root_cell_id_str not in pt.index:
        logger.warning(
            "root_cell_id '%s' not found in pseudotime index; skipping orientation check.",
            root_cell_id_str,
        )
        return pt, edge_list, False

    root_pt = float(pt.loc[root_cell_id_str])

    if not np.isfinite(root_pt):
        logger.warning(
            "root_cell_id '%s' has non-finite pseudotime (%s); skipping orientation check.",
            root_cell_id_str, root_pt,
        )
        return pt, edge_list, False

    if root_pt <= 0.5:
        logger.info(
            "SCORPIUS orientation check: root '%s' at pt=%.4f; no flip needed.",
            root_cell_id_str, root_pt,
        )
        return pt, edge_list, False

    logger.warning(
        "SCORPIUS orientation check: root '%s' at pt=%.4f (> 0.5); flipping pseudotime.",
        root_cell_id_str, root_pt,
    )

    pt_max = float(pt.max())
    pt_flipped = pt_max - pt
    pt_flipped.name = pt.name

    el_corrected: Optional[pd.DataFrame] = edge_list
    if edge_list is not None and not edge_list.empty:
        if {"source", "target"}.issubset(edge_list.columns):
            el_corrected = edge_list.copy()
            el_corrected["source"], el_corrected["target"] = (
                edge_list["target"].values.copy(),
                edge_list["source"].values.copy(),
            )
            logger.info("Edge list orientation corrected by swapping source/target.")
        else:
            logger.warning("edge_list missing source/target columns; orientation not corrected.")

    return pt_flipped, el_corrected, True


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
    run_dir: Optional[str] = None,
    rscript: str = "Rscript",
    scorpius_n_dims: int = 20,
    scorpius_edge_mode: str = "auto",
    scorpius_n_waypoints: int = 50,
    scorpius_weight_mode: str = "unit",
    scorpius_directed: int = 0,
    group_key: Optional[str] = None,
    **kwargs: Any,
) -> TIOutput:
    """
    Run SCORPIUS trajectory inference and return a root-oriented TIOutput.
    """
    if rep_key not in adata.obsm:
        raise KeyError(
            f"rep_key '{rep_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}"
        )

    base = Path(run_dir) if run_dir is not None else Path.cwd()
    tag = "main" if bootstrap_index is None else f"bootstrap_{int(bootstrap_index)}"
    tmp_dir = base / "logs" / "tmp" / "scorpius" / tag
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Embedding ─────────────────────────────────────────────────────────────
    X = np.asarray(adata.obsm[rep_key])
    if X.ndim != 2 or X.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Invalid embedding at obsm['{rep_key}']: shape={getattr(X, 'shape', None)}"
        )

    n_dims = int(min(int(scorpius_n_dims), int(X.shape[1])))
    if n_dims < 2:
        raise RuntimeError(
            f"Need at least 2 dims for SCORPIUS; got n_dims={n_dims} from embedding shape={X.shape}"
        )

    emb = pd.DataFrame(X[:, :n_dims], index=adata.obs_names.astype(str))
    emb_path = tmp_dir / "embedding.csv"
    emb.to_csv(emb_path)

    # ── Meta: auto-export from adata.obs[group_key] ───────────────────────────
    group_key_eff = group_key or kwargs.get("group_key") or kwargs.get("cluster_key")

    meta_path: Optional[Path] = None
    if group_key_eff and group_key_eff in adata.obs.columns:
        meta_df = pd.DataFrame({
            "cell_id": adata.obs_names.astype(str),
            group_key_eff: adata.obs[group_key_eff].astype(str).values,
        })
        meta_path = tmp_dir / "meta.csv"
        meta_df.to_csv(meta_path, index=False)
        logger.info("Wrote SCORPIUS meta_csv: %s (group_key=%s)", meta_path, group_key_eff)
    else:
        logger.warning(
            "No valid group_key found for SCORPIUS topology. "
            "group_key=%r kwargs.group_key=%r kwargs.cluster_key=%r. "
            "Falling back to waypoint/cell topology.",
            group_key, kwargs.get("group_key"), kwargs.get("cluster_key"),
        )

    # ── Decide edge_mode ──────────────────────────────────────────────────────
    edge_mode = (scorpius_edge_mode or "auto").lower()
    if edge_mode == "auto":
        edge_mode = "group" if meta_path is not None else "waypoints"
    if edge_mode == "group" and meta_path is None:
        logger.warning("Requested edge_mode=group but meta is missing; using waypoints.")
        edge_mode = "waypoints"

    # ── R call ────────────────────────────────────────────────────────────────
    r_out = tmp_dir / "r_out"
    r_out.mkdir(parents=True, exist_ok=True)

    r_script = THIS_DIR / "R" / "scorpius_run.R"
    log_path = tmp_dir / "scorpius_R.log"

    r_args: Dict[str, Any] = {
        "embedding": emb_path,
        "seed": int(seed),
        "out_dir": r_out,
        "edge_mode": edge_mode,
        "n_waypoints": int(max(2, scorpius_n_waypoints)),
        "weight_mode": str(scorpius_weight_mode),
        "directed": int(scorpius_directed),
        "require_groups": 1 if edge_mode == "group" else 0,
    }
    if meta_path is not None:
        r_args["meta_csv"] = meta_path
        r_args["group_key"] = str(group_key_eff)

    _run_rscript(
        rscript=str(rscript),
        script_path=r_script,
        args=r_args,
        log_path=log_path,
    )

    # ── Read pseudotime ───────────────────────────────────────────────────────
    pt_path = r_out / "pseudotime.csv"
    if not pt_path.exists():
        raise RuntimeError(f"R completed but pseudotime.csv not found at: {pt_path}")

    pt_df = pd.read_csv(pt_path)
    if not {"cell_id", "pseudotime"}.issubset(pt_df.columns):
        raise RuntimeError(
            f"pseudotime.csv missing required columns. Found: {list(pt_df.columns)}"
        )

    pt = pd.Series(
        pd.to_numeric(pt_df["pseudotime"], errors="coerce").astype(float).values,
        index=pt_df["cell_id"].astype(str).values,
        name="pseudotime",
    )

    pt = pt.reindex(adata.obs_names.astype(str))
    pt.index = adata.obs_names

    if pt.isna().any():
        n_bad = int(pt.isna().sum())
        raise RuntimeError(
            f"SCORPIUS pseudotime contains {n_bad} NaN values after aligning to adata.obs_names. "
            f"This usually means mismatched cell IDs between embedding.csv and pseudotime.csv. "
            f"See R log: {log_path}"
        )
    if not np.isfinite(pt.values).all():
        raise RuntimeError(
            f"SCORPIUS pseudotime contains non-finite values (inf/-inf). See R log: {log_path}"
        )

    # ── Read edges ────────────────────────────────────────────────────────────
    edges_path = r_out / "edges.csv"
    edge_list = pd.read_csv(edges_path) if edges_path.exists() else pd.DataFrame()
    edge_list = _ensure_edge_list_schema(edge_list)

    # ── Post-hoc pseudotime orientation ──────────────────────────────────────
    pt, edge_list, was_flipped = _orient_pseudotime_to_root(
        pt=pt,
        root_cell_id=str(root_cell_id),
        edge_list=edge_list,
    )

    if was_flipped:
        logger.info(
            "SCORPIUS pseudotime was flipped to anchor root '%s' at early pseudotime.",
            root_cell_id,
        )
    else:
        logger.info("SCORPIUS pseudotime orientation: no flip required.")

    if not np.isfinite(pt.values).all():
        raise RuntimeError("SCORPIUS pseudotime contains non-finite values after orientation correction.")
    if int(pd.Series(pt.values).nunique()) < 2:
        raise RuntimeError("SCORPIUS pseudotime is constant after orientation correction.")

    return TIOutput(
        pseudotime=pt,
        edge_list=edge_list,
        method_name=METHOD_NAME,
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    p = build_arg_parser()
    add_method_args(p)
    a = p.parse_args()

    spec = args_to_run_spec(a)
    spec.method_name = METHOD_NAME

    runner_extra_kwargs = {
        "rscript": str(a.rscript),
        "scorpius_n_dims": int(a.scorpius_n_dims),
        "scorpius_edge_mode": str(a.scorpius_edge_mode),
        "scorpius_n_waypoints": int(a.scorpius_n_waypoints),
        "scorpius_weight_mode": str(a.scorpius_weight_mode),
        "scorpius_directed": int(a.scorpius_directed),
    }

    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()